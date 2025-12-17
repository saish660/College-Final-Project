"""
Zone-based human detection with optional OpenVINO inference.
- If OPENVINO_XML_PATH is set and openvino.runtime is available, uses OpenVINO.
- Otherwise falls back to Ultraytics YOLO Python inference.
"""

import cv2
import threading
import json
import numpy as np
from queue import Queue

# Only consider these class IDs as "humans"
# For COCO-trained YOLO models, person class id is usually 0.
ALLOWED_CLASSES = [0]   # change if your model uses a different id for "person"

IP_CAM_URL_FILE = "ip_camera_url.txt"


def load_ip_cam_url(path=IP_CAM_URL_FILE):
    """Load IP camera URL from a text file; exit if missing/empty."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            url = f.read().strip()
        if not url:
            raise ValueError("IP camera URL file is empty")
        return url
    except Exception as e:
        print(f"❌ Could not load IP camera URL from {path}:", e)
        raise SystemExit(1)


IP_CAM_URL = load_ip_cam_url()

# Optional: enable some debug printing to verify class ids seen
DEBUG_PRINT_CLASSES_SEEN = True
_seen_class_ids = set()


# Try to import OpenVINO runtime
try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except Exception:
    OPENVINO_AVAILABLE = False

# Try to import ultralytics as fallback
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# =======================
# CONFIG
# =======================

# If you have exported an OpenVINO model (xml), set path here.
# Example: "yolov9e_openvino_model/model.xml"
OPENVINO_XML_PATH = "yolov9e_openvino_model/model.xml"  # set to None to skip OpenVINO

# Fallback PyTorch model path (used if OpenVINO not available)
FALLBACK_YOLO_PATH = "yolov9e.pt"

# Inference params
PROCESS_EVERY = 8
# Lowered threshold to improve recall for person detections, especially with OpenVINO raw outputs
CONF_THRESHOLD = 0.30

# =======================
# Initialization
# =======================
cap = cv2.VideoCapture(IP_CAM_URL)
if not cap.isOpened():
    print("❌ Could not open camera:", IP_CAM_URL)
    raise SystemExit(1)

frame_queue = Queue(maxsize=1)
result_lock = threading.Lock()
last_result = None  # unified canonical result object: dict with 'boxes' list of (x1,y1,x2,y2,score,class_id)
running = True

# =======================
# Load zones.json (same as your format)
# =======================
def load_zones(path="zones.json"):
    with open(path, "r") as f:
        data = json.load(f)
    zones = []
    for z in data:
        poly = np.array(z["polygon"], dtype=np.int32)
        if len(poly) > 2 and np.array_equal(poly[0], poly[-1]):
            poly = poly[:-1]
        zones.append({
            "id": z["id"],
            "name": z.get("name", f"Zone {z['id']}"),
            "poly": poly,
            "device_id": z.get("device_id", "")
        })
    return zones

zones = load_zones("zones.json")
print(f"📌 Loaded {len(zones)} zones")

# =======================
# Helper: point-in-polygon
# =======================
def point_in_poly(point, poly):
    return cv2.pointPolygonTest(poly, (int(point[0]), int(point[1])), False) >= 0

# =======================
# OpenVINO loader + inference helper
# =======================
use_openvino = False
ov_core = None
ov_compiled = None
ov_input_shape = None
ov_input_port = None
ov_output_ports = None

# OPENVINO_AVAILABLE = False

if OPENVINO_AVAILABLE and OPENVINO_XML_PATH:
    try:
        ov_core = Core()
        ov_model = ov_core.read_model(OPENVINO_XML_PATH)
        ov_compiled = ov_core.compile_model(ov_model, "CPU")
        # Input/output info (avoid relying on tensor names — some exports have no names)
        ov_input_port = ov_compiled.input(0)
        ov_input_shape = ov_input_port.shape  # e.g. [1, 3, 640, 640]
        ov_output_ports = list(ov_compiled.outputs)
        use_openvino = True
        print("✅ OpenVINO model loaded:", OPENVINO_XML_PATH)
        try:
            # Best-effort prints without requiring names
            print("  input shape:", ov_input_shape, "num_outputs:", len(ov_output_ports))
        except Exception:
            pass
    except Exception as e:
        print("❌ Failed to load OpenVINO model:", e)
        use_openvino = False

# =======================
# Fallback Ultralyitcs model (if not using OpenVINO)
# =======================
yolo_model = None
if not use_openvino:
    if ULTRALYTICS_AVAILABLE:
        try:
            model_to_load = FALLBACK_YOLO_PATH
            print("ℹ️ Using Ultraytics YOLO model as fallback:", model_to_load)
            yolo_model = YOLO(model_to_load)
        except Exception as e:
            print("❌ Could not load Ultraytics model:", e)
            yolo_model = None
    else:
        print("⚠ Neither OpenVINO nor Ultraytics available. Install one of them.")
        raise SystemExit(1)

# =======================
# Utility: convert OpenVINO raw outputs to unified boxes
# (IMPORTANT: exported OpenVINO from Ultralytics often includes NMS and returns [N,6] boxes: [x1,y1,x2,y2,score,class])
# We'll try to handle both cases.
# =======================
def _nms_numpy(xyxy, scores, iou_thresh=0.45):
    """Simple NMS in numpy. xyxy: (N,4), scores: (N,) -> keep indices list"""
    if len(xyxy) == 0:
        return []
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]

    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.zeros_like(inter)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def ov_postprocess(outputs, orig_w, orig_h, input_w, input_h, conf_thresh=CONF_THRESHOLD):
    """
    outputs: can be a dict (keys may be Output objects) or a list/tuple of ndarrays
    returns: list of boxes (x1,y1,x2,y2,score,class_id)
    """
    boxes = []
    # Normalize to iterable of arrays
    if isinstance(outputs, dict):
        arr_iter = outputs.values()
    elif isinstance(outputs, (list, tuple)):
        arr_iter = outputs
    else:
        arr_iter = [outputs]

    for arr in arr_iter:
        arr = np.array(arr)
        if arr.ndim == 2 and arr.shape[1] >= 6:
            # assume format [x1, y1, x2, y2, score, class]
            for row in arr:
                x1, y1, x2, y2, score = row[:5]
                cls = int(row[5]) if row.shape[1] > 5 else 0
                if score >= conf_thresh:
                    # filter by allowed classes (only keep persons)
                    if cls not in ALLOWED_CLASSES:
                        continue
                    boxes.append((float(x1), float(y1), float(x2), float(y2), float(score), int(cls)))
            if len(boxes) > 0:
                return boxes

        # Handle raw Ultralytics OpenVINO exports without NMS, typically shapes:
        # (1, 84, N) or (1, N, 84), where first 4 are xywh and remaining are class confidences
        if arr.ndim == 3 and (arr.shape[1] in (84, 85) or arr.shape[2] in (84, 85)):
            a = arr[0]
            if a.shape[0] in (84, 85):
                a = a.transpose(1, 0)  # (N,84/85)
            # Now a is (N, 84/85)
            if a.shape[1] >= 6:
                # Split into bbox + class scores
                xywh = a[:, :4]
                cls_scores = a[:, 4:]
                # Best class per anchor
                best_cls = np.argmax(cls_scores, axis=1)
                best_conf = cls_scores[np.arange(cls_scores.shape[0]), best_cls]
                # Filter by allowed classes and confidence
                mask_conf = best_conf >= conf_thresh
                if mask_conf.any():
                    xywh = xywh[mask_conf]
                    best_cls = best_cls[mask_conf]
                    best_conf = best_conf[mask_conf]

                    # Keep only allowed classes (e.g., person=0)
                    allowed_mask = np.isin(best_cls, np.array(ALLOWED_CLASSES, dtype=np.int32))
                    if allowed_mask.any():
                        xywh = xywh[allowed_mask]
                        best_cls = best_cls[allowed_mask]
                        best_conf = best_conf[allowed_mask]
                    else:
                        continue

                    # Convert xywh (center-based) to xyxy on model input scale
                    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                    # Scale to original image size
                    sx = float(orig_w) / float(input_w)
                    sy = float(orig_h) / float(input_h)
                    x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy

                    xyxy = np.stack([x1, y1, x2, y2], axis=1)
                    # Apply a light NMS to reduce duplicates
                    keep = _nms_numpy(xyxy, best_conf, iou_thresh=0.5)
                    for i in keep:
                        boxes.append((float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3]), float(best_conf[i]), int(best_cls[i])))
                    if len(boxes) > 0:
                        return boxes

    return boxes


# =======================
# Worker thread
# =======================
def yolo_worker():
    global last_result, running
    while running:
        try:
            frame = frame_queue.get(timeout=1)
        except Exception:
            continue

        try:
            if use_openvino:
                # Preprocess frame to model input
                # ov_input_shape is e.g. [1,3,H,W] or [1,H,W,3] depending on export. We handle common case [1,3,H,W].
                inp = frame.copy()
                inp_h, inp_w = inp.shape[:2]

                # target input size
                _, c, h_in, w_in = ov_input_shape if len(ov_input_shape) == 4 else (1,3,ov_input_shape[1],ov_input_shape[2])

                # Resize and convert BGR -> RGB
                img = cv2.resize(inp, (w_in, h_in))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                # HWC -> CHW
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, 0)

                # Run inference (avoid relying on names)
                # Use a new request to be explicit and robust across OV versions
                try:
                    results = ov_compiled.infer_new_request({ov_input_port: img})
                except Exception:
                    # Fallback: some OV versions support calling compiled model directly with dict
                    results = ov_compiled({ov_input_port: img})

                # Normalize outputs to a simple list in the same order as ov_output_ports
                outputs = []
                try:
                    for port in ov_output_ports:
                        outputs.append(results[port])
                except Exception:
                    # If indexing by port fails, try values()
                    if isinstance(results, dict):
                        outputs = list(results.values())
                    elif isinstance(results, (list, tuple)):
                        outputs = list(results)
                    else:
                        outputs = [results]

                # Try to postprocess outputs to boxes
                boxes = ov_postprocess(outputs, inp_w, inp_h, w_in, h_in, conf_thresh=CONF_THRESHOLD)
                # boxes: list of (x1,y1,x2,y2,score,class)
                unified = []
                for b in boxes:
                    x1, y1, x2, y2, score, cls = b
                    unified.append({"xyxy": (x1, y1, x2, y2), "conf": float(score), "cls": int(cls)})
                with result_lock:
                    last_result = {"boxes": unified}
            else:
                # Ultralyitcs fallback: leverages model to return structured result
                res = yolo_model(frame)  # returns list of Results; pick first

                # debug: print/inspect model class names (Ultralytics exposes model.names)
                try:
                    model_names = getattr(yolo_model, "names", None)
                    if DEBUG_PRINT_CLASSES_SEEN and model_names is not None:
                        print("Ultralytics model class mapping sample (id:name):")
                        # print only person id mapping + few entries for sanity
                        for i in sorted(list(set(ALLOWED_CLASSES))):
                            if i < len(model_names):
                                print(f"  {i}: {model_names[i]}")
                except Exception as _e:
                    print("Could not print model names:", _e)

                # Convert Ultralyitcs result into unified simple dict, but only keep allowed classes
                unified = []
                try:
                    r = res[0]  # result object
                    # quick debug: how many raw boxes returned
                    raw_box_count = len(getattr(r, "boxes", []))
                    if DEBUG_PRINT_CLASSES_SEEN:
                        print("Ultralytics raw boxes count:", raw_box_count)

                    for box in r.boxes:
                        # xyxy are tensors; convert to numpy
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf[0], "cpu") else float(box.conf[0])
                        cls = int(box.cls[0].cpu().numpy()) if hasattr(box.cls[0], "cpu") else int(box.cls[0])

                        # record seen class for debugging
                        if DEBUG_PRINT_CLASSES_SEEN:
                            _seen_class_ids.add(cls)

                        if conf < CONF_THRESHOLD:
                            continue
                        # filter to only person / allowed classes
                        if cls not in ALLOWED_CLASSES:
                            continue

                        unified.append({"xyxy": (float(x1), float(y1), float(x2), float(y2)), "conf": conf, "cls": cls})

                    # IMPORTANT: write unified results back to last_result inside lock
                    with result_lock:
                        last_result = {"boxes": unified}
                except Exception as e:
                    print("Ultralytics branch postprocess error:", e)


        except Exception as e:
            print("Inference error:", e)


threading.Thread(target=yolo_worker, daemon=True).start()

# =======================
# Window / loop
# =======================
cv2.namedWindow("Zone-Based Human Detection", cv2.WINDOW_NORMAL)
screen = cv2.getWindowImageRect("Zone-Based Human Detection")
sw, sh = screen[2], screen[3]

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        frame_count += 1
        if frame_count % PROCESS_EVERY == 0:
            if not frame_queue.full():
                frame_queue.put(frame.copy())

        # Read last_result safely
        with result_lock:
            res = last_result

        annotated = frame.copy()
        zone_occupancy = {z["id"]: False for z in zones}

        if res and "boxes" in res:
            for b in res["boxes"]:
                x1, y1, x2, y2 = b["xyxy"]
                conf = b.get("conf", 0.0)
                cls = b.get("cls", 0)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # draw detection
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 2)
                cv2.circle(annotated, (cx, cy), 4, (0,255,0), -1)
                cv2.putText(annotated, f"{conf:.2f}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # map to zones
                for z in zones:
                    if point_in_poly((cx, cy), z["poly"]):
                        zone_occupancy[z["id"]] = True
                        break

        # Draw zones
        for z in zones:
            poly = z["poly"]
            occupied = zone_occupancy[z["id"]]
            color = (0,255,0) if occupied else (0,0,255)
            cv2.polylines(annotated, [poly], True, color, 3)
            cx = int(np.mean(poly[:,0])); cy = int(np.mean(poly[:,1]))
            cv2.putText(annotated, z["name"], (cx-40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Zone-Based Human Detection", annotated)
        print(zone_occupancy)

        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            running = False
            break

except KeyboardInterrupt:
    running = False
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete")
