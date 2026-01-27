import cv2
import json
import numpy as np
from ultralytics import YOLO

# =======================
# CONFIG
# =======================
IP_CAM_URL_FILE = "ip_camera_url.txt"
ZONES_FILE = "zones.json"
MODEL_PATH = "yolov9e.pt"

CONF_THRESHOLD = 0.40
PROCESS_EVERY = 30
PERSON_CLASS_ID = 0  # COCO person class

# =======================
# Load IP Camera URL
# =======================
with open(IP_CAM_URL_FILE, "r") as f:
    IP_CAM_URL = f.read().strip()

cap = cv2.VideoCapture(IP_CAM_URL)
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit(1)

# =======================
# Load Zones
# =======================
with open(ZONES_FILE, "r") as f:
    raw_zones = json.load(f)

zones = []
for z in raw_zones:
    zones.append({
        "id": z["id"],
        "name": z.get("name", f"Zone {z['id']}"),
        "poly": np.array(z["polygon"], dtype=np.int32)
    })

print(f"📌 Loaded {len(zones)} zones")

# Track previous occupancy to only signal on changes
prev_zone_occupancy = {z["id"]: False for z in zones}

# =======================
# Load YOLO Model
# =======================
model = YOLO(MODEL_PATH)

# =======================
# Point in Polygon
# =======================
def point_in_poly(point, poly):
    return cv2.pointPolygonTest(poly, point, False) >= 0

# =======================
# Main Loop
# =======================
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera read failed")
            break

        frame_count += 1
        if frame_count % PROCESS_EVERY != 0:
            continue

        results = model(frame, classes=[PERSON_CLASS_ID], conf=CONF_THRESHOLD)[0]

        # Initialize zone states
        zone_occupancy = {z["id"]: False for z in zones}

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != PERSON_CLASS_ID or conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            for z in zones:
                if point_in_poly((cx, cy), z["poly"]):
                    zone_occupancy[z["id"]] = True

        # =======================
        # Terminal Output (Lights)
        # =======================
        changed = any(zone_occupancy[z["id"]] != prev_zone_occupancy[z["id"]] for z in zones)
        if changed:
            print("\nZone Status (changed):")
            for z in zones:
                state = "💡 ON" if zone_occupancy[z["id"]] else "❌ OFF"
                print(f"{z['name']} : {state}")
            prev_zone_occupancy = zone_occupancy.copy()

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

finally:
    cap.release()
    print("✅ Shutdown complete")