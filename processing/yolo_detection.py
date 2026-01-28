import cv2
import json
import numpy as np
from ultralytics import YOLO
import serial
import time

# =======================
# CONFIG
# =======================
IP_CAM_URL_FILE = "ip_camera_url.txt"
ZONES_FILE = "zones.json"
MODEL_PATH = "yolov9e.pt"

CONF_THRESHOLD = 0.40
PROCESS_EVERY = 30
PERSON_CLASS_ID = 0  # COCO person class

SERIAL_PORT = "COM4"
BAUD_RATE = 9600

OFF_DELAY_SECONDS = 4.0  # ⏳ delay before turning OFF lights

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
# Connect to Arduino
# =======================
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(1)
    print("ARDUINO CONNECTED")
except Exception as e:
    print("ERROR CONNECTING TO ARDUINO:", e)
    arduino = None

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

# =======================
# State Tracking
# =======================
prev_zone_state = {z["id"]: False for z in zones}
last_active_time = {z["id"]: 0.0 for z in zones}

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

        # Current detection state
        zone_detected = {z["id"]: False for z in zones}

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
                    zone_detected[z["id"]] = True

        now = time.time()

        # =======================
        # Light Control Logic
        # =======================
        for z in zones:
            zid = z["id"]
            detected = zone_detected[zid]
            was_on = prev_zone_state[zid]

            # 🔆 Person detected → ON immediately
            if detected:
                last_active_time[zid] = now
                if not was_on:
                    if arduino:
                        arduino.write(f"Z{zid}:1\n".encode())
                    print(f"{z['name']} : ON")
                    prev_zone_state[zid] = True

            # ⏳ No person → delayed OFF
            else:
                if was_on and (now - last_active_time[zid]) >= OFF_DELAY_SECONDS:
                    if arduino:
                        arduino.write(f"Z{zid}:0\n".encode())
                    print(f"{z['name']} : OFF")
                    prev_zone_state[zid] = False

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

finally:
    cap.release()
    if arduino:
        arduino.close()
    print("✅ Shutdown complete")
