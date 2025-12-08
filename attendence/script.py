import cv2
import numpy as np
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis
import faiss
import os

IP_CAMERA_VIDEO_URL = "http://192.168.137.120:8080/video"
STUDENTS_DIR = "students/"
YOLO_WEIGHTS = r"C:\Users\SANIYA\Desktop\CollegeProject\attendence\yolov8n.pt"

NUM_FRAMES = 5

# Load YOLO detector
face_detector = YOLO(YOLO_WEIGHTS)

# Load ArcFace
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# -------- BUILD STUDENT DATABASE --------
student_names = []
student_embeddings = []

for student_name in os.listdir(STUDENTS_DIR):
    student_folder = os.path.join(STUDENTS_DIR, student_name)
    if not os.path.isdir(student_folder):
        continue

    for img_name in os.listdir(student_folder):
        img_path = os.path.join(student_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = face_app.get(img)
        if len(faces) == 0:
            print("No face found in:", img_path)
            continue

        emb = faces[0].normed_embedding
        student_names.append(student_name)
        student_embeddings.append(emb)

print("Loaded students:", student_names)

student_embeddings = np.array(student_embeddings).astype("float32")

index = faiss.IndexFlatL2(student_embeddings.shape[1])
index.add(student_embeddings)


# -------- CAPTURE FRAMES --------
cap = cv2.VideoCapture(IP_CAMERA_VIDEO_URL)
captured_images = []

for i in range(NUM_FRAMES):
    ret, img = cap.read()
    if ret:
        captured_images.append(img)
        print(f"Captured frame {i+1}/{NUM_FRAMES}")
    else:
        print("Failed to capture frame")

cap.release()


# -------- RECOGNITION --------
recognized_students = set()

for img in captured_images:

    results = face_detector(img)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Enlarge crop margin
        h, w = img.shape[:2]
        margin = 25
        x1m = max(0, x1 - margin)
        y1m = max(0, y1 - margin)
        x2m = min(w, x2 + margin)
        y2m = min(h, y2 + margin)

        face_crop = img[x1m:x2m, y1m:y2m]

        if face_crop.shape[0] < 120 or face_crop.shape[1] < 120:
            print("Face too small, skipping")
            continue

        faces = face_app.get(face_crop)
        if len(faces) == 0:
            print("ArcFace could not extract face")
            continue

        emb = faces[0].normed_embedding.astype("float32")

        D, I = index.search(np.array([emb]), 1)
        distance = D[0][0]
        matched_idx = I[0][0]

        print("Distance:", distance)

        if distance < 1.6:  # relaxed threshold
            recognized_students.add(student_names[matched_idx])


# -------- OUTPUT --------
print("\nRecognized Students:")
if len(recognized_students) == 0:
    print("❌ No faces recognized.")
else:
    for s in recognized_students:
        print("✔", s)
