import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import cv2
import math
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from database import SessionLocal
import models

IP_CAM_URL_FILE = BASE_DIR / "ip_camera_url.txt"


def load_ip_cam_url(path=IP_CAM_URL_FILE):
    """Load IP camera URL from a text file; exit if missing/empty."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            url = f.read().strip()
        if not url:
            raise ValueError("IP camera URL file is empty")
        return url
    except Exception as e:
        print(f"ERROR: Could not load IP camera URL from {path}:", e)
        raise SystemExit(1)


IP_CAMERA_VIDEO_URL = load_ip_cam_url()


STUDENTS_DIR = BASE_DIR / "students"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
FRAMES_PER_TRIGGER = 5           # capture 3–5 frames per button press
MIN_FACE_SIZE = 40               # px (bare-minimum gate)
LAPLACIAN_VAR_THRESHOLD = 20.0   # lower = blurrier (bare-minimum gate)
MAX_POSE_DEG = 45.0              # allow larger pose range (bare-minimum gate)
QUALITY_CHECK_ENABLED = True     # set False to skip all quality gating
STRICT_QUALITY = False           # when True, drop faces that fail quality; when False, keep but mark
COSINE_MEAN_THRESHOLD = 0.40
COSINE_MAX_THRESHOLD = 0.45
TOPK_MEAN = 3                    # mean over top-K similarities
ENABLE_LIVENESS = False          # liveness disabled
LIVENESS_YAW_RANGE_MIN = 5.0
LIVENESS_YAW_RANGE_MAX = 35.0
SAME_FACE_MERGE_THRESHOLD = 0.65 # merge detections of the same person across frames

# GPU first, CPU fallback
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Local timezone for timestamping
IST = ZoneInfo("Asia/Kolkata")


# ------------------------------
# Helpers
# ------------------------------
def build_face_app() -> FaceAnalysis:
    """Load buffalo_l (SCRFD + ArcFace) with GPU-first fallback."""
    for provider in PROVIDERS:
        try:
            app = FaceAnalysis(name="buffalo_l", providers=[provider])
            ctx_id = 0 if provider == "CUDAExecutionProvider" else -1
            app.prepare(ctx_id=ctx_id)
            print(f"READY: FaceAnalysis ready with provider={provider}")
            return app
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: Failed to load provider {provider}: {exc}")
    raise RuntimeError("No valid onnxruntime provider available")


def capture_frames(cap: cv2.VideoCapture, num_frames: int) -> List[np.ndarray]:
    frames = []
    for i in range(num_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
            print(f"CAPTURED FRAME: {i+1}/{num_frames}")
        else:
            print("Failed to read frame from camera")
        time.sleep(0.02)
    return frames


def laplacian_var(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_pose_deg(kps: np.ndarray) -> Tuple[float, float]:
    """
    Rough yaw/pitch from 5-point landmarks.
    kps: array shape (5,2) -> [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    left_eye, right_eye, nose, left_mouth, right_mouth = kps
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0

    # Yaw: horizontal asymmetry of nose vs eyes
    yaw_num = (right_eye[0] - nose[0]) - (nose[0] - left_eye[0])
    yaw_den = (right_eye[0] - left_eye[0]) + 1e-6
    yaw = math.degrees(math.atan2(yaw_num, yaw_den))

    # Pitch: vertical balance between eyes, nose, mouth
    pitch_num = (nose[1] - eye_center[1]) - (mouth_center[1] - nose[1])
    pitch_den = (mouth_center[1] - eye_center[1]) + 1e-6
    pitch = math.degrees(math.atan2(pitch_num, pitch_den))

    return yaw, pitch


def passes_quality(face, image: np.ndarray) -> Tuple[bool, dict]:
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False, {"reason": "small"}

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False, {"reason": "empty_crop"}

    blur_score = laplacian_var(crop)
    yaw, pitch = estimate_pose_deg(face.kps)

    if blur_score < LAPLACIAN_VAR_THRESHOLD:
        return False, {"reason": "blurry", "blur": blur_score, "yaw": yaw, "pitch": pitch}

    return True, {"blur": blur_score, "yaw": yaw, "pitch": pitch}


def align_face(image: np.ndarray, face) -> np.ndarray:
    # 112x112 ArcFace standard crop using 5-point landmarks
    return face_align.norm_crop(image, landmark=face.kps, image_size=112)


def _load_precomputed_embeddings() -> Dict[str, List[np.ndarray]]:
    """Load saved embeddings from disk to avoid recomputing each run."""
    db: Dict[str, List[np.ndarray]] = {}
    if not EMBEDDINGS_DIR.is_dir():
        return db

    for npz_path in EMBEDDINGS_DIR.glob("*.npz"):
        roll_no = npz_path.stem
        try:
            data = np.load(npz_path)
            embs = data.get("embeddings")
            if embs is None:
                print(f"WARNING: Missing 'embeddings' key in {npz_path.name}; skipping")
                continue
            embs = np.asarray(embs, dtype="float32")
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
            db[roll_no] = [emb for emb in embs]
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: Failed to load embeddings for {roll_no} from {npz_path.name}: {exc}")
    if db:
        total = sum(len(v) for v in db.values())
        print(f"Loaded {total} precomputed embeddings for {len(db)} students from {EMBEDDINGS_DIR}")
    return db


def _load_student_db_from_images(app: FaceAnalysis) -> Dict[str, List[np.ndarray]]:
    """Legacy path: compute embeddings directly from student images."""
    db: Dict[str, List[np.ndarray]] = {}
    for student_dir in STUDENTS_DIR.iterdir():
        if not student_dir.is_dir():
            continue
        roll_no = student_dir.name
        for fpath in student_dir.iterdir():
            if not fpath.is_file():
                continue
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            faces = app.get(img)
            if len(faces) == 0:
                print(f"ERROR: No face found for {roll_no} in {fpath.name}")
                continue
            face = faces[0]
            emb = face.normed_embedding.astype("float32")
            db.setdefault(roll_no, []).append(emb)
    print(f"Loaded {sum(len(v) for v in db.values())} embeddings for {len(db)} students (computed from images)")
    return db


def load_student_db(app: FaceAnalysis) -> Dict[str, List[np.ndarray]]:
    """Load embeddings, preferring precomputed cache over on-the-fly computation."""
    db = _load_precomputed_embeddings()
    if db:
        return db

    print("No precomputed embeddings found; computing from student images")
    return _load_student_db_from_images(app)


def cosine_similarity_matrix(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    # queries: MxD, gallery: NxD, both already L2-normalized
    return queries @ gallery.T


def match_student(query_embs: List[np.ndarray], db: Dict[str, List[np.ndarray]]):
    if not query_embs:
        return None, {"reason": "no_query"}

    q = np.stack(query_embs).astype("float32")
    best_name, best_max = None, -1.0
    for name, embs in db.items():
        g = np.stack(embs).astype("float32")
        sims = cosine_similarity_matrix(q, g)  # shape MxN
        max_sim = float(sims.max())
        flat = sims.flatten()
        topk = min(TOPK_MEAN, flat.size)
        mean_topk = float(np.partition(flat, -topk)[-topk:].mean())

        if max_sim >= COSINE_MAX_THRESHOLD and mean_topk >= COSINE_MEAN_THRESHOLD:
            if max_sim > best_max:
                best_max = max_sim
                best_name = name

    if best_name is None:
        return None, {"reason": "threshold"}
    return best_name, {"max_sim": best_max}


def liveness_head_turn(yaws: List[float]) -> bool:
    if not yaws:
        return False
    yaw_range = max(yaws) - min(yaws)
    return LIVENESS_YAW_RANGE_MIN <= yaw_range <= LIVENESS_YAW_RANGE_MAX


def cluster_candidates(candidates: List[dict], threshold: float) -> List[dict]:
    """Merge detections that likely belong to the same person based on embedding similarity."""
    clusters: List[dict] = []
    for cand in candidates:
        emb = cand["embedding"]
        placed = False
        for cluster in clusters:
            centroid = cluster["centroid"]
            sim = float(np.dot(centroid, emb))
            if sim >= threshold:
                cluster["items"].append(cand)
                # Recompute centroid to stay representative of the cluster
                stack = np.stack([item["embedding"] for item in cluster["items"]])
                centroid = stack.mean(axis=0)
                norm = np.linalg.norm(centroid) + 1e-6
                cluster["centroid"] = (centroid / norm).astype("float32")
                placed = True
                break
        if not placed:
            clusters.append({"items": [cand], "centroid": emb})
    return clusters


def load_student_lookup(session):
    students = session.query(models.Student).all()
    return {student.roll_no: student for student in students}


def record_attendance(
    session,
    roll_no: str,
    teacher_email: str,
    confidence: float,
    status: str,
    student_id: Optional[int] = None,
):
    now = datetime.now(IST)
    session.add(
        models.Attendance(
            student_id=student_id,
            roll_no=roll_no,
            date=now.date(),
            status=status,
            captured_at=now,
            teacher_email=teacher_email,
            confidence=confidence,
        )
    )


def generate_embeddings_for_student(roll_no: str, app: FaceAnalysis | None = None) -> dict:
    """Compute and persist embeddings for a single student's folder."""
    student_dir = STUDENTS_DIR / roll_no
    if not student_dir.is_dir():
        raise FileNotFoundError(f"Student folder not found: {student_dir}")

    own_app = False
    if app is None:
        app = build_face_app()
        own_app = True

    embeddings: List[np.ndarray] = []
    for fpath in sorted(student_dir.iterdir()):
        if not fpath.is_file():
            continue
        img = cv2.imread(str(fpath))
        if img is None:
            print(f"WARNING: Could not read image {fpath.name} for {roll_no}; skipping")
            continue
        faces = app.get(img)
        if not faces:
            print(f"WARNING: No face detected in {fpath.name} for {roll_no}; skipping")
            continue
        embeddings.append(faces[0].normed_embedding.astype("float32"))

    if own_app:
        app = None

    if not embeddings:
        raise ValueError(f"No embeddings generated for {roll_no}; ensure images contain faces")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EMBEDDINGS_DIR / f"{roll_no}.npz"
    np.savez_compressed(out_path, embeddings=np.stack(embeddings, axis=0))
    print(f"Saved {len(embeddings)} embeddings for {roll_no} -> {out_path}")
    return {"count": len(embeddings), "path": out_path}


# ------------------------------
# Main pipeline
# ------------------------------
def run_once(
    app: FaceAnalysis,
    student_db: Dict[str, List[np.ndarray]],
    session,
    teacher_email: str,
    status: str = "present",
):
    cap = cv2.VideoCapture(IP_CAMERA_VIDEO_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at {IP_CAMERA_VIDEO_URL}")

    frames = capture_frames(cap, FRAMES_PER_TRIGGER)
    cap.release()

    candidates = []
    for frame in frames:
        faces = app.get(frame)
        for face in faces:
            if QUALITY_CHECK_ENABLED:
                ok, info = passes_quality(face, frame)
                info["quality_ok"] = ok
                if not ok:
                    if STRICT_QUALITY:
                        print(f"WARNING: Rejected by quality: {info}")
                        continue
                    else:
                        print(f"WARNING: Quality fail but keeping (STRICT_QUALITY=False): {info}")
            else:
                # Collect metrics but do not gate
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                blur_score = laplacian_var(crop) if crop.size > 0 else 0.0
                yaw, pitch = estimate_pose_deg(face.kps)
                info = {"blur": blur_score, "yaw": yaw, "pitch": pitch, "quality_ok": True}
            aligned = align_face(frame, face)
            emb = face.normed_embedding.astype("float32")
            blur_val = info.get("blur", 0.0)
            yaw_val = info.get("yaw", 0.0)
            pitch_val = info.get("pitch", 0.0)
            candidates.append({
                "aligned": aligned,
                "embedding": emb,
                "blur": blur_val,
                "yaw": yaw_val,
                "pitch": pitch_val,
                "quality_ok": info.get("quality_ok", True),
            })

    if not candidates:
        print("ERROR: No faces passed detection/quality")
        return

    clusters = cluster_candidates(candidates, SAME_FACE_MERGE_THRESHOLD)

    attendance: Dict[str, Dict[str, float]] = {}
    for cluster in clusters:
        embs = [item["embedding"] for item in cluster["items"]]
        matched, meta = match_student(embs, student_db)
        if matched is None:
            continue
        current = attendance.get(matched)
        if current is None or meta["max_sim"] > current["max_sim"]:
            attendance[matched] = {"max_sim": meta["max_sim"], "samples": len(embs)}

    if not attendance:
        print("ERROR: No matches across detected faces")
        return

    student_lookup = load_student_lookup(session)
    saved = 0
    records = []
    missing_students = 0
    for roll_no, meta in attendance.items():
        student = student_lookup.get(roll_no)
        if not student:
            print(
                f"WARNING: No student found with roll number {roll_no}; storing attendance with roll_no only"
            )
            missing_students += 1

        record_attendance(
            session=session,
            roll_no=roll_no,
            student_id=student.id if student else None,
            teacher_email=teacher_email,
            confidence=meta["max_sim"],
            status=status,
        )
        saved += 1
        records.append(
            {
                "roll_no": roll_no,
                "student_id": student.id if student else None,
                "name": student.name if student else None,
                "confidence": meta["max_sim"],
                "samples": meta["samples"],
                "status": status,
            }
        )
        print(
            f"SUCCESS: Marked attendance for {roll_no} "
            f"(max_sim={meta['max_sim']:.3f}, samples={meta['samples']})"
        )

    if saved:
        session.commit()
        print(f"INFO: Saved {saved} attendance record(s) to the database (missing students: {missing_students})")
    return {"saved": saved, "records": records, "missing_students": missing_students}


def run_pipeline(teacher_email: str, status: str = "present", session=None):
    """Entry point for API callers to capture and store attendance.

    Builds the face model, loads embeddings, runs detection once, and writes
    recognized students to the database with teacher email and confidence.
    """
    own_session = False
    if session is None:
        session = SessionLocal()
        own_session = True

    # Ensure tables exist when running standalone (roll_no column is required).
    models.Base.metadata.create_all(bind=session.get_bind())

    try:
        face_app = build_face_app()
        student_db = load_student_db(face_app)
        return run_once(
            app=face_app,
            student_db=student_db,
            session=session,
            teacher_email=teacher_email,
            status=status,
        )
    finally:
        if own_session:
            session.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run attendance capture and store results")
    parser.add_argument(
        "--teacher-email",
        help="Email ID of the teacher capturing attendance (for manual/debug runs)",
    )
    parser.add_argument(
        "--status",
        default="present",
        help="Attendance status label to store (default: present)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.teacher_email:
        raise SystemExit("--teacher-email is required when running manually")
    run_pipeline(teacher_email=args.teacher_email, status=args.status)
