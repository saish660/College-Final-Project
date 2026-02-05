from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo
import subprocess
import sys
import time
from pathlib import Path
from typing import List
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Response
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models, schemas, auth

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Smart Classroom Backend")

# Smart lights simple in-memory state
light_state = {"on": False}
light_process = None

# Local timezone
IST = ZoneInfo("Asia/Kolkata")

# Fixed script locations (do not auto-guess)
PROJECT_ROOT = Path(__file__).resolve().parent  # smart_classroom_backend
PROCESSING_DIR = PROJECT_ROOT / "processing"
ATTENDANCE_SCRIPT_PATH = PROCESSING_DIR / "mark_attendance.py"
LIGHT_SCRIPT_PATH = PROCESSING_DIR / "yolo_detection.py"
ZONE_CREATOR_SCRIPT_PATH = PROCESSING_DIR / "zone_creator.py"
ADMIN_PAGE_PATH = PROJECT_ROOT / "static" / "admin.html"
ZONE_CREATOR_SCRIPT_PATH = PROCESSING_DIR / "zone_creator.py"

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_admin(request: Request, db: Session = Depends(get_db)):
    """Guard that ensures an admin cookie is present and valid."""
    email = request.cookies.get("admin_email")
    if not email:
        raise HTTPException(status_code=401, detail="Admin login required")

    admin = db.query(models.Admin).filter(models.Admin.email == email).first()
    if not admin:
        raise HTTPException(status_code=401, detail="Admin not found")

    return admin

# ---------------------------
# ADMIN AUTH
# ---------------------------
@app.post("/admin/create")
def create_admin(data: schemas.AdminCreate, db: Session = Depends(get_db)):
    existing_admin = db.query(models.Admin).filter(models.Admin.email == data.email).first()
    if existing_admin:
        return {"error": "Admin already exists"}

    admin = models.Admin(
        name=data.name,
        email=data.email,
        password=auth.hash_password(data.password),
    )
    db.add(admin)
    db.commit()
    return {"message": "Admin created successfully"}


@app.post("/admin/login")
def admin_login(data: schemas.AdminLogin, response: Response, db: Session = Depends(get_db)):
    admin = db.query(models.Admin).filter(models.Admin.email == data.email).first()

    if not admin or not auth.verify_password(data.password, admin.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    response.set_cookie(
        key="admin_email",
        value=admin.email,
        httponly=True,
        samesite="lax",
        max_age=7200,
    )
    return {"message": "Logged in"}


@app.post("/admin/logout")
def admin_logout(response: Response):
    response.delete_cookie("admin_email")
    return {"message": "Logged out"}


@app.get("/admin/me")
def admin_me(admin=Depends(require_admin)):
    return {"email": admin.email, "name": admin.name}


# ---------------------------
# TEACHER ALLOWLIST (ADMIN ONLY)
# ---------------------------
@app.get("/admin/teacher-allowlist", response_model=List[schemas.AllowedTeacherEmail])
def get_teacher_allowlist(admin=Depends(require_admin), db: Session = Depends(get_db)):
    return db.query(models.AllowedTeacherEmail).order_by(models.AllowedTeacherEmail.email).all()


@app.post("/admin/teacher-allowlist", response_model=schemas.AllowedTeacherEmail)
def add_teacher_allowlist(
    data: schemas.AllowedTeacherEmailCreate,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    existing = db.query(models.AllowedTeacherEmail).filter(models.AllowedTeacherEmail.email == data.email).first()
    if existing:
        return existing

    entry = models.AllowedTeacherEmail(email=data.email)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@app.delete("/admin/teacher-allowlist")
def remove_teacher_allowlist(
    data: schemas.AllowedTeacherEmailCreate,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    entry = db.query(models.AllowedTeacherEmail).filter(models.AllowedTeacherEmail.email == data.email).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Email not found in allowlist")

    db.delete(entry)
    db.commit()
    return {"message": "Email removed from allowlist"}


# ---------------------------
# CREATE TEACHER API
# ---------------------------
@app.post("/create-teacher")
def create_teacher(data: schemas.TeacherCreate, db: Session = Depends(get_db)):
    allowed = db.query(models.AllowedTeacherEmail).filter(models.AllowedTeacherEmail.email == data.email).first()
    if not allowed:
        raise HTTPException(status_code=403, detail="Email not allowed for teacher signup")

    # Check if teacher already exists
    existing_teacher = db.query(models.Teacher).filter(
        models.Teacher.email == data.email
    ).first()

    if existing_teacher:
        return {"error": "Teacher already exists"}

    teacher = models.Teacher(
        name=data.name,
        email=data.email,
        password=auth.hash_password(data.password)
    )

    db.add(teacher)
    db.commit()
    return {"message": "Teacher created successfully"}


# ---------------------------
# CREATE STUDENT API
# ---------------------------
@app.post("/create-student")
def create_student(data: schemas.StudentCreate, db: Session = Depends(get_db)):
    # Check if student already exists by email or roll_no
    existing_student = db.query(models.Student).filter(
        (models.Student.email == data.email) | (models.Student.roll_no == data.roll_no)
    ).first()

    if existing_student:
        return {"error": "Student already exists"}

    student = models.Student(
        name=data.name,
        roll_no=data.roll_no,
        email=data.email,
        password=auth.hash_password(data.password)
    )

    db.add(student)
    db.commit()
    return {"message": "Student created successfully"}

# ---------------------------
# LOGIN API
# ---------------------------
@app.post("/login")
def login(data: schemas.Login, db: Session = Depends(get_db)):
    teacher = db.query(models.Teacher).filter(
        models.Teacher.email == data.email
    ).first()

    if not teacher:
        return {"error": "User not found"}

    if not auth.verify_password(data.password, teacher.password):
        return {"error": "Wrong password"}

    token = auth.create_token({"user": teacher.email})
    return {"token": token}


# ---------------------------
# STUDENT LOGIN API
# ---------------------------
@app.post("/student/login")
def student_login(data: schemas.Login, db: Session = Depends(get_db)):
    student = db.query(models.Student).filter(
        models.Student.email == data.email
    ).first()

    if not student:
        return {"error": "Student not found"}

    if not auth.verify_password(data.password, student.password):
        return {"error": "Wrong password"}

    token = auth.create_token({"user": student.email, "role": "student"})
    return {"token": token, "roll_no": student.roll_no}

# ---------------------------
# CONFIRM ATTENDANCE
# ---------------------------
@app.post("/attendance/confirm")
def confirm_attendance(data: schemas.AttendanceConfirm, db: Session = Depends(get_db)):
    attendance_date = data.date
    if attendance_date is None:
        if data.timestamp:
            attendance_date = data.timestamp.astimezone(IST).date()
        else:
            attendance_date = datetime.now(IST).date()
    record = models.Attendance(
        student_id=data.student_id,
        roll_no=data.roll_no,
        date=attendance_date,
        status=data.status,
        captured_at=data.timestamp.astimezone(IST) if data.timestamp else datetime.now(IST),
        teacher_email=data.teacher_email,
        confidence=data.confidence,
    )
    db.add(record)
    db.commit()
    return {"message": "Attendance saved"}

# ---------------------------
# GET STUDENT ATTENDANCE
# ---------------------------
@app.get("/attendance/student/{student_id}")
def get_attendance(student_id: int, db: Session = Depends(get_db)):
    return db.query(models.Attendance).filter(
        models.Attendance.student_id == student_id
    ).all()


@app.get("/attendance/roll/{roll_no}")
def get_attendance_by_roll(roll_no: str, db: Session = Depends(get_db)):
    """Return attendance records using roll number (covers unregistered students)."""
    return (
        db.query(models.Attendance)
        .filter(models.Attendance.roll_no == roll_no)
        .order_by(models.Attendance.date.desc(), models.Attendance.captured_at.desc())
        .all()
    )

def _ensure_script_exists(path: Path, label: str) -> Path:
    script = path.expanduser().resolve()
    if not script.is_file():
        raise HTTPException(status_code=500, detail=f"{label} script missing at {script}")
    return script


def _resolve_script_path(req_path: str | None, default_path: Path) -> Path:
    if not req_path:
        return default_path

    candidate = Path(req_path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _run_script_or_error(script_path: Path, args: list[str]):
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), *args],
            check=True,
            cwd=script_path.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
        return {
            "message": "Script executed successfully",
            "output": result.stdout,
        }
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="", flush=True)
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr, flush=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Script execution failed",
                "returncode": exc.returncode,
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            },
        )


def _stop_light_process():
    global light_process

    # If nothing is running, do nothing
    if light_process is None:
        return "not running"

    # ---- STOP YOLO PROCESS CLEANLY ----
    try:
        if light_process.poll() is None:
            light_process.terminate()
            try:
                light_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                light_process.kill()
                light_process.wait()
    finally:
        light_process = None

    # ---- VERY IMPORTANT (Windows COM port release) ----
    time.sleep(1.5)  # allow Windows to fully release COM4

    # ---- ARDUINO CLEANUP ----
    arduino = None
    try:
        import serial
        import json

        zones_path = (PROCESSING_DIR / "zones.json").resolve()
        if not zones_path.is_file():
            print("⚠️ zones.json not found, skipping Arduino cleanup")
            return "stopped"

        arduino = serial.Serial("COM4", 9600, timeout=1)
        time.sleep(2)  # Arduino reset delay (VERY IMPORTANT)

        with open(zones_path, "r", encoding="utf-8") as f:
            raw_zones = json.load(f)

        print("Shutting down all zones...")
        for z in raw_zones:
            zid = z["id"]
            arduino.write(f"Z{zid}:0\n".encode())
            time.sleep(0.05)  # prevent serial buffer overflow
            print(f"{z.get('name', f'Zone {zid}')} : OFF (forced shutdown)")

        print("✅ Arduino cleanup complete")

    except PermissionError as e:
        print("⚠️ COM4 still busy, cleanup skipped:", e)

    except Exception as e:
        print(f"⚠️ Arduino cleanup error: {e}")

    finally:
        if arduino and arduino.is_open:
            arduino.close()
            time.sleep(0.5)

    return "stopped"


def _start_light_process(script_path: Path, args: list[str]):
    global light_process
    if light_process is not None and light_process.poll() is None:
        return "already running"

    # Run inside the processing directory so relative files (ip_camera_url.txt, zones.json, weights) resolve.
    work_dir = script_path.parent

    light_process = subprocess.Popen(
        [sys.executable, str(script_path), *args],
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=work_dir,
    )

    # If the script dies immediately (e.g., missing ip_camera_url.txt), surface the failure instead of silently returning.
    time.sleep(0.5)
    if light_process.poll() is not None:
        code = light_process.returncode
        light_process = None
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Light script exited immediately",
                "returncode": code,
            },
        )
    return "started"


# ---------------------------
# RUN ATTENDANCE AI SCRIPT (API-triggered)
# ---------------------------
@app.post("/attendance/process")
def run_attendance_ai(req: schemas.AttendanceProcessRequest, db: Session = Depends(get_db)):
    try:
        from processing import mark_attendance
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to load attendance pipeline: {exc}")

    result = mark_attendance.run_pipeline(
        teacher_email=req.teacher_email,
        status=req.status,
        session=db,
    )
    return {"message": "Attendance run completed", **(result or {})}


# ---------------------------
# SMART LIGHTS TOGGLE
# ---------------------------
@app.post("/lights/toggle")
def toggle_lights(req: schemas.LightToggle):
    light_state["on"] = bool(req.status)
    args = req.arguments or []

    if light_state["on"]:
        script_path = _ensure_script_exists(LIGHT_SCRIPT_PATH, "Light")
        state = _start_light_process(script_path, args)
        return {"lights_on": True, "process": state}

    state = _stop_light_process()
    return {"lights_on": False, "process": state}


# ---------------------------
# ADMIN PANEL (UI + ACTIONS)
# ---------------------------
@app.get("/", response_class=FileResponse)
@app.get("/admin", response_class=FileResponse)
def admin_page():
    if not ADMIN_PAGE_PATH.is_file():
        raise HTTPException(status_code=500, detail="Admin page is missing")
    return FileResponse(ADMIN_PAGE_PATH, media_type="text/html")


@app.post("/admin/zones/create")
def trigger_zone_creator(admin=Depends(require_admin)):
    script_path = _ensure_script_exists(ZONE_CREATOR_SCRIPT_PATH, "Zone creator")
    return _run_script_or_error(script_path, [])


@app.post("/admin/students/upload")
async def upload_student_images(
    student_folder: str = Form(...),
    files: List[UploadFile] = File(...),
    admin=Depends(require_admin),
):
    folder = (student_folder or "").strip()
    if not folder:
        raise HTTPException(status_code=400, detail="student_folder is required")

    safe_folder = Path(folder).name
    if not safe_folder:
        raise HTTPException(status_code=400, detail="Invalid folder name")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    dest_dir = PROCESSING_DIR / "students" / safe_folder
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for upload in files:
        filename = Path(upload.filename or "").name
        if not filename:
            continue
        content = await upload.read()
        dest_path = dest_dir / filename
        dest_path.write_bytes(content)
        saved_paths.append(str(dest_path.relative_to(PROJECT_ROOT)))

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    try:
        from processing import mark_attendance

        face_app = mark_attendance.build_face_app()
        embedding_info = mark_attendance.generate_embeddings_for_student(
            roll_no=safe_folder,
            app=face_app,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {exc}")

    return {
        "message": f"Uploaded {len(saved_paths)} file(s)",
        "paths": saved_paths,
        "embeddings": {
            "count": embedding_info["count"],
            "path": str(Path(embedding_info["path"]).relative_to(PROJECT_ROOT)),
        },
    }
