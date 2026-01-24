from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models, schemas, auth

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Smart Classroom Backend")

# Smart lights simple in-memory state
light_state = {"on": False}
light_process = None

# Fixed script locations (do not auto-guess)
PROJECT_ROOT = Path(__file__).resolve().parent  # smart_classroom_backend
PROCESSING_DIR = PROJECT_ROOT / "processing"
ATTENDANCE_SCRIPT_PATH = PROCESSING_DIR / "mark_attendance.py"
LIGHT_SCRIPT_PATH = PROCESSING_DIR / "yolo_detection.py"

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------
# CREATE TEACHER API
# ---------------------------
@app.post("/create-teacher")
def create_teacher(data: schemas.TeacherCreate, db: Session = Depends(get_db)):
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
    return {"token": token}

# ---------------------------
# CONFIRM ATTENDANCE
# ---------------------------
@app.post("/attendance/confirm")
def confirm_attendance(data: schemas.AttendanceConfirm, db: Session = Depends(get_db)):
    record = models.Attendance(
        student_id=data.student_id,
        date=data.date,
        status=data.status
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
    if light_process is None:
        return "not running"

    if light_process.poll() is None:
        light_process.terminate()
        try:
            light_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            light_process.kill()
    light_process = None
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
# RUN ATTENDANCE AI SCRIPT
# ---------------------------
@app.post("/attendance/process")
def run_attendance_ai(req: schemas.AttendanceScriptTrigger):
    # Resolve and validate script path in one step
    resolved_path = _resolve_script_path(req.script_path, ATTENDANCE_SCRIPT_PATH)
    script_path = _ensure_script_exists(resolved_path, "Attendance")
    args = req.arguments or []

    return _run_script_or_error(script_path, args)


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
