# Smart Classroom Backend

FastAPI backend for teacher/student accounts, attendance tracking, and AI-powered classroom utilities (attendance capture script and smart light control).

## Requirements

- Python 3.10+ recommended
- SQLite (bundled, no extra install needed)
- Recommended: virtual environment (venv/conda)

## Setup

1. Clone or download this repository and open it in a shell.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   **NOTE:** _This will install all the requirements for the input processing also, and might take a long time for complete installation_

## Running the API

- Start the server from the project root:
  ```bash
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
  ```
- Open interactive docs at http://localhost:8000/docs.
- Database is SQLite at `attendance.db` (auto-created on first run).

## Admin Panel (web UI)

- UI lives at [static/admin.html](static/admin.html) and is served from `/` or `/admin` once the server is running.
- Create an admin account first (one-time):
  ```bash
  curl -X POST http://localhost:8000/admin/create \
    -H "Content-Type: application/json" \
    -d '{"name":"Admin","email":"admin@example.com","password":"changeme"}'
  ```
- Log in through the Admin Panel page; it sets an `admin_email` cookie for subsequent actions.
- Available admin actions from the UI:
  - Launch zone creator (opens the local `processing/zone_creator.py`; requires display access on the server).
  - Upload student images into `processing/students/<folder>/` via `/admin/students/upload`.
  - The panel also checks `/admin/me` to restore a session and `/admin/logout` to sign out.

## Data Models (SQLite)

- `teachers(id, name, email, password)`
- `students(id, name, roll_no, email, password)`
- `attendance(id, student_id, roll_no, date, status, captured_at, teacher_email, confidence)`

## REST API (2026)

All request/response bodies are JSON unless otherwise noted.

### Admin Endpoints

- **POST /admin/create**
  - Create a new admin account.
  - Request: `{ "name": str, "email": str, "password": str }`
  - Response: `{ "message": "Admin created successfully" }` or `{ "error": "Admin already exists" }`

- **POST /admin/login**
  - Log in as admin. Sets an `admin_email` cookie.
  - Request: `{ "email": str, "password": str }`
  - Response: `{ "message": "Logged in" }` (cookie set)

- **POST /admin/logout**
  - Log out admin (removes cookie).
  - Response: `{ "message": "Logged out" }`

- **GET /admin/me**
  - Get current admin info (requires cookie).
  - Response: `{ "email": str, "name": str }`

#### Teacher Allowlist (Admin Only)

- **GET /admin/teacher-allowlist**
  - List allowed teacher emails.
  - Response: `[ { "email": str }, ... ]`

- **POST /admin/teacher-allowlist**
  - Add a teacher email to allowlist.
  - Request: `{ "email": str }`
  - Response: `{ "email": str }`

- **DELETE /admin/teacher-allowlist**
  - Remove a teacher email from allowlist.
  - Request: `{ "email": str }`
  - Response: `{ "message": "Email removed from allowlist" }`

#### Admin Utilities

- **POST /admin/zones/create**
  - Launch the zone creator script (for marking classroom zones).
  - Response: `{ "message": "Script executed successfully", "output": str }` or error details.

- **POST /admin/students/upload**
  - Upload student images for face recognition.
  - Form-data: `student_folder` (str), `files` (multiple images)
  - Response: `{ "message": "Images uploaded" }` or error details.

### Teacher & Student Endpoints

- **POST /create-teacher**
  - Register a new teacher (email must be allowlisted).
  - Request: `{ "name": str, "email": str, "password": str }`
  - Response: `{ "message": "Teacher created successfully" }` or `{ "error": "Teacher already exists" }`

- **POST /create-student**
  - Register a new student.
  - Request: `{ "name": str, "roll_no": str, "email": str, "password": str }`
  - Response: `{ "message": "Student created successfully" }` or `{ "error": "Student already exists" }`

- **POST /login**
  - Teacher login.
  - Request: `{ "email": str, "password": str }`
  - Response: `{ "token": str }` (JWT)

- **POST /student/login**
  - Student login.
  - Request: `{ "email": str, "password": str }`
  - Response: `{ "token": str }` (JWT)

### Attendance Endpoints

- **POST /attendance/confirm**
  - Save a student's attendance record.
  - Request: `{ "student_id"?: int, "roll_no"?: str, "date"?: "YYYY-MM-DD", "status": str, "teacher_email": str, "confidence": float, "timestamp"?: ISO8601 }`
  - Response: `{ "message": "Attendance saved" }`

- **GET /attendance/student/{student_id}**
  - Get all attendance records for a student.
  - Response: `[ { ...attendance fields... } ]`

- **GET /attendance/roll/{roll_no}**
  - Get all attendance records by roll number (works even if the student has not registered yet).
  - Response: `[ { ...attendance fields... } ]`

- **POST /attendance/process**
  - Trigger the attendance AI script (face recognition over IP camera).
  - Request: `{ "script_path"?: str, "arguments"?: [str, ...] }`
  - Defaults to `processing/mark_attendance.py` if omitted.
  - Response: `{ "message": "Script executed successfully", "output": str }` or error details.

### Smart Lights (YOLO-based Detector)

- **POST /lights/toggle**
  - Start or stop the YOLO-based light detection script.
  - Request: `{ "status": bool, "script_path"?: str, "arguments"?: [str, ...] }`
  - `status=true` starts detection; `status=false` stops it.
  - Response: `{ "lights_on": bool, "process": "started"|"stopped"|"already running" }`

### Admin Panel (Web UI)

- **GET /** and **GET /admin**
  - Serves the admin panel HTML page.
  - Response: HTML file (not JSON)

---

## API Usage Notes

- All endpoints expect and return JSON unless otherwise specified.
- Admin endpoints require authentication via cookie (set by `/admin/login`).
- Teacher creation requires the email to be allowlisted by an admin.
- Attendance and light control endpoints may trigger scripts; errors are returned as HTTP 500 with details.
- For uploading student images, use multipart/form-data with `student_folder` and one or more image files.
- The `/attendance/process` and `/lights/toggle` endpoints stream script output to the server log; the API response is a short confirmation or error.

For more details, see the interactive docs at `/docs` when the server is running.

## AI Scripts (processing/)

- **IP camera URL**: put the stream URL in `processing/ip_camera_url.txt` (one line, e.g., `http://<ip>:<port>/video`).
- **Zone setup**: run `python processing/zone_creator.py` to draw polygons and save `zones.json`.
- **Detection**: run `python processing/yolo_detection.py` to visualize detections and drive smart lights; first run will download the YOLO weights.
- **Face attendance**: the server's `/attendance/process` calls `processing/mark_attendance.py` (face recognition over IP camera). Script expects labeled face images under `processing/students/<student_name>/`.

## Example Requests (curl)

- Create teacher:
  ```bash
  curl -X POST http://localhost:8000/create-teacher \
    -H "Content-Type: application/json" \
    -d '{"name":"Alice","email":"alice@example.com","password":"secret"}'
  ```
- Teacher login:
  ```bash
  curl -X POST http://localhost:8000/login \
    -H "Content-Type: application/json" \
    -d '{"email":"alice@example.com","password":"secret"}'
  ```
- Confirm attendance:
  ```bash
  curl -X POST http://localhost:8000/attendance/confirm \
    -H "Content-Type: application/json" \
    -d '{"student_id":1,"date":"2024-12-12","status":"present"}'
  ```
- Trigger attendance AI script with extra args:
  ```bash
  curl -X POST http://localhost:8000/attendance/process \
    -H "Content-Type: application/json" \
    -d '{"arguments":["--dry-run","--max-frames","50"]}'
  ```
- Start smart light detection:
  ```bash
  curl -X POST http://localhost:8000/lights/toggle \
    -H "Content-Type: application/json" \
    -d '{"status":true}'
  ```

## Notes

- JWT secret is hardcoded in `auth.py` as `SMARTCLASSROOM`; change it for production.
- `/attendance/process` and `/lights/toggle` stream script output to the server log; client receives a short confirmation JSON.
- If a script fails to start (missing files, bad path), the API returns HTTP 500 with details.
