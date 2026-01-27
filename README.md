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
- `attendance(id, student_id, date, status)`

## REST API

All request/response bodies are JSON.

### Auth & Users

- **POST /create-teacher** — body: `{ name, email, password }` → creates a teacher.
- **POST /create-student** — body: `{ name, roll_no, email, password }` → creates a student.
- **POST /login** — body: `{ email, password }` → `{ token }` (teacher JWT).
- **POST /student/login** — body: `{ email, password }` → `{ token }` (student JWT).

### Attendance

- **POST /attendance/confirm** — body: `{ student_id, date, status }` → stores a record.
- **GET /attendance/student/{student_id}** — returns all records for that student.
- **POST /attendance/process** — trigger the attendance AI script.
  - Body: `{ script_path?: string, arguments?: string[] }`
  - Defaults to `processing/mark_attendance.py` if `script_path` is omitted.
  - Streams script output to the server log; response: `{ message }`.

### Smart Lights (YOLO-based detector)

- **POST /lights/toggle** — start/stop the light detection script.
  - Body: `{ status: boolean, script_path?: string, arguments?: string[] }`
  - `status=true` starts `processing/yolo_detection.py` (or provided `script_path`).
  - `status=false` stops a running process.
  - Response: `{ lights_on: bool, process: "started"|"stopped"|"already running" }`.

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
