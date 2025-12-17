# Attendance Capture (Face Recognition)

This script captures frames from the IP camera feed, runs InsightFace for detection/embeddings, and tries to match faces against the `students/` gallery.

## Requirements
- Python 3.8+
- Install dependencies:

```powershell
pip install -r requirements.txt
```

## Setup
- Set your IP camera stream URL in `../ip_camera_url.txt` (one line, e.g., `http://<ip>:<port>/video`). The script reads from this file; no code edits needed.
- Populate `students/<student_name>/` folders with reference face images (one or more per student). Images should contain a clear, frontal face.

## Run

```powershell
python script.py
```

The script grabs a small burst of frames from the camera, extracts faces, and matches them to the gallery. Console logs report which student matched (if any) along with basic quality info.

## Notes
- Uses GPU if available (`CUDAExecutionProvider`), otherwise CPU.
- Basic quality gates (blur, face size, pose) are enabled by default; tune thresholds in the config section of `script.py` if needed.
