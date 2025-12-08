# YOLO Human Detection (person)

This script uses the `ultralytics` YOLO package to detect humans (COCO class "person").

Requirements
- Python 3.8+
- Install dependencies:

```powershell
pip install -r requirements.txt
```

Basic usage

Run webcam (default model `yolov8n.pt`):

```powershell
python script.py --source 0 --show
```

Run a video file and save output:

```powershell
python script.py --source path\to\video.mp4 --output out.mp4 --show
```

Run a folder of images:

```powershell
python script.py --source c:\path\to\images\ --output out.mp4 --show
```

Notes
- The script filters detections to only the `person` class.
- Press `q` in the display window to stop early.
- If you use a model name like `yolov8n.pt`, the ultralytics package will download it automatically on first use.
