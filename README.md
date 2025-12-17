# AI Based System for Automated Classroom Management


## Setting up

- `pip install -r requirements.txt`
- On your phone, install [IP Camera for Android](https://play.google.com/store/apps/details?id=com.pas.webcam) or any other IP camera app of your choice.
- Connect your phone and machine to the same wifi network.
- Start your IP camera and place its stream URL in `ip_camera_url.txt` (one line, e.g., `http://<ip>:<port>/video`). The scripts now read the URL from this file—no hardcoded changes needed.

## Creating Zones

- Position your camera(phone) into it's intended detection position.
- Execute the `zone_creator.py` script in the terminal, this will open a new window with a picture from camera. Start creating your zones in this window. A zone is any polygon, left click on the screen to mark the points of the zone polygon, after zone polygon is complete, right click to save the polygon. follow these steps to create all the zones, after which, click on the 'save zones' button and close the window.

```
The window might not close and stop responding. Terminate the script from the terminal or kill the process from task manager.
```

## Detection System

- Execute the `yolo_detection.py` script in the terminal; it will open a window with the IP camera stream and show human detections within zones.

```
Wait for the YOLO model to download when you execute the script the first time.
```

- Any zone that has humans in it will turn green.
- A human counts as in a zone when the center of their bounding box is inside that zone polygon.
- To stop the program, terminate the Python script running in the terminal.
