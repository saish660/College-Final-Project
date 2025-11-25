# AI Based System for Automated Classroom Management

# Using the project

## Setting up

- `pip install -r requirements.txt`
- On your phone, install [IP Camera for Android](https://play.google.com/store/apps/details?id=com.pas.webcam) or any other IP camera app of your choice.
- Connect your phone and machine to the same wifi network.
- Type the IP address of your IP camera in the variables of both scripts.

## Creating Zones

- Position your camera(phone) into it's intended detection position.
- Execute the `zone_creator.py` script in the terminal, this will open a new window with a picture from camera. Start creating your zones in this window. A zone is any polygon, left click on the screen to mark the points of the zone polygon, after zone polygon is complete, right click to save the polygon. follow these steps to create all the zones, after which, click on the 'save zones' button and close the window.

```
The window might not close and stop responding. Terminate the script from the terminal or kill the process from task manager.
```

## Detection System

- Execute the `yolo_detection.py` script in the terminal, this will open a new window with the video stream from your ip camera with human detection into zones.

```
Wait for the yolo model to download when you execute the script the first time.
```

- Any zone that has humans in it will turn green.
- For a human to be in a zone, the center of the bounding box of the detected human should be in the zone.
- To stop the program, terminate the python script running in the terminal.
