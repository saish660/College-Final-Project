"""
Zone Creation Tool
"""

import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import json

# ------------------------------
# CONFIG
# ------------------------------
IP_CAMERA_URL = "http://10.152.211.115:8080/video"

zones = []
current_zone = []


# ------------------------------
# CAPTURE ONE FRAME FROM IP CAMERA
# ------------------------------
def capture_frame(width, height):
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not connect to IP camera")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Error", "Failed to read frame from IP camera")
        return None

    frame = cv2.resize(frame, (width, height))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ------------------------------
# TKINTER WINDOW (FULLSCREEN)
# ------------------------------
root = tk.Tk()
root.title("Zone Creation Tool")

# Make fullscreen
root.attributes("-fullscreen", True)

# Get screen size
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()

# Capture frame according to fullscreen resolution
frame = capture_frame(SCREEN_WIDTH, SCREEN_HEIGHT)
if frame is None:
    exit()

frame_img = Image.fromarray(frame)

# Create Canvas the size of the full screen
canvas = tk.Canvas(root, width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
canvas.pack(fill="both", expand=True)

# Convert image and display it
tk_frame = ImageTk.PhotoImage(frame_img)
canvas.create_image(0, 0, anchor="nw", image=tk_frame)


# ------------------------------
# SAVE ZONES
# ------------------------------
def save_zones():
    if not zones:
        messagebox.showerror("Error", "No zones created.")
        return

    with open("zones.json", "w") as f:
        json.dump(zones, f, indent=4)

    messagebox.showinfo("Saved", "zones.json saved successfully!")


# Save button
save_btn = tk.Button(root, text="Save Zones", font=("Arial", 14), command=save_zones)
save_btn.place(x=20, y=20)  # place button at top-left corner


# ------------------------------
# DRAWING LOGIC
# ------------------------------
def on_left_click(event):
    """Add points to current zone"""
    current_zone.append((event.x, event.y))

    # Small point
    canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3,
                       fill="yellow", outline="")

    # Connect points
    if len(current_zone) > 1:
        x1, y1 = current_zone[-2]
        x2, y2 = current_zone[-1]
        canvas.create_line(x1, y1, x2, y2, fill="yellow", width=2)


def on_right_click(event):
    """Finish zone"""
    global current_zone

    if len(current_zone) < 3:
        messagebox.showwarning("Invalid Zone", "A zone must have at least 3 points.")
        return

    # Close polygon
    x1, y1 = current_zone[-1]
    x2, y2 = current_zone[0]
    canvas.create_line(x1, y1, x2, y2, fill="cyan", width=3)

    zones.append({
        "id": len(zones),
        "name": f"Zone {len(zones)}",
        "polygon": current_zone.copy(),
        "device_id": ""
    })

    current_zone = []


canvas.bind("<Button-1>", on_left_click)
canvas.bind("<Button-3>", on_right_click)


# Press ESC to exit fullscreen
def exit_fullscreen(event):
    root.attributes("-fullscreen", False)


root.bind("<Escape>", exit_fullscreen)

root.mainloop()
