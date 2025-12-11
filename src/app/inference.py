import math
import os
import sys
import threading
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "preprocessing"))

import analysis
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "../models/yolov9m.pt"
WEBCAM_ID = 0

# Try to find the labels directory for the analysis script
POSSIBLE_PATHS = [
    "../../data/train/labels",
    "../data/train/labels",
    "data/train/labels",
    "./data/train/labels",
]
LABELS_DIR = "../../data/train/labels"
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        LABELS_DIR = path
        print(f"Found labels directory at: {LABELS_DIR}")
        break

# Global variables
outputFrame = None
lock = threading.Lock()
conf_threshold = 0.5
app = Flask(__name__)

# Initialize YOLO Model
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def get_error_frame(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Error:",
        (50, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        (50, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def capture_frames():
    global outputFrame, lock, conf_threshold
    print(f"Attempting to open Webcam ID: {WEBCAM_ID}...")
    cap = cv2.VideoCapture(WEBCAM_ID)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        with lock:
            outputFrame = get_error_frame("Error: Could not open webcam")
        return

    print("Webcam opened successfully.")

    while True:
        success, frame = cap.read()
        if not success:
            with lock:
                outputFrame = get_error_frame("Camera disconnected")
            time.sleep(1)
            continue

        if model:
            try:
                results = model(frame, stream=True, conf=conf_threshold, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        current_class = model.names[cls]

                        # Draw Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw Label
                        label = f"{current_class} {conf}"
                        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[
                            0
                        ]
                        c2 = x1 + t_size[0], y1 - t_size[1] - 3
                        cv2.rectangle(frame, (x1, y1), c2, (0, 255, 0), -1, cv2.LINE_AA)
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 2),
                            0,
                            0.5,
                            [255, 255, 255],
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )
            except Exception as e:
                print(f"Inference Error: {e}")

        with lock:
            outputFrame = frame.copy()
        time.sleep(0.01)


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                frame_to_encode = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame_to_encode,
                    "Waiting for Camera...",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
            else:
                frame_to_encode = outputFrame.copy()

        (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode)
        if not flag:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/update_settings", methods=["POST"])
def update_settings():
    global conf_threshold
    data = request.json
    if "conf" in data:
        conf_threshold = float(data["conf"])
    return jsonify({"status": "success"})


@app.route("/dataset_stats.png")
def dataset_stats_img():
    """Calls analysis.py to generate the image."""
    img_buffer = analysis.generate_analysis_image(LABELS_DIR)
    return send_file(img_buffer, mimetype="image/png")


if __name__ == "__main__":
    t = threading.Thread(target=capture_frames)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
