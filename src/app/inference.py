import base64
import datetime
import math
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

# Logger setup
from logger import get_logger
from ultralytics import YOLO

log = get_logger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "preprocessing"))

# Analysis module import
import analysis

# --- Configuration ---
MODEL_PATH = "../models/optimized_model.pt"
WEBCAM_ID = 0
LOG_FILE_PATH = "logs/app_activity.log"

# Try to find the labels directory
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
        log.info(f"Found labels directory at: {LABELS_DIR}")
        break

# Global variables
outputFrame = None
lock = threading.Lock()
conf_threshold = 0.5
inference_enabled = True
app = Flask(__name__)

# Store recent detections for the UI log (Max 50 items)
detection_history = deque(maxlen=50)

# Initialize YOLO Model
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    log.info("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    log.error(f"Error loading model: {e}")
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


def process_frame(frame, conf_thresh):
    """
    Runs inference on a frame and draws bounding boxes.
    Returns the annotated frame.
    """
    if model is None:
        return frame

    try:
        # Run inference
        results = model(frame, conf=conf_thresh, verbose=False)

        current_detections = []

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
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
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

                # Add to current detections list
                detection_info = {
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "class": current_class,
                    "conf": f"{conf:.2f}",
                }
                current_detections.append(detection_info)

        # Update global history with unique detections from this frame (to avoid spamming the log)
        # We assume if the class and conf are identical to the last entry, it's the same object
        for det in current_detections:
            if not detection_history or (detection_history[0]["class"] != det["class"]):
                detection_history.appendleft(det)
                # Log detection to the central logging system
                log.info(f"Detected: {det['class']} (Conf: {det['conf']})")

    except Exception as e:
        log.error(f"Inference Error in process_frame: {e}")

    return frame


def capture_frames():
    global outputFrame, lock, conf_threshold, inference_enabled
    print(f"Attempting to open Webcam ID: {WEBCAM_ID}...")
    log.info(f"Attempting to open Webcam ID: {WEBCAM_ID}...")
    cap = cv2.VideoCapture(WEBCAM_ID)

    if not cap.isOpened():
        log.error("Could not open webcam.")
        with lock:
            outputFrame = get_error_frame("Error: Could not open webcam")
        return

    print("Webcam opened successfully.")

    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            with lock:
                outputFrame = get_error_frame("Camera disconnected")
                log.warning("Camera disconnected")
            time.sleep(1)
            continue

        # Only run inference if enabled in settings
        if inference_enabled:
            frame = process_frame(frame, conf_threshold)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Draw FPS on frame (Green text with black outline for visibility)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4
        )  # Outline
        cv2.putText(
            frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )  # Text

        with lock:
            outputFrame = frame.copy()

        # Minimal sleep to prevent pure busy loop, but kept low for high FPS
        time.sleep(0.001)


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                frame_to_encode = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame_to_encode,
                    "Waiting...",
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


# --- Routes ---


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/update_settings", methods=["POST"])
def update_settings():
    global conf_threshold, inference_enabled
    data = request.json

    if "conf" in data:
        conf_threshold = float(data["conf"])

    if "inference_enabled" in data:
        inference_enabled = bool(data["inference_enabled"])
        log.info(f"Inference enabled set to: {inference_enabled}")

    return jsonify(
        {
            "status": "success",
            "conf": conf_threshold,
            "inference_enabled": inference_enabled,
        }
    )


@app.route("/upload_image", methods=["POST"])
def upload_image():
    log.info("Started image upload")
    if "file" not in request.files:
        log.warning("No file part")
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        log.warning("No selected file")
        return jsonify({"error": "No selected file"})

    try:
        # Read image from memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Process image with current confidence threshold
        processed_img = process_frame(img, conf_threshold)

        # Encode back to jpg to send to browser
        _, buffer = cv2.imencode(".jpg", processed_img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        log.info("Image processed successfully")
        return jsonify({"status": "success", "image": img_base64})

    except Exception as e:
        log.error(f"Error processing uploaded image: {e}")
        return jsonify({"error": str(e)})


@app.route("/get_detections")
def get_detections():
    """Returns the list of recent detections."""
    return jsonify(list(detection_history))


@app.route("/get_app_log")
def get_app_log():
    """Reads the log file and returns its content."""
    content = ""
    try:
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                # Read the last 200 lines to keep the payload size manageable
                lines = f.readlines()
                content = "".join(lines[-200:])
        else:
            content = "Log file not found."
    except Exception as e:
        content = f"Error reading log file: {e}"

    return jsonify({"content": content})


@app.route("/download_logs")
def download_logs():
    """Downloads the current log file."""
    try:
        if os.path.exists(LOG_FILE_PATH):
            return send_file("../../" + LOG_FILE_PATH, as_attachment=True)
        else:
            return "Log file not found", 404
    except Exception as e:
        return str(e), 500


@app.route("/snapshot")
def snapshot():
    """Takes a snapshot of the current video frame."""
    with lock:
        if outputFrame is None:
            return "No frame available", 404
        # Encode the current frame as a JPEG
        _, buffer = cv2.imencode(".jpg", outputFrame)

    # Generate a filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.jpg"

    return Response(
        buffer.tobytes(),
        mimetype="image/jpeg",
        headers={"Content-Disposition": f"attachment;filename={filename}"},
    )


@app.route("/dataset_stats.png")
def dataset_stats_img():
    img_buffer = analysis.generate_analysis_image(LABELS_DIR)
    return send_file(img_buffer, mimetype="image/png")


if __name__ == "__main__":
    t = threading.Thread(target=capture_frames)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
