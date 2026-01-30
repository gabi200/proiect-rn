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

# Vehicle State (Simulated)
vehicle_state = {
    "speed_limit": 50,  # Default limit
    "current_speed": 45,  # Simulated car speed
    "action": "CRUISING",  # Cruising, Braking, Turning
    "turn_signal": "none",  # left, right, none
    "last_sign_class": None,  # To show the icon
}

# Timestamp of last valid detection for reset logic
last_detection_time = 0
RESET_TIMEOUT = 5.0  # Seconds before resetting state

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


def update_vehicle_logic(detected_class):
    """
    Translates a traffic sign class into vehicle control commands.
    """
    global vehicle_state, last_detection_time

    last_detection_time = time.time()

    # 1. Speed Limits
    if "speed_over" in detected_class:
        try:
            # Extract number from string like 'forb_speed_over_30'
            limit = int(detected_class.split("_")[-1])
            vehicle_state["speed_limit"] = limit
            vehicle_state["action"] = f"SET LIMIT: {limit}"
            vehicle_state["current_speed"] = min(vehicle_state["current_speed"], limit)
        except:
            pass  # Failed to parse

    # 2. Stop / Give Way
    elif "stop" in detected_class or "forb_ahead" in detected_class:
        vehicle_state["current_speed"] = 0
        vehicle_state["action"] = "STOP"
    elif "give_way" in detected_class:
        vehicle_state["current_speed"] = max(0, vehicle_state["current_speed"] - 10)
        vehicle_state["action"] = "YIELDING"

    # 3. Turn Signals
    elif (
        "left" in detected_class
        and "mand" in detected_class
        and not "straight" in detected_class
    ):
        vehicle_state["turn_signal"] = "left"
        vehicle_state["action"] = "PREPARE TURN LEFT"
    elif (
        "right" in detected_class
        and "mand" in detected_class
        and not "straight" in detected_class
    ):
        vehicle_state["turn_signal"] = "right"
        vehicle_state["action"] = "PREPARE TURN RIGHT"

    # 4. Warnings
    elif (
        "warn" in detected_class
        or "crosswalk" in detected_class
        or "roundabout" in detected_class
    ):
        vehicle_state["action"] = "CAUTION"
        if vehicle_state["current_speed"] > 20:
            vehicle_state["current_speed"] = max(
                20, vehicle_state["current_speed"] - 20
            )
    elif "highway" in detected_class:
        vehicle_state["action"] = "HIGHWAY"
        vehicle_state["speed_limit"] = 130
        vehicle_state["current_speed"] = max(130, vehicle_state["current_speed"])

    # Update last sign for UI display
    vehicle_state["last_sign_class"] = detected_class


def check_reset_condition():
    """Resets vehicle state if no signs detected for RESET_TIMEOUT."""
    global vehicle_state, last_detection_time
    if time.time() - last_detection_time > RESET_TIMEOUT:
        if (
            vehicle_state["action"] != "CRUISING"
        ):  # Only log/change if we are actually resetting something
            vehicle_state = {
                "speed_limit": 50,
                "current_speed": 50,  # Return to normal cruise speed
                "action": "CRUISING",
                "turn_signal": "none",
                "last_sign_class": None,
            }


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

        # Check reset logic every frame
        check_reset_condition()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = model.names[cls]

                # Update Vehicle Logic ONLY if confidence meets threshold
                if conf >= conf_thresh:
                    update_vehicle_logic(current_class)

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

        # Update global history
        for det in current_detections:
            if not detection_history or (detection_history[0]["class"] != det["class"]):
                detection_history.appendleft(det)
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
            time.sleep(1)
            continue

        # Only run inference if enabled in settings
        if inference_enabled:
            frame = process_frame(frame, conf_threshold)
        else:
            # Even if inference disabled, we might want to reset state if timeout passed
            check_reset_condition()

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Draw FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4
        )
        cv2.putText(
            frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        with lock:
            outputFrame = frame.copy()

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
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
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

        return jsonify({"status": "success", "image": img_base64})

    except Exception as e:
        log.error(f"Error processing uploaded image: {e}")
        return jsonify({"error": str(e)})


@app.route("/get_detections")
def get_detections():
    return jsonify(list(detection_history))


@app.route("/get_app_log")
def get_app_log():
    content = ""
    try:
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                content = "".join(lines[-200:])
        else:
            content = "Log file not found."
    except Exception as e:
        content = f"Error reading log file: {e}"
    return jsonify({"content": content})


@app.route("/download_logs")
def download_logs():
    try:
        if os.path.exists(LOG_FILE_PATH):
            return send_file("../../" + LOG_FILE_PATH, as_attachment=True)
        else:
            return "Log file not found", 404
    except Exception as e:
        return str(e), 500


@app.route("/snapshot")
def snapshot():
    with lock:
        if outputFrame is None:
            return "No frame available", 404
        _, buffer = cv2.imencode(".jpg", outputFrame)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        buffer.tobytes(),
        mimetype="image/jpeg",
        headers={"Content-Disposition": f"attachment;filename=snap_{timestamp}.jpg"},
    )


@app.route("/dataset_stats.png")
def dataset_stats_img():
    img_buffer = analysis.generate_analysis_image(LABELS_DIR)
    return send_file(img_buffer, mimetype="image/png")


# New route for vehicle dashboard
@app.route("/get_vehicle_state")
def get_vehicle_state():
    return jsonify(vehicle_state)


if __name__ == "__main__":
    t = threading.Thread(target=capture_frames)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
