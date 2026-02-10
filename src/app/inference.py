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
import serial
import serial.tools.list_ports
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
    "current_speed": 50,  # Default cruising speed
    "action": "CRUISING",  # Cruising, Braking, Turning
    "turn_signal": "none",  # left, right, none, hazard
    "steering": "forward",  # forward, left, right
    "last_sign_class": None,  # To show the icon
    # Internal state for delayed actions
    "pending_turn": None,  # 'left' or 'right'
    "signal_start_time": 0,  # Timestamp when signal started blinking
    "camera_blocked": False,  # New state for blocked camera
}

# Timestamp of last valid detection for reset logic
last_detection_time = 0
RESET_TIMEOUT = 5.0  # Seconds before resetting state
STEERING_DELAY = 2.0  # Seconds between signal start and steering change


# Serial Port Manager
class SerialManager:
    def __init__(self):
        self.ser = None
        self.port = None
        self.baudrate = 9600
        self.is_connected = False
        self.last_sent_command = ""

    def connect(self, port_name):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()

            self.ser = serial.Serial(port_name, self.baudrate, timeout=1)
            self.port = port_name
            self.is_connected = True
            log.info(f"Connected to serial port: {port_name}")
            return True, "Connected"
        except Exception as e:
            self.is_connected = False
            log.error(f"Serial connection failed: {e}")
            return False, str(e)

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.is_connected = False
        log.info("Disconnected from serial port")
        return True

    def send_command(self, speed, turn_signal, steering):
        """
        Sends command in format "x;y;z"
        x: speed (km/h)
        y: turn signal (0-off, 1-left, 2-right, 3-hazards)
        z: steering (0-forward, 1-left, 2-right)
        """
        if not self.is_connected or not self.ser:
            return

        # Map turn signal string to int
        turn_map = {"none": 0, "left": 1, "right": 2, "hazard": 3}
        y = turn_map.get(turn_signal, 0)

        # Map steering string to int
        steer_map = {"forward": 0, "left": 1, "right": 2}
        z = steer_map.get(steering, 0)

        command = f"{int(speed)};{y};{z}"

        # Only send if state changed to avoid flooding serial buffer
        if command != self.last_sent_command:
            try:
                self.ser.write(f"{command}\n".encode("utf-8"))
                self.last_sent_command = command
            except Exception as e:
                log.error(f"Failed to send serial command: {e}")
                self.is_connected = False  # Assume disconnected on write error


serial_manager = SerialManager()

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


def check_camera_obstruction(frame):
    """
    Checks if the camera is obstructed (dark/black image).
    Returns True if obstructed, False otherwise.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    # Threshold for "darkness" - adjust based on environment
    # 30 is quite dark, typical for a covered lens
    if avg_brightness < 30:
        return True
    return False


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

            # Logic: Set current speed to limit but ensure >= 0.
            vehicle_state["current_speed"] = limit
            vehicle_state["action"] = f"SET LIMIT: {limit}"
        except:
            pass

    # 2. Stop / Give Way
    elif "stop" in detected_class:
        vehicle_state["current_speed"] = 0
        vehicle_state["action"] = "EMERGENCY BRAKE"
    elif "give_way" in detected_class:
        vehicle_state["current_speed"] = max(0, vehicle_state["current_speed"] - 10)
        vehicle_state["action"] = "YIELDING"

    # 3. Turn Signals (Immediate Activation, Delayed Steering)
    elif "left" in detected_class:
        # If this is a new turn intent
        if vehicle_state["pending_turn"] != "left":
            vehicle_state["pending_turn"] = "left"
            vehicle_state["signal_start_time"] = time.time()
            # Turn ON signal immediately
            vehicle_state["turn_signal"] = "left"
            vehicle_state["action"] = "SIGNALING LEFT"

    elif "right" in detected_class:
        # If this is a new turn intent
        if vehicle_state["pending_turn"] != "right":
            vehicle_state["pending_turn"] = "right"
            vehicle_state["signal_start_time"] = time.time()
            # Turn ON signal immediately
            vehicle_state["turn_signal"] = "right"
            vehicle_state["action"] = "SIGNALING RIGHT"

    # 4. Warnings
    elif "warn" in detected_class:
        vehicle_state["action"] = "CAUTION"

    # Update last sign for UI display
    vehicle_state["last_sign_class"] = detected_class


def check_delayed_actions():
    """Checks if enough time has passed to activate steering."""
    global vehicle_state

    if vehicle_state["pending_turn"]:
        # If signal has been on for > STEERING_DELAY seconds
        if time.time() - vehicle_state["signal_start_time"] >= STEERING_DELAY:
            vehicle_state["steering"] = vehicle_state["pending_turn"]
            vehicle_state["action"] = f"TURNING {vehicle_state['pending_turn'].upper()}"
            # Keep pending_turn set so we don't reset the timer or re-trigger logic
            # until the sign disappears/times out.


def check_reset_condition():
    """Resets vehicle state if no signs detected for RESET_TIMEOUT."""
    global vehicle_state, last_detection_time

    # Only reset if we are not already in default state
    is_default = (
        vehicle_state["action"] == "CRUISING"
        and vehicle_state["current_speed"] == 50  # Default speed logic (50)
        and vehicle_state["turn_signal"] == "none"
        and vehicle_state["steering"] == "forward"
        and not vehicle_state["camera_blocked"]
    )

    if not is_default and (time.time() - last_detection_time > RESET_TIMEOUT):
        vehicle_state = {
            "speed_limit": 50,
            "current_speed": 50,  # Reset to default 50
            "action": "CRUISING",
            "turn_signal": "none",
            "steering": "forward",
            "last_sign_class": None,
            "pending_turn": None,
            "signal_start_time": 0,
            "camera_blocked": False,
        }

    # Check steering delay logic regardless of reset timer (as long as sign is active)
    check_delayed_actions()

    # Send Command to Hardware (Sync every cycle)
    # Here we implement the -5 logic for the hardware command
    sent_speed = max(0, vehicle_state["current_speed"] - 5)

    serial_manager.send_command(
        sent_speed, vehicle_state["turn_signal"], vehicle_state["steering"]
    )


def process_frame(frame, conf_thresh):
    """
    Runs inference on a frame and draws bounding boxes.
    Returns the annotated frame.
    """
    global vehicle_state

    if model is None:
        return frame

    # --- 1. Camera Obstruction Check ---
    is_obstructed = check_camera_obstruction(frame)
    if is_obstructed:
        # Emergency Override
        vehicle_state["camera_blocked"] = True
        vehicle_state["action"] = "CAMERA BLOCKED! STOPPING"
        vehicle_state["current_speed"] = 0
        vehicle_state["turn_signal"] = "hazard"  # ENABLE HAZARDS
        vehicle_state["steering"] = "forward"

        # Visual Warning on Frame
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1
        )
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(
            frame,
            "CAMERA OBSTRUCTED",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
        )

        # Log incident
        if not detection_history or detection_history[0]["class"] != "CAMERA_BLOCKED":
            detection_history.appendleft(
                {
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "class": "CAMERA_BLOCKED",
                    "conf": "1.00",
                }
            )
            log.warning("Camera obstruction detected! Hazards enabled.")

        # Send command immediately (0 speed, hazards on)
        serial_manager.send_command(0, "hazard", "forward")
        return frame  # Skip inference
    else:
        # Only reset camera blocked state if it was previously blocked
        if vehicle_state["camera_blocked"]:
            vehicle_state["camera_blocked"] = False
            # Force reset state to normal/cruising immediately
            vehicle_state["turn_signal"] = "none"
            vehicle_state["action"] = "CRUISING"
            vehicle_state["current_speed"] = 50

    try:
        # Run inference
        results = model(frame, conf=conf_thresh, verbose=False)

        current_detections = []

        # Determine if we found any RELEVANT sign this frame to reset the timeout counter
        found_relevant_sign = False

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
                    found_relevant_sign = True

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

        # Check reset logic every frame
        check_reset_condition()

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
            # Even if inference disabled, we check reset/delayed actions
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


# --- Serial Port Routes ---
@app.route("/list_serial_ports")
def list_serial_ports():
    ports = [port.device for port in serial.tools.list_ports.comports()]
    return jsonify(ports)


@app.route("/connect_serial", methods=["POST"])
def connect_serial():
    data = request.json
    port = data.get("port")
    success, msg = serial_manager.connect(port)
    return jsonify({"success": success, "message": msg})


@app.route("/disconnect_serial", methods=["POST"])
def disconnect_serial():
    success = serial_manager.disconnect()
    return jsonify({"success": success})


if __name__ == "__main__":
    t = threading.Thread(target=capture_frames)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
