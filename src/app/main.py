import os
import subprocess
import sys

from logger import get_logger

current_dir = os.path.dirname(os.path.abspath(__file__))

inference_path = os.path.join(current_dir, "inference.py")
train_path = os.path.join(current_dir, "..", "neural_network", "train.py")
get_dataset_path = os.path.join(current_dir, "..", "data_acquisition", "get_dataset.py")
data_gen_path = os.path.join(
    current_dir, "..", "data_acquisition", "data_generation_main.py"
)
eval_path = os.path.join(current_dir, "..", "neural_network", "evaluate.py")

log = get_logger(__name__)

log.info("Started main app")
print("Traffic Sign Recognition System")
print("Author: Gabriel Georgescu, 632AB")

print("[1] Run web UI")
print("[2] Download dataset and generate data")
print("[3] Train model")
print("[4] Evaluate model")
sel = input("Selection: ")

if sel == "1":
    log.info("Starting web UI module...")
    subprocess.run([sys.executable, inference_path], check=True)
elif sel == "2":
    log.info("Starting data acquisition module...")
    subprocess.run([sys.executable, get_dataset_path], check=True)
    log.info("Starting data generation module...")
    subprocess.run([sys.executable, data_gen_path], check=True)
elif sel == "3":
    log.info("Starting training module...")
    subprocess.run([sys.executable, train_path], check=True)
elif sel == "4":
    log.info("Starting evaluation module...")
    subprocess.run([sys.executable, eval_path], check=True)

else:
    log.error("Invalid user selection")
    print("Invalid selection")
