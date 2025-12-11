import os
import subprocess
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

inference_path = os.path.join(current_dir, "inference.py")
train_path = os.path.join(current_dir, "..", "neural_network", "train.py")
get_dataset_path = os.path.join(current_dir, "..", "data_acquisition", "get_dataset.py")
data_gen_path = os.path.join(
    current_dir, "..", "data_acquisition", "data_generation_main.py"
)

print("Traffic Sign Recognition System")
print("Author: Gabriel Georgescu, 632AB")

print("[1] Run web UI")
print("[2] Download dataset and generate data")
print("[3] Train model")
sel = input("Selection: ")

if sel == "1":
    subprocess.run([sys.executable, inference_path], check=True)
elif sel == "2":
    subprocess.run([sys.executable, get_dataset_path], check=True)
    subprocess.run([sys.executable, data_gen_path], check=True)
elif sel == "3":
    subprocess.run([sys.executable, train_path], check=True)

else:
    print("Invalid selection")
