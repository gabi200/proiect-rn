import os

import torch
from ultralytics import YOLO

# --- Configuration ---
# Path to your trained weights
MODEL_PATH = "models/trained_model.pt"
# Dataset config is required by the benchmark utility to define the task/classes
DATA_CONFIG = "dataset_config.yaml"
IMG_SIZE = 640


def main():
    # 1. Setup Device
    # Ultralytics accepts int 0 for GPU 0, or 'cpu' string
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # 2. Check Paths
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    if not os.path.exists(DATA_CONFIG):
        print(f"⚠️ Warning: Dataset config '{DATA_CONFIG}' not found.")
        print("The benchmark might default to COCO or fail.")

    print(f"Loading model from: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("\nStarting Ultralytics Built-in Benchmark...")
    print("---------------------------------------------------------------")
    print("Benchmarking inference speed using TorchScript.")
    print("(TorchScript is the standard deployable format for PyTorch models)")
    print("---------------------------------------------------------------")

    # 3. Run Benchmark
    # We use format='torchscript' because 'pt' is not a supported export format for benchmarking.
    # TorchScript is the native PyTorch deployment format and runs without extra DLLs.
    try:
        model.benchmark(
            data=DATA_CONFIG,
            imgsz=IMG_SIZE,
            device=device,
            half=False,  # Set to True to test FP16 performance (Recommended for GPUs)
            verbose=True,
            format="torchscript",  # <--- UPDATED: Use TorchScript instead of 'pt'
        )
    except Exception as e:
        print(f"\n❌ Benchmark execution failed: {e}")


if __name__ == "__main__":
    main()
