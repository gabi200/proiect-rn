import os
import sys

import torch
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "app"))

from logger import get_logger

log = get_logger(__name__)

log.info("Started training module")


def main():
    # 1. Setup Device (GPU is highly recommended)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    print("Loading YOLOv9m model...")
    log.info("Loading YOLOv9m model...")
    model = YOLO("models/untrained_model.yaml")

    choice = input(
        "Enter custom training parameters? Choosing 'no' will use default parameters (y/n) "
    )

    if choice.lower() == "y":
        print("Please enter training parameters:")
        epochs = int(input("Number of epochs: "))
        batch = int(input("Batch size: "))
        patience = int(input("Early stopping patience: "))
        optimizer = input("Optimizer: ")
        cls = float(input("Weight of classification loss: "))

        log.info(
            f"Using custom training parameters: epochs={epochs}, batch={batch}, patience={patience}"
        )
    else:
        epochs = 50
        batch = 8
        patience = 5
        optimizer = "Adam"
        cls = 1.5
        log.info(
            f"Using default training parameters: epochs={epochs}, batch={batch}, patience={patience}"
        )

    print("Starting training...")
    log.info("Starting training...")

    results = model.train(
        data="dataset_config.yaml",
        epochs=epochs,
        imgsz=640,
        device=device,
        batch=batch,  # Adjust based on your GPU VRAM
        name="../../saved_models",  # Name of the run folder
        patience=patience,  # Early stopping patience
        plots=True,  # Save plots of training metrics
        exist_ok=True,
        optimizer=optimizer,
        cls=cls,
        cos_lr=True,
        hsv_h=0.015,  # augmentations start
        hsv_s=0.6,
        hsv_v=0.5,
        scale=0.8,
        shear=2,
        perspective=0.001,
        fliplr=0,  # CRITICAL
        degrees=3,
        lr0=0.001,
        label_smoothing=0.1,
        copy_paste=0.3,
    )

    print("Validating model...")
    log.info("Validating model...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    log.info(f"mAP50: {metrics.box.map50}")
    log.info(f"mAP50-95: {metrics.box.map}")

    print("Running evaluation on Test set...")
    log.info("Running evaluation on Test set...")
    try:
        test_metrics = model.val(split="test")
        print(f"Test Set mAP50: {test_metrics.box.map50}")
        print(f"Test Set mAP50-95: {test_metrics.box.map}")
        log.info(f"Test Set mAP50: {test_metrics.box.map50}")
        log.info(f"Test Set mAP50-95: {test_metrics.box.map}")
    except Exception as e:
        print(
            f"Could not run test set evaluation (check if 'test' path exists in yaml). Error: {e}"
        )
        log.error(
            f"Could not run test set evaluation (check if 'test' path exists in yaml). Error: {e}"
        )


if __name__ == "__main__":
    # Required for Windows multiprocessing safety
    main()
