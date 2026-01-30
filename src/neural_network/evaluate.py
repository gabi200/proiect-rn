import glob
import json
import os
import shutil

import numpy as np
import pandas as pd
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "models/optimized_model.pt"
DATA_CONFIG = "dataset_config.yaml"
OUTPUT_DIR = "evaluation_results"
TEST_IMAGES_DIR = "data/test/images"  # Point this to your actual test images folder
MAX_VISUAL_SAMPLES = 150  # How many prediction images you want to save


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model from: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 1. Run Standard Evaluation (Metrics & Confusion Matrix)
    print("\n--- Phase 1: Standard Evaluation ---")
    metrics = model.val(
        data=DATA_CONFIG,
        split="test",
        plots=True,
        save_json=True,
        project=OUTPUT_DIR,
        name="metrics_run",
        exist_ok=True,
    )

    # 2. Run Visual Prediction (Generate More Samples)
    print("\n--- Phase 2: Generating Visual Samples ---")
    visual_output_dir = os.path.join(OUTPUT_DIR, "visual_predictions")
    if os.path.exists(visual_output_dir):
        shutil.rmtree(visual_output_dir)  # Clean previous run

    # Get list of test images
    test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.*"))
    test_images = [
        f for f in test_images if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if len(test_images) > 0:
        # Limit to MAX_VISUAL_SAMPLES to avoid filling disk if test set is huge
        sample_images = test_images[:MAX_VISUAL_SAMPLES]
        print(f"Running inference on {len(sample_images)} test images...")

        # Run prediction
        model.predict(
            source=sample_images,
            save=True,
            project=OUTPUT_DIR,
            name="visual_predictions",
            conf=0.25,  # Confidence threshold for visualization
            iou=0.45,
            exist_ok=True,
        )
        print(
            f"✅ Prediction images saved to: {os.path.join(OUTPUT_DIR, 'visual_predictions')}"
        )
    else:
        print(f"⚠️ No images found in {TEST_IMAGES_DIR}. Skipping visual generation.")

    # 3. Save Metrics & CSVs (Same as before)
    print("\n--- Phase 3: Saving Metrics ---")

    map50_95 = metrics.box.map
    map50 = metrics.box.map50
    precision = metrics.box.mp
    recall = metrics.box.mr

    print(f"mAP (50-95): {map50_95:.4f}")
    print(f"mAP (50)   : {map50:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")

    metrics_dict = {
        "map50_95": map50_95,
        "map50": map50,
        "precision": precision,
        "recall": recall,
        "fitness": metrics.fitness,
    }

    json_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # Confusion Matrix Extraction
    try:
        cm_array = metrics.confusion_matrix.matrix
        names = model.names
        class_names = [names[i] for i in sorted(names.keys())]

        if cm_array.shape[0] == len(class_names) + 1:
            class_names.append("background")

        df_cm = pd.DataFrame(cm_array, index=class_names, columns=class_names)
        df_cm.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))

        row_sums = cm_array.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm_cm_array = cm_array / row_sums
        df_norm = pd.DataFrame(norm_cm_array, index=class_names, columns=class_names)
        df_norm.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.csv"))
        print(f"✅ Confusion Matrix CSVs saved.")

    except Exception as e:
        print(f"❌ Error extracting confusion matrix: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
