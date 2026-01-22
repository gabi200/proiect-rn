import json
import os

import numpy as np
import pandas as pd
from ultralytics import YOLO

# --- Configuration ---
# Path to your trained weights
MODEL_PATH = "models/optimized_model.pt"
# Dataset config
DATA_CONFIG = "dataset_config.yaml"
# Directory to save results
OUTPUT_DIR = "evaluation_results"


def main():
    # 1. Setup
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

    # 2. Run Evaluation (on Test set)
    print("\nStarting evaluation on 'test' split...")
    # split='test' forces it to use the test set defined in yaml.
    # plots=True ensures standard confusion matrix images are generated internally if you want them later.
    metrics = model.val(data=DATA_CONFIG, split="test", plots=True, save_json=True)

    # 3. Extract and Print Standard Metrics
    print("\n" + "=" * 30)
    print("   EVALUATION RESULTS")
    print("=" * 30)

    # mAP50-95 (Mean Average Precision at IoU 0.50:0.95)
    map50_95 = metrics.box.map
    # mAP50 (Mean Average Precision at IoU 0.50)
    map50 = metrics.box.map50
    # Precision and Recall (averaged)
    precision = metrics.box.mp
    recall = metrics.box.mr

    print(f"mAP (50-95): {map50_95:.4f}")
    print(f"mAP (50)   : {map50:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")

    # Save scalar metrics to JSON
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
    print(f"Scalar metrics saved to: {json_path}")

    # 4. Extract Confusion Matrix to CSV
    print("\nGenerating Confusion Matrix CSV...")

    try:
        # Access the raw confusion matrix from the validator metrics
        # The matrix is typically shape (nc + 1, nc + 1) to include background
        cm_array = metrics.confusion_matrix.matrix

        # Get class names
        names = model.names
        # Ensure names dictionary is sorted by ID to match matrix index
        class_names = [names[i] for i in sorted(names.keys())]

        # The confusion matrix in Ultralytics often includes a "background" row/col at the end
        # representing False Positives (background detected as object)
        # and False Negatives (object detected as background/missed).
        if cm_array.shape[0] == len(class_names) + 1:
            class_names.append("background")

        # Create DataFrame
        df_cm = pd.DataFrame(cm_array, index=class_names, columns=class_names)

        # Save to CSV
        csv_path = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
        df_cm.to_csv(csv_path)
        print(f" Confusion Matrix CSV saved to: {csv_path}")

        # Optional: Save a normalized version as well
        # Normalize by row (True Label counts)
        row_sums = cm_array.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        norm_cm_array = cm_array / row_sums

        df_norm = pd.DataFrame(norm_cm_array, index=class_names, columns=class_names)
        norm_csv_path = os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.csv")
        df_norm.to_csv(norm_csv_path)
        print(f" Normalized Confusion Matrix CSV saved to: {norm_csv_path}")

    except AttributeError:
        print(
            " Warning: Could not extract confusion matrix directly. Ensure you are using a recent version of Ultralytics."
        )
    except Exception as e:
        print(f"❌ Error generating CSV: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
