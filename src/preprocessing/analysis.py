import os
from collections import defaultdict

import matplotlib.pyplot as plt

# 1. Define the dataset classes exactly as provided
CLASS_NAMES = [
    "forb_ahead",
    "forb_left",
    "forb_overtake",
    "forb_right",
    "forb_speed_over_10",
    "forb_speed_over_100",
    "forb_speed_over_130",
    "forb_speed_over_20",
    "forb_speed_over_30",
    "forb_speed_over_40",
    "forb_speed_over_5",
    "forb_speed_over_50",
    "forb_speed_over_60",
    "forb_speed_over_70",
    "forb_speed_over_80",
    "forb_speed_over_90",
    "forb_stopping",
    "forb_trucks",
    "forb_u_turn",
    "forb_weight_over_3.5t",
    "forb_weight_over_7.5t",
    "info_bus_station",
    "info_crosswalk",
    "info_highway",
    "info_one_way_traffic",
    "info_parking",
    "info_taxi_parking",
    "mand_bike_lane",
    "mand_left",
    "mand_left_right",
    "mand_pass_left",
    "mand_pass_left_right",
    "mand_pass_right",
    "mand_right",
    "mand_roundabout",
    "mand_straigh_left",
    "mand_straight",
    "mand_straight_right",
    "prio_give_way",
    "prio_priority_road",
    "prio_stop",
    "warn_children",
    "warn_construction",
    "warn_crosswalk",
    "warn_cyclists",
    "warn_domestic_animals",
    "warn_other_dangers",
    "warn_poor_road_surface",
    "warn_roundabout",
    "warn_slippery_road",
    "warn_speed_bumper",
    "warn_traffic_light",
    "warn_tram",
    "warn_two_way_traffic",
    "warn_wild_animals",
]

# 2. Configuration
LABELS_DIR = "../../data/train/labels"  # Path to your labels
CATEGORIES = ["warn", "mand", "info", "forb"]
COLORS = ["#ffcc00", "#3366cc", "#33cc33", "#cc3333"]  # Yellow, Blue, Green, Red


def analyze_labels():
    # Counters
    category_counts = defaultdict(int)
    total_annotations = 0

    # Check if directory exists
    if not os.path.exists(LABELS_DIR):
        print(f"Error: Directory '{LABELS_DIR}' not found.")
        return

    print(f"Scanning files in {LABELS_DIR}...")

    # 3. Iterate over all .txt files
    files = [f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")]

    if not files:
        print("No .txt files found.")
        return

    for filename in files:
        filepath = os.path.join(LABELS_DIR, filename)
        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                # The first element is the class ID
                try:
                    class_id = int(parts[0])

                    if 0 <= class_id < len(CLASS_NAMES):
                        name = CLASS_NAMES[class_id]

                        # Determine category based on prefix
                        matched = False
                        for cat in CATEGORIES:
                            if name.startswith(cat):
                                category_counts[cat] += 1
                                matched = True
                                break

                        if not matched:
                            # Handle classes like 'prio' that weren't requested but exist
                            category_counts["other"] += 1

                        total_annotations += 1
                    else:
                        print(
                            f"Warning: Class ID {class_id} in {filename} is out of range."
                        )

                except ValueError:
                    print(
                        f"Warning: Could not parse line in {filename}: {line.strip()}"
                    )

    # 4. Print Results
    print("\n--- Analysis Results ---")
    print(f"Total Files Scanned: {len(files)}")
    print(f"Total Annotations: {total_annotations}")
    for cat in CATEGORIES:
        print(f"  {cat.upper()}: {category_counts[cat]}")
    if category_counts["other"] > 0:
        print(f"  OTHER (e.g., prio): {category_counts['other']}")

    # 5. Plot Histogram
    values = [category_counts[cat] for cat in CATEGORIES]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(CATEGORIES, values, color=COLORS, edgecolor="black")

    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(values) * 0.01),
            int(yval),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.title("Traffic Sign Distribution by Category", fontsize=16)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Number of Instances (Annotations)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_labels()
