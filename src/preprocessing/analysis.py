import os
from collections import defaultdict

import matplotlib

# Use Agg backend for non-interactive (server/web) plotting
matplotlib.use("Agg")
import io

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
DEFAULT_LABELS_DIR = "../../data/train/labels"
CATEGORIES = ["warn", "mand", "info", "forb", "prio"]
COLORS = ["#ffcc00", "#3366cc", "#33cc33", "#cc3333", "#ff9900"]
CAT_COLOR_MAP = dict(zip(CATEGORIES, COLORS))


def generate_analysis_image(labels_dir=None):
    """
    Scans the directory, counts classes, and returns a PNG image bytes buffer.
    """
    target_dir = labels_dir if labels_dir else DEFAULT_LABELS_DIR

    # Counters
    category_counts = defaultdict(int)
    individual_counts = defaultdict(int)
    total_annotations = 0

    # Handle missing directory gracefully for the web app
    if not os.path.exists(target_dir):
        return create_error_image(f"Directory not found: {target_dir}")

    files = [f for f in os.listdir(target_dir) if f.endswith(".txt")]

    if not files:
        return create_error_image("No .txt files found in directory.")

    for filename in files:
        filepath = os.path.join(target_dir, filename)
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    try:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(CLASS_NAMES):
                            name = CLASS_NAMES[class_id]
                            individual_counts[name] += 1
                            total_annotations += 1

                            matched = False
                            for cat in CATEGORIES:
                                if name.startswith(cat):
                                    category_counts[cat] += 1
                                    matched = True
                                    break
                            if not matched:
                                category_counts["other"] += 1
                    except ValueError:
                        continue
        except Exception:
            continue

    # Plotting
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14))
    plt.subplots_adjust(hspace=0.6)

    # Plot 1: Categories
    cat_values = [category_counts[cat] for cat in CATEGORIES]
    bars1 = ax1.bar(CATEGORIES, cat_values, color=COLORS, edgecolor="white")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            int(yval),
            ha="center",
            va="bottom",
            color="white",
            fontweight="bold",
        )

    ax1.set_title(
        f"Traffic Sign Distribution (Total: {total_annotations})",
        fontsize=16,
        color="white",
    )
    ax1.set_ylabel("Count", fontsize=12, color="white")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Plot 2: Individual Classes
    ind_values = [individual_counts[name] for name in CLASS_NAMES]
    ind_colors = []
    for name in CLASS_NAMES:
        color = "#888888"
        for cat in CATEGORIES:
            if name.startswith(cat):
                color = CAT_COLOR_MAP[cat]
                break
        ind_colors.append(color)

    ax2.bar(CLASS_NAMES, ind_values, color=ind_colors, edgecolor="white", alpha=0.8)
    ax2.set_title(
        "Detailed Distribution by Individual Class", fontsize=16, color="white"
    )
    ax2.set_ylabel("Count", fontsize=12, color="white")
    ax2.set_xticklabels(
        CLASS_NAMES, rotation=90, ha="center", fontsize=8, color="white"
    )
    ax2.set_xlim(-1, len(CLASS_NAMES))
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    # Save to IO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor="#1a1a1a", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def create_error_image(msg):
    """Helper to return an image with text if something fails."""
    fig, ax = plt.subplots(figsize=(8, 2))
    plt.style.use("dark_background")
    ax.text(0.5, 0.5, f"Error: {msg}", ha="center", va="center", color="red")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor="#1a1a1a")
    plt.close(fig)
    buf.seek(0)
    return buf
