import os
import shutil
from modulefinder import test
from pathlib import Path
from random import shuffle

from generate_img import *

original_img_path = "data/raw"
final_target_img_path = "data/train"
proc_target_img_path = "data/generated"
valid_path = "data/validation"
test_path = "data/test"

# copy labels and original images to trainiing folder
print("Copying labels...")
shutil.copytree(
    original_img_path + "/labels", final_target_img_path + "/labels", dirs_exist_ok=True
)

print("Copying original images...")
shutil.copytree(
    original_img_path + "/images", final_target_img_path + "/images", dirs_exist_ok=True
)

print("Generating additional images...")

# create folders if not existing
Path(proc_target_img_path + "/labels").mkdir(parents=True, exist_ok=True)
Path(proc_target_img_path + "/images").mkdir(parents=True, exist_ok=True)
Path(test_path + "/labels").mkdir(parents=True, exist_ok=True)
Path(test_path + "/images").mkdir(parents=True, exist_ok=True)

# generate labels for new images
for name in os.listdir(original_img_path + "/labels"):
    shutil.copyfile(
        original_img_path + "/labels/" + name,
        proc_target_img_path + "/labels/" + "proc-" + name,
    )

# process images
for name in os.listdir(original_img_path + "/images"):
    augument(
        original_img_path + "/images/" + name,
        proc_target_img_path + "/images/" + "proc-" + name,
    )

print("Copying generated images to training dataset...")
# copy generated images to training folder
shutil.copytree(
    proc_target_img_path + "/images",
    final_target_img_path + "/images",
    dirs_exist_ok=True,
)

# copy generated labels to training folder
shutil.copytree(
    proc_target_img_path + "/labels",
    final_target_img_path + "/labels",
    dirs_exist_ok=True,
)

# move half of the validation data to the test dataset
print("Moving files for test dataset...")

all_files = [f for f in os.listdir(valid_path + "/images") if f.endswith(".jpg")]

half_count = len(all_files) // 2

for name in all_files[:half_count]:
    shutil.move(
        valid_path + "/images/" + name,
        test_path + "/images/" + name,
    )
    shutil.move(
        valid_path + "/labels/" + name.replace(".jpg", ".txt"),
        test_path + "/labels/" + name.replace(".jpg", ".txt"),
    )

print(f"Moved {half_count} files.")

print("Done.")
