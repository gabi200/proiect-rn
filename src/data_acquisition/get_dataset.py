import shutil

import kagglehub

# Download latest version
try:
    cache_path = kagglehub.dataset_download("raduoprea/traffic-signs")
except Exception as e:
    print(f"Error downloading dataset: {e}")

print("Path to dataset files:", cache_path)

print("Copying to dataset folder from cache...")

try:
    shutil.copytree(
        cache_path + "/V2 - Traffic Signs/train", "data/raw", dirs_exist_ok=True
    )
    shutil.copytree(
        cache_path + "/V2 - Traffic Signs/valid",
        "data/validation",
        dirs_exist_ok=True,
    )
    print("Done copying dataset.")
except Exception as e:
    print(f"Error copying dataset: {e}")

print("Cleaning up...")

shutil.rmtree(cache_path)
