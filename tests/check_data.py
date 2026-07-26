import os, sys, cv2
import numpy as np

sys.path.insert(0, "D:\\CloudRemoval_Project")

patches_dir = "data/processed/patches/test"

# Check first 5 label (cloud-free) and cloudy patches
label_files = sorted(os.listdir(f"{patches_dir}/label"))[:5]
cloud_files = sorted(os.listdir(f"{patches_dir}/cloudy"))[:5]

print("Ground Truth (cloudfree) patch statistics:")
for f in label_files:
    img = cv2.imread(f"{patches_dir}/cloudfree/{f}")
    if img is not None:
        print(f"  {f}: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
    else:
        print(f"  {f}: Failed to load image.")

print("\nCloudy patch statistics:")
for f in cloud_files:
    img = cv2.imread(f"{patches_dir}/cloudy/{f}")
    if img is not None:
        print(f"  {f}: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
    else:
        print(f"  {f}: Failed to load image.")
