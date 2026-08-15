"""
Run: conda activate cloudremoval
python view_patches.py

Builds one PNG grid showing N random patches (cloudy | clean | mask side by side)
from patches_liss4/train, so you can visually check quality without opening
hundreds of individual files.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
import cv2
import numpy as np

PATCHES_DIR = r"D:\CloudRemoval_Project\data\processed\patches_liss4_realcloud\train"
N_SAMPLES = 6
OUT_PATH = r"D:\CloudRemoval_Project\patches_preview.png"

with open(os.path.join(PATCHES_DIR, "patch_manifest.json")) as f:
    manifest = json.load(f)

patch_ids = random.sample(list(manifest.keys()), min(N_SAMPLES, len(manifest)))
print(f"Showing {len(patch_ids)} random patches out of {len(manifest)} total")

rows = []
for pid in patch_ids:
    cloud = cv2.imread(os.path.join(PATCHES_DIR, "cloud", f"{pid}.png"))
    label = cv2.imread(os.path.join(PATCHES_DIR, "label", f"{pid}.png"))
    mask = cv2.imread(os.path.join(PATCHES_DIR, "mask", f"{pid}.png"))
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask

    # label each column
    for img, tag in [(cloud, "cloudy"), (label, "clean"), (mask_bgr, "mask")]:
        cv2.putText(img, tag, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    row = np.hstack([cloud, label, mask_bgr])
    cv2.putText(
        row, pid, (5, row.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
    )
    rows.append(row)

grid = np.vstack(rows)
cv2.imwrite(OUT_PATH, grid)
print(f"Saved: {OUT_PATH}")
print("Upload this file to chat to view it.")
