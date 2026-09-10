import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"D:\CloudRemoval_Project")
MANIFEST = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "patch_manifest.json"
RICE2_CLOUD_DIR = PROJECT_ROOT / "datasets" / "RICE2" / "RICE2" / "cloud"
OUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics" / "rice2_agriculture_inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(MANIFEST) as f:
    manifest = json.load(f)

source_ids = sorted(set(int(meta["source_id"]) for meta in manifest.values()))
print(f"Unique test source scenes: {len(source_ids)}")

# Load images
images = []
missing = []
for sid in source_ids:
    img_path = RICE2_CLOUD_DIR / f"{sid}.png"
    if img_path.exists():
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append((sid, img))
    else:
        missing.append(sid)
        images.append((sid, None))

if missing:
    print(f"WARNING: {len(missing)} source images not found: {missing}")

# Grid layout: aim for ~10 columns
n = len(images)
cols = 10
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = np.atleast_2d(axes).reshape(rows, cols)
fig.suptitle("RICE2 Test Source Scenes Contact Sheet", fontsize=16, y=0.995)

for idx, (sid, img) in enumerate(images):
    r, c = divmod(idx, cols)
    ax = axes[r][c]
    if img is not None:
        ax.imshow(img)
        ax.set_title(f"source_id={sid}", fontsize=8, pad=2)
    else:
        ax.text(0.5, 0.5, f"MISSING\n{sid}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.axis("off")

# Hide empty subplots
for idx in range(n, rows * cols):
    r, c = divmod(idx, cols)
    axes[r][c].axis("off")

plt.tight_layout()
contact_sheet_path = OUT_DIR / "rice2_test_source_scenes_contact_sheet.png"
plt.savefig(contact_sheet_path, dpi=150, bbox_inches="tight")
plt.close()

# Save source_id list
list_path = OUT_DIR / "rice2_test_source_ids.txt"
with open(list_path, "w") as f:
    for sid in source_ids:
        f.write(f"{sid}\n")

print(f"Contact sheet saved: {contact_sheet_path}")
print(f"Source ID list saved: {list_path}")
print(f"Total source scenes: {len(source_ids)}")
print(f"Grid: {rows} rows x {cols} cols")
