import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\CloudRemoval_Project")

from src.models.segmentation import AttentionUNet

PROJECT_ROOT = Path(r"D:\CloudRemoval_Project")
CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation" / "best_iou0.7935_ep39.pt"
TEST_CLOUD_DIR = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "cloud"
TEST_MASK_DIR = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "mask"
TEST_MANIFEST = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "patch_manifest.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "seg_config.yaml"
OUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics" / "fn_visual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH, "r") as f:
    seg_config = yaml.safe_load(f)["segmentation"]

seg_transform = A.Compose(
    [
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

seg_model = AttentionUNet(seg_config).to(DEVICE)
ckpt = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=False)
if isinstance(ckpt, dict) and "model_state" in ckpt:
    seg_model.load_state_dict(ckpt["model_state"])
else:
    seg_model.load_state_dict(ckpt)
seg_model.eval()

with open(TEST_MANIFEST, "r") as f:
    manifest = json.load(f)

image_paths = sorted(TEST_CLOUD_DIR.glob("*.png"))

# Evaluate all images and collect per-image metrics
results = []
for img_path in image_paths:
    pid = img_path.stem
    if pid not in manifest:
        continue
    cov = float(manifest[pid]["cloud_coverage"])
    if cov >= 0.30:
        continue  # only light cloud for worst-case

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_path = TEST_MASK_DIR / f"{pid}.png"
    gt_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    gt = (gt_raw > 127).astype(np.uint8)

    transformed = seg_transform(image=img)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = seg_model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    if prob.shape != gt.shape:
        prob = cv2.resize(prob, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    pred = (prob > 0.50).astype(np.uint8)

    tp_mask = np.logical_and(pred == 1, gt == 1)
    fn_mask = np.logical_and(pred == 0, gt == 1)
    fp_mask = np.logical_and(pred == 1, gt == 0)

    tp = int(tp_mask.sum())
    fn = int(fn_mask.sum())
    fp = int(fp_mask.sum())
    tn = int((np.logical_and(pred == 0, gt == 0)).sum())

    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 0.0
    dice = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    results.append({
        "pid": pid,
        "coverage": cov,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "img": img,
        "gt": gt,
        "prob": prob,
        "pred": pred,
        "fn_mask": fn_mask,
        "fp_mask": fp_mask,
    })

# Sort by IoU ascending, take worst 10
results.sort(key=lambda x: x["iou"])
worst = results[:10]

print(f"Worst 10 light-cloud images by IoU:")
for r in worst:
    print(f"  {r['pid']}: IoU={r['iou']:.4f} Dice={r['dice']:.4f} Recall={r['recall']:.4f} FN={r['fn']} TP={r['tp']}")

# Generate visualizations
for idx, r in enumerate(worst, 1):
    pid = r["pid"]
    img = r["img"]
    gt = r["gt"]
    prob = r["prob"]
    pred = r["pred"]
    fn_mask = r["fn_mask"]
    fp_mask = r["fp_mask"]

    # Create 5-panel figure: original | GT mask | prob heatmap | pred mask | FN overlay
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"Worst Light-Cloud Case #{idx}: {pid} | IoU={r['iou']:.3f} Recall={r['recall']:.3f} Coverage={r['coverage']:.1%}")

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("GT Mask")
    axes[1].axis("off")

    im = axes[2].imshow(prob, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Probability")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    axes[3].imshow(pred, cmap="gray")
    axes[3].set_title("Predicted Mask")
    axes[3].axis("off")

    overlay = img.copy()
    overlay[fn_mask] = [255, 0, 0]  # red for FN
    alpha = 0.5
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    axes[4].imshow(blended)
    axes[4].set_title("FN Highlighted")
    axes[4].axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / f"worst_light_cloud_{idx:02d}_{pid}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

print(f"\nSaved {len(worst)} visualizations to {OUT_DIR}")
