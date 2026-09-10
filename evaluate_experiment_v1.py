import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, r"D:\CloudRemoval_Project")
from src.models.segmentation import AttentionUNet

PROJECT_ROOT = Path(r"D:\CloudRemoval_Project")
TEST_CLOUD_DIR = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "cloud"
TEST_MASK_DIR = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "mask"
TEST_MANIFEST = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "patch_manifest.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "seg_config.yaml"
OUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
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

with open(TEST_MANIFEST, "r") as f:
    manifest = json.load(f)

image_paths = sorted(TEST_CLOUD_DIR.glob("*.png"))


def load_checkpoint(path):
    model = AttentionUNet(seg_config).to(DEVICE)
    ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        # Lightning checkpoints prefix model keys with "model."
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_state[k[len("model."):]] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def evaluate(model, threshold=0.50):
    global_tp = 0
    global_fn = 0
    global_fp = 0
    global_tn = 0
    cat_stats = defaultdict(lambda: {"images": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0})
    per_image = []

    for img_path in image_paths:
        pid = img_path.stem
        if pid not in manifest:
            continue
        cov = float(manifest[pid]["cloud_coverage"])
        if cov < 0.30:
            cat = "light"
        elif cov < 0.60:
            cat = "medium"
        else:
            cat = "heavy"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_path = TEST_MASK_DIR / f"{pid}.png"
        gt_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt = (gt_raw > 127).astype(np.uint8)

        transformed = seg_transform(image=img)
        tensor = transformed["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        if prob.shape != gt.shape:
            prob = cv2.resize(prob, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        pred = (prob > threshold).astype(np.uint8)

        tp = int(np.logical_and(pred == 1, gt == 1).sum())
        fn = int(np.logical_and(pred == 0, gt == 1).sum())
        fp = int(np.logical_and(pred == 1, gt == 0).sum())
        tn = int(np.logical_and(pred == 0, gt == 0).sum())

        global_tp += tp
        global_fn += fn
        global_fp += fp
        global_tn += tn

        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else 0.0
        dice = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        cat_stats[cat]["images"] += 1
        cat_stats[cat]["tp"] += tp
        cat_stats[cat]["fn"] += fn
        cat_stats[cat]["fp"] += fp
        cat_stats[cat]["tn"] += tn

        per_image.append({
            "pid": pid,
            "coverage": cov,
            "category": cat,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "iou": iou, "dice": dice,
            "precision": precision, "recall": recall,
        })

    return {
        "global_tp": global_tp,
        "global_fn": global_fn,
        "global_fp": global_fp,
        "global_tn": global_tn,
        "iou": float(global_tp / (global_tp + global_fp + global_fn)) if (global_tp + global_fp + global_fn) > 0 else 0.0,
        "dice": float((2 * global_tp) / (2 * global_tp + global_fp + global_fn)) if (2 * global_tp + global_fp + global_fn) > 0 else 0.0,
        "precision": float(global_tp / (global_tp + global_fp)) if (global_tp + global_fp) > 0 else 0.0,
        "recall": float(global_tp / (global_tp + global_fn)) if (global_tp + global_fn) > 0 else 0.0,
        "categories": {cat: {
            "images": stats["images"],
            "tp": stats["tp"], "fn": stats["fn"], "fp": stats["fp"], "tn": stats["tn"],
            "iou": float(stats["tp"] / (stats["tp"] + stats["fp"] + stats["fn"])) if (stats["tp"] + stats["fp"] + stats["fn"]) > 0 else 0.0,
            "recall": float(stats["tp"] / (stats["tp"] + stats["fn"])) if (stats["tp"] + stats["fn"]) > 0 else 0.0,
        } for cat, stats in cat_stats.items()},
        "per_image": per_image,
    }


baseline_path = PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation" / "best_iou0.7935_ep39.pt"
new_path = PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation" / "best_iou0.7703_ep27_brightness_aug.ckpt"

print("Loading baseline model...")
baseline_model = load_checkpoint(baseline_path)
print("Loading new model...")
new_model = load_checkpoint(new_path)

print("Evaluating baseline...")
baseline_results = evaluate(baseline_model)
print("Evaluating new model...")
new_results = evaluate(new_model)

# Save results
json_path = OUT_DIR / "brightness_contrast_aug_v1_results.json"
with open(json_path, "w") as f:
    json.dump({
        "baseline": baseline_results,
        "new": new_results,
        "changes": {
            "iou": new_results["iou"] - baseline_results["iou"],
            "dice": new_results["dice"] - baseline_results["dice"],
            "precision": new_results["precision"] - baseline_results["precision"],
            "recall": new_results["recall"] - baseline_results["recall"],
        }
    }, f, indent=2)

csv_rows = []
for model_name, results in [("baseline", baseline_results), ("new", new_results)]:
    csv_rows.append({
        "model": model_name,
        "group": "overall",
        "category": "all",
        "images": len(results["per_image"]),
        "tp": results["global_tp"],
        "fn": results["global_fn"],
        "fp": results["global_fp"],
        "tn": results["global_tn"],
        "iou": results["iou"],
        "dice": results["dice"],
        "precision": results["precision"],
        "recall": results["recall"],
    })
    for cat, stats in results["categories"].items():
        csv_rows.append({
            "model": model_name,
            "group": "category",
            "category": cat,
            "images": stats["images"],
            "tp": stats["tp"],
            "fn": stats["fn"],
            "fp": stats["fp"],
            "tn": stats["tn"],
            "iou": stats["iou"],
            "dice": float("nan"),
            "precision": float("nan"),
            "recall": stats["recall"],
        })

csv_path = OUT_DIR / "brightness_contrast_aug_v1_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(csv_rows)

# Text summary
lines = []
lines.append("=" * 80)
lines.append("BRIGHTNESS/CONTRAST AUGMENTATION V1 RESULTS")
lines.append("=" * 80)
lines.append("")
lines.append("BASELINE (best_iou0.7935_ep39.pt):")
lines.append(f"  IoU={baseline_results['iou']:.4f} Dice={baseline_results['dice']:.4f}")
lines.append(f"  Precision={baseline_results['precision']:.4f} Recall={baseline_results['recall']:.4f}")
lines.append("")
lines.append("NEW (best_iou0.7703_ep27_brightness_aug.ckpt):")
lines.append(f"  IoU={new_results['iou']:.4f} Dice={new_results['dice']:.4f}")
lines.append(f"  Precision={new_results['precision']:.4f} Recall={new_results['recall']:.4f}")
lines.append("")
lines.append("CHANGE:")
lines.append(f"  IoU: {new_results['iou'] - baseline_results['iou']:+.4f}")
lines.append(f"  Dice: {new_results['dice'] - baseline_results['dice']:+.4f}")
lines.append(f"  Precision: {new_results['precision'] - baseline_results['precision']:+.4f}")
lines.append(f"  Recall: {new_results['recall'] - baseline_results['recall']:+.4f}")
lines.append("")
lines.append("CATEGORY BREAKDOWN (baseline -> new):")
for cat in ["light", "medium", "heavy"]:
    if cat in baseline_results["categories"] and cat in new_results["categories"]:
        b = baseline_results["categories"][cat]
        n = new_results["categories"][cat]
        lines.append(f"  {cat}:")
        lines.append(f"    IoU: {b['iou']:.4f} -> {n['iou']:.4f}")
        lines.append(f"    Recall: {b['recall']:.4f} -> {n['recall']:.4f}")
lines.append("")
lines.append("=" * 80)
lines.append("CONCLUSION:")
if new_results["iou"] > baseline_results["iou"]:
    lines.append("The brightness/contrast augmentation IMPROVED overall IoU.")
else:
    lines.append("The brightness/contrast augmentation DID NOT improve overall IoU.")
if new_results["recall"] > baseline_results["recall"]:
    lines.append("Recall improved, suggesting better cloud detection.")
else:
    lines.append("Recall did not improve.")
lines.append("=" * 80)

txt_path = OUT_DIR / "brightness_contrast_aug_v1_results.txt"
with open(txt_path, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\nSaved: {json_path}")
print(f"Saved: {csv_path}")
print(f"Saved: {txt_path}")
