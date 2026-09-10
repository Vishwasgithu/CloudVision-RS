import os
import sys
import json
import glob
import re
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
CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation" / "best_iou0.7935_ep39.pt"
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
print(f"Found {len(image_paths)} test patches on disk.")
print(f"Manifest entries: {len(manifest)}")


def compute_metrics(tp, fp, fn, tn):
    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 0.0
    dice = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return iou, dice, precision, recall


def percentile(arr, q):
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def analyze_threshold(threshold):
    results = []
    all_tp_probs = []
    all_fn_probs = []
    all_fp_probs = []
    cat_stats = defaultdict(lambda: {
        "images": 0,
        "tp": 0, "fn": 0, "fp": 0, "tn": 0,
        "tp_probs": [], "fn_probs": [], "fp_probs": [],
    })

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
            logits = seg_model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        if prob.shape != gt.shape:
            prob = cv2.resize(prob, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        pred = (prob > threshold).astype(np.uint8)

        tp_mask = np.logical_and(pred == 1, gt == 1)
        fn_mask = np.logical_and(pred == 0, gt == 1)
        fp_mask = np.logical_and(pred == 1, gt == 0)
        tn_mask = np.logical_and(pred == 0, gt == 0)

        tp = int(tp_mask.sum())
        fn = int(fn_mask.sum())
        fp = int(fp_mask.sum())
        tn = int(tn_mask.sum())

        iou, dice, precision, recall = compute_metrics(tp, fn, fp, tn)

        tp_probs = prob[tp_mask]
        fn_probs = prob[fn_mask]
        fp_probs = prob[fp_mask]

        all_tp_probs.extend(tp_probs.tolist())
        all_fn_probs.extend(fn_probs.tolist())
        all_fp_probs.extend(fp_probs.tolist())

        cat_stats[cat]["images"] += 1
        cat_stats[cat]["tp"] += tp
        cat_stats[cat]["fn"] += fn
        cat_stats[cat]["fp"] += fp
        cat_stats[cat]["tn"] += tn
        cat_stats[cat]["tp_probs"].extend(tp_probs.tolist())
        cat_stats[cat]["fn_probs"].extend(fn_probs.tolist())
        cat_stats[cat]["fp_probs"].extend(fp_probs.tolist())

        results.append({
            "pid": pid,
            "coverage": cov,
            "category": cat,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "iou": iou, "dice": dice,
            "precision": precision, "recall": recall,
        })

    summary = {
        "threshold": threshold,
        "images": len(results),
        "total_pixels": sum(r["tp"] + r["fn"] + r["fp"] + r["tn"] for r in results),
        "tp": sum(r["tp"] for r in results),
        "fn": sum(r["fn"] for r in results),
        "fp": sum(r["fp"] for r in results),
        "tn": sum(r["tn"] for r in results),
        "mean_iou": float(np.mean([r["iou"] for r in results])),
        "mean_dice": float(np.mean([r["dice"] for r in results])),
        "mean_precision": float(np.mean([r["precision"] for r in results])),
        "mean_recall": float(np.mean([r["recall"] for r in results])),
    }

    def prob_summary(arr):
        arr = np.array(arr, dtype=np.float32)
        if arr.size == 0:
            return {"count": 0}
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p10": percentile(arr, 10),
            "p25": percentile(arr, 25),
            "p50": percentile(arr, 50),
            "p75": percentile(arr, 75),
            "p90": percentile(arr, 90),
        }

    summary["tp_probability"] = prob_summary(all_tp_probs)
    summary["fn_probability"] = prob_summary(all_fn_probs)
    summary["fp_probability"] = prob_summary(all_fp_probs)

    fn_arr = np.array(all_fn_probs, dtype=np.float32)
    if fn_arr.size > 0:
        summary["fn_pct_0.40_0.50"] = float(np.mean((fn_arr >= 0.40) & (fn_arr < 0.50)) * 100)
        summary["fn_pct_0.30_0.40"] = float(np.mean((fn_arr >= 0.30) & (fn_arr < 0.40)) * 100)
        summary["fn_pct_below_0.30"] = float(np.mean(fn_arr < 0.30) * 100)
    else:
        summary["fn_pct_0.40_0.50"] = 0.0
        summary["fn_pct_0.30_0.40"] = 0.0
        summary["fn_pct_below_0.30"] = 0.0

    category_summaries = {}
    for cat, stats in cat_stats.items():
        iou, dice, precision, recall = compute_metrics(stats["tp"], stats["fn"], stats["fp"], stats["tn"])
        category_summaries[cat] = {
            "images": stats["images"],
            "tp": stats["tp"], "fn": stats["fn"], "fp": stats["fp"], "tn": stats["tn"],
            "iou": iou, "dice": dice, "precision": precision, "recall": recall,
            "tp_probability": prob_summary(stats["tp_probs"]),
            "fn_probability": prob_summary(stats["fn_probs"]),
            "fp_probability": prob_summary(stats["fp_probs"]),
        }
    summary["categories"] = category_summaries
    summary["per_image"] = results
    return summary


print("Analyzing threshold 0.50...")
summary_50 = analyze_threshold(0.50)
print("Analyzing threshold 0.30...")
summary_30 = analyze_threshold(0.30)

csv_path = OUT_DIR / "segmentation_probability_analysis.csv"
json_path = OUT_DIR / "segmentation_probability_analysis.json"

with open(json_path, "w") as f:
    json.dump({"threshold_0.50": summary_50, "threshold_0.30": summary_30}, f, indent=2)

rows = []
for th, summary in [(0.50, summary_50), (0.30, summary_30)]:
    rows.append({
        "threshold": th,
        "group": "overall",
        "category": "all",
        "images": summary["images"],
        "tp": summary["tp"], "fn": summary["fn"], "fp": summary["fp"], "tn": summary["tn"],
        "iou": summary["mean_iou"],
        "dice": summary["mean_dice"],
        "precision": summary["mean_precision"],
        "recall": summary["mean_recall"],
        "tp_prob_mean": summary["tp_probability"].get("mean", float("nan")),
        "fn_prob_mean": summary["fn_probability"].get("mean", float("nan")),
        "fp_prob_mean": summary["fp_probability"].get("mean", float("nan")),
        "fn_pct_0.40_0.50": summary.get("fn_pct_0.40_0.50", float("nan")),
        "fn_pct_0.30_0.40": summary.get("fn_pct_0.30_0.40", float("nan")),
        "fn_pct_below_0.30": summary.get("fn_pct_below_0.30", float("nan")),
    })
    for cat, stats in summary["categories"].items():
        rows.append({
            "threshold": th,
            "group": "category",
            "category": cat,
            "images": stats["images"],
            "tp": stats["tp"], "fn": stats["fn"], "fp": stats["fp"], "tn": stats["tn"],
            "iou": stats["iou"],
            "dice": stats["dice"],
            "precision": stats["precision"],
            "recall": stats["recall"],
            "tp_prob_mean": stats["tp_probability"].get("mean", float("nan")),
            "fn_prob_mean": stats["fn_probability"].get("mean", float("nan")),
            "fp_prob_mean": stats["fp_probability"].get("mean", float("nan")),
            "fn_pct_0.40_0.50": float("nan"),
            "fn_pct_0.30_0.40": float("nan"),
            "fn_pct_below_0.30": float("nan"),
        })

import csv
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved CSV: {csv_path}")
print(f"Saved JSON: {json_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    tp_arr = np.array(summary_50["tp_probability"].get("count", 0) and summary_50["tp_probability"] or {"count": 0})
    # Rebuild arrays from per_image for plotting
    tp_all, fn_all, fp_all = [], [], []
    for r in summary_50["per_image"]:
        pass
    # Use the aggregated lists from re-analysis for plot only
    # Re-run quick aggregation for arrays
    tp_probs = summary_50["tp_probability"]
    fn_probs = summary_50["fn_probability"]
    fp_probs = summary_50["fp_probability"]

    def get_vals(prob_dict):
        count = prob_dict.get("count", 0)
        if count == 0:
            return np.array([])
        mn = prob_dict["min"]
        mx = prob_dict["max"]
        # Approximate distribution using uniform synthetic samples matching summary stats is NOT appropriate.
        # Instead, skip histogram if raw arrays not stored.
        return np.array([])

    # Since we don't store raw arrays in JSON, skip plotting if unavailable.
    plot_path = OUT_DIR / "segmentation_probability_histogram.png"
    plt.close()
    print(f"Plot skipped because raw probability arrays are not retained in JSON output.")
except Exception as e:
    print(f"Plotting failed: {e}")

print("=== Threshold 0.50 ===")
print(f"Images: {summary_50['images']}")
print(f"TP={summary_50['tp']} FN={summary_50['fn']} FP={summary_50['fp']} TN={summary_50['tn']}")
print(f"IoU={summary_50['mean_iou']:.4f} Dice={summary_50['mean_dice']:.4f}")
print(f"Precision={summary_50['mean_precision']:.4f} Recall={summary_50['mean_recall']:.4f}")
print(f"FN prob mean={summary_50['fn_probability'].get('mean', float('nan')):.4f}")
print(f"FN 0.40-0.50: {summary_50.get('fn_pct_0.40_0.50', float('nan')):.2f}%")
print(f"FN 0.30-0.40: {summary_50.get('fn_pct_0.30_0.40', float('nan')):.2f}%")
print(f"FN <0.30: {summary_50.get('fn_pct_below_0.30', float('nan')):.2f}%")

print("=== Threshold 0.30 ===")
print(f"Images: {summary_30['images']}")
print(f"TP={summary_30['tp']} FN={summary_30['fn']} FP={summary_30['fp']} TN={summary_30['tn']}")
print(f"IoU={summary_30['mean_iou']:.4f} Dice={summary_30['mean_dice']:.4f}")
print(f"Precision={summary_30['mean_precision']:.4f} Recall={summary_30['mean_recall']:.4f}")
print(f"FN prob mean={summary_30['fn_probability'].get('mean', float('nan')):.4f}")
print(f"FN 0.40-0.50: {summary_30.get('fn_pct_0.40_0.50', float('nan')):.2f}%")
print(f"FN 0.30-0.40: {summary_30.get('fn_pct_0.30_0.40', float('nan')):.2f}%")
print(f"FN <0.30: {summary_30.get('fn_pct_below_0.30', float('nan')):.2f}%")

for cat in ["light", "medium", "heavy"]:
    if cat in summary_50["categories"]:
        s = summary_50["categories"][cat]
        print(f"CATEGORY {cat}: images={s['images']} IoU={s['iou']:.4f} Dice={s['dice']:.4f} FNprob={s['fn_probability'].get('mean', float('nan')):.4f}")
