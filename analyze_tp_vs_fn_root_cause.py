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
from scipy.ndimage import uniform_filter

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
print(f"Test patches: {len(image_paths)}")


def compute_metrics(tp, fp, fn, tn):
    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 0.0
    dice = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return iou, dice, precision, recall


def prob_summary(arr):
    arr = np.array(arr, dtype=np.float32)
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "p10": float("nan"), "p25": float("nan"),
                "p50": float("nan"), "p75": float("nan"), "p90": float("nan")}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


# Collect per-pixel diagnostics
tp_records = []
fn_records = []
fp_records = []

per_image_records = []
cat_stats = defaultdict(lambda: {"images": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0, "fn_probs": []})

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

    pred = (prob > 0.50).astype(np.uint8)

    tp_mask = np.logical_and(pred == 1, gt == 1)
    fn_mask = np.logical_and(pred == 0, gt == 1)
    fp_mask = np.logical_and(pred == 1, gt == 0)

    tp = int(tp_mask.sum())
    fn = int(fn_mask.sum())
    fp = int(fp_mask.sum())
    tn = int((np.logical_and(pred == 0, gt == 0)).sum())

    iou, dice, precision, recall = compute_metrics(tp, fp, fn, tn)

    # Per-pixel diagnostics
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = np.abs(texture)

    gt_float = gt.astype(np.float32)
    neighborhood_gt = uniform_filter(gt_float, size=5, mode="nearest")
    neighborhood_pred = uniform_filter(prob, size=5, mode="nearest")

    # Cloud fragmentation: count connected components in GT cloud mask
    num_gt_components = 0
    if gt.max() > 0:
        num_labels, labels = cv2.connectedComponents(gt.astype(np.uint8))
        num_gt_components = num_labels - 1  # exclude background

    def record_pixels(mask, records, label):
        if mask.sum() == 0:
            return
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            records.append({
                "pid": pid,
                "category": cat,
                "label": label,
                "probability": float(prob[y, x]),
                "brightness": float(gray[y, x]),
                "r": float(img[y, x, 0]),
                "g": float(img[y, x, 1]),
                "b": float(img[y, x, 2]),
                "texture": float(texture[y, x]),
                "neighborhood_gt_frac": float(neighborhood_gt[y, x]),
                "neighborhood_pred_prob": float(neighborhood_pred[y, x]),
                "gt_components": int(num_gt_components),
                "coverage": float(cov),
            })

    record_pixels(tp_mask, tp_records, "TP")
    record_pixels(fn_mask, fn_records, "FN")
    record_pixels(fp_mask, fp_records, "FP")

    cat_stats[cat]["images"] += 1
    cat_stats[cat]["tp"] += tp
    cat_stats[cat]["fn"] += fn
    cat_stats[cat]["fp"] += fp
    cat_stats[cat]["tn"] += tn
    cat_stats[cat]["fn_probs"].extend(prob[fn_mask].tolist())

    per_image_records.append({
        "pid": pid,
        "coverage": cov,
        "category": cat,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "iou": iou, "dice": dice,
        "precision": precision, "recall": recall,
    })

# Global metrics
global_tp = sum(r["tp"] for r in per_image_records)
global_fn = sum(r["fn"] for r in per_image_records)
global_fp = sum(r["fp"] for r in per_image_records)
global_tn = sum(r["tn"] for r in per_image_records)
iou, dice, precision, recall = compute_metrics(global_tp, global_fp, global_fn, global_tn)

# Per-pixel summaries
def group_summary(records):
    if not records:
        return {}
    keys = ["probability", "brightness", "r", "g", "b", "texture", "neighborhood_gt_frac", "neighborhood_pred_prob"]
    result = {}
    for k in keys:
        vals = [r[k] for r in records]
        result[k] = prob_summary(vals)
    return result

tp_summary = group_summary(tp_records)
fn_summary = group_summary(fn_records)
fp_summary = group_summary(fp_records)

# Category summaries
category_summaries = {}
for cat, stats in cat_stats.items():
    iou_c, dice_c, prec_c, rec_c = compute_metrics(stats["tp"], stats["fp"], stats["fn"], stats["tn"])
    category_summaries[cat] = {
        "images": stats["images"],
        "tp": stats["tp"], "fn": stats["fn"], "fp": stats["fp"], "tn": stats["tn"],
        "iou": iou_c, "dice": dice_c, "precision": prec_c, "recall": rec_c,
    }

# Save JSON
json_path = OUT_DIR / "tp_vs_fn_root_cause.json"
with open(json_path, "w") as f:
    json.dump({
        "threshold": 0.50,
        "global_metrics": {
            "tp": global_tp, "fn": global_fn, "fp": global_fp, "tn": global_tn,
            "iou": iou, "dice": dice, "precision": precision, "recall": recall,
        },
        "tp_summary": tp_summary,
        "fn_summary": fn_summary,
        "fp_summary": fp_summary,
        "categories": category_summaries,
        "per_image": per_image_records,
    }, f, indent=2)

# Save CSV
csv_rows = []
for r in per_image_records:
    csv_rows.append(r)
csv_path = OUT_DIR / "tp_vs_fn_root_cause.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(csv_rows)

# Text summary
lines = []
lines.append("=" * 80)
lines.append("TP vs FN ROOT CAUSE ANALYSIS (RGB pipeline, threshold=0.50)")
lines.append("=" * 80)
lines.append("")
lines.append(f"Global: TP={global_tp} FN={global_fn} FP={global_fp} TN={global_tn}")
lines.append(f"IoU={iou:.4f} Dice={dice:.4f} Precision={precision:.4f} Recall={recall:.4f}")
lines.append("")
lines.append("PROBABILITY")
lines.append(f"  TP: mean={tp_summary.get('probability', {}).get('mean', float('nan')):.4f} median={tp_summary.get('probability', {}).get('median', float('nan')):.4f} std={tp_summary.get('probability', {}).get('std', float('nan')):.4f}")
lines.append(f"  FN: mean={fn_summary.get('probability', {}).get('mean', float('nan')):.4f} median={fn_summary.get('probability', {}).get('median', float('nan')):.4f} std={fn_summary.get('probability', {}).get('std', float('nan')):.4f}")
lines.append("")
lines.append("BRIGHTNESS")
lines.append(f"  TP: mean={tp_summary.get('brightness', {}).get('mean', float('nan')):.2f}")
lines.append(f"  FN: mean={fn_summary.get('brightness', {}).get('mean', float('nan')):.2f}")
lines.append("")
lines.append("RGB MEANS")
for ch in ["r", "g", "b"]:
    lines.append(f"  {ch.upper()} TP={tp_summary.get(ch, {}).get('mean', float('nan')):.2f} FN={fn_summary.get(ch, {}).get('mean', float('nan')):.2f}")
lines.append("")
lines.append("TEXTURE (Laplacian magnitude)")
lines.append(f"  TP: mean={tp_summary.get('texture', {}).get('mean', float('nan')):.2f}")
lines.append(f"  FN: mean={fn_summary.get('texture', {}).get('mean', float('nan')):.2f}")
lines.append("")
lines.append("NEIGHBOURHOOD (5x5)")
lines.append(f"  TP GT cloud frac: mean={tp_summary.get('neighborhood_gt_frac', {}).get('mean', float('nan')):.4f}")
lines.append(f"  FN GT cloud frac: mean={fn_summary.get('neighborhood_gt_frac', {}).get('mean', float('nan')):.4f}")
lines.append(f"  TP pred prob: mean={tp_summary.get('neighborhood_pred_prob', {}).get('mean', float('nan')):.4f}")
lines.append(f"  FN pred prob: mean={fn_summary.get('neighborhood_pred_prob', {}).get('mean', float('nan')):.4f}")
lines.append("")
lines.append("CATEGORY RECALL")
for cat in ["light", "medium", "heavy"]:
    if cat in category_summaries:
        s = category_summaries[cat]
        lines.append(f"  {cat}: images={s['images']} recall={s['recall']:.4f} IoU={s['iou']:.4f}")
lines.append("")
lines.append("=" * 80)
lines.append("STRONGEST FINDING")
lines.append("=" * 80)
tp_mean_prob = tp_summary.get("probability", {}).get("mean", float("nan"))
fn_mean_prob = fn_summary.get("probability", {}).get("mean", float("nan"))
fn_pct_low = float("nan")
fn_prob_vals = [r["probability"] for r in fn_records] if fn_records else []
if fn_prob_vals:
    fn_arr = np.array(fn_prob_vals)
    fn_pct_low = float(np.mean(fn_arr < 0.30) * 100)
lines.append(f"- TP mean prob={tp_mean_prob:.4f}, FN mean prob={fn_mean_prob:.4f}")
lines.append(f"- {fn_pct_low:.1f}% of FN pixels have prob<0.30")
lines.append("- Recall is lowest for light cloud.")
lines.append("=" * 80)

text_path = OUT_DIR / "tp_vs_fn_root_cause_summary.txt"
with open(text_path, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\nSaved: {json_path}")
print(f"Saved: {csv_path}")
print(f"Saved: {text_path}")
