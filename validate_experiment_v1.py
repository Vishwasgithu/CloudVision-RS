import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\CloudRemoval_Project")
from src.models.segmentation import AttentionUNet

PROJECT_ROOT = Path(r"D:\CloudRemoval_Project")
BASELINE_CKPT = PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation" / "best_iou0.7935_ep39.pt"
V1_CKPT = PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation" / "best_iou0.7703_ep27_brightness_aug.ckpt"
TEST_CLOUD_DIR = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "cloud"
TEST_MASK_DIR = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "mask"
TEST_MANIFEST = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "patch_manifest.json"
RICE1_CLOUD_DIR = PROJECT_ROOT / "datasets" / "RICE2" / "RICE1" / "cloud"
CONFIG_PATH = PROJECT_ROOT / "configs" / "seg_config.yaml"
OUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics" / "brightness_aug_v1_validation"
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


def load_seg_model(path):
    model = AttentionUNet(seg_config).to(DEVICE)
    ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
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


print("Loading models...")
baseline_model = load_seg_model(BASELINE_CKPT)
v1_model = load_seg_model(V1_CKPT)

with open(TEST_MANIFEST, "r") as f:
    manifest = json.load(f)

image_paths = sorted(TEST_CLOUD_DIR.glob("*.png"))


def compute_metrics(tp, fp, fn, tn):
    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 0.0
    dice = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return iou, dice, precision, recall


# 1. Evaluate both models on all 536 test patches
print("Evaluating on test set...")
baseline_test = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "per_image": []}
v1_test = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "per_image": []}

for img_path in image_paths:
    pid = img_path.stem
    if pid not in manifest:
        continue
    cov = float(manifest[pid]["cloud_coverage"])
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_path = TEST_MASK_DIR / f"{pid}.png"
    gt_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    gt = (gt_raw > 127).astype(np.uint8)

    transformed = seg_transform(image=img)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits_b = baseline_model(tensor)
        prob_b = torch.sigmoid(logits_b)[0, 0].cpu().numpy()

        logits_v = v1_model(tensor)
        prob_v = torch.sigmoid(logits_v)[0, 0].cpu().numpy()

    if prob_b.shape != gt.shape:
        prob_b = cv2.resize(prob_b, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        prob_v = cv2.resize(prob_v, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    pred_b = (prob_b > 0.50).astype(np.uint8)
    pred_v = (prob_v > 0.50).astype(np.uint8)

    for pred, results in [(pred_b, baseline_test), (pred_v, v1_test)]:
        tp = int(np.logical_and(pred == 1, gt == 1).sum())
        fn = int(np.logical_and(pred == 0, gt == 1).sum())
        fp = int(np.logical_and(pred == 1, gt == 0).sum())
        tn = int(np.logical_and(pred == 0, gt == 0).sum())
        results["tp"] += tp
        results["fn"] += fn
        results["fp"] += fp
        results["tn"] += tn
        iou, dice, precision, recall = compute_metrics(tp, fp, fn, tn)
        results["per_image"].append({
            "pid": pid, "coverage": cov,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "iou": iou, "dice": dice, "precision": precision, "recall": recall
        })

# Compute global metrics
def global_metrics(r):
    iou, dice, precision, recall = compute_metrics(r["tp"], r["fp"], r["fn"], r["tn"])
    return {"tp": r["tp"], "fn": r["fn"], "fp": r["fp"], "tn": r["tn"],
            "iou": iou, "dice": dice, "precision": precision, "recall": recall}

baseline_global = global_metrics(baseline_test)
v1_global = global_metrics(v1_test)

# 2. Generate comparison visualizations for worst 10 light-cloud cases
print("Generating comparison visualizations...")
worst_light = [r for r in baseline_test["per_image"] if r["coverage"] < 0.30]
worst_light.sort(key=lambda x: x["iou"])
worst_10 = worst_light[:10]

fig_dir = OUT_DIR / "comparison_plots"
fig_dir.mkdir(parents=True, exist_ok=True)

for idx, r in enumerate(worst_10, 1):
    pid = r["pid"]
    img = cv2.imread(str(TEST_CLOUD_DIR / f"{pid}.png"), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_path = TEST_MASK_DIR / f"{pid}.png"
    gt_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    gt = (gt_raw > 127).astype(np.uint8)

    transformed = seg_transform(image=img)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits_b = baseline_model(tensor)
        prob_b = torch.sigmoid(logits_b)[0, 0].cpu().numpy()
        logits_v = v1_model(tensor)
        prob_v = torch.sigmoid(logits_v)[0, 0].cpu().numpy()

    if prob_b.shape != gt.shape:
        prob_b = cv2.resize(prob_b, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        prob_v = cv2.resize(prob_v, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    pred_b = (prob_b > 0.50).astype(np.uint8)
    pred_v = (prob_v > 0.50).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{pid} | Coverage={r['coverage']:.1%} | Baseline IoU={r['iou']:.3f}")

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("GT Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_b, cmap="gray")
    axes[2].set_title("Baseline Prediction")
    axes[2].axis("off")

    axes[3].imshow(pred_v, cmap="gray")
    axes[3].set_title("V1 Prediction")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(fig_dir / f"case_{idx:02d}_{pid}.png", dpi=150, bbox_inches="tight")
    plt.close()

# 3. RICE1 comparison
print("Running RICE1 comparison...")
rice1_paths = sorted(RICE1_CLOUD_DIR.glob("*.png")) if RICE1_CLOUD_DIR.exists() else []
rice1_stats = {"baseline": {"probs": [], "coverage": []}, "v1": {"probs": [], "coverage": []}}

for img_path in rice1_paths:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = seg_transform(image=img)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits_b = baseline_model(tensor)
        prob_b = torch.sigmoid(logits_b)[0, 0].cpu().numpy()
        logits_v = v1_model(tensor)
        prob_v = torch.sigmoid(logits_v)[0, 0].cpu().numpy()
    pred_b = (prob_b > 0.50).astype(np.uint8)
    pred_v = (prob_v > 0.50).astype(np.uint8)
    rice1_stats["baseline"]["probs"].append(prob_b.mean())
    rice1_stats["baseline"]["coverage"].append(pred_b.mean())
    rice1_stats["v1"]["probs"].append(prob_v.mean())
    rice1_stats["v1"]["coverage"].append(pred_v.mean())

# Save results
results = {
    "baseline_global": baseline_global,
    "v1_global": v1_global,
    "changes": {
        "iou": v1_global["iou"] - baseline_global["iou"],
        "dice": v1_global["dice"] - baseline_global["dice"],
        "precision": v1_global["precision"] - baseline_global["precision"],
        "recall": v1_global["recall"] - baseline_global["recall"],
    },
    "rice1_stats": {
        "baseline_mean_prob": float(np.mean(rice1_stats["baseline"]["probs"])) if rice1_stats["baseline"]["probs"] else 0.0,
        "baseline_mean_coverage": float(np.mean(rice1_stats["baseline"]["coverage"])) if rice1_stats["baseline"]["coverage"] else 0.0,
        "v1_mean_prob": float(np.mean(rice1_stats["v1"]["probs"])) if rice1_stats["v1"]["probs"] else 0.0,
        "v1_mean_coverage": float(np.mean(rice1_stats["v1"]["coverage"])) if rice1_stats["v1"]["coverage"] else 0.0,
        "num_images": len(rice1_paths),
    }
}

with open(OUT_DIR / "validation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary text
lines = []
lines.append("=" * 80)
lines.append("BRIGHTNESS AUGMENTATION V1 VALIDATION")
lines.append("=" * 80)
lines.append("")
lines.append("TEST SET RESULTS (536 patches):")
lines.append(f"  Baseline: IoU={baseline_global['iou']:.4f} Dice={baseline_global['dice']:.4f} Prec={baseline_global['precision']:.4f} Rec={baseline_global['recall']:.4f}")
lines.append(f"  V1:       IoU={v1_global['iou']:.4f} Dice={v1_global['dice']:.4f} Prec={v1_global['precision']:.4f} Rec={v1_global['recall']:.4f}")
lines.append(f"  Change:   IoU={results['changes']['iou']:+.4f} Dice={results['changes']['dice']:+.4f} Prec={results['changes']['precision']:+.4f} Rec={results['changes']['recall']:+.4f}")
lines.append("")
lines.append("RICE1 COMPARISON (no GT masks):")
lines.append(f"  Images: {len(rice1_paths)}")
lines.append(f"  Baseline mean prob: {results['rice1_stats']['baseline_mean_prob']:.4f}")
lines.append(f"  Baseline mean coverage: {results['rice1_stats']['baseline_mean_coverage']:.4f}")
lines.append(f"  V1 mean prob: {results['rice1_stats']['v1_mean_prob']:.4f}")
lines.append(f"  V1 mean coverage: {results['rice1_stats']['v1_mean_coverage']:.4f}")
lines.append("")
lines.append("COMPARISON PLOTS SAVED TO:")
lines.append(f"  {fig_dir}")
lines.append("")

if v1_global["iou"] > baseline_global["iou"]:
    lines.append("CONCLUSION: V1 improves overall segmentation on test set.")
else:
    lines.append("CONCLUSION: V1 does not improve overall segmentation on test set.")

if results["rice1_stats"]["v1_mean_coverage"] > results["rice1_stats"]["baseline_mean_coverage"]:
    lines.append("V1 predicts MORE cloud on RICE1 than baseline.")
else:
    lines.append("V1 predicts LESS or EQUAL cloud on RICE1 than baseline.")

lines.append("=" * 80)

with open(OUT_DIR / "validation_summary.txt", "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\nSaved results to: {OUT_DIR}")
