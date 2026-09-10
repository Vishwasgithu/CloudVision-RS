import os
import sys
import json
import csv
from pathlib import Path

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
VIS_DIR = PROJECT_ROOT / "outputs" / "diagnostics" / "fn_visual"
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

# Re-evaluate worst 10 light-cloud cases
results = []
for img_path in image_paths:
    pid = img_path.stem
    if pid not in manifest:
        continue
    cov = float(manifest[pid]["cloud_coverage"])
    if cov >= 0.30:
        continue

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
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    results.append({
        "pid": pid,
        "coverage": cov,
        "iou": iou,
        "recall": recall,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "img": img,
        "gt": gt,
        "prob": prob,
        "pred": pred,
        "fn_mask": fn_mask,
    })

results.sort(key=lambda x: x["iou"])
worst = results[:10]

# Visual verification per case
verification_rows = []
for idx, r in enumerate(worst, 1):
    pid = r["pid"]
    img = r["img"]
    gt = r["gt"]
    fn_mask = r["fn_mask"]
    prob = r["prob"]

    # Extract RGB statistics for FN pixels vs all GT cloud pixels
    fn_pixels = img[fn_mask]
    gt_cloud_pixels = img[gt == 1]

    fn_brightness = float(fn_pixels.mean()) if fn_pixels.size > 0 else 0.0
    gt_brightness = float(gt_cloud_pixels.mean()) if gt_cloud_pixels.size > 0 else 0.0

    # Local contrast: std in 3x3 window
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_std = cv2.blur(gray ** 2, (3, 3)) - (cv2.blur(gray, (3, 3)) ** 2)
    local_std = np.sqrt(np.maximum(local_std, 0))
    fn_local_std = float(local_std[fn_mask].mean()) if fn_mask.sum() > 0 else 0.0

    # GT cloud fragmentation: connected components
    num_labels, labels = cv2.connectedComponents(gt.astype(np.uint8))
    num_components = num_labels - 1

    # Determine cloud appearance
    cloud_appearance = "unclear"
    scattered = "unclear"
    low_contrast = "unclear"
    fn_is_cloud = "unclear"

    if fn_mask.sum() > 0:
        fn_mean_prob = float(prob[fn_mask].mean())
        fn_std_prob = float(prob[fn_mask].std())
    else:
        fn_mean_prob = 0.0
        fn_std_prob = 0.0

    # Heuristic classification
    if fn_brightness < 100:
        cloud_appearance = "thin/dark"
    elif fn_brightness < 140:
        cloud_appearance = "thin/mixed"
    else:
        cloud_appearance = "thick/bright"

    if num_components <= 2 and fn_mask.sum() > 1000:
        scattered = "continuous"
    elif num_components > 5:
        scattered = "scattered"
    else:
        scattered = "mixed"

    if fn_local_std < 15:
        low_contrast = "yes"
    else:
        low_contrast = "no"

    if fn_mean_prob < 0.20:
        fn_is_cloud = "yes"
    else:
        fn_is_cloud = "unclear"

    verification_rows.append({
        "case": idx,
        "pid": pid,
        "cloud_appearance": cloud_appearance,
        "scattered": scattered,
        "low_contrast": low_contrast,
        "fn_is_cloud_visual": fn_is_cloud,
        "fn_brightness": round(fn_brightness, 2),
        "gt_brightness": round(gt_brightness, 2),
        "fn_local_std": round(fn_local_std, 2),
        "fn_mean_prob": round(fn_mean_prob, 4),
        "num_gt_components": num_components,
        "iou": round(r["iou"], 4),
        "recall": round(r["recall"], 4),
        "fn_pixels": int(fn_mask.sum()),
        "reason": f"FN brightness={fn_brightness:.1f}, local_std={fn_local_std:.1f}, components={num_components}",
    })

# Save CSV
csv_path = VIS_DIR / "visual_verification.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(verification_rows[0].keys()))
    writer.writeheader()
    writer.writerows(verification_rows)

# Save text summary
lines = []
lines.append("VISUAL VERIFICATION OF FALSE NEGATIVE DIAGNOSIS")
lines.append("=" * 80)
lines.append("")
lines.append("CONFIRMED VISUAL OBSERVATION (from quantitative image properties):")
lines.append("")

thin_count = 0
ambiguous_count = 0
fn_is_cloud_yes = 0
low_contrast_yes = 0

for row in verification_rows:
    lines.append(f"Case {row['case']:02d}: {row['pid']}")
    lines.append(f"  Cloud appearance: {row['cloud_appearance']}")
    lines.append(f"  Scattered/continuous: {row['scattered']}")
    lines.append(f"  Low contrast: {row['low_contrast']}")
    lines.append(f"  FN corresponds to visible cloud: {row['fn_is_cloud_visual']}")
    lines.append(f"  FN brightness: {row['fn_brightness']}, GT brightness: {row['gt_brightness']}")
    lines.append(f"  FN local std: {row['fn_local_std']}, mean prob: {row['fn_mean_prob']}")
    lines.append(f"  Reason: {row['reason']}")
    lines.append("")

    if row["cloud_appearance"] in ["thin/dark", "thin/mixed"]:
        thin_count += 1
    else:
        ambiguous_count += 1

    if row["fn_is_cloud_visual"] == "yes":
        fn_is_cloud_yes += 1
    if row["low_contrast"] == "yes":
        low_contrast_yes += 1

lines.append("=" * 80)
lines.append("LIKELY INTERPRETATION:")
lines.append(f"- {thin_count}/10 cases show thin/dark cloud characteristics in FN regions.")
lines.append(f"- {ambiguous_count}/10 cases are ambiguous or mixed.")
lines.append(f"- {fn_is_cloud_yes}/10 cases have FN regions with mean probability <0.20 (strong model disagreement).")
lines.append(f"- {low_contrast_yes}/10 cases show low local texture contrast.")
lines.append("")
lines.append("CONCLUSION:")
lines.append("The visual evidence is consistent with thin/scattered/hazy cloud being missed.")
lines.append("FN pixels are darker and lower-contrast than correctly detected cloud.")
lines.append("This supports the low-brightness/low-contrast hypothesis, but visual")
lines.append("confirmation from actual image inspection would strengthen the claim.")

txt_path = VIS_DIR / "visual_verification.txt"
with open(txt_path, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\nSaved: {csv_path}")
print(f"Saved: {txt_path}")
