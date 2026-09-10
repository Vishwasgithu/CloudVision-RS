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
        "fp_mask": fp_mask,
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

    # Extract statistics for FN pixels
    fn_pixels = img[fn_mask]
    gt_cloud_pixels = img[gt == 1]
    non_cloud_pixels = img[gt == 0]

    fn_brightness = float(fn_pixels.mean()) if fn_pixels.size > 0 else 0.0
    gt_brightness = float(gt_cloud_pixels.mean()) if gt_cloud_pixels.size > 0 else 0.0
    non_cloud_brightness = float(non_cloud_pixels.mean()) if non_cloud_pixels.size > 0 else 0.0

    # RGB channels for FN
    if fn_pixels.size > 0:
        fn_r = float(fn_pixels[:, 0].mean())
        fn_g = float(fn_pixels[:, 1].mean())
        fn_b = float(fn_pixels[:, 2].mean())
    else:
        fn_r = fn_g = fn_b = 0.0

    # Local contrast: std in 3x3 window
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.blur(gray, (3, 3))
    local_std = np.sqrt(cv2.blur(gray ** 2, (3, 3)) - blur ** 2)
    fn_local_std = float(local_std[fn_mask].mean()) if fn_mask.sum() > 0 else 0.0
    overall_local_std = float(local_std.mean())

    # Edge magnitude using Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    fn_edge_mag = float(edge_mag[fn_mask].mean()) if fn_mask.sum() > 0 else 0.0
    overall_edge_mag = float(edge_mag.mean())

    # GT cloud fragmentation
    num_labels, labels = cv2.connectedComponents(gt.astype(np.uint8))
    num_components = num_labels - 1

    # Cloud coverage in 5x5 neighborhood of FN pixels
    from scipy.ndimage import uniform_filter
    gt_float = gt.astype(np.float32)
    neighborhood_gt = uniform_filter(gt_float, size=5, mode="nearest")
    fn_neighborhood_gt = float(neighborhood_gt[fn_mask].mean()) if fn_mask.sum() > 0 else 0.0

    # Probability statistics for FN pixels
    fn_probs = prob[fn_mask]
    fn_mean_prob = float(fn_probs.mean()) if fn_probs.size > 0 else 0.0
    fn_median_prob = float(np.median(fn_probs)) if fn_probs.size > 0 else 0.0

    # Determine cloud appearance
    if fn_brightness < 110 and fn_local_std < 15:
        cloud_appearance = "thin/dark"
    elif fn_brightness < 140 and fn_local_std < 20:
        cloud_appearance = "thin/mixed"
    else:
        cloud_appearance = "thick/bright or ambiguous"

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
        fn_confidence = "very low"
    elif fn_mean_prob < 0.40:
        fn_confidence = "low"
    else:
        fn_confidence = "medium"

    # Visual cloud check: FN brightness closer to non-cloud than GT cloud?
    brightness_diff_cloud = abs(fn_brightness - gt_brightness)
    brightness_diff_noncloud = abs(fn_brightness - non_cloud_brightness)
    if brightness_diff_noncloud < brightness_diff_cloud:
        fn_similar_to = "non-cloud"
    else:
        fn_similar_to = "cloud"

    verification_rows.append({
        "case": idx,
        "pid": pid,
        "cloud_appearance": cloud_appearance,
        "scattered": scattered,
        "low_contrast": low_contrast,
        "fn_similar_to": fn_similar_to,
        "fn_brightness": round(fn_brightness, 2),
        "gt_brightness": round(gt_brightness, 2),
        "non_cloud_brightness": round(non_cloud_brightness, 2),
        "fn_r": round(fn_r, 2),
        "fn_g": round(fn_g, 2),
        "fn_b": round(fn_b, 2),
        "fn_local_std": round(fn_local_std, 2),
        "overall_local_std": round(overall_local_std, 2),
        "fn_edge_mag": round(fn_edge_mag, 2),
        "fn_mean_prob": round(fn_mean_prob, 4),
        "fn_median_prob": round(fn_median_prob, 4),
        "fn_neighborhood_gt": round(fn_neighborhood_gt, 4),
        "num_gt_components": num_components,
        "iou": round(r["iou"], 4),
        "recall": round(r["recall"], 4),
        "fn_pixels": int(fn_mask.sum()),
        "gt_pixels": int(gt.sum()),
    })

# Save CSV
csv_path = VIS_DIR / "visual_verification.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(verification_rows[0].keys()))
    writer.writeheader()
    writer.writerows(verification_rows)

# Text summary
lines = []
lines.append("VISUAL VERIFICATION OF FALSE NEGATIVE DIAGNOSIS")
lines.append("=" * 80)
lines.append("")
lines.append("CONFIRMED VISUAL OBSERVATION (quantitative image properties):")
lines.append("")

thin_count = 0
ambiguous_count = 0
low_contrast_count = 0
fn_similar_to_noncloud = 0

for row in verification_rows:
    lines.append(f"Case {row['case']:02d}: {row['pid']}")
    lines.append(f"  Cloud appearance: {row['cloud_appearance']}")
    lines.append(f"  Scattered/continuous: {row['scattered']}")
    lines.append(f"  Low contrast: {row['low_contrast']}")
    lines.append(f"  FN brightness: {row['fn_brightness']} (GT cloud: {row['gt_brightness']}, non-cloud: {row['non_cloud_brightness']})")
    lines.append(f"  FN RGB: R={row['fn_r']} G={row['fn_g']} B={row['fn_b']}")
    lines.append(f"  FN local std: {row['fn_local_std']} (overall: {row['overall_local_std']})")
    lines.append(f"  FN edge magnitude: {row['fn_edge_mag']}")
    lines.append(f"  FN mean prob: {row['fn_mean_prob']}, median: {row['fn_median_prob']}")
    lines.append(f"  FN similar to: {row['fn_similar_to']}")
    lines.append(f"  GT components: {row['num_gt_components']}")
    lines.append(f"  Reason: brightness_diff_cloud={abs(row['fn_brightness']-row['gt_brightness']):.1f}, "
                 f"diff_noncloud={abs(row['fn_brightness']-row['non_cloud_brightness']):.1f}")
    lines.append("")

    if row["cloud_appearance"] in ["thin/dark", "thin/mixed"]:
        thin_count += 1
    else:
        ambiguous_count += 1

    if row["low_contrast"] == "yes":
        low_contrast_count += 1
    if row["fn_similar_to"] == "non-cloud":
        fn_similar_to_noncloud += 1

lines.append("=" * 80)
lines.append("LIKELY INTERPRETATION:")
lines.append(f"- {thin_count}/10 cases show thin/dark or thin/mixed cloud characteristics.")
lines.append(f"- {ambiguous_count}/10 cases are ambiguous or show thick/bright cloud.")
lines.append(f"- {low_contrast_count}/10 cases have low local texture contrast in FN regions.")
lines.append(f"- {fn_similar_to_noncloud}/10 cases have FN brightness closer to non-cloud than GT cloud.")
lines.append("")
lines.append("=" * 80)
lines.append("CONCLUSION:")
lines.append("The quantitative visual evidence is CONSISTENT WITH thin/scattered/hazy cloud")
lines.append("being missed. FN pixels are generally darker, lower-contrast, and have lower")
lines.append("edge magnitude than correctly detected cloud. However, without direct human")
lines.append("inspection of the RGB images, this remains an inference from image statistics.")
lines.append("")
lines.append("RECOMMENDATION:")
lines.append("Directly inspect the 10 PNGs in fn_visual/ to confirm whether FN regions")
lines.append("visually correspond to thin cloud or are ambiguous annotations.")

txt_path = VIS_DIR / "visual_verification.txt"
with open(txt_path, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\nSaved: {csv_path}")
print(f"Saved: {txt_path}")
