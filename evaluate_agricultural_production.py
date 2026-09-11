"""
evaluate_agricultural_production.py

Evaluates the production cloud-removal pipeline on 116 agricultural patches
selected from 19 manually identified source scenes in the RICE2 test set.

This script:
  1. Reuses the EXACT production inference pipeline from inference_production.py
     (V1 segmentation checkpoint, threshold=0.5, Gaussian sliding-window
      segmentation, binary cloud mask, RGB+mask+Sobel → 5-channel GAN input,
      best_generator.pt, Gaussian reconstruction stitching).
  2. Reads patch_manifest.json and filters source_id to 19 agricultural scenes.
  3. For each patch: runs production inference, loads GT mask & GT clear image,
     and computes IoU, Dice, Precision, Recall, PSNR, SSIM, VARI-RMSE, LPIPS(if available).
  4. Saves per-patch CSV and summary text file.
  5. Saves worst Heavy-cloud cases for visual inspection.

No model checkpoints, training code, frontend, or dataset files are modified.
No filenames are hard-coded — all patches are selected dynamically from the manifest.
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml
from scipy.ndimage import gaussian_filter
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Project setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(PROJECT_ROOT))

# ── Reuse production model loading ──────────────────────────────────────────
# Importing inference_production loads the EXACT production models at module level:
#   - V1 segmentation checkpoint: outputs/checkpoints/segmentation/v1_brightness_aug_model_state.pt
#   - Production GAN checkpoint: outputs/checkpoints/gan/best_generator.pt
#   - Same transforms, constants (PATCH_SIZE=256, STRIDE=128, threshold=0.5, GAUSS_W)
from inference_production import (
    seg_model,
    gen_model,
    GAUSS_W,
    seg_transform,
    gan_transform,
    DEVICE,
    PATCH_SIZE,
    STRIDE,
    pad_image,
)

# ── Metric functions from training code ─────────────────────────────────────
from run_gan_training import (
    compute_psnr,
    compute_ssim,
    compute_vari_rmse,
)

# ── LPIPS (optional) ────────────────────────────────────────────────────────
LPIPS_AVAILABLE = False
lpips_model = None
try:
    import lpips

    lpips_model = lpips.LPIPS(net="vgg").to(DEVICE).eval()
    LPIPS_AVAILABLE = True
    print("LPIPS: available")
except ImportError:
    print("LPIPS: not installed — LPIPS column will be NaN")
    lpips_model = None
except Exception as e:
    print(f"LPIPS: import failed ({e}) — LPIPS column will be NaN")
    lpips_model = None

# ── Config ──────────────────────────────────────────────────────────────────
with open(PROJECT_ROOT / "configs/seg_config.yaml") as f:
    seg_config = yaml.safe_load(f)["segmentation"]
with open(PROJECT_ROOT / "configs/gan_config.yaml") as f:
    gan_config = yaml.safe_load(f)["gan"]

# ── Paths ───────────────────────────────────────────────────────────────────
PATCHES_DIR = PROJECT_ROOT / "data" / "processed" / "patches" / "test"
MANIFEST_PATH = PATCHES_DIR / "patch_manifest.json"
CLOUD_DIR = PATCHES_DIR / "cloud"
MASK_DIR = PATCHES_DIR / "mask"
LABEL_DIR = PATCHES_DIR / "label"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics" / "agricultural_production_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SMOKE_DIR = OUTPUT_DIR / "smoke_test"
SMOKE_DIR.mkdir(parents=True, exist_ok=True)
WORST_DIR = OUTPUT_DIR / "worst_heavy_cases"
WORST_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "agricultural_production_per_patch.csv"
SUMMARY_PATH = OUTPUT_DIR / "agricultural_production_summary.txt"

# ── Agricultural source IDs ─────────────────────────────────────────────────
AGRICULTURAL_SOURCE_IDS = {
    "154", "163", "168", "190", "197", "216", "223",
    "386", "414", "421", "428", "433", "439", "471",
    "498", "504", "523", "530", "533",
}

# ── Coverage bins ───────────────────────────────────────────────────────────
def classify_coverage(coverage_frac):
    """coverage_frac is 0-1 fraction from manifest."""
    pct = coverage_frac * 100
    if pct < 30:
        return "Light"
    elif pct < 60:
        return "Medium"
    else:
        return "Heavy"


# ── Core production inference (replicates inference_production.run_inference
#    Steps 2-4 exactly, without file I/O or uncertainty map) ─────────────────
def run_production_inference_core(img_data: np.ndarray):
    """
    Run the EXACT production inference pipeline (seg + GAN + stitching)
    on a single RGB image, returning the predicted binary cloud mask and
    reconstructed cloud-free image.

    This mirrors the core logic of run_inference() in inference_production.py
    (Steps 2-4):
      - Gaussian sliding-window segmentation (PATCH_SIZE=256, STRIDE=128)
      - threshold = 0.5
      - binary cloud mask
      - RGB + binary mask + Sobel edge → 5-channel GAN input
      - GAN reconstruction with Gaussian stitching

    No files are written. No uncertainty map is computed.
    """
    # Ensure uint8
    if img_data.dtype != np.uint8:
        img_data = ((img_data - img_data.min()) /
                    (img_data.max() - img_data.min() + 1e-8) * 255).astype(np.uint8)

    H_orig, W_orig = img_data.shape[:2]

    # Pad to fit sliding window
    img_padded, H_orig, W_orig = pad_image(img_data)
    H_pad, W_pad = img_padded.shape[:2]

    # ── Step 2: Segmentation (Gaussian sliding window) ─────────────────────
    mask_accum = np.zeros((H_pad, W_pad), dtype=np.float32)
    weight_accum = np.zeros((H_pad, W_pad), dtype=np.float32)

    with torch.no_grad():
        for r in range(0, H_pad - PATCH_SIZE + 1, STRIDE):
            for c in range(0, W_pad - PATCH_SIZE + 1, STRIDE):
                patch = img_padded[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
                t = seg_transform(image=patch)["image"].unsqueeze(0).to(DEVICE)
                logit = seg_model(t)
                prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
                mask_accum[r:r + PATCH_SIZE, c:c + PATCH_SIZE] += prob * GAUSS_W
                weight_accum[r:r + PATCH_SIZE, c:c + PATCH_SIZE] += GAUSS_W

    cloud_mask = (mask_accum / (weight_accum + 1e-8))[:H_orig, :W_orig]
    cloud_mask_bin = (cloud_mask > 0.5).astype(np.uint8)

    # ── Step 3: GAN cloud removal (5-channel input, Gaussian stitching) ────
    output_accum = np.zeros((H_pad, W_pad, 3), dtype=np.float32)
    weight_out = np.zeros((H_pad, W_pad), dtype=np.float32)

    mask_padded = np.pad(
        cloud_mask_bin,
        ((0, H_pad - H_orig), (0, W_pad - W_orig)),
        mode="reflect",
    )

    with torch.no_grad():
        for r in range(0, H_pad - PATCH_SIZE + 1, STRIDE):
            for c in range(0, W_pad - PATCH_SIZE + 1, STRIDE):
                patch_img = img_padded[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
                patch_mask = mask_padded[r:r + PATCH_SIZE, c:c + PATCH_SIZE]

                img_t = gan_transform(image=patch_img)["image"]
                mask_t = torch.from_numpy(patch_mask.astype(np.float32)).unsqueeze(0)

                # Sobel edge map (same as production)
                m = (patch_mask * 255).astype(np.float32)
                gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
                G = np.sqrt(gx ** 2 + gy ** 2)
                edge_t = torch.from_numpy(
                    (G / G.max() if G.max() > 0 else G).astype(np.float32)
                ).unsqueeze(0)

                gen_input = torch.cat([img_t, mask_t, edge_t], dim=0)
                gen_input = gen_input.unsqueeze(0).to(DEVICE)

                fake = gen_model(gen_input)
                out_patch = np.clip(
                    ((fake[0].cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255),
                    0, 255,
                ).astype(np.float32)

                w3d = GAUSS_W[:, :, np.newaxis]
                output_accum[r:r + PATCH_SIZE, c:c + PATCH_SIZE] += out_patch * w3d
                weight_out[r:r + PATCH_SIZE, c:c + PATCH_SIZE] += GAUSS_W

    # ── Step 4: Stitch and crop ────────────────────────────────────────────
    output_full = (output_accum / (weight_out[:, :, np.newaxis] + 1e-8))
    output_crop = np.clip(output_full[:H_orig, :W_orig], 0, 255).astype(np.uint8)

    return output_crop, cloud_mask_bin


# ── Metric functions ────────────────────────────────────────────────────────

def compute_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Compute IoU, Dice, Precision, Recall from binary masks.
    Masks should be {0, 1} uint8 arrays of the same shape.
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)

    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    fp = int(np.logical_and(pred_bin, ~gt_bin.astype(bool)).sum())
    fn = int(np.logical_and(~pred_bin.astype(bool), gt_bin).sum())
    tn = int(np.logical_and(~pred_bin.astype(bool), ~gt_bin.astype(bool)).sum())

    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_lpips(pred_np: np.ndarray, gt_np: np.ndarray):
    """Compute LPIPS if the lpips package is available."""
    if not LPIPS_AVAILABLE or lpips_model is None:
        return float("nan")
    # Convert numpy [H, W, 3] uint8 → tensor [1, 3, H, W] in [-1, 1]
    pred_t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    pred_t = pred_t.to(DEVICE)
    gt_t = gt_t.to(DEVICE)
    with torch.no_grad():
        score = lpips_model(pred_t, gt_t)
    return score.item()


def compute_gan_metrics(pred_np: np.ndarray, gt_np: np.ndarray, pred_mask_bin: np.ndarray):
    """
    Compute PSNR, SSIM, VARI-RMSE using the project's existing metric functions.
    These functions expect tensors in [-1, 1] range.
    """
    # Convert numpy [H, W, 3] uint8 → tensor [1, 3, H, W] in [-1, 1]
    pred_t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

    pred_t = pred_t.to(DEVICE)
    gt_t = gt_t.to(DEVICE)

    # Mask for VARI-RMSE: use predicted cloud mask region (where reconstruction matters)
    mask_t = torch.from_numpy(pred_mask_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)

    psnr = compute_psnr(pred_t, gt_t)
    ssim_val = compute_ssim(pred_t, gt_t)
    vari_rmse = compute_vari_rmse(pred_t, gt_t, mask_t)

    lpips_val = compute_lpips(pred_np, gt_np)

    return {
        "psnr": psnr,
        "ssim": ssim_val,
        "vari_rmse": vari_rmse,
        "lpips": lpips_val,
    }


# ── Load manifest & filter ──────────────────────────────────────────────────

def load_agricultural_patches():
    """
    Read patch_manifest.json and filter to the 19 agricultural source IDs.
    Returns a list of (patch_id, entry) sorted by patch_id.
    Does NOT hard-code filenames — all selection is dynamic from the manifest.
    """
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    selected = []
    for pid, info in manifest.items():
        src_id = str(info.get("source_id", ""))
        if src_id in AGRICULTURAL_SOURCE_IDS:
            selected.append((pid, info))

    selected.sort(key=lambda x: x[0])
    return selected


# ── Smoke test ──────────────────────────────────────────────────────────────

def smoke_test(patches):
    """
    Run a single-patch smoke test to verify:
      - Production inference produces correct output dimensions
      - GT mask and GT label load correctly
      - Metric calculation works end-to-end
    """
    print("=" * 70)
    print("SINGLE-PATCH SMOKE TEST")
    print("=" * 70)

    pid, info = patches[0]
    cloud_path = CLOUD_DIR / f"{pid}.png"
    mask_path = MASK_DIR / f"{pid}.png"
    label_path = LABEL_DIR / f"{pid}.png"

    print(f"  Patch ID:     {pid}")
    print(f"  Source ID:     {info['source_id']}")
    print(f"  Cloud coverage: {info['cloud_coverage']:.4f} ({info['cloud_coverage']*100:.1f}%)")
    print(f"  Bucket:         {classify_coverage(info['cloud_coverage'])}")

    # Load cloudy image
    img = cv2.cvtColor(cv2.imread(str(cloud_path)), cv2.COLOR_BGR2RGB)
    print(f"  Cloudy image:   shape={img.shape}, dtype={img.dtype}")

    # Load GT mask
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    print(f"  GT mask:        shape={gt_mask.shape}, dtype={gt_mask.dtype}")

    # Load GT clear image (label)
    gt_label = cv2.cvtColor(cv2.imread(str(label_path)), cv2.COLOR_BGR2RGB)
    print(f"  GT label:       shape={gt_label.shape}, dtype={gt_label.dtype}")

    # Run production inference
    pred_recon, pred_mask = run_production_inference_core(img)
    print(f"  Pred mask:      shape={pred_mask.shape}, dtype={pred_mask.dtype}")
    print(f"  Pred recon:     shape={pred_recon.shape}, dtype={pred_recon.dtype}")
    print(f"  Pred mask unique values: {np.unique(pred_mask)}")

    # Verify dimensions
    assert pred_recon.shape == img.shape, f"Recon shape {pred_recon.shape} != image shape {img.shape}"
    assert pred_mask.shape == (img.shape[0], img.shape[1]), \
        f"Mask shape {pred_mask.shape} != expected {(img.shape[0], img.shape[1])}"

    # Compute metrics
    seg_metrics = compute_segmentation_metrics(pred_mask, gt_mask)
    print(f"\n  Segmentation metrics:")
    print(f"    IoU:         {seg_metrics['iou']:.4f}")
    print(f"    Dice:        {seg_metrics['dice']:.4f}")
    print(f"    Precision:   {seg_metrics['precision']:.4f}")
    print(f"    Recall:      {seg_metrics['recall']:.4f}")

    gan_metrics = compute_gan_metrics(pred_recon, gt_label, pred_mask)
    print(f"\n  GAN reconstruction metrics:")
    print(f"    PSNR:        {gan_metrics['psnr']:.2f} dB")
    print(f"    SSIM:        {gan_metrics['ssim']:.4f}")
    print(f"    VARI-RMSE:   {gan_metrics['vari_rmse']:.4f}")
    print(f"    LPIPS:       {gan_metrics['lpips']}")

    # Save smoke test artifacts
    cv2.imwrite(str(SMOKE_DIR / "smoke_cloudy.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(SMOKE_DIR / "smoke_pred_mask.png"), pred_mask * 255)
    cv2.imwrite(str(SMOKE_DIR / "smoke_gt_mask.png"), gt_mask)
    cv2.imwrite(str(SMOKE_DIR / "smoke_pred_recon.png"), cv2.cvtColor(pred_recon, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(SMOKE_DIR / "smoke_gt_label.png"), cv2.cvtColor(gt_label, cv2.COLOR_RGB2BGR))

    print(f"\n  Smoke test artifacts saved to: {SMOKE_DIR}")
    print("  SMOKE TEST PASSED — dimensions and metrics verified.")
    print("=" * 70 + "\n")


# ── Full evaluation ─────────────────────────────────────────────────────────

def run_full_evaluation(patches):
    """
    Run production inference on all selected patches and compute metrics.
    Returns a list of per-patch result dictionaries.
    """
    results = []
    total = len(patches)

    for i, (pid, info) in enumerate(patches):
        cloud_path = CLOUD_DIR / f"{pid}.png"
        mask_path = MASK_DIR / f"{pid}.png"
        label_path = LABEL_DIR / f"{pid}.png"

        coverage_frac = info["cloud_coverage"]
        coverage_pct = coverage_frac * 100
        bucket = classify_coverage(coverage_frac)
        source_id = info["source_id"]

        # Load inputs
        img = cv2.cvtColor(cv2.imread(str(cloud_path)), cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_label = cv2.cvtColor(cv2.imread(str(label_path)), cv2.COLOR_BGR2RGB)

        if img is None or gt_mask is None or gt_label is None:
            print(f"  WARNING: {pid} — could not load files, skipping.")
            continue

        # Run production inference
        pred_recon, pred_mask = run_production_inference_core(img)

        # Compute metrics
        seg_m = compute_segmentation_metrics(pred_mask, gt_mask)
        gan_m = compute_gan_metrics(pred_recon, gt_label, pred_mask)

        result = {
            "patch_id": pid,
            "source_id": source_id,
            "cloud_coverage_pct": round(coverage_pct, 4),
            "bucket": bucket,
            "iou": round(seg_m["iou"], 6),
            "dice": round(seg_m["dice"], 6),
            "precision": round(seg_m["precision"], 6),
            "recall": round(seg_m["recall"], 6),
            "tp": seg_m["tp"],
            "fp": seg_m["fp"],
            "fn": seg_m["fn"],
            "tn": seg_m["tn"],
            "psnr": round(gan_m["psnr"], 6),
            "ssim": round(gan_m["ssim"], 6),
            "vari_rmse": round(gan_m["vari_rmse"], 6),
            "lpips": round(gan_m["lpips"], 6) if not np.isnan(gan_m["lpips"]) else float("nan"),
        }
        results.append(result)

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  Progress: {i + 1}/{total} patches processed")

    return results


# ── Save outputs ────────────────────────────────────────────────────────────

def save_csv(results):
    fieldnames = [
        "patch_id", "source_id", "cloud_coverage_pct", "bucket",
        "iou", "dice", "precision", "recall",
        "tp", "fp", "fn", "tn",
        "psnr", "ssim", "vari_rmse", "lpips",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Per-patch CSV saved: {CSV_PATH}")


def save_summary(results):
    """Save summary text file with Light/Medium/Heavy/Overall stats."""
    with open(SUMMARY_PATH, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("AGRICULTURAL PRODUCTION EVALUATION — SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Pipeline: inference_production.py (V1 segmentation + best_generator.pt)\n")
        f.write("  Segmentation checkpoint: outputs/checkpoints/segmentation/v1_brightness_aug_model_state.pt\n")
        f.write("  GAN checkpoint:          outputs/checkpoints/gan/best_generator.pt\n")
        f.write("  Segmentation threshold:  0.5\n")
        f.write("  Sliding window:          256×256, stride 128, Gaussian feathering\n")
        f.write("  GAN input:               5-channel (RGB + binary mask + Sobel edge)\n")
        f.write("  Reconstruction stitching: Gaussian weighted overlap accumulation\n")
        f.write(f"  LPIPS available:         {LPIPS_AVAILABLE}\n\n")

        f.write("=" * 80 + "\n")
        f.write("SELECTION CRITERIA\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Source IDs (19 agricultural scenes): {sorted(AGRICULTURAL_SOURCE_IDS, key=int)}\n")
        f.write(f"  Total patches selected: {len(results)}\n")
        f.write(f"  Light (<30%):    {sum(1 for r in results if r['bucket']=='Light')}\n")
        f.write(f"  Medium (30-60%): {sum(1 for r in results if r['bucket']=='Medium')}\n")
        f.write(f"  Heavy (>=60%):   {sum(1 for r in results if r['bucket']=='Heavy')}\n\n")

        f.write("=" * 80 + "\n")
        f.write("SEGMENTATION METRICS (Mask: Pred vs GT)\n")
        f.write("=" * 80 + "\n\n")

        for bucket_name in ["Light", "Medium", "Heavy", "Overall"]:
            bucket_results = [r for r in results if r["bucket"] == bucket_name] if bucket_name != "Overall" else results
            if not bucket_results:
                continue

            f.write(f"  --- {bucket_name} (N={len(bucket_results)}) ---\n")

            for metric_name in ["iou", "dice", "precision", "recall", "psnr", "ssim", "vari_rmse", "lpips"]:
                vals = [r[metric_name] for r in bucket_results if not np.isnan(r[metric_name])]
                if vals:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    f.write(f"    {metric_name:20s}: {mean_val:.6f} ± {std_val:.6f}\n")
                else:
                    f.write(f"    {metric_name:20s}: N/A\n")
            f.write("\n")

        # ── Heavy cloud detailed PSNR statistics ───────────────────────────
        f.write("=" * 80 + "\n")
        f.write("HEAVY CLOUD DETAILED PSNR STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        heavy_results = [r for r in results if r["bucket"] == "Heavy"]
        heavy_psnrs = [r["psnr"] for r in heavy_results]

        f.write(f"  N:                      {len(heavy_psnrs)}\n")
        f.write(f"  Mean PSNR:             {np.mean(heavy_psnrs):.4f} dB\n")
        f.write(f"  Std PSNR:              {np.std(heavy_psnrs):.4f} dB\n")
        f.write(f"  Minimum PSNR:          {np.min(heavy_psnrs):.4f} dB\n")
        f.write(f"  Maximum PSNR:          {np.max(heavy_psnrs):.4f} dB\n")
        f.write(f"  Median PSNR:           {np.median(heavy_psnrs):.4f} dB\n")
        n_below_20 = sum(1 for p in heavy_psnrs if p < 20)
        f.write(f"  Samples with PSNR < 20 dB: {n_below_20}\n\n")

        # Worst 5 PSNR samples
        sorted_heavy = sorted(heavy_results, key=lambda r: r["psnr"])
        f.write("  Worst 5 PSNR samples:\n")
        f.write(f"    {'Patch ID':<20} {'Source ID':<12} {'Coverage %':<12} {'PSNR (dB)':<12}\n")
        for r in sorted_heavy[:5]:
            f.write(f"    {r['patch_id']:<20} {r['source_id']:<12} {r['cloud_coverage_pct']:<12.2f} {r['psnr']:<12.4f}\n")
        f.write("\n")

    print(f"Summary saved: {SUMMARY_PATH}")


def save_worst_heavy_cases(results):
    """Save worst Heavy-cloud patches for visual inspection."""
    heavy_results = [r for r in results if r["bucket"] == "Heavy"]
    sorted_heavy = sorted(heavy_results, key=lambda r: r["psnr"])

    for rank, r in enumerate(sorted_heavy[:5], 1):
        pid = r["patch_id"]
        cloud_path = CLOUD_DIR / f"{pid}.png"
        mask_path = MASK_DIR / f"{pid}.png"
        label_path = LABEL_DIR / f"{pid}.png"

        img = cv2.cvtColor(cv2.imread(str(cloud_path)), cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_label = cv2.cvtColor(cv2.imread(str(label_path)), cv2.COLOR_BGR2RGB)

        pred_recon, pred_mask = run_production_inference_core(img)

        case_dir = WORST_DIR / f"rank{rank}_psnr{r['psnr']:.2f}_{pid}"
        case_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(case_dir / "01_cloudy_input.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(case_dir / "02_predicted_mask.png"), pred_mask * 255)
        cv2.imwrite(str(case_dir / "03_reconstruction.png"), cv2.cvtColor(pred_recon, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(case_dir / "04_gt_clear.png"), cv2.cvtColor(gt_label, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(case_dir / "05_gt_mask.png"), gt_mask)

        # Also save a composite image
        composite = np.hstack([
            img,
            cv2.cvtColor(pred_mask * 255, cv2.COLOR_GRAY2RGB),
            pred_recon,
            gt_label,
        ])
        cv2.imwrite(str(case_dir / "06_composite.png"), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

        coverage = r["cloud_coverage_pct"]
        info_path = case_dir / "info.txt"
        with open(info_path, "w") as info_f:
            info_f.write(f"Patch ID: {pid}\n")
            info_f.write(f"Source ID: {r['source_id']}\n")
            info_f.write(f"Cloud coverage: {coverage:.2f}% ({r['bucket']})\n")
            info_f.write(f"PSNR: {r['psnr']:.4f} dB\n")
            info_f.write(f"SSIM: {r['ssim']:.6f}\n")
            info_f.write(f"VARI-RMSE: {r['vari_rmse']:.6f}\n")
            info_f.write(f"IoU: {r['iou']:.6f}\n")
            info_f.write(f"Dice: {r['dice']:.6f}\n")
            info_f.write(f"LPIPS: {r['lpips']}\n")
            info_f.write(f"\nFiles: 01_cloudy_input.png, 02_predicted_mask.png, 03_reconstruction.png, 04_gt_clear.png, 05_gt_mask.png, 06_composite.png\n")

    print(f"Worst heavy cases saved: {WORST_DIR}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("AGRICULTURAL PRODUCTION EVALUATION")
    print("=" * 80)
    print(f"  Device:               {DEVICE}")
    print(f"  Segmentation ckpt:    outputs/checkpoints/segmentation/v1_brightness_aug_model_state.pt")
    print(f"  GAN ckpt:             outputs/checkpoints/gan/best_generator.pt")
    print(f"  Segmentation threshold: 0.5")
    print(f"  LPIPS available:      {LPIPS_AVAILABLE}")
    print()

    # Step 1: Load manifest and filter agricultural patches
    patches = load_agricultural_patches()
    print(f"Selected {len(patches)} agricultural patches from manifest.")

    # Verify expected counts
    light_count = sum(1 for _, v in patches if classify_coverage(v["cloud_coverage"]) == "Light")
    medium_count = sum(1 for _, v in patches if classify_coverage(v["cloud_coverage"]) == "Medium")
    heavy_count = sum(1 for _, v in patches if classify_coverage(v["cloud_coverage"]) == "Heavy")
    print(f"  Light (<30%):    {light_count}")
    print(f"  Medium (30-60%): {medium_count}")
    print(f"  Heavy (>=60%):   {heavy_count}")
    assert len(patches) == 116, f"Expected 116 patches, got {len(patches)}"
    assert light_count == 72, f"Expected 72 Light patches, got {light_count}"
    assert medium_count == 30, f"Expected 30 Medium patches, got {medium_count}"
    assert heavy_count == 14, f"Expected 14 Heavy patches, got {heavy_count}"
    print("  Patch count verification passed.\n")

    # Step 2: Smoke test
    smoke_test(patches)

    # Step 3: Full evaluation
    print("=" * 80)
    print("FULL 116-PATCH EVALUATION")
    print("=" * 80 + "\n")
    results = run_full_evaluation(patches)
    print(f"\nProcessed {len(results)} patches.\n")

    # Step 4: Save outputs
    save_csv(results)
    save_summary(results)
    save_worst_heavy_cases(results)

    # Print quick summary
    print("\n" + "=" * 80)
    print("QUICK RESULTS")
    print("=" * 80)
    for bucket_name in ["Light", "Medium", "Heavy", "Overall"]:
        bucket_results = [r for r in results if r["bucket"] == bucket_name] if bucket_name != "Overall" else results
        if not bucket_results:
            continue
        psnrs = [r["psnr"] for r in bucket_results]
        ssims = [r["ssim"] for r in bucket_results]
        ious = [r["iou"] for r in bucket_results]
        print(f"  {bucket_name:10s} (N={len(bucket_results):4d}): "
              f"IoU={np.mean(ious):.4f}  PSNR={np.mean(psnrs):.2f}  SSIM={np.mean(ssims):.4f}")

    print("=" * 80)
    print("Evaluation complete.")
    print(f"  CSV:    {CSV_PATH}")
    print(f"  Summary: {SUMMARY_PATH}")
    print(f"  Worst cases: {WORST_DIR}")


if __name__ == "__main__":
    main()
