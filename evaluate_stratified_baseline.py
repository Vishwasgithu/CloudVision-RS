"""
evaluate_stratified_baseline.py

Evaluates the BASELINE cGAN model across the ENTIRE RICE2 test set.

Pipeline:
    Cloudy Image
        ↓
    Attention U-Net Segmentation
        ↓
    Predicted Cloud Mask
        ↓
    Mask Edge Map
        ↓
    Baseline Pix2Pix cGAN Generator
        ↓
    Reconstructed Cloud-Free Image

Baseline training objective:
    ✓ Adversarial loss
    ✓ L1 reconstruction loss (weight = 100)
    ✗ VARI physics loss
    ✗ Spectral ratio loss
    ✗ Edge coherence loss

Metrics:
    Reconstruction:
        - PSNR
        - SSIM
        - LPIPS
        - VARI-RMSE

    Cloud Detection:
        - IoU
        - Dice

Results are stratified into:
    - Light Cloud  (<30%)
    - Medium Cloud (30–60%)
    - Heavy Cloud  (>60%)

Purpose:
    Fair ablation comparison against the Physics-Informed cGAN.

IMPORTANT:
    Everything except the Generator checkpoint/config is kept identical
    to evaluate_stratified_full.py so the comparison remains fair.

Run from project root:

    conda activate cloudremoval
    python evaluate_stratified_baseline.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import glob
import re

import cv2
import numpy as np
import torch
import yaml

from pathlib import Path

# ============================================================
# PROJECT IMPORTS
# ============================================================

sys.path.insert(0, "D:\\CloudRemoval_Project")

from src.models.segmentation import AttentionUNet
from src.models.generator import Generator

import lpips

# ============================================================
# PATHS AND DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(".")

TEST_DIR = Path("data/processed/patches/test")

SEG_CONFIG_PATH = Path("configs/seg_config.yaml")

BASELINE_GAN_CONFIG_PATH = Path("configs/gan_baseline_config.yaml")

BASELINE_GENERATOR_CKPT = Path(
    "outputs/checkpoints/gan_baseline/best_generator_baseline.pt"
)


# ============================================================
# CLOUD COVERAGE BINS
# ============================================================

BINS = [
    ("Light (<30%)", 0.0, 0.30),
    ("Medium (30-60%)", 0.30, 0.60),
    ("Heavy (>60%)", 0.60, 1.01),
]


# ============================================================
# CHECKPOINT LOADER
# ============================================================


def load_state(path):
    """
    Loads model state dictionary.

    Supports different checkpoint formats used across
    the project.
    """

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    # Segmentation checkpoint format
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]

    # GAN checkpoint format
    if isinstance(ckpt, dict) and "G_state" in ckpt:
        return ckpt["G_state"]

    # Direct state_dict
    return ckpt


# ============================================================
# FIND BEST SEGMENTATION CHECKPOINT
# ============================================================


def pick_seg_checkpoint():
    """
    Automatically finds the segmentation checkpoint
    with the highest IoU value in its filename.
    """

    checkpoints = sorted(
        glob.glob("outputs/checkpoints/segmentation/best_*.pt")
        + glob.glob("outputs/checkpoints/segmentation/*.pt")
    )

    if not checkpoints:
        raise FileNotFoundError(
            "No segmentation checkpoint found in " "outputs/checkpoints/segmentation/"
        )

    def extract_iou(path):

        match = re.search(r"iou([\d.]+)", os.path.basename(path))

        return float(match.group(1)) if match else 0.0

    return max(checkpoints, key=extract_iou)


# ============================================================
# SEGMENTATION PREPROCESSING
# IMPORTANT: MUST MATCH TRAINING
# ============================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)

IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_for_segmentation(img_rgb_u8):
    """
    Preprocess image exactly as done during
    segmentation model training.

    Steps:

        uint8 RGB [0,255]
                ↓
        float [0,1]
                ↓
        ImageNet normalization

    This is critical.

    The segmentation model was trained using ImageNet
    normalization, so evaluation must use the same
    preprocessing.
    """

    img_01 = img_rgb_u8.astype(np.float32) / 255.0

    img_norm = (img_01 - IMAGENET_MEAN) / IMAGENET_STD

    return img_norm


# ============================================================
# METRIC: PSNR
# ============================================================


def compute_psnr(pred01, target01):
    """
    Peak Signal-to-Noise Ratio.

    Higher is better.
    """

    mse = ((pred01 - target01) ** 2).mean()

    if mse > 1e-10:
        return 10 * np.log10(1.0 / mse)

    return 100.0


# ============================================================
# METRIC: SSIM
# ============================================================


def compute_ssim(pred01, target01):
    """
    Structural Similarity Index.

    Higher is better.
    """

    mu_p = pred01.mean()
    mu_t = target01.mean()

    sp = pred01.std()
    st = target01.std()

    spt = ((pred01 - mu_p) * (target01 - mu_t)).mean()

    C1 = 0.01**2
    C2 = 0.03**2

    numerator = (2 * mu_p * mu_t + C1) * (2 * spt + C2)

    denominator = (mu_p**2 + mu_t**2 + C1) * (sp**2 + st**2 + C2)

    return numerator / denominator


# ============================================================
# METRIC: VARI-RMSE
# ============================================================


def compute_vari_rmse(pred01, target01):
    """
    Computes RMSE between predicted and ground-truth
    VARI values.

    VARI =
        (Green - Red)
        ----------------
        (Green + Red - Blue)

    Lower is better.

    Used to evaluate vegetation-related spectral
    consistency.
    """

    Rp = pred01[..., 0]
    Gp = pred01[..., 1]
    Bp = pred01[..., 2]

    Rt = target01[..., 0]
    Gt = target01[..., 1]
    Bt = target01[..., 2]

    # Stable denominators
    dp = np.clip(Gp + Rp - Bp, 0.1, None)

    dt = np.clip(Gt + Rt - Bt, 0.1, None)

    vari_pred = np.clip((Gp - Rp) / dp, -1, 1)

    vari_target = np.clip((Gt - Rt) / dt, -1, 1)

    return float(np.sqrt(((vari_pred - vari_target) ** 2).mean()))


# ============================================================
# MASK ACCURACY
# ============================================================


def compute_mask_accuracy(pred_bin, true_bin):
    """
    Computes:

        IoU  = Intersection / Union

        Dice = 2 * Intersection /
               (Prediction + Ground Truth)
    """

    intersection = (pred_bin & true_bin).sum()

    union = (pred_bin | true_bin).sum()

    if union > 0:
        iou = float(intersection / union)
    else:
        iou = 1.0

    denominator = pred_bin.sum() + true_bin.sum()

    if denominator > 0:
        dice = float(2 * intersection / denominator)
    else:
        dice = 1.0

    return iou, dice


# ============================================================
# MAIN EVALUATION
# ============================================================


def main():

    print("=" * 75)
    print("CloudVision-RS")
    print("Phase 3 — Baseline cGAN Full Stratified Evaluation")
    print("=" * 75)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print()

    # ========================================================
    # LOAD CONFIGURATIONS
    # ========================================================

    print("Loading configurations...")

    with open(SEG_CONFIG_PATH) as f:

        seg_config = yaml.safe_load(f)["segmentation"]

    with open(BASELINE_GAN_CONFIG_PATH) as f:

        gan_config = yaml.safe_load(f)["gan"]

    # ========================================================
    # LOAD SEGMENTATION MODEL
    # ========================================================

    print("\nLoading segmentation model...")

    seg_model = AttentionUNet(seg_config).to(DEVICE)

    seg_ckpt = pick_seg_checkpoint()

    seg_model.load_state_dict(load_state(seg_ckpt))

    seg_model.eval()

    print(f"Segmentation checkpoint:\n" f"  {seg_ckpt}")

    # ========================================================
    # LOAD BASELINE GENERATOR
    # ========================================================

    print("\nLoading BASELINE Generator...")

    if not BASELINE_GENERATOR_CKPT.exists():

        raise FileNotFoundError(
            f"\nBaseline Generator checkpoint not found:\n"
            f"{BASELINE_GENERATOR_CKPT}\n\n"
            f"Expected:\n"
            f"outputs/checkpoints/gan_baseline/"
            f"best_generator_baseline.pt"
        )

    gen_model = Generator(
        in_channels=gan_config["in_channels"], features=gan_config["features_g"]
    ).to(DEVICE)

    gen_model.load_state_dict(load_state(BASELINE_GENERATOR_CKPT))

    gen_model.eval()

    print(f"Baseline Generator checkpoint:\n" f"  {BASELINE_GENERATOR_CKPT}")

    print("\nBaseline objective:")

    print("  Adversarial loss : ENABLED")

    print("  L1 loss          : ENABLED")

    print("  VARI loss        : DISABLED")

    print("  Spectral loss    : DISABLED")

    print("  Edge physics     : DISABLED")

    # ========================================================
    # LOAD LPIPS
    # ========================================================

    print("\nLoading LPIPS perceptual metric " "(AlexNet backbone)...")

    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    lpips_fn.eval()

    # ========================================================
    # LOAD TEST MANIFEST
    # ========================================================

    manifest_path = TEST_DIR / "patch_manifest.json"

    if not manifest_path.exists():

        raise FileNotFoundError(f"Test manifest not found:\n" f"{manifest_path}")

    with open(manifest_path) as f:

        manifest = json.load(f)

    patch_ids = sorted(manifest.keys())

    print(f"\nEvaluating {len(patch_ids)} " f"test patches...")

    # ========================================================
    # RESULTS STORAGE
    # ========================================================

    results = {bin_name: [] for bin_name, _, _ in BINS}

    results["Overall"] = []

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    with torch.no_grad():

        for i, pid in enumerate(patch_ids):

            # ------------------------------------------------
            # LOAD CLOUDY INPUT
            # ------------------------------------------------

            cloudy_path = TEST_DIR / "cloud" / f"{pid}.png"

            cloudy = cv2.cvtColor(cv2.imread(str(cloudy_path)), cv2.COLOR_BGR2RGB)

            # ------------------------------------------------
            # LOAD GROUND TRUTH CLEAN IMAGE
            # ------------------------------------------------

            label_path = TEST_DIR / "label" / f"{pid}.png"

            label = cv2.cvtColor(cv2.imread(str(label_path)), cv2.COLOR_BGR2RGB)

            # ------------------------------------------------
            # LOAD GROUND TRUTH CLOUD MASK
            # ------------------------------------------------

            mask_path = TEST_DIR / "mask" / f"{pid}.png"

            true_mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            true_mask_bin = (true_mask_raw > 127).astype(np.uint8)

            # ------------------------------------------------
            # CLOUD COVERAGE
            # ------------------------------------------------

            coverage = float(manifest[pid].get("cloud_coverage", true_mask_bin.mean()))

            # =================================================
            # STEP 1 — CLOUD SEGMENTATION
            # =================================================

            seg_in_np = preprocess_for_segmentation(cloudy)

            seg_in = (
                torch.from_numpy(seg_in_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            )

            pred_prob = torch.sigmoid(seg_model(seg_in))[0, 0].cpu().numpy()

            # Binary predicted cloud mask
            pred_mask_bin = (pred_prob > 0.5).astype(np.uint8)

            # ------------------------------------------------
            # MASK METRICS
            # ------------------------------------------------

            iou, dice = compute_mask_accuracy(pred_mask_bin, true_mask_bin)

            # =================================================
            # STEP 2 — CREATE MASK EDGE MAP
            # =================================================

            mask_float = (pred_mask_bin * 255).astype(np.float32)

            gx = cv2.Sobel(mask_float, cv2.CV_32F, 1, 0, ksize=3)

            gy = cv2.Sobel(mask_float, cv2.CV_32F, 0, 1, ksize=3)

            edge = np.sqrt(gx**2 + gy**2)

            edge = edge / (edge.max() + 1e-8)

            # =================================================
            # STEP 3 — GENERATOR INPUT
            # =================================================

            # RGB cloudy image:
            # [0,255] → [-1,1]

            cloudy_norm = (cloudy.astype(np.float32) / 127.5) - 1.0

            # Generator receives:
            #
            # 3 channels → Cloudy RGB
            # 1 channel  → Predicted cloud mask
            # 1 channel  → Mask edge map
            #
            # Total = 5 channels

            gen_input = np.concatenate(
                [
                    cloudy_norm,
                    pred_mask_bin[:, :, None].astype(np.float32),
                    edge[:, :, None],
                ],
                axis=2,
            )

            gen_tensor = (
                torch.from_numpy(gen_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            )

            # =================================================
            # STEP 4 — BASELINE GENERATOR INFERENCE
            # =================================================

            fake = gen_model(gen_tensor)

            # Convert generator output:
            #
            # [-1,1] → [0,1]

            fake_01 = (fake[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2

            fake_01 = np.clip(fake_01, 0, 1)

            # Ground truth:
            #
            # [0,255] → [0,1]

            label_01 = label.astype(np.float32) / 255.0

            # =================================================
            # STEP 5 — RECONSTRUCTION METRICS
            # =================================================

            psnr = compute_psnr(fake_01, label_01)

            ssim = compute_ssim(fake_01, label_01)

            vari = compute_vari_rmse(fake_01, label_01)

            # =================================================
            # LPIPS PREPARATION
            # =================================================

            # LPIPS expects tensors in [-1,1]

            fake_lpips = (
                torch.from_numpy(fake_01 * 2 - 1)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(DEVICE)
            )

            label_lpips = (
                torch.from_numpy(label_01 * 2 - 1)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(DEVICE)
            )

            lp = lpips_fn(fake_lpips, label_lpips).item()

            # =================================================
            # STORE RESULTS
            # =================================================

            row = {
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lp,
                "iou": iou,
                "dice": dice,
                "vari": vari,
            }

            # Overall results

            results["Overall"].append(row)

            # Stratified bin results

            for name, lo, hi in BINS:

                if lo <= coverage < hi:

                    results[name].append(row)

                    break

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if (i + 1) % 100 == 0:

                print(f"  {i + 1}/" f"{len(patch_ids)} " f"processed...")

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()

    print("=" * 95)

    print("BASELINE cGAN — FULL STRATIFIED EVALUATION")

    print("=" * 95)

    print(
        f"{'Bin':<20}"
        f"{'N':>5}"
        f"{'PSNR':>10}"
        f"{'SSIM':>9}"
        f"{'LPIPS':>9}"
        f"{'IoU':>8}"
        f"{'Dice':>8}"
        f"{'VARI-RMSE':>12}"
    )

    print("=" * 95)

    for name in [b[0] for b in BINS] + ["Overall"]:

        rows = results[name]

        if not rows:

            print(f"{name:<20}" f"{'(no samples in this bin)':>70}")

            continue

        n = len(rows)

        def avg(key):

            return sum(row[key] for row in rows) / n

        print(
            f"{name:<20}"
            f"{n:>5}"
            f"{avg('psnr'):>9.2f}dB"
            f"{avg('ssim'):>9.4f}"
            f"{avg('lpips'):>9.4f}"
            f"{avg('iou'):>8.4f}"
            f"{avg('dice'):>8.4f}"
            f"{avg('vari'):>12.4f}"
        )

    print("=" * 95)

    print(
        "\nTargets for reference:\n"
        "PSNR > 30 dB | "
        "SSIM > 0.85 | "
        "LPIPS < 0.15 | "
        "IoU > 0.85 | "
        "Dice > 0.93 | "
        "VARI-RMSE < 0.08"
    )

    print()

    print("Evaluation complete.")

    print(
        "This result can now be directly compared "
        "against the Physics-Informed cGAN evaluation."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
