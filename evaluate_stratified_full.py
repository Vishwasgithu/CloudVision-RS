"""
evaluate_stratified_full.py

Runs the baseline models (segmentation + generator) across the ENTIRE RICE2
test set (not a single spot-check image), splits results into the master
plan's Light/Medium/Heavy cloud coverage bins, and reports PSNR, SSIM, LPIPS,
IoU, Dice, and VARI-RMSE for each bin plus an overall average.

Run: conda activate CloudRemoval
pip install lpips --break-system-packages
python evaluate_stratified_full.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, json, glob, re
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path

sys.path.insert(0, "D:\\CloudRemoval_Project")
from src.models.segmentation import AttentionUNet
from src.models.generator import Generator

import lpips

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = Path("data/processed/patches/test")

BINS = [
    ("Light (<30%)", 0.0, 0.30),
    ("Medium (30-60%)", 0.30, 0.60),
    ("Heavy (>60%)", 0.60, 1.01),
]


def load_state(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if isinstance(ckpt, dict) and "G_state" in ckpt:
        return ckpt["G_state"]
    return ckpt


def pick_seg_checkpoint():
    cks = sorted(
        glob.glob("outputs/checkpoints/segmentation/best_*.pt")
        + glob.glob("outputs/checkpoints/segmentation/*.pt")
    )

    def iou(p):
        m = re.search(r"iou([\d.]+)", os.path.basename(p))
        return float(m.group(1)) if m else 0.0

    return max(cks, key=iou)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_for_segmentation(img_rgb_u8):
    """Must match training exactly: dataset.py documents segmentation inputs as
    ImageNet-normalized, not just scaled to [0,1]."""
    img_01 = img_rgb_u8.astype(np.float32) / 255.0
    img_norm = (img_01 - IMAGENET_MEAN) / IMAGENET_STD
    return img_norm


def compute_psnr(pred01, target01):
    mse = ((pred01 - target01) ** 2).mean()
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0


def compute_ssim(pred01, target01):
    mu_p, mu_t = pred01.mean(), target01.mean()
    sp, st = pred01.std(), target01.std()
    spt = ((pred01 - mu_p) * (target01 - mu_t)).mean()
    C1, C2 = 0.01**2, 0.03**2
    return ((2 * mu_p * mu_t + C1) * (2 * spt + C2)) / (
        (mu_p**2 + mu_t**2 + C1) * (sp**2 + st**2 + C2)
    )


def compute_vari_rmse(pred01, target01):
    Rp, Gp, Bp = pred01[..., 0], pred01[..., 1], pred01[..., 2]
    Rt, Gt, Bt = target01[..., 0], target01[..., 1], target01[..., 2]
    dp = np.clip(Gp + Rp - Bp, 0.1, None)
    dt = np.clip(Gt + Rt - Bt, 0.1, None)
    vp = np.clip((Gp - Rp) / dp, -1, 1)
    vt = np.clip((Gt - Rt) / dt, -1, 1)
    return float(np.sqrt(((vp - vt) ** 2).mean()))


def compute_mask_accuracy(pred_bin, true_bin):
    inter = (pred_bin & true_bin).sum()
    union = (pred_bin | true_bin).sum()
    iou = float(inter / union) if union > 0 else 1.0
    denom = pred_bin.sum() + true_bin.sum()
    dice = float(2 * inter / denom) if denom > 0 else 1.0
    return iou, dice


def main():
    with open("configs/seg_config.yaml") as f:
        seg_config = yaml.safe_load(f)["segmentation"]
    with open("configs/gan_config.yaml") as f:
        gan_config = yaml.safe_load(f)["gan"]

    seg_model = AttentionUNet(seg_config).to(DEVICE)
    seg_ckpt = pick_seg_checkpoint()
    seg_model.load_state_dict(load_state(seg_ckpt))
    seg_model.eval()
    print(f"Segmentation checkpoint: {seg_ckpt}")

    gen_model = Generator(
        in_channels=gan_config["in_channels"], features=gan_config["features_g"]
    ).to(DEVICE)
    gen_model.load_state_dict(load_state("outputs/checkpoints/gan/best_generator.pt"))
    gen_model.eval()
    print("Generator checkpoint: outputs/checkpoints/gan/best_generator.pt")

    print("Loading LPIPS perceptual metric (AlexNet backbone, pretrained)...")
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
    lpips_fn.eval()

    with open(TEST_DIR / "patch_manifest.json") as f:
        manifest = json.load(f)
    patch_ids = sorted(manifest.keys())
    print(f"Evaluating {len(patch_ids)} test patches...\n")

    results = {b[0]: [] for b in BINS}
    results["Overall"] = []

    with torch.no_grad():
        for i, pid in enumerate(patch_ids):
            cloudy = cv2.cvtColor(
                cv2.imread(str(TEST_DIR / "cloud" / f"{pid}.png")), cv2.COLOR_BGR2RGB
            )
            label = cv2.cvtColor(
                cv2.imread(str(TEST_DIR / "label" / f"{pid}.png")), cv2.COLOR_BGR2RGB
            )
            true_mask_raw = cv2.imread(
                str(TEST_DIR / "mask" / f"{pid}.png"), cv2.IMREAD_GRAYSCALE
            )
            true_mask_bin = (true_mask_raw > 127).astype(np.uint8)
            coverage = float(manifest[pid].get("cloud_coverage", true_mask_bin.mean()))

            # Segmentation -- MUST use ImageNet normalization to match training (see fix above)
            seg_in_np = preprocess_for_segmentation(cloudy)
            seg_in = (
                torch.from_numpy(seg_in_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            )
            pred_prob = torch.sigmoid(seg_model(seg_in))[0, 0].cpu().numpy()
            pred_mask_bin = (pred_prob > 0.5).astype(np.uint8)
            iou, dice = compute_mask_accuracy(pred_mask_bin, true_mask_bin)

            # Edge map + generator input
            m = (pred_mask_bin * 255).astype(np.float32)
            gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
            edge = np.sqrt(gx**2 + gy**2)
            edge = edge / (edge.max() + 1e-8)

            cloudy_norm = (cloudy.astype(np.float32) / 127.5) - 1.0
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
            fake = gen_model(gen_tensor)
            fake_01 = ((fake[0].cpu().numpy().transpose(1, 2, 0)) + 1) / 2
            fake_01 = np.clip(fake_01, 0, 1)
            label_01 = label.astype(np.float32) / 255.0

            psnr = compute_psnr(fake_01, label_01)
            ssim = compute_ssim(fake_01, label_01)
            vari = compute_vari_rmse(fake_01, label_01)

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

            row = dict(psnr=psnr, ssim=ssim, lpips=lp, iou=iou, dice=dice, vari=vari)
            results["Overall"].append(row)
            for name, lo, hi in BINS:
                if lo <= coverage < hi:
                    results[name].append(row)
                    break

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(patch_ids)} processed...")

    print(f"\n{'='*90}")
    print(
        f"{'Bin':<20}{'N':>5}{'PSNR':>10}{'SSIM':>9}{'LPIPS':>9}{'IoU':>8}{'Dice':>8}{'VARI-RMSE':>12}"
    )
    print("=" * 90)
    for name in [b[0] for b in BINS] + ["Overall"]:
        rows = results[name]
        if not rows:
            print(f"{name:<20}{'(no samples in this bin)':>60}")
            continue
        n = len(rows)
        avg = lambda k: sum(r[k] for r in rows) / n
        print(
            f"{name:<20}{n:>5}{avg('psnr'):>9.2f}dB{avg('ssim'):>9.4f}{avg('lpips'):>9.4f}"
            f"{avg('iou'):>8.4f}{avg('dice'):>8.4f}{avg('vari'):>12.4f}"
        )
    print("=" * 90)
    print(
        "\nTargets for reference: PSNR>30dB, SSIM>0.85, LPIPS<0.15, IoU>0.85, Dice>0.93, VARI-RMSE<0.08"
    )


if __name__ == "__main__":
    main()
