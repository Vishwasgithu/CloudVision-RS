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
from scipy.ndimage import gaussian_filter
import lpips
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\CloudRemoval_Project")
from src.models.segmentation import AttentionUNet
from src.models.generator import Generator

PROJECT_ROOT = Path(r"D:\CloudRemoval_Project")
BASELINE_SEG = PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation" / "best_iou0.7935_ep39.pt"
V1_SEG = PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation" / "v1_brightness_aug_model_state.pt"
GAN_CKPT = PROJECT_ROOT / "outputs" / "checkpoints" / "gan_baseline" / "best_generator_baseline.pt"
TEST_CLOUD_DIR = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "cloud"
TEST_MASK_DIR = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "mask"
TEST_LABEL_DIR = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "label"
TEST_MANIFEST = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "patch_manifest.json"
SEG_CONFIG_PATH = PROJECT_ROOT / "configs" / "seg_config.yaml"
GAN_CONFIG_PATH = PROJECT_ROOT / "configs" / "gan_config.yaml"
OUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics" / "segmentation_ab_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 256
STRIDE = 128

with open(SEG_CONFIG_PATH, "r") as f:
    seg_config = yaml.safe_load(f)["segmentation"]
with open(GAN_CONFIG_PATH, "r") as f:
    gan_config = yaml.safe_load(f)["gan"]

seg_transform = A.Compose([
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
    ToTensorV2()
])

gan_transform = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
    ToTensorV2()
])

GAUSS_W = None

def gaussian_weight(size=256):
    center = size // 2
    w = np.zeros((size, size))
    w[center, center] = 1.0
    w = gaussian_filter(w, sigma=center * 0.35)
    return (w / w.max()).astype(np.float32)


def load_seg_model(path):
    model = AttentionUNet(seg_config).to(DEVICE)
    ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_state[k[len("model."):]] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def load_gan_model(path):
    model = Generator(
        in_channels=gan_config["in_channels"],
        features=gan_config["features_g"]
    ).to(DEVICE)
    ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "G_state" in ckpt:
        model.load_state_dict(ckpt["G_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def compute_psnr(pred, target):
    p = pred.astype(np.float32) / 255.0
    t = target.astype(np.float32) / 255.0
    mse = ((p - t) ** 2).mean()
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0


def compute_ssim(pred, target):
    p = pred.astype(np.float32) / 255.0
    t = target.astype(np.float32) / 255.0
    mu_p, mu_t = p.mean(), t.mean()
    sp, st = p.std(), t.std()
    spt = ((p - mu_p) * (t - mu_t)).mean()
    C1, C2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_p * mu_t + C1) * (2 * spt + C2)) / (
        (mu_p**2 + mu_t**2 + C1) * (sp**2 + st**2 + C2)
    )
    return float(ssim)


def compute_vari_rmse(pred, target):
    p = pred.astype(np.float32) / 255.0
    t = target.astype(np.float32) / 255.0
    Rp, Gp, Bp = p[..., 0], p[..., 1], p[..., 2]
    Rt, Gt, Bt = t[..., 0], t[..., 1], t[..., 2]
    dp = np.clip(Gp + Rp - Bp, 0.1, None)
    dt = np.clip(Gt + Rt - Bt, 0.1, None)
    vari_p = np.clip((Gp - Rp) / dp, -1, 1)
    vari_t = np.clip((Gt - Rt) / dt, -1, 1)
    return float(np.sqrt(((vari_p - vari_t) ** 2).mean()))


def compute_mask_metrics(pred_bin, gt_bin):
    tp = int(np.logical_and(pred_bin == 1, gt_bin == 1).sum())
    fn = int(np.logical_and(pred_bin == 0, gt_bin == 1).sum())
    fp = int(np.logical_and(pred_bin == 1, gt_bin == 0).sum())
    tn = int(np.logical_and(pred_bin == 0, gt_bin == 0).sum())
    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 0.0
    dice = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "iou": iou, "dice": dice, "precision": precision, "recall": recall}


def run_inference_on_image(img, seg_model, gan_model):
    global GAUSS_W
    if GAUSS_W is None:
        GAUSS_W = gaussian_weight(PATCH_SIZE)

    H_orig, W_orig = img.shape[:2]
    pad_h = (STRIDE - (H_orig - PATCH_SIZE) % STRIDE) % STRIDE
    pad_w = (STRIDE - (W_orig - PATCH_SIZE) % STRIDE) % STRIDE
    if pad_h > 0 or pad_w > 0:
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        img_padded = img
    H_pad, W_pad = img_padded.shape[:2]

    mask_accum = np.zeros((H_pad, W_pad), dtype=np.float32)
    weight_accum = np.zeros((H_pad, W_pad), dtype=np.float32)

    with torch.no_grad():
        for r in range(0, H_pad - PATCH_SIZE + 1, STRIDE):
            for c in range(0, W_pad - PATCH_SIZE + 1, STRIDE):
                patch = img_padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                t = seg_transform(image=patch)['image'].unsqueeze(0).to(DEVICE)
                logit = seg_model(t)
                prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
                mask_accum[r:r+PATCH_SIZE, c:c+PATCH_SIZE] += prob * GAUSS_W
                weight_accum[r:r+PATCH_SIZE, c:c+PATCH_SIZE] += GAUSS_W

    cloud_mask = (mask_accum / (weight_accum + 1e-8))[:H_orig, :W_orig]
    cloud_mask_bin = (cloud_mask > 0.5).astype(np.uint8)

    output_accum = np.zeros((H_pad, W_pad, 3), dtype=np.float32)
    weight_out = np.zeros((H_pad, W_pad, 3), dtype=np.float32)

    mask_padded = np.pad(cloud_mask_bin, ((0, H_pad - H_orig), (0, W_pad - W_orig)), mode='reflect')

    with torch.no_grad():
        for r in range(0, H_pad - PATCH_SIZE + 1, STRIDE):
            for c in range(0, W_pad - PATCH_SIZE + 1, STRIDE):
                patch_img = img_padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                patch_mask = mask_padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE]

                img_t = gan_transform(image=patch_img)['image']
                mask_t = torch.from_numpy(patch_mask.astype(np.float32)).unsqueeze(0)

                m = (patch_mask * 255).astype(np.float32)
                gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
                G = np.sqrt(gx**2 + gy**2)
                edge_t = torch.from_numpy(
                    (G / G.max() if G.max() > 0 else G).astype(np.float32)
                ).unsqueeze(0)

                gen_input = torch.cat([img_t, mask_t, edge_t], dim=0).unsqueeze(0).to(DEVICE)
                fake = gan_model(gen_input)
                out_patch = np.clip(
                    ((fake[0].cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255),
                    0, 255
                ).astype(np.float32)

                output_accum[r:r+PATCH_SIZE, c:c+PATCH_SIZE] += out_patch
                weight_out[r:r+PATCH_SIZE, c:c+PATCH_SIZE] += 1.0

    output_full = output_accum / (weight_out + 1e-8)
    output_crop = np.clip(output_full[:H_orig, :W_orig], 0, 255).astype(np.uint8)

    return output_crop, cloud_mask_bin, cloud_mask


def main():
    print("Loading models...")
    baseline_seg = load_seg_model(BASELINE_SEG)
    v1_seg = load_seg_model(V1_SEG)
    gan_model = load_gan_model(GAN_CKPT)
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
    lpips_fn.eval()

    with open(TEST_MANIFEST, "r") as f:
        manifest = json.load(f)

    image_paths = sorted(TEST_CLOUD_DIR.glob("*.png"))
    print(f"Test images: {len(image_paths)}")

    results = []
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
        gt_mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        label_path = TEST_LABEL_DIR / f"{pid}.png"
        has_label = label_path.exists()
        if has_label:
            label_img = cv2.imread(str(label_path), cv2.IMREAD_COLOR)
            label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)

        # Baseline
        out_b, mask_b, prob_b = run_inference_on_image(img, baseline_seg, gan_model)
        mask_metrics_b = compute_mask_metrics(mask_b, gt_mask)
        if has_label:
            psnr_b = compute_psnr(out_b, label_img)
            ssim_b = compute_ssim(out_b, label_img)
            vari_b = compute_vari_rmse(out_b, label_img)
            fake_lpips = (torch.from_numpy(out_b.astype(np.float32) / 255.0 * 2 - 1)
                          .permute(2, 0, 1).unsqueeze(0).float().to(DEVICE))
            label_lpips = (torch.from_numpy(label_img.astype(np.float32) / 255.0 * 2 - 1)
                           .permute(2, 0, 1).unsqueeze(0).float().to(DEVICE))
            lpips_b = lpips_fn(fake_lpips, label_lpips).item()
        else:
            psnr_b = ssim_b = vari_b = lpips_b = None

        # V1
        out_v, mask_v, prob_v = run_inference_on_image(img, v1_seg, gan_model)
        mask_metrics_v = compute_mask_metrics(mask_v, gt_mask)
        if has_label:
            psnr_v = compute_psnr(out_v, label_img)
            ssim_v = compute_ssim(out_v, label_img)
            vari_v = compute_vari_rmse(out_v, label_img)
            fake_lpips = (torch.from_numpy(out_v.astype(np.float32) / 255.0 * 2 - 1)
                          .permute(2, 0, 1).unsqueeze(0).float().to(DEVICE))
            lpips_v = lpips_fn(fake_lpips, label_lpips).item()
        else:
            psnr_v = ssim_v = vari_v = lpips_v = None

        results.append({
            "pid": pid, "coverage": cov, "category": cat, "has_label": has_label,
            "baseline": {**mask_metrics_b, "psnr": psnr_b, "ssim": ssim_b, "vari_rmse": vari_b, "lpips": lpips_b},
            "v1": {**mask_metrics_v, "psnr": psnr_v, "ssim": ssim_v, "vari_rmse": vari_v, "lpips": lpips_v},
        })

    # Aggregate
    agg = {"baseline": defaultdict(list), "v1": defaultdict(list)}
    cat_agg = {c: {"baseline": defaultdict(list), "v1": defaultdict(list)} for c in ["light", "medium", "heavy"]}

    for r in results:
        for model in ["baseline", "v1"]:
            for k, v in r[model].items():
                if v is not None:
                    agg[model][k].append(v)
            cat_agg[r["category"]][model]["iou"].append(r[model]["iou"])
            cat_agg[r["category"]][model]["dice"].append(r[model]["dice"])
            cat_agg[r["category"]][model]["precision"].append(r[model]["precision"])
            cat_agg[r["category"]][model]["recall"].append(r[model]["recall"])
            if r["has_label"]:
                for k in ["psnr", "ssim", "vari_rmse", "lpips"]:
                    if r[model][k] is not None:
                        cat_agg[r["category"]][model][k].append(r[model][k])

    def summarize(vals):
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}

    summary = {
        "overall": {
            "baseline": {k: summarize(v) for k, v in agg["baseline"].items()},
            "v1": {k: summarize(v) for k, v in agg["v1"].items()},
        },
        "categories": {}
    }
    for cat in ["light", "medium", "heavy"]:
        summary["categories"][cat] = {
            "baseline": {k: summarize(v) for k, v in cat_agg[cat]["baseline"].items()},
            "v1": {k: summarize(v) for k, v in cat_agg[cat]["v1"].items()},
        }

    with open(OUT_DIR / "ab_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV
    csv_rows = []
    for model in ["baseline", "v1"]:
        for cat in ["overall", "light", "medium", "heavy"]:
            if cat == "overall":
                d = summary["overall"][model]
            else:
                d = summary["categories"][cat][model]
            csv_rows.append({
                "model": model, "category": cat,
                "iou_mean": d["iou"]["mean"] if d.get("iou") else None,
                "dice_mean": d["dice"]["mean"] if d.get("dice") else None,
                "precision_mean": d["precision"]["mean"] if d.get("precision") else None,
                "recall_mean": d["recall"]["mean"] if d.get("recall") else None,
                "psnr_mean": d["psnr"]["mean"] if d.get("psnr") else None,
                "ssim_mean": d["ssim"]["mean"] if d.get("ssim") else None,
                "vari_rmse_mean": d["vari_rmse"]["mean"] if d.get("vari_rmse") else None,
                "lpips_mean": d["lpips"]["mean"] if d.get("lpips") else None,
            })
    with open(OUT_DIR / "ab_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # Text report
    lines = []
    lines.append("=" * 80)
    lines.append("A/B INFERENCE EXPERIMENT: BASELINE vs V1")
    lines.append("=" * 80)
    lines.append("")
    for model in ["baseline", "v1"]:
        d = summary["overall"][model]
        lines.append(f"{model.upper()}:")
        lines.append(f"  IoU: {d['iou']['mean']:.4f} ± {d['iou']['std']:.4f}" if d.get("iou") else "  IoU: N/A")
        lines.append(f"  Dice: {d['dice']['mean']:.4f} ± {d['dice']['std']:.4f}" if d.get("dice") else "  Dice: N/A")
        lines.append(f"  Precision: {d['precision']['mean']:.4f} ± {d['precision']['std']:.4f}" if d.get("precision") else "  Precision: N/A")
        lines.append(f"  Recall: {d['recall']['mean']:.4f} ± {d['recall']['std']:.4f}" if d.get("recall") else "  Recall: N/A")
        lines.append(f"  PSNR: {d['psnr']['mean']:.2f} ± {d['psnr']['std']:.2f}" if d.get("psnr") else "  PSNR: N/A")
        lines.append(f"  SSIM: {d['ssim']['mean']:.4f} ± {d['ssim']['std']:.4f}" if d.get("ssim") else "  SSIM: N/A")
        lines.append(f"  VARI-RMSE: {d['vari_rmse']['mean']:.4f} ± {d['vari_rmse']['std']:.4f}" if d.get("vari_rmse") else "  VARI-RMSE: N/A")
        lines.append(f"  LPIPS: {d['lpips']['mean']:.4f} ± {d['lpips']['std']:.4f}" if d.get("lpips") else "  LPIPS: N/A")
        lines.append("")

    lines.append("CATEGORY BREAKDOWN:")
    for cat in ["light", "medium", "heavy"]:
        lines.append(f"  {cat.upper()}:")
        for model in ["baseline", "v1"]:
            d = summary["categories"][cat][model]
            lines.append(f"    {model}: IoU={d['iou']['mean']:.4f} Recall={d['recall']['mean']:.4f}" if d.get("iou") else f"    {model}: N/A")
        lines.append("")

    lines.append("=" * 80)
    with open(OUT_DIR / "ab_report.txt", "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
