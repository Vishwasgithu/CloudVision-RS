"""
evaluate_gan.py
Evaluates the trained GAN on test patches (filtering out empty/black patches for visualization).
"""

import os, sys, glob, json, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, "D:\\CloudRemoval_Project")

from src.models.generator import Generator
from run_gan_training import GANDataset, psnr, ssim_simple, vari_rmse


# ── Denormalization Helpers ──────────────────────────
def to_img(t):
    """Converts a [1, 3, H, W] tensor in [-1, 1] to a [H, W, 3] numpy array in [0, 1]."""
    arr = t[0].cpu().permute(1, 2, 0).numpy()
    return np.clip((arr + 1.0) / 2.0, 0, 1)


def is_valid_patch(gt_tensor, min_mean_threshold=0.05):
    """Returns True if the patch contains actual scene content (not pure black/edge)."""
    gt_normalized = (gt_tensor + 1.0) / 2.0
    return gt_normalized.mean().item() >= min_mean_threshold


# ── Load model ───────────────────────────────────────
CKPT = "outputs/checkpoints/gan/best_generator.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("configs/gan_config.yaml") as f:
    config = yaml.safe_load(f)["gan"]

G = Generator(in_channels=config["in_channels"], features=config["features_g"]).to(
    DEVICE
)
G.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False))
G.eval()
print(f"Model loaded from {CKPT}")

# ── Test DataLoader ──────────────────────────────────
test_ds = GANDataset(config["patches_dir"], "test", augment=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
print(f"Test patches: {len(test_ds)}")

# ── Evaluate all test patches ────────────────────────
all_psnr, all_ssim, all_vari = [], [], []
samples = []
skipped_count = 0

with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        gi = batch["gen_input"].to(DEVICE)
        cf = batch["cloudfree"].to(DEVICE)
        mk = batch["mask"].to(DEVICE)
        cl = batch["cloudy"].to(DEVICE)

        fake = G(gi)

        p = psnr(fake, cf)
        if isinstance(p, torch.Tensor):
            p = p.item()
        s = ssim_simple(fake, cf)
        v = vari_rmse(fake, cf, mk)

        all_psnr.append(p)
        all_ssim.append(s)
        all_vari.append(v)

        # Save first 6 VALID (NON-BLACK) patches for visualization
        if is_valid_patch(cf):
            if len(samples) < 6:
                samples.append(
                    {
                        "cloudy": to_img(cl),
                        "generated": to_img(fake),
                        "real": to_img(cf),
                        "psnr": p,
                        "ssim": s,
                        "vari": v,
                    }
                )
        else:
            skipped_count += 1

# ── Print summary ────────────────────────────────────
print(f"\n{'='*55}")
print(f"PHASE 3 EVALUATION — TEST SET RESULTS")
print(f"{'='*55}")
print(f"  Total test patches: {len(all_psnr)}")
print(f"  Skipped (black):    {skipped_count}")
print(f"  PSNR:               {np.mean(all_psnr):.2f} dB ± {np.std(all_psnr):.2f}")
print(f"  SSIM:               {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
print(f"  VARI-RMSE:          {np.mean(all_vari):.4f} ± {np.std(all_vari):.4f}")
print(f"{'='*55}")

# ── Visualize 6 samples ──────────────────────────────
if len(samples) > 0:
    fig, axes = plt.subplots(len(samples), 4, figsize=(18, 4 * len(samples)))

    fig.suptitle(
        f"Phase 3 Results — Physics-Informed cGAN Cloud Removal\n"
        f"PSNR: {np.mean(all_psnr):.2f} dB  |  "
        f"SSIM: {np.mean(all_ssim):.4f}  |  "
        f"VARI-RMSE: {np.mean(all_vari):.4f}",
        fontsize=14,
        fontweight="bold",
        y=0.999,
    )

    titles = [
        "Cloudy Input",
        "Generated (Cloud-Free)",
        "Ground Truth",
        "Difference Map",
    ]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for idx, s in enumerate(samples):
        # Col 0: Cloudy input
        axes[idx, 0].imshow(s["cloudy"])
        axes[idx, 0].set_ylabel(
            f"PSNR:{s['psnr']:.1f}dB\nSSIM:{s['ssim']:.3f}\nVARI:{s['vari']:.3f}",
            fontsize=8,
            rotation=0,
            labelpad=70,
            va="center",
        )
        axes[idx, 0].axis("off")

        # Col 1: Generated output
        axes[idx, 1].imshow(s["generated"])
        axes[idx, 1].axis("off")

        # Col 2: Ground truth
        axes[idx, 2].imshow(s["real"])
        axes[idx, 2].axis("off")

        # Col 3: Difference map
        diff = np.abs(s["generated"] - s["real"]).mean(axis=2)
        im = axes[idx, 3].imshow(diff, cmap="hot", vmin=0, vmax=0.3)
        axes[idx, 3].axis("off")

    plt.colorbar(im, ax=axes[:, 3], shrink=0.8, label="Mean absolute error")
    plt.tight_layout()

    os.makedirs("outputs/results/gan", exist_ok=True)
    save_path = "outputs/results/gan/phase3_evaluation.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()

    print(f"\nVisualization saved: {save_path}")
print("Phase 3 evaluation complete.")
