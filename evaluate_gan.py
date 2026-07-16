"""
evaluate_gan.py
Evaluates the trained GAN on test patches COMPLETELY SELF-CONTAINED.
Shows: cloudy input | generated output | ground truth | difference map
Computes: PSNR, SSIM, VARI-RMSE on all test patches.

Self-contained: GANDataset + metric functions are copied locally so there
is no circular import from run_gan_training.py.
"""
import os, sys, glob, json, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, "D:\\CloudRemoval_Project")

from src.models.generator import Generator

# ═══════════════════════════════════════════════════════
# SELF-CONTAINED DATASET (no circular import)
# ═══════════════════════════════════════════════════════
class GANDataset(Dataset):
    def __init__(self, patches_dir, split, augment=False):
        self.split_dir = Path(patches_dir) / split
        with open(self.split_dir / "patch_manifest.json") as f:
            self.manifest = json.load(f)
        self.patch_ids = sorted(self.manifest.keys())

        self.transform = A.Compose(
            [
                A.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0
                ),
                ToTensorV2(),
            ],
            additional_targets={"cloudfree": "image", "mask": "mask"},
        )

    def _compute_edge_map(self, mask_bin):
        m = (mask_bin * 255).astype(np.float32)
        gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
        G = np.sqrt(gx**2 + gy**2)
        return (G / G.max()).astype(np.float32) if G.max() > 0 else G

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        cloudy = cv2.cvtColor(
            cv2.imread(str(self.split_dir / "cloud" / f"{pid}.png")), cv2.COLOR_BGR2RGB
        )
        cloudfree = cv2.cvtColor(
            cv2.imread(str(self.split_dir / "label" / f"{pid}.png")), cv2.COLOR_BGR2RGB
        )
        mask_raw = cv2.imread(
            str(self.split_dir / "mask" / f"{pid}.png"), cv2.IMREAD_GRAYSCALE
        )
        mask_bin = (mask_raw > 127).astype(np.uint8)

        out = self.transform(image=cloudy, cloudfree=cloudfree, mask=mask_bin)
        cloudy_t = out["image"]
        cloudfree_t = out["cloudfree"]
        mask_t = out["mask"].unsqueeze(0).float()

        edge = self._compute_edge_map(out["mask"].numpy())
        edge_t = torch.from_numpy(edge).unsqueeze(0)

        gen_input = torch.cat([cloudy_t, mask_t, edge_t], dim=0)
        return {
            "gen_input": gen_input,
            "cloudy": cloudy_t,
            "cloudfree": cloudfree_t,
            "mask": mask_t,
        }


# ═══════════════════════════════════════════════════════
# SELF-CONTAINED METRICS
# ═══════════════════════════════════════════════════════
def compute_psnr(pred, target):
    p = (pred + 1) / 2
    t = (target + 1) / 2
    mse = ((p - t) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim_simple(pred, target):
    p = (pred + 1) / 2
    t = (target + 1) / 2
    mu_p, mu_t = p.mean(), t.mean()
    sigma_p = ((p - mu_p) ** 2).mean().sqrt()
    sigma_t = ((t - mu_t) ** 2).mean().sqrt()
    sigma_pt = ((p - mu_p) * (t - mu_t)).mean()
    C1, C2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / (
        (mu_p**2 + mu_t**2 + C1) * (sigma_p**2 + sigma_t**2 + C2)
    )
    return ssim.item()


def compute_vari_rmse(pred, target, mask):
    p01 = torch.clamp((pred + 1) / 2, 0.0, 1.0)
    t01 = torch.clamp((target + 1) / 2, 0.0, 1.0)
    Rp, Gp, Bp = p01[:, 0:1], p01[:, 1:2], p01[:, 2:3]
    Rt, Gt, Bt = t01[:, 0:1], t01[:, 1:2], t01[:, 2:3]
    denom_p = torch.clamp(Gp + Rp - Bp, min=0.1)
    denom_t = torch.clamp(Gt + Rt - Bt, min=0.1)
    vp = torch.clamp((Gp - Rp) / denom_p, -1.0, 1.0)
    vt = torch.clamp((Gt - Rt) / denom_t, -1.0, 1.0)
    diff_sq = (vp - vt) ** 2
    mask_sum = mask.sum()
    if mask_sum < 1e-5:
        return 0.0
    mse = (diff_sq * mask).sum() / mask_sum
    return torch.sqrt(torch.clamp(mse, min=0.0, max=1.0)).item()


# ── Load model ───────────────────────────────────────
CKPT = "outputs/checkpoints/gan/best_generator.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

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

with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        gi = batch["gen_input"].to(DEVICE)
        cf = batch["cloudfree"].to(DEVICE)
        mk = batch["mask"].to(DEVICE)
        cl = batch["cloudy"].to(DEVICE)

        fake = G(gi)

        p = compute_psnr(fake, cf)
        s = compute_ssim_simple(fake, cf)
        v = compute_vari_rmse(fake, cf, mk)

        all_psnr.append(p)
        all_ssim.append(s)
        all_vari.append(v)

        if len(samples) < 6:

            def to_img(t):
                arr = (t[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
                return np.clip(arr, 0, 1)

            samples.append(
                {
                    "cloudy": to_img(cl),
                    "generated": to_img(fake),
                    "real": to_img(cf),
                    "mask": mk[0, 0].cpu().numpy(),
                    "psnr": p,
                    "ssim": s,
                    "vari": v,
                }
            )

# ── Print summary ────────────────────────────────────
print(f"\n{'='*55}")
print(f"PHASE 3 EVALUATION — TEST SET RESULTS")
print(f"{'='*55}")
print(f"  Test patches:  {len(all_psnr)}")
print(f"  PSNR:          {np.mean(all_psnr):.2f} dB +/- {np.std(all_psnr):.2f}")
print(f"  SSIM:          {np.mean(all_ssim):.4f} +/- {np.std(all_ssim):.4f}")
print(f"  VARI-RMSE:     {np.mean(all_vari):.4f} +/- {np.std(all_vari):.4f}")
print(f"{'='*55}")
print(f"\n  Best PSNR patch:  {max(all_psnr):.2f} dB")
print(f"  Worst PSNR patch: {min(all_psnr):.2f} dB")
print(f"  Best SSIM:        {max(all_ssim):.4f}")
print(f"  Best VARI-RMSE:   {min(all_vari):.4f}")
print(f"{'='*55}")

# ── Visualize 6 samples ──────────────────────────────
fig, axes = plt.subplots(6, 4, figsize=(18, 26))
fig.suptitle(
    f"Phase 3 Results — Physics-Informed cGAN Cloud Removal\n"
    f"PSNR: {np.mean(all_psnr):.2f} dB  |  "
    f"SSIM: {np.mean(all_ssim):.4f}  |  "
    f"VARI-RMSE: {np.mean(all_vari):.4f}",
    fontsize=14,
    fontweight="bold",
    y=0.999,
)

for ax, title in zip(
    axes[0], ["Cloudy Input", "Generated (Cloud-Free)", "Ground Truth", "Difference Map"]
):
    ax.set_title(title, fontsize=11, fontweight="bold")

for idx, s in enumerate(samples):
    axes[idx, 0].imshow(np.clip(s["cloudy"], 0, 1))
    axes[idx, 0].set_ylabel(
        f"PSNR:{s['psnr']:.1f}dB\nSSIM:{s['ssim']:.3f}\nVARI:{s['vari']:.3f}",
        fontsize=8,
        rotation=0,
        labelpad=70,
        va="center",
    )
    axes[idx, 0].axis("off")

    axes[idx, 1].imshow(np.clip(s["generated"], 0, 1))
    axes[idx, 1].axis("off")

    axes[idx, 2].imshow(np.clip(s["real"], 0, 1))
    axes[idx, 2].axis("off")

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
