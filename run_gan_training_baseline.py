"""
run_gan_training_baseline.py
CloudVision-RS | Phase 3 Ablation Experiment

BASELINE cGAN — WITHOUT PHYSICS CONSTRAINTS

Purpose:
    Establish a controlled baseline for comparison with the
    Physics-Informed cGAN.

Generator:
    cloudy RGB + cloud mask + Sobel edge map
        -> reconstructed cloud-free RGB

Discriminator:
    cloudy RGB + target/generated RGB
        -> PatchGAN real/fake prediction

Generator loss:
    Adversarial Loss + L1 Reconstruction Loss

IMPORTANT:
    This experiment intentionally DOES NOT use:
      1. VARI loss
      2. Spectral-ratio loss
      3. Edge-coherence physics loss

Everything else is kept as close as possible to the
Physics-Informed cGAN experiment so that the comparison
isolates the contribution of the physics constraints.

This is an ABLATION experiment.
"""

import os
import sys
import glob
import json
from pathlib import Path

import cv2
import yaml
import torch
import torch.nn as nn
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path("D:/CloudRemoval_Project")
sys.path.insert(0, str(PROJECT_ROOT))

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.generator import Generator
from src.models.discriminator import Discriminator

# ============================================================
# DATASET
# ============================================================


class GANDataset(Dataset):
    """
    Dataset used by the baseline cGAN.

    Generator input:
        RGB cloudy image  -> 3 channels
        Cloud mask        -> 1 channel
        Sobel edge map    -> 1 channel

        Total = 5 channels

    Discriminator input:
        Cloudy RGB        -> 3 channels
        Target/generated  -> 3 channels

        Total = 6 channels

    Images are normalized to [-1, 1].
    """

    def __init__(self, patches_dir: str, split: str, augment: bool = True):

        self.split_dir = Path(patches_dir) / split

        with open(self.split_dir / "patch_manifest.json") as f:
            self.manifest = json.load(f)

        self.patch_ids = sorted(self.manifest.keys())

        # ----------------------------------------------------
        # Spatial augmentation
        # ----------------------------------------------------

        spatial = []

        if augment:
            spatial = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]

        # ----------------------------------------------------
        # Normalize images to [-1, 1]
        # ----------------------------------------------------

        self.transform = A.Compose(
            spatial
            + [
                A.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0
                ),
                ToTensorV2(),
            ],
            additional_targets={"cloudfree": "image", "mask": "mask"},
        )

    # --------------------------------------------------------
    # Sobel edge map
    # --------------------------------------------------------

    def _compute_edge_map(self, mask_bin: np.ndarray) -> np.ndarray:

        m = (mask_bin * 255).astype(np.float32)

        gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)

        gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)

        edge = cv2.magnitude(gx, gy)

        edge = edge / (edge.max() + 1e-8)

        return edge.astype(np.float32)

    # --------------------------------------------------------

    def __len__(self):

        return len(self.patch_ids)

    # --------------------------------------------------------

    def __getitem__(self, idx):

        pid = self.patch_ids[idx]

        # ----------------------------------------------------
        # Load cloudy image
        # OpenCV reads BGR -> convert to RGB
        # ----------------------------------------------------

        cloud_path = self.split_dir / "cloud" / f"{pid}.png"

        cloudy = cv2.imread(str(cloud_path))

        if cloudy is None:
            raise FileNotFoundError(f"Could not read cloudy image: {cloud_path}")

        cloudy = cv2.cvtColor(cloudy, cv2.COLOR_BGR2RGB)

        # ----------------------------------------------------
        # Load cloud-free ground truth
        # ----------------------------------------------------

        label_path = self.split_dir / "label" / f"{pid}.png"

        cloudfree = cv2.imread(str(label_path))

        if cloudfree is None:
            raise FileNotFoundError(f"Could not read label: {label_path}")

        cloudfree = cv2.cvtColor(cloudfree, cv2.COLOR_BGR2RGB)

        # ----------------------------------------------------
        # Load cloud mask
        # ----------------------------------------------------

        mask_path = self.split_dir / "mask" / f"{pid}.png"

        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask_raw is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        mask_bin = (mask_raw > 127).astype(np.uint8)

        # ----------------------------------------------------
        # Apply synchronized augmentation
        # ----------------------------------------------------

        out = self.transform(image=cloudy, cloudfree=cloudfree, mask=mask_bin)

        cloudy_t = out["image"]
        cloudfree_t = out["cloudfree"]

        mask_t = out["mask"].unsqueeze(0).float()

        # ----------------------------------------------------
        # Recompute edge after augmentation
        # ----------------------------------------------------

        aug_mask_np = out["mask"].numpy().astype(np.uint8)

        edge = self._compute_edge_map(aug_mask_np)

        edge_t = torch.from_numpy(edge).unsqueeze(0)

        # ----------------------------------------------------
        # Generator input
        # RGB + mask + edge = 5 channels
        # ----------------------------------------------------

        gen_input = torch.cat([cloudy_t, mask_t, edge_t], dim=0)

        return {
            "gen_input": gen_input,
            "cloudy": cloudy_t,
            "cloudfree": cloudfree_t,
            "mask": mask_t,
        }


# ============================================================
# EVALUATION METRICS
# ============================================================


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    PSNR computed in [0,1] image space.

    Higher = better.
    """

    p = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)

    t = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)

    mse = ((p - t) ** 2).mean().item()

    if mse < 1e-10:
        return 100.0

    return float(10.0 * np.log10(1.0 / mse))


# ------------------------------------------------------------


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Global SSIM-style calculation matching the
    existing Phase 3 training implementation.

    Higher = better.
    """

    p = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)

    t = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)

    mu_p = p.mean()
    mu_t = t.mean()

    sp = ((p - mu_p) ** 2).mean().sqrt()

    st = ((t - mu_t) ** 2).mean().sqrt()

    spt = ((p - mu_p) * (t - mu_t)).mean()

    C1 = 0.01**2
    C2 = 0.03**2

    numerator = (2 * mu_p * mu_t + C1) * (2 * spt + C2)

    denominator = (mu_p**2 + mu_t**2 + C1) * (sp**2 + st**2 + C2)

    return float((numerator / denominator).clamp(0.0, 1.0).item())


# ------------------------------------------------------------


def compute_vari_rmse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    """
    VARI-RMSE is NOT used for training in this baseline.

    It is retained ONLY as an evaluation metric so that
    the baseline can be compared fairly with the
    Physics-Informed model.

    Lower = better.
    """

    p = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)

    t = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)

    Rp, Gp, Bp = (p[:, 0:1], p[:, 1:2], p[:, 2:3])

    Rt, Gt, Bt = (t[:, 0:1], t[:, 1:2], t[:, 2:3])

    # Same safe denominator treatment
    # used in the existing evaluation pipeline.

    dp = torch.clamp(Gp + Rp - Bp, min=0.1)

    dt = torch.clamp(Gt + Rt - Bt, min=0.1)

    vp = torch.clamp((Gp - Rp) / dp, -1.0, 1.0)

    vt = torch.clamp((Gt - Rt) / dt, -1.0, 1.0)

    diff_sq = (vp - vt) ** 2

    mask_sum = mask.sum()

    if mask_sum < 1e-5:
        return 0.0

    mse = (diff_sq * mask).sum() / mask_sum

    return float(torch.sqrt(torch.clamp(mse, min=0.0, max=1.0)).item())


# ============================================================
# MAIN TRAINING
# ============================================================


def train():

    # --------------------------------------------------------
    # Load BASELINE GAN configuration
    # --------------------------------------------------------

    config_path = PROJECT_ROOT / "configs" / "gan_baseline_config.yaml"

    with open(config_path, "r") as f:

        config = yaml.safe_load(f)["gan"]

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Separate output directories
    # --------------------------------------------------------

    CKPT_DIR = PROJECT_ROOT / config["checkpoint_dir"]

    RES_DIR = PROJECT_ROOT / config["results_dir"]

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    RES_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 70)
    print("CloudVision-RS")
    print("Phase 3 — Baseline cGAN Ablation")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":

        print(f"GPU: " f"{torch.cuda.get_device_name(0)}")

        print(
            f"VRAM: " f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_ds = GANDataset(config["patches_dir"], "train", augment=True)

    val_ds = GANDataset(config["patches_dir"], "val", augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    print(f"Training patches: {len(train_ds)}")

    print(f"Validation patches: {len(val_ds)}")

    print(f"Train batches: {len(train_loader)}")

    print(f"Val batches: {len(val_loader)}")

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print("\nCreating models...")

    G = Generator(in_channels=config["in_channels"], features=config["features_g"]).to(
        DEVICE
    )

    D = Discriminator(
        in_channels=config["in_channels_d"], features=config["features_d"]
    ).to(DEVICE)

    g_params = sum(p.numel() for p in G.parameters())

    d_params = sum(p.numel() for p in D.parameters())

    print(f"Generator parameters: " f"{g_params / 1e6:.2f}M")

    print(f"Discriminator parameters: " f"{d_params / 1e6:.2f}M")

    # --------------------------------------------------------
    # Loss functions
    # --------------------------------------------------------

    criterion_GAN = nn.BCEWithLogitsLoss()

    criterion_L1 = nn.L1Loss()

    # IMPORTANT:
    # Physics losses are deliberately absent.

    lambda_l1 = config["lambda_l1"]

    print("\nTraining objective:")

    print("  Adversarial loss : ENABLED")

    print(f"  L1 loss          : ENABLED " f"(weight={lambda_l1})")

    print("  VARI loss        : DISABLED")
    print("  Spectral loss    : DISABLED")
    print("  Edge physics     : DISABLED")

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

    opt_G = torch.optim.Adam(
        G.parameters(),
        lr=config["learning_rate_g"],
        betas=(config["beta1"], config["beta2"]),
    )

    opt_D = torch.optim.Adam(
        D.parameters(),
        lr=config["learning_rate_d"],
        betas=(config["beta1"], config["beta2"]),
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    start_epoch = 0
    best_psnr = 0.0
    no_improve = 0

    history = {"G_loss": [], "D_loss": [], "psnr": [], "ssim": [], "vari": []}

    # --------------------------------------------------------
    # Resume ONLY from baseline checkpoints
    # --------------------------------------------------------

    existing_ckpts = sorted(glob.glob(str(CKPT_DIR / "baseline_ep*.pt")))

    if existing_ckpts:

        latest = existing_ckpts[-1]

        print(f"\nFound baseline checkpoint: " f"{os.path.basename(latest)}")

        ckpt = torch.load(latest, map_location=DEVICE, weights_only=False)

        G.load_state_dict(ckpt["G_state"])

        D.load_state_dict(ckpt["D_state"])

        opt_G.load_state_dict(ckpt["opt_G"])

        opt_D.load_state_dict(ckpt["opt_D"])

        start_epoch = ckpt["epoch"]

        best_psnr = ckpt.get("best_psnr", 0.0)

        history = ckpt.get("history", history)

        print(f"Resuming from epoch " f"{start_epoch}")

        print(f"Best PSNR so far: " f"{best_psnr:.2f} dB")

    else:

        print("\nNo baseline checkpoint found.")

        print("Starting baseline training from scratch.")

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(start_epoch, config["max_epochs"]):

        G.train()
        D.train()

        sum_G = 0.0
        sum_D = 0.0

        n_batches = 0

        # ----------------------------------------------------
        # Batch loop
        # ----------------------------------------------------

        for batch_idx, batch in enumerate(train_loader):

            gen_input = batch["gen_input"].to(DEVICE)

            cloudy = batch["cloudy"].to(DEVICE)

            cloudfree = batch["cloudfree"].to(DEVICE)

            # =================================================
            # DISCRIMINATOR UPDATE
            # =================================================

            with torch.no_grad():

                fake_img = G(gen_input)

            # Real pair:
            # cloudy image + real cloud-free image

            real_score = D(cloudy, cloudfree)

            # Label smoothing:
            # real = 0.9

            real_label = torch.ones_like(real_score) * 0.9

            loss_D_real = criterion_GAN(real_score, real_label)

            # Fake pair:
            # cloudy image + generated image

            fake_score = D(cloudy, fake_img.detach())

            fake_label = torch.zeros_like(fake_score)

            loss_D_fake = criterion_GAN(fake_score, fake_label)

            loss_D = (loss_D_real + loss_D_fake) * 0.5

            opt_D.zero_grad()

            loss_D.backward()

            # Same gradient clipping
            # as Physics-Informed experiment

            nn.utils.clip_grad_norm_(D.parameters(), 1.0)

            opt_D.step()

            # =================================================
            # GENERATOR UPDATE
            # =================================================

            # Same one-generator-update-per-batch
            # strategy as Physics-Informed experiment.

            fake_img = G(gen_input)

            fake_score = D(cloudy, fake_img)

            # -------------------------------------------------
            # Adversarial loss
            # -------------------------------------------------

            loss_G_adv = criterion_GAN(fake_score, torch.ones_like(fake_score))

            # -------------------------------------------------
            # L1 reconstruction loss
            # -------------------------------------------------

            loss_G_L1 = criterion_L1(fake_img, cloudfree) * lambda_l1

            # -------------------------------------------------
            # NO PHYSICS LOSS
            # -------------------------------------------------

            loss_G = loss_G_adv + loss_G_L1

            opt_G.zero_grad()

            loss_G.backward()

            nn.utils.clip_grad_norm_(G.parameters(), 1.0)

            opt_G.step()

            sum_G += loss_G.item()
            sum_D += loss_D.item()

            n_batches += 1

        # ====================================================
        # AVERAGE TRAINING LOSSES
        # ====================================================

        avg_G = sum_G / n_batches

        avg_D = sum_D / n_batches

        # ====================================================
        # VALIDATION
        # ====================================================

        G.eval()

        val_psnr = 0.0
        val_ssim = 0.0
        val_vari = 0.0

        n_val = 0

        with torch.no_grad():

            for batch in val_loader:

                gi = batch["gen_input"].to(DEVICE)

                cf = batch["cloudfree"].to(DEVICE)

                mk = batch["mask"].to(DEVICE)

                fake = G(gi)

                for i in range(fake.shape[0]):

                    val_psnr += compute_psnr(fake[i : i + 1], cf[i : i + 1])

                    val_ssim += compute_ssim(fake[i : i + 1], cf[i : i + 1])

                    # Evaluation metric only.
                    # NOT used during training.

                    val_vari += compute_vari_rmse(
                        fake[i : i + 1], cf[i : i + 1], mk[i : i + 1]
                    )

                    n_val += 1

        val_psnr /= n_val
        val_ssim /= n_val
        val_vari /= n_val

        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        print(
            f"\nEpoch "
            f"{epoch + 1:03d}/"
            f"{config['max_epochs']} | "
            f"G:{avg_G:.4f} | "
            f"D:{avg_D:.4f} | "
            f"PSNR:{val_psnr:.2f} dB | "
            f"SSIM:{val_ssim:.4f} | "
            f"VARI-RMSE:{val_vari:.4f}"
        )

        # ----------------------------------------------------
        # GAN health checks
        # ----------------------------------------------------

        if avg_D > 0.85:

            print(
                f"  WARNING: D_loss={avg_D:.3f} " f"— discriminator may be too strong."
            )

        if avg_D < 0.30:

            print(
                f"  WARNING: D_loss={avg_D:.3f} "
                f"— discriminator may be too weak/collapsed."
            )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history["G_loss"].append(avg_G)

        history["D_loss"].append(avg_D)

        history["psnr"].append(val_psnr)

        history["ssim"].append(val_ssim)

        history["vari"].append(val_vari)

        # ====================================================
        # PERIODIC CHECKPOINT
        # ====================================================

        if (epoch + 1) % config["save_every_epochs"] == 0:

            ckpt_path = (
                CKPT_DIR / f"baseline_ep"
                f"{epoch + 1:03d}"
                f"_psnr"
                f"{val_psnr:.2f}.pt"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "G_state": G.state_dict(),
                    "D_state": D.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(),
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "val_vari": val_vari,
                    "best_psnr": best_psnr,
                    "history": history,
                },
                ckpt_path,
            )

            print(f"  ✓ Baseline checkpoint saved: " f"{ckpt_path.name}")

        # ====================================================
        # BEST GENERATOR + EARLY STOPPING
        # ====================================================

        if val_psnr > best_psnr:

            best_psnr = val_psnr
            no_improve = 0

            best_path = CKPT_DIR / "best_generator_baseline.pt"

            torch.save(G.state_dict(), best_path)

            print(f"  ✓ Best baseline Generator saved " f"(PSNR: {best_psnr:.2f} dB)")

        else:

            no_improve += 1

            if no_improve >= config["early_stopping_patience"]:

                print(f"\nEarly stopping at epoch " f"{epoch + 1}")

                print(
                    f"No PSNR improvement for "
                    f"{config['early_stopping_patience']} epochs."
                )

                break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print("BASELINE cGAN TRAINING COMPLETE")

    print("=" * 70)

    print(f"Best validation PSNR: " f"{best_psnr:.2f} dB")

    print("Best Generator:")

    print(CKPT_DIR / "best_generator_baseline.pt")

    print(f"Results directory:")

    print(RES_DIR)

    print("=" * 70)

    # ========================================================
    # TRAINING CURVES
    # ========================================================

    if len(history["psnr"]) > 0:

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        fig.suptitle(
            "CloudVision-RS — Baseline cGAN Training", fontsize=14, fontweight="bold"
        )

        # ----------------------------------------------------
        # GAN losses
        # ----------------------------------------------------

        axes[0, 0].plot(history["G_loss"], label="Generator Loss")

        axes[0, 0].plot(history["D_loss"], label="Discriminator Loss")

        axes[0, 0].set_title("GAN Training Losses")

        axes[0, 0].set_xlabel("Epoch")

        axes[0, 0].set_ylabel("Loss")

        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # ----------------------------------------------------
        # PSNR
        # ----------------------------------------------------

        axes[0, 1].plot(history["psnr"])

        axes[0, 1].set_title("Validation PSNR")

        axes[0, 1].set_xlabel("Epoch")

        axes[0, 1].set_ylabel("PSNR (dB)")

        axes[0, 1].grid(True)

        # ----------------------------------------------------
        # SSIM
        # ----------------------------------------------------

        axes[0, 2].plot(history["ssim"])

        axes[0, 2].set_title("Validation SSIM")

        axes[0, 2].set_xlabel("Epoch")

        axes[0, 2].set_ylabel("SSIM")

        axes[0, 2].grid(True)

        # ----------------------------------------------------
        # VARI-RMSE
        # ----------------------------------------------------

        axes[1, 0].plot(history["vari"])

        axes[1, 0].set_title("Validation VARI-RMSE")

        axes[1, 0].set_xlabel("Epoch")

        axes[1, 0].set_ylabel("VARI-RMSE")

        axes[1, 0].grid(True)

        # ----------------------------------------------------
        # Experiment information
        # ----------------------------------------------------

        axes[1, 1].axis("off")

        summary_text = (
            "BASELINE cGAN\n\n"
            "Generator input:\n"
            "  RGB + Mask + Edge\n\n"
            "Discriminator:\n"
            "  Cloudy RGB + Target RGB\n\n"
            "Generator loss:\n"
            "  Adversarial + L1\n\n"
            f"L1 weight: {lambda_l1}\n\n"
            "Physics losses:\n"
            "  VARI       OFF\n"
            "  Spectral   OFF\n"
            "  Edge       OFF\n\n"
            f"Best PSNR: {best_psnr:.2f} dB\n"
            f"Epochs: {len(history['psnr'])}"
        )

        axes[1, 1].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 1].transAxes,
            verticalalignment="top",
            fontsize=11,
            family="monospace",
        )

        # ----------------------------------------------------
        # Empty panel
        # ----------------------------------------------------

        axes[1, 2].axis("off")

        plt.tight_layout()

        curves_path = RES_DIR / "baseline_training_curves.png"

        plt.savefig(curves_path, dpi=120, bbox_inches="tight")

        plt.close()

        print("\nTraining curves saved to:")

        print(curves_path)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()
