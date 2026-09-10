"""
run_gan_training.py — Phase 3 Physics-Informed cGAN
CloudVision-RS | Team NirmalDrishti | ISRO BAH 2026

WHAT THIS DOES:
Trains a conditional GAN to remove clouds from satellite imagery.
Generator: takes (cloudy_RGB + cloud_mask + edge_map) → outputs cloud-free image
Discriminator: judges whether output is real or generated

PHYSICS LOSSES (all inline — no module import to avoid caching issues):
1. VARI Loss      — vegetation index preservation (G-R)/(G+R-B)
2. Spectral Ratio — R/G, B/G, R/B ratio consistency
3. Edge Coherence — smooth transitions at cloud boundaries

KEY DESIGN DECISIONS:
- Train G once per batch (not 2x) — prevents discriminator collapse
- Label smoothing: real=0.9 not 1.0 — keeps D learning
- Gradient clipping at 1.0 on both G and D
- L1 weight=100 anchors pixel accuracy, prevents mode collapse
- All VARI denominators clamped to min=0.1 — prevents overflow
- Physics losses capped at max=5.0 — prevents exploding loss
- num_workers=0 — required on Windows
- weights_only=False in torch.load — required for older checkpoints
"""

import os
import sys
import glob
import json
import cv2
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, 'D:\\CloudRemoval_Project')

import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.generator     import Generator
from src.models.discriminator import Discriminator


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class GANDataset(Dataset):
    """
    PyTorch Dataset for GAN training.

    Returns per sample:
      gen_input : [5, 256, 256]  — RGB(-1,1) + mask(0,1) + edge(0,1)
      cloudy    : [3, 256, 256]  — cloudy RGB in [-1, 1]  (discriminator condition)
      cloudfree : [3, 256, 256]  — cloud-free target in [-1, 1]
      mask      : [1, 256, 256]  — binary cloud mask {0, 1}

    NORMALISATION:
    GAN uses [-1, 1] (not [0, 1] like segmentation).
    Generator output is tanh → [-1, 1].
    Target must match → normalise with mean=0.5, std=0.5.
    Formula: (x/255 - 0.5) / 0.5 = 2*(x/255) - 1
    """

    def __init__(self, patches_dir: str, split: str, augment: bool = True):
        self.split_dir = Path(patches_dir) / split

        with open(self.split_dir / 'patch_manifest.json') as f:
            self.manifest = json.load(f)
        self.patch_ids = sorted(self.manifest.keys())

        # Spatial augmentations only for training split
        spatial = []
        if augment:
            spatial = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]

        # Normalise images to [-1, 1]
        # additional_targets: 'cloudfree' gets same image transform
        #                     'mask' gets only spatial transforms (no normalisation)
        self.transform = A.Compose(
            spatial + [
                A.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ],
            additional_targets={'cloudfree': 'image', 'mask': 'mask'}
        )

    def _compute_edge_map(self, mask_bin: np.ndarray) -> np.ndarray:
        """
        Sobel edge map of cloud mask.
        High values = cloud boundary pixels.
        Tells generator exactly WHERE to create smooth transitions.
        """
        m  = (mask_bin * 255).astype(np.float32)
        gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
        G  = np.sqrt(gx ** 2 + gy ** 2)
        if G.max() > 0:
            G = G / G.max()
        return G.astype(np.float32)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict:
        pid = self.patch_ids[idx]

        # Load images — OpenCV reads BGR, convert to RGB immediately
        cloudy = cv2.cvtColor(
            cv2.imread(str(self.split_dir / 'cloud' / f'{pid}.png')),
            cv2.COLOR_BGR2RGB
        )
        cloudfree = cv2.cvtColor(
            cv2.imread(str(self.split_dir / 'label' / f'{pid}.png')),
            cv2.COLOR_BGR2RGB
        )
        mask_raw = cv2.imread(
            str(self.split_dir / 'mask' / f'{pid}.png'),
            cv2.IMREAD_GRAYSCALE
        )
        mask_bin = (mask_raw > 127).astype(np.uint8)  # {0, 1}

        # Apply synchronized transforms
        out = self.transform(image=cloudy, cloudfree=cloudfree, mask=mask_bin)

        cloudy_t    = out['image']                          # [3, H, W] in [-1, 1]
        cloudfree_t = out['cloudfree']                      # [3, H, W] in [-1, 1]
        mask_t      = out['mask'].unsqueeze(0).float()      # [1, H, W] in {0, 1}

        # Recompute edge from augmented mask (after spatial transforms)
        aug_mask_np = out['mask'].numpy().astype(np.uint8)
        edge        = self._compute_edge_map(aug_mask_np)
        edge_t      = torch.from_numpy(edge).unsqueeze(0)  # [1, H, W] in [0, 1]

        # Generator input: 5 channels
        gen_input = torch.cat([cloudy_t, mask_t, edge_t], dim=0)  # [5, H, W]

        return {
            'gen_input': gen_input,     # [5, H, W]
            'cloudy':    cloudy_t,      # [3, H, W] — discriminator condition
            'cloudfree': cloudfree_t,   # [3, H, W] — real target
            'mask':      mask_t,        # [1, H, W] — for physics loss
        }


# ═══════════════════════════════════════════════════════════════
# PHYSICS LOSSES — INLINE
# All computed directly here, no external module.
# This avoids Python module caching which caused VARI overflow bug.
# ═══════════════════════════════════════════════════════════════

def physics_loss(fake_img, real_img, mask, config):
    """
    Physics-informed losses enforcing spectral correctness.

    fake_img : [B, 3, H, W] in [-1, 1]  — generator output
    real_img : [B, 3, H, W] in [-1, 1]  — ground truth cloud-free
    mask     : [B, 1, H, W] in {0, 1}   — cloud mask (1 = reconstructed region)
    config   : dict with lambda weights

    All losses computed only over cloud-masked pixels (mask=1).
    These are the pixels the generator actually reconstructed.
    Applying to unchanged clear-sky pixels would penalize correctly
    preserved pixels.

    WHY INLINE NOT IMPORTED:
    Python caches module imports. If physics_loss.py is modified and
    reimported in the same session, the old version may still run.
    Inline code always runs the current version.
    """

    # Denormalise from [-1, 1] to [0, 1]
    # Clamp strictly to valid range — denorm of tanh output may slightly exceed [0,1]
    pred = torch.clamp((fake_img + 1.0) / 2.0, 0.0, 1.0)
    tgt  = torch.clamp((real_img + 1.0) / 2.0, 0.0, 1.0)

    Rp, Gp, Bp = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    Rt, Gt, Bt = tgt[:, 0:1],  tgt[:, 1:2],  tgt[:, 2:3]

    # ── 1. VARI Loss ───────────────────────────────────────────
    # VARI (Visible Atmospherically Resistant Index):
    #   VARI = (Green - Red) / (Green + Red - Blue)
    # Published: Gitelson et al. 2002. Correlation with NDVI r=0.92.
    # Standard RGB-accessible vegetation index.
    #
    # WHY clamp denominator to min=0.1:
    # When G + R - B ≈ 0 (water, shadow pixels), division produces
    # values in thousands. Squaring and summing = VARI-RMSE 15713.
    # Clamping to 0.1 bounds VARI to [-10, 10] before clamping to [-1,1].
    dp = torch.clamp(Gp + Rp - Bp, min=0.1)
    dt = torch.clamp(Gt + Rt - Bt, min=0.1)

    vari_pred = torch.clamp((Gp - Rp) / dp, -1.0, 1.0)
    vari_tgt  = torch.clamp((Gt - Rt) / dt, -1.0, 1.0)

    l_vari = ((vari_pred - vari_tgt) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    l_vari = torch.clamp(l_vari, max=5.0)  # hard safety cap

    # ── 2. Spectral Ratio Loss ─────────────────────────────────
    # R/G, B/G, R/B ratios encode spectral signature of land cover.
    # Water: high B/G, low R/G.
    # Vegetation: low R/G, low B/G (chlorophyll absorption).
    # Urban: R/G ≈ B/G ≈ 1 (spectrally flat).
    #
    # Using ratios not absolute values: a forest at noon and at different
    # sun angle has the same R/G but different absolute brightness.
    # Ratios are illumination-invariant.
    eps   = 0.1
    l_spec = torch.tensor(0.0, device=fake_img.device)

    for rp, rt in [
        (Rp / (Gp + eps), Rt / (Gt + eps)),  # R/G
        (Bp / (Gp + eps), Bt / (Gt + eps)),  # B/G
        (Rp / (Bp + eps), Rt / (Bt + eps)),  # R/B
    ]:
        diff    = torch.clamp((rp - rt) ** 2, max=5.0)
        l_spec += (diff * mask).sum() / (mask.sum() + 1e-8)

    l_spec = l_spec / 3.0

    # ── 3. Edge Coherence Loss ─────────────────────────────────
    # The surface should be spatially smooth at cloud boundaries.
    # Cloud removal artifact: visible seam where reconstructed region
    # meets original clear-sky region.
    # Penalises gradient magnitude difference between fake and real.
    def grad_mag(img):
        dx = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs()
        dy = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs()
        return dx.mean() + dy.mean()

    l_edge = (grad_mag(fake_img) - grad_mag(real_img)).abs()

    # ── Total physics loss ─────────────────────────────────────
    total = (config['lambda_vari']     * l_vari  +
             config['lambda_spectral'] * l_spec  +
             config['lambda_edge']     * l_edge)

    return {
        'total':    total,
        'vari':     l_vari.item(),
        'spectral': l_spec.item(),
        'edge':     l_edge.item(),
    }


# ═══════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.
    Both tensors in [-1, 1], converted to [0, 1] first.
    Higher = better. Above 30 dB = excellent. 24-28 dB = good.
    """
    p   = torch.clamp((pred   + 1) / 2, 0, 1)
    t   = torch.clamp((target + 1) / 2, 0, 1)
    mse = ((p - t) ** 2).mean().item()
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Structural Similarity Index.
    Range [0, 1]. Higher = better.
    Measures luminance, contrast, and structural similarity.
    More perceptually meaningful than PSNR.
    """
    p     = torch.clamp((pred   + 1) / 2, 0, 1)
    t     = torch.clamp((target + 1) / 2, 0, 1)
    mu_p  = p.mean();  mu_t = t.mean()
    sp    = ((p - mu_p) ** 2).mean().sqrt()
    st    = ((t - mu_t) ** 2).mean().sqrt()
    spt   = ((p - mu_p) * (t - mu_t)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_p * mu_t + C1) * (2 * spt + C2)) /
            ((mu_p ** 2 + mu_t ** 2 + C1) * (sp ** 2 + st ** 2 + C2))).item()


def compute_vari_rmse(pred: torch.Tensor,
                      target: torch.Tensor,
                      mask: torch.Tensor) -> float:
    """
    VARI-RMSE: Root Mean Squared Error of vegetation index
    computed ONLY over cloud-masked (reconstructed) pixels.

    This is your unique scientific metric.
    No published RICE2 cloud removal paper reports this.
    Lower = better spectral correctness in reconstructed vegetation.
    """
    p  = torch.clamp((pred   + 1) / 2, 0, 1)
    t  = torch.clamp((target + 1) / 2, 0, 1)
    Rp, Gp, Bp = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    Rt, Gt, Bt = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    dp = torch.clamp(Gp + Rp - Bp, min=0.1)
    dt = torch.clamp(Gt + Rt - Bt, min=0.1)
    vp = torch.clamp((Gp - Rp) / dp, -1, 1)
    vt = torch.clamp((Gt - Rt) / dt, -1, 1)
    mse = ((vp - vt) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    return mse.sqrt().item()


# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════

def train():
    # ── Load config ───────────────────────────────────────────
    with open('configs/gan_config.yaml') as f:
        config = yaml.safe_load(f)['gan']

    DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CKPT_DIR = config['checkpoint_dir']
    RES_DIR  = config['results_dir']
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RES_DIR,  exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Datasets ──────────────────────────────────────────────
    train_ds = GANDataset(config['patches_dir'], 'train', augment=True)
    val_ds   = GANDataset(config['patches_dir'], 'val',   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,     # Windows: multiprocessing spawn causes crash with num_workers>0
        drop_last=True     # drop incomplete last batch to keep batch size consistent
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # ── Models ────────────────────────────────────────────────
    G = Generator(
        in_channels=config['in_channels'],    # 5 (RGB + mask + edge)
        features=config['features_g']          # 64
    ).to(DEVICE)

    D = Discriminator(
        in_channels=config['in_channels_d'],  # 6 (condition + target)
        features=config['features_d']          # 64
    ).to(DEVICE)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator:     {g_params / 1e6:.1f}M parameters")
    print(f"Discriminator: {d_params / 1e6:.1f}M parameters")

    # ── VARI sanity check before training ────────────────────
    # If VARI returns values > 1.0, there is a normalisation bug.
    # Catch it before wasting GPU time on a broken training run.
    print("\nVARI sanity check...")
    dummy_pred = torch.rand(2, 3, 64, 64).to(DEVICE) * 2 - 1
    dummy_tgt  = torch.rand(2, 3, 64, 64).to(DEVICE) * 2 - 1
    dummy_mask = torch.ones(2, 1, 64, 64).to(DEVICE)
    phys_test  = physics_loss(dummy_pred, dummy_tgt, dummy_mask, config)

    if phys_test['vari'] > 2.0:
        print(f"ERROR: VARI test value = {phys_test['vari']:.4f} — should be < 1.0")
        print("Check denorm logic in physics_loss function")
        return

    print(f"VARI test: {phys_test['vari']:.4f} (OK — must be < 2.0)")
    print("Sanity check passed\n")

    # ── Loss functions ────────────────────────────────────────
    # GAN adversarial loss — BCEWithLogitsLoss for numerical stability
    criterion_GAN = nn.BCEWithLogitsLoss()
    # L1 pixel loss — high weight (100) anchors output to correct pixel values
    criterion_L1  = nn.L1Loss()

    # ── Optimizers ────────────────────────────────────────────
    # Adam with beta1=0.5 is standard for GANs (from Pix2Pix paper)
    # Lower beta1 (0.5 vs 0.9) reduces momentum, prevents oscillation
    opt_G = torch.optim.Adam(
        G.parameters(),
        lr=config['learning_rate_g'],
        betas=(config['beta1'], config['beta2'])
    )
    opt_D = torch.optim.Adam(
        D.parameters(),
        lr=config['learning_rate_d'],
        betas=(config['beta1'], config['beta2'])
    )

    # ── Resume from checkpoint ────────────────────────────────
    start_epoch = 0
    best_psnr   = 0.0
    no_improve  = 0
    history = {
        'G_loss': [], 'D_loss': [],
        'psnr': [],   'ssim': [],   'vari': [],
        'vari_loss': [], 'spec_loss': [], 'edge_loss': []
    }

    existing_ckpts = sorted(glob.glob(f'{CKPT_DIR}/gan_ep*.pt'))
    if existing_ckpts:
        latest = existing_ckpts[-1]
        print(f"Found checkpoint: {os.path.basename(latest)}")
        ckpt = torch.load(latest, map_location=DEVICE, weights_only=False)
        G.load_state_dict(ckpt['G_state'])
        D.load_state_dict(ckpt['D_state'])
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt['epoch']
        best_psnr   = ckpt.get('best_psnr', 0.0)
        history     = ckpt.get('history', history)
        print(f"Resumed from epoch {start_epoch} | Best PSNR so far: {best_psnr:.2f} dB")
    else:
        print("No checkpoint found — starting fresh training")

    print(f"\n{'='*65}")
    print("Phase 3: Physics-Informed cGAN Training")
    print(f"{'='*65}\n")

    # ── Training loop ─────────────────────────────────────────
    for epoch in range(start_epoch, config['max_epochs']):

        G.train()
        D.train()

        sum_G, sum_D          = 0.0, 0.0
        sum_vari, sum_spec, sum_edge = 0.0, 0.0, 0.0
        n_batches             = 0

        for batch_idx, batch in enumerate(train_loader):
            gen_input = batch['gen_input'].to(DEVICE)   # [B, 5, H, W]
            cloudy    = batch['cloudy'].to(DEVICE)       # [B, 3, H, W]
            cloudfree = batch['cloudfree'].to(DEVICE)    # [B, 3, H, W]
            mask      = batch['mask'].to(DEVICE)         # [B, 1, H, W]

            # ── Discriminator update ──────────────────────────
            # Generate fake image (detached — no gradient into G yet)
            with torch.no_grad():
                fake_img = G(gen_input)

            # Real pair: (cloudy_condition, real_cloudfree)
            real_score = D(cloudy, cloudfree)
            # Label smoothing: 0.9 not 1.0 — prevents D overconfidence
            real_label = torch.ones_like(real_score) * 0.9
            loss_D_real = criterion_GAN(real_score, real_label)

            # Fake pair: (cloudy_condition, generated_cloudfree)
            fake_score = D(cloudy, fake_img.detach())
            fake_label = torch.zeros_like(fake_score)
            loss_D_fake = criterion_GAN(fake_score, fake_label)

            loss_D = (loss_D_real + loss_D_fake) * 0.5

            opt_D.zero_grad()
            loss_D.backward()
            nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_D.step()

            # ── Generator update ──────────────────────────────
            # Train G ONCE per batch.
            # Two-timescale: D trained once, G trained once.
            # Training G twice caused D_loss → 1.0 (discriminator collapse).
            fake_img   = G(gen_input)
            fake_score = D(cloudy, fake_img)

            # G wants D to say its output is real
            loss_G_adv = criterion_GAN(fake_score, torch.ones_like(fake_score))

            # L1 pixel reconstruction — weight=100 anchors output
            # Without this high weight, G ignores pixel accuracy
            loss_G_L1 = criterion_L1(fake_img, cloudfree) * config['lambda_l1']

            # Physics losses — spectral correctness constraints
            phys        = physics_loss(fake_img, cloudfree, mask, config)
            loss_G_phys = phys['total']

            loss_G = loss_G_adv + loss_G_L1 + loss_G_phys

            opt_G.zero_grad()
            loss_G.backward()
            nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

            sum_G    += loss_G.item()
            sum_D    += loss_D.item()
            sum_vari += phys['vari']
            sum_spec += phys['spectral']
            sum_edge += phys['edge']
            n_batches += 1

        avg_G    = sum_G    / n_batches
        avg_D    = sum_D    / n_batches
        avg_vari = sum_vari / n_batches
        avg_spec = sum_spec / n_batches
        avg_edge = sum_edge / n_batches

        # ── Validation ────────────────────────────────────────
        G.eval()
        val_psnr, val_ssim, val_vari = 0.0, 0.0, 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                gi = batch['gen_input'].to(DEVICE)
                cf = batch['cloudfree'].to(DEVICE)
                mk = batch['mask'].to(DEVICE)
                fk = G(gi)

                for i in range(fk.shape[0]):
                    val_psnr += compute_psnr(fk[i:i+1], cf[i:i+1])
                    val_ssim += compute_ssim(fk[i:i+1], cf[i:i+1])
                    val_vari += compute_vari_rmse(fk[i:i+1], cf[i:i+1], mk[i:i+1])
                    n_val    += 1

        val_psnr /= n_val
        val_ssim /= n_val
        val_vari /= n_val

        # ── Epoch summary ─────────────────────────────────────
        print(
            f"Epoch {epoch+1:03d}/{config['max_epochs']} | "
            f"G:{avg_G:.4f}  D:{avg_D:.4f} | "
            f"PSNR:{val_psnr:.2f}dB  SSIM:{val_ssim:.4f}  "
            f"VARI-RMSE:{val_vari:.4f} | "
            f"[vari:{avg_vari:.3f} spec:{avg_spec:.3f} edge:{avg_edge:.4f}]"
        )

        # D_loss health check
        if avg_D > 0.85:
            print(f"  WARNING: D_loss={avg_D:.3f} — discriminator too strong. "
                  "Consider reducing D learning rate.")
        if avg_D < 0.30:
            print(f"  WARNING: D_loss={avg_D:.3f} — discriminator collapsed. "
                  "G is too easily fooling D.")

        # Update history
        history['G_loss'].append(avg_G)
        history['D_loss'].append(avg_D)
        history['psnr'].append(val_psnr)
        history['ssim'].append(val_ssim)
        history['vari'].append(val_vari)
        history['vari_loss'].append(avg_vari)
        history['spec_loss'].append(avg_spec)
        history['edge_loss'].append(avg_edge)

        # ── Save checkpoint every N epochs ────────────────────
        if (epoch + 1) % config['save_every_epochs'] == 0:
            ckpt_path = f"{CKPT_DIR}/gan_ep{epoch+1:03d}_psnr{val_psnr:.2f}.pt"
            torch.save({
                'epoch':    epoch + 1,
                'G_state':  G.state_dict(),
                'D_state':  D.state_dict(),
                'opt_G':    opt_G.state_dict(),
                'opt_D':    opt_D.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_vari': val_vari,
                'best_psnr': best_psnr,
                'history':  history
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved: {os.path.basename(ckpt_path)}")

        # ── Best generator ────────────────────────────────────
        if val_psnr > best_psnr:
            best_psnr  = val_psnr
            no_improve = 0
            torch.save(G.state_dict(), f"{CKPT_DIR}/best_generator.pt")
            print(f"  ✓ Best generator saved (PSNR: {best_psnr:.2f} dB)")
        else:
            no_improve += 1
            if no_improve >= config['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no improvement for {config['early_stopping_patience']} epochs)")
                break

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Phase 3 Training Complete")
    print(f"Best PSNR:   {best_psnr:.2f} dB")
    print(f"Generator:   {CKPT_DIR}/best_generator.pt")
    print(f"{'='*65}")

    # ── Training curves ───────────────────────────────────────
    if len(history['psnr']) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Phase 3 Training Curves — Physics-Informed cGAN',
                     fontsize=14, fontweight='bold')

        axes[0,0].plot(history['G_loss'], label='G loss', color='steelblue')
        axes[0,0].plot(history['D_loss'], label='D loss', color='orange')
        axes[0,0].set_title('GAN Losses')
        axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='D healthy zone')

        axes[0,1].plot(history['psnr'], color='green')
        axes[0,1].set_title('Validation PSNR (dB) — higher is better')
        axes[0,1].grid(True, alpha=0.3)

        axes[0,2].plot(history['ssim'], color='purple')
        axes[0,2].set_title('Validation SSIM — higher is better')
        axes[0,2].grid(True, alpha=0.3)

        axes[1,0].plot(history['vari'], color='teal')
        axes[1,0].set_title('VARI-RMSE (validation) — lower is better')
        axes[1,0].grid(True, alpha=0.3)

        axes[1,1].plot(history['vari_loss'], label='VARI', color='green')
        axes[1,1].plot(history['spec_loss'], label='Spectral', color='blue')
        axes[1,1].plot(history['edge_loss'], label='Edge', color='red')
        axes[1,1].set_title('Physics Loss Components (training)')
        axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

        axes[1,2].axis('off')
        summary_text = (
            f"Training Summary\n\n"
            f"Best PSNR:     {best_psnr:.2f} dB\n"
            f"Best SSIM:     {max(history['ssim']):.4f}\n"
            f"Best VARI-RMSE:{min(history['vari']):.4f}\n\n"
            f"Total epochs:  {len(history['psnr'])}\n"
            f"Batch size:    {config['batch_size']}\n"
            f"LR (G):        {config['learning_rate_g']}\n"
            f"LR (D):        {config['learning_rate_d']}\n\n"
            f"Loss weights:\n"
            f"  L1:       {config['lambda_l1']}\n"
            f"  VARI:     {config['lambda_vari']}\n"
            f"  Spectral: {config['lambda_spectral']}\n"
            f"  Edge:     {config['lambda_edge']}"
        )
        axes[1,2].text(0.1, 0.9, summary_text, transform=axes[1,2].transAxes,
                       verticalalignment='top', fontsize=11,
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        curves_path = f"{RES_DIR}/training_curves.png"
        plt.savefig(curves_path, dpi=120, bbox_inches='tight')
        print(f"Training curves saved: {curves_path}")
        plt.close()


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# Windows requires if __name__ == '__main__' guard for multiprocessing
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    train()
