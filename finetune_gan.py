"""
finetune_gan.py — Fine-tune Phase 3 cGAN on real LISS-IV patches

Reuses GANDataset, physics_loss, and metrics functions VERBATIM from
run_gan_training.py -- only the data source, learning rate, and epoch
count change. Starts from gan_ep030_psnr23.87.pt (best checkpoint that
has BOTH Generator and Discriminator weights saved together).

Run: conda activate cloudremoval
python finetune_gan.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # fixes OMP duplicate-library crash

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

sys.path.insert(0, 'D:\\CloudRemoval_Project')

import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.generator import Generator
from src.models.discriminator import Discriminator


# ═══════════════════════════════════════════════════════════════
# DATASET — identical to run_gan_training.py's GANDataset
# ═══════════════════════════════════════════════════════════════

class GANDataset(Dataset):
    def __init__(self, patches_dir: str, split: str, augment: bool = True):
        self.split_dir = Path(patches_dir) / split
        with open(self.split_dir / 'patch_manifest.json') as f:
            self.manifest = json.load(f)
        self.patch_ids = sorted(self.manifest.keys())

        spatial = []
        if augment:
            spatial = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        self.transform = A.Compose(
            spatial + [
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
                ToTensorV2()
            ],
            additional_targets={'cloudfree': 'image', 'mask': 'mask'}
        )

    def _compute_edge_map(self, mask_bin: np.ndarray) -> np.ndarray:
        m = (mask_bin * 255).astype(np.float32)
        gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
        G = np.sqrt(gx ** 2 + gy ** 2)
        if G.max() > 0:
            G = G / G.max()
        return G.astype(np.float32)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict:
        pid = self.patch_ids[idx]
        cloudy = cv2.cvtColor(cv2.imread(str(self.split_dir / 'cloud' / f'{pid}.png')), cv2.COLOR_BGR2RGB)
        cloudfree = cv2.cvtColor(cv2.imread(str(self.split_dir / 'label' / f'{pid}.png')), cv2.COLOR_BGR2RGB)
        mask_raw = cv2.imread(str(self.split_dir / 'mask' / f'{pid}.png'), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask_raw > 127).astype(np.uint8)

        out = self.transform(image=cloudy, cloudfree=cloudfree, mask=mask_bin)
        cloudy_t = out['image']
        cloudfree_t = out['cloudfree']
        mask_t = out['mask'].unsqueeze(0).float()

        aug_mask_np = out['mask'].numpy().astype(np.uint8)
        edge = self._compute_edge_map(aug_mask_np)
        edge_t = torch.from_numpy(edge).unsqueeze(0)

        gen_input = torch.cat([cloudy_t, mask_t, edge_t], dim=0)
        return {'gen_input': gen_input, 'cloudy': cloudy_t, 'cloudfree': cloudfree_t, 'mask': mask_t}


# ═══════════════════════════════════════════════════════════════
# PHYSICS LOSSES — identical to run_gan_training.py, inline
# ═══════════════════════════════════════════════════════════════

def physics_loss(fake_img, real_img, mask, config):
    pred = torch.clamp((fake_img + 1.0) / 2.0, 0.0, 1.0)
    tgt = torch.clamp((real_img + 1.0) / 2.0, 0.0, 1.0)
    Rp, Gp, Bp = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    Rt, Gt, Bt = tgt[:, 0:1], tgt[:, 1:2], tgt[:, 2:3]

    dp = torch.clamp(Gp + Rp - Bp, min=0.1)
    dt = torch.clamp(Gt + Rt - Bt, min=0.1)
    vari_pred = torch.clamp((Gp - Rp) / dp, -1.0, 1.0)
    vari_tgt = torch.clamp((Gt - Rt) / dt, -1.0, 1.0)
    l_vari = ((vari_pred - vari_tgt) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    l_vari = torch.clamp(l_vari, max=5.0)

    eps = 0.1
    l_spec = torch.tensor(0.0, device=fake_img.device)
    for rp, rt in [
        (Rp / (Gp + eps), Rt / (Gt + eps)),
        (Bp / (Gp + eps), Bt / (Gt + eps)),
        (Rp / (Bp + eps), Rt / (Bt + eps)),
    ]:
        diff = torch.clamp((rp - rt) ** 2, max=5.0)
        l_spec += (diff * mask).sum() / (mask.sum() + 1e-8)
    l_spec = l_spec / 3.0

    def grad_mag(img):
        dx = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs()
        dy = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs()
        return dx.mean() + dy.mean()
    l_edge = (grad_mag(fake_img) - grad_mag(real_img)).abs()

    total = (config['lambda_vari'] * l_vari +
             config['lambda_spectral'] * l_spec +
             config['lambda_edge'] * l_edge)

    return {'total': total, 'vari': l_vari.item(), 'spectral': l_spec.item(), 'edge': l_edge.item()}


# ═══════════════════════════════════════════════════════════════
# METRICS — identical to run_gan_training.py
# ═══════════════════════════════════════════════════════════════

def compute_psnr(pred, target):
    p = torch.clamp((pred + 1) / 2, 0, 1)
    t = torch.clamp((target + 1) / 2, 0, 1)
    mse = ((p - t) ** 2).mean().item()
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0


def compute_ssim(pred, target):
    p = torch.clamp((pred + 1) / 2, 0, 1)
    t = torch.clamp((target + 1) / 2, 0, 1)
    mu_p = p.mean(); mu_t = t.mean()
    sp = ((p - mu_p) ** 2).mean().sqrt()
    st = ((t - mu_t) ** 2).mean().sqrt()
    spt = ((p - mu_p) * (t - mu_t)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_p * mu_t + C1) * (2 * spt + C2)) /
            ((mu_p ** 2 + mu_t ** 2 + C1) * (sp ** 2 + st ** 2 + C2))).item()


def compute_vari_rmse(pred, target, mask):
    p = torch.clamp((pred + 1) / 2, 0, 1)
    t = torch.clamp((target + 1) / 2, 0, 1)
    Rp, Gp, Bp = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    Rt, Gt, Bt = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    dp = torch.clamp(Gp + Rp - Bp, min=0.1)
    dt = torch.clamp(Gt + Rt - Bt, min=0.1)
    vp = torch.clamp((Gp - Rp) / dp, -1, 1)
    vt = torch.clamp((Gt - Rt) / dt, -1, 1)
    mse = ((vp - vt) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    return mse.sqrt().item()


# ═══════════════════════════════════════════════════════════════
# FINE-TUNE CONFIG — same architecture/loss weights as original,
# only data source, learning rate, and epoch count differ
# ═══════════════════════════════════════════════════════════════

BASE_CKPT = "outputs/checkpoints/gan/gan_ep030_psnr23.87.pt"  # matched G+D pair, best available
FT_CKPT_DIR = "outputs/checkpoints/gan_liss4_finetune"
FT_RES_DIR = "outputs/results/gan_liss4_finetune"

config = {
    'in_channels': 5, 'features_g': 64,
    'in_channels_d': 6, 'features_d': 64,
    'learning_rate_g': 0.00002,   # 10x lower than original 0.0002 -- small nudge, not a rewrite
    'learning_rate_d': 0.00002,
    'beta1': 0.5, 'beta2': 0.999,
    'max_epochs': 8,               # small dataset (864) + low LR -- more risks overfitting
    'batch_size': 2,
    'early_stopping_patience': 4,
    'save_every_epochs': 2,
    'lambda_l1': 100.0,
    'lambda_vari': 0.5,
    'lambda_spectral': 1.0,
    'lambda_edge': 5.0,
    'patches_dir': "data/processed/patches_liss4",
}


def train():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(FT_CKPT_DIR, exist_ok=True)
    os.makedirs(FT_RES_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = GANDataset(config['patches_dir'], 'train', augment=True)
    val_ds = GANDataset(config['patches_dir'], 'val', augment=False)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                               num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    G = Generator(in_channels=config['in_channels'], features=config['features_g']).to(DEVICE)
    D = Discriminator(in_channels=config['in_channels_d'], features=config['features_d']).to(DEVICE)

    # ── Load base checkpoint (matched G+D pair) ──────────────────
    existing_ft_ckpts = sorted(glob.glob(f'{FT_CKPT_DIR}/liss4_ep*.pt'))
    if existing_ft_ckpts:
        # resume our own fine-tuning if it was already started
        latest = existing_ft_ckpts[-1]
        print(f"Resuming fine-tune from: {os.path.basename(latest)}")
        ckpt = torch.load(latest, map_location=DEVICE, weights_only=False)
        G.load_state_dict(ckpt['G_state'])
        D.load_state_dict(ckpt['D_state'])
        start_epoch = ckpt['epoch']
        best_psnr = ckpt.get('best_psnr', 0.0)
    else:
        print(f"Loading base checkpoint: {BASE_CKPT}")
        ckpt = torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False)
        G.load_state_dict(ckpt['G_state'])
        D.load_state_dict(ckpt['D_state'])
        start_epoch = 0
        best_psnr = 0.0
        print(f"Base checkpoint val PSNR was: {ckpt.get('val_psnr', 'unknown')}")

    # Fresh optimizers at the new low LR -- NOT resuming old optimizer momentum,
    # since it was tuned for the original (10x higher) learning rate
    opt_G = torch.optim.Adam(G.parameters(), lr=config['learning_rate_g'],
                              betas=(config['beta1'], config['beta2']))
    opt_D = torch.optim.Adam(D.parameters(), lr=config['learning_rate_d'],
                              betas=(config['beta1'], config['beta2']))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    # Mixed precision: runs forward/backward in 16-bit where safe, roughly
    # halving activation memory -- needed since 4GB VRAM is already near its
    # ceiling running two networks (G+D) simultaneously.
    use_amp = (DEVICE.type == 'cuda')
    scaler_G = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_D = torch.cuda.amp.GradScaler(enabled=use_amp)

    no_improve = 0
    history = {'G_loss': [], 'D_loss': [], 'psnr': [], 'ssim': [], 'vari': []}

    print(f"\n{'='*65}\nLISS-IV Fine-Tuning\n{'='*65}\n")

    for epoch in range(start_epoch, config['max_epochs']):
        G.train(); D.train()
        sum_G, sum_D, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            gen_input = batch['gen_input'].to(DEVICE)
            cloudy = batch['cloudy'].to(DEVICE)
            cloudfree = batch['cloudfree'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                fake_img_nograd = G(gen_input)

            with torch.cuda.amp.autocast(enabled=use_amp):
                real_score = D(cloudy, cloudfree)
                real_label = torch.ones_like(real_score) * 0.9
                loss_D_real = criterion_GAN(real_score, real_label)

                fake_score = D(cloudy, fake_img_nograd)
                fake_label = torch.zeros_like(fake_score)
                loss_D_fake = criterion_GAN(fake_score, fake_label)
                loss_D = (loss_D_real + loss_D_fake) * 0.5

            opt_D.zero_grad(set_to_none=True)
            scaler_D.scale(loss_D).backward()
            scaler_D.unscale_(opt_D)
            nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            scaler_D.step(opt_D)
            scaler_D.update()

            del fake_img_nograd, real_score, fake_score  # free ASAP, don't wait for GC
            if use_amp:
                torch.cuda.empty_cache()

            with torch.cuda.amp.autocast(enabled=use_amp):
                fake_img = G(gen_input)
                fake_score = D(cloudy, fake_img)
                loss_G_adv = criterion_GAN(fake_score, torch.ones_like(fake_score))
                loss_G_L1 = criterion_L1(fake_img, cloudfree) * config['lambda_l1']
                phys = physics_loss(fake_img, cloudfree, mask, config)
                loss_G = loss_G_adv + loss_G_L1 + phys['total']

            opt_G.zero_grad(set_to_none=True)
            scaler_G.scale(loss_G).backward()
            scaler_G.unscale_(opt_G)
            nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler_G.step(opt_G)
            scaler_G.update()

            sum_G += loss_G.item(); sum_D += loss_D.item(); n_batches += 1
            del fake_img, fake_score
            if use_amp and n_batches % 20 == 0:
                torch.cuda.empty_cache()

        avg_G, avg_D = sum_G / n_batches, sum_D / n_batches

        G.eval()
        val_psnr, val_ssim, val_vari, n_val = 0.0, 0.0, 0.0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for batch in val_loader:
                gi = batch['gen_input'].to(DEVICE)
                cf = batch['cloudfree'].to(DEVICE)
                mk = batch['mask'].to(DEVICE)
                fk = G(gi)
                for i in range(fk.shape[0]):
                    val_psnr += compute_psnr(fk[i:i+1], cf[i:i+1])
                    val_ssim += compute_ssim(fk[i:i+1], cf[i:i+1])
                    val_vari += compute_vari_rmse(fk[i:i+1], cf[i:i+1], mk[i:i+1])
                    n_val += 1
        val_psnr /= n_val; val_ssim /= n_val; val_vari /= n_val

        print(f"Epoch {epoch+1:02d}/{config['max_epochs']} | G:{avg_G:.4f} D:{avg_D:.4f} | "
              f"PSNR:{val_psnr:.2f}dB SSIM:{val_ssim:.4f} VARI-RMSE:{val_vari:.4f}")

        if avg_D > 0.85:
            print(f"  WARNING: D_loss={avg_D:.3f} — discriminator too strong.")
        if avg_D < 0.30:
            print(f"  WARNING: D_loss={avg_D:.3f} — discriminator collapsed.")

        history['G_loss'].append(avg_G); history['D_loss'].append(avg_D)
        history['psnr'].append(val_psnr); history['ssim'].append(val_ssim); history['vari'].append(val_vari)

        if (epoch + 1) % config['save_every_epochs'] == 0:
            ckpt_path = f"{FT_CKPT_DIR}/liss4_ep{epoch+1:02d}_psnr{val_psnr:.2f}.pt"
            torch.save({'epoch': epoch + 1, 'G_state': G.state_dict(), 'D_state': D.state_dict(),
                        'val_psnr': val_psnr, 'best_psnr': best_psnr, 'history': history}, ckpt_path)
            print(f"  Checkpoint saved: {os.path.basename(ckpt_path)}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr; no_improve = 0
            torch.save(G.state_dict(), f"{FT_CKPT_DIR}/best_generator_liss4.pt")
            print(f"  New best generator saved (PSNR: {best_psnr:.2f} dB)")
        else:
            no_improve += 1
            if no_improve >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*65}\nFine-Tuning Complete\nBest PSNR on real LISS-IV val data: {best_psnr:.2f} dB")
    print(f"Compare against original RICE2-only test PSNR of 24.74 dB\n{'='*65}")

    if len(history['psnr']) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(history['G_loss'], label='G'); axes[0].plot(history['D_loss'], label='D')
        axes[0].set_title('Losses'); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(history['psnr'], color='green'); axes[1].set_title('Val PSNR (dB)'); axes[1].grid(alpha=0.3)
        axes[2].plot(history['vari'], color='teal'); axes[2].set_title('Val VARI-RMSE'); axes[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FT_RES_DIR}/finetune_curves.png", dpi=120, bbox_inches='tight')
        print(f"Curves saved: {FT_RES_DIR}/finetune_curves.png")


if __name__ == '__main__':
    train()
