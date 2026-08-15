"""
finetune_gan_generator_only.py — continues from liss4_ep08, Generator-only
(L1 + physics losses, no adversarial term -- avoids the D-collapse shortcut).

Run: conda activate cloudremoval
python finetune_gan_generator_only.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, glob, json, cv2, torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "D:\\CloudRemoval_Project")
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.generator import Generator


# ── Dataset (same as finetune_gan.py) ─────────────────────────
class GANDataset(Dataset):
    def __init__(self, patches_dir, split, augment=True):
        self.split_dir = Path(patches_dir) / split
        with open(self.split_dir / "patch_manifest.json") as f:
            self.manifest = json.load(f)
        self.patch_ids = sorted(self.manifest.keys())
        spatial = (
            [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
            if augment
            else []
        )
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

    def _edge(self, mask_bin):
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
        edge = self._edge(out["mask"].numpy().astype(np.uint8))
        gen_input = torch.cat(
            [
                out["image"],
                out["mask"].unsqueeze(0).float(),
                torch.from_numpy(edge).unsqueeze(0),
            ],
            dim=0,
        )
        return {
            "gen_input": gen_input,
            "cloudfree": out["cloudfree"],
            "mask": out["mask"].unsqueeze(0).float(),
        }


# ── Physics loss (identical to run_gan_training.py) ───────────
def physics_loss(fake_img, real_img, mask, cfg):
    pred = torch.clamp((fake_img + 1) / 2, 0, 1)
    tgt = torch.clamp((real_img + 1) / 2, 0, 1)
    Rp, Gp, Bp = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    Rt, Gt, Bt = tgt[:, 0:1], tgt[:, 1:2], tgt[:, 2:3]
    dp = torch.clamp(Gp + Rp - Bp, min=0.1)
    dt = torch.clamp(Gt + Rt - Bt, min=0.1)
    vp = torch.clamp((Gp - Rp) / dp, -1, 1)
    vt = torch.clamp((Gt - Rt) / dt, -1, 1)
    l_vari = torch.clamp(((vp - vt) ** 2 * mask).sum() / (mask.sum() + 1e-8), max=5.0)
    eps = 0.1
    l_spec = torch.tensor(0.0, device=fake_img.device)
    for rp, rt in [
        (Rp / (Gp + eps), Rt / (Gt + eps)),
        (Bp / (Gp + eps), Bt / (Gt + eps)),
        (Rp / (Bp + eps), Rt / (Bt + eps)),
    ]:
        l_spec += (torch.clamp((rp - rt) ** 2, max=5.0) * mask).sum() / (
            mask.sum() + 1e-8
        )
    l_spec /= 3.0

    def gm(img):
        dx = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs()
        dy = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs()
        return dx.mean() + dy.mean()

    l_edge = (gm(fake_img) - gm(real_img)).abs()
    total = (
        cfg["lambda_vari"] * l_vari
        + cfg["lambda_spectral"] * l_spec
        + cfg["lambda_edge"] * l_edge
    )
    return {
        "total": total,
        "vari": l_vari.item(),
        "spectral": l_spec.item(),
        "edge": l_edge.item(),
    }


def compute_psnr(pred, target):
    p = torch.clamp((pred + 1) / 2, 0, 1)
    t = torch.clamp((target + 1) / 2, 0, 1)
    mse = ((p - t) ** 2).mean().item()
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0


def compute_ssim(pred, target):
    p = torch.clamp((pred + 1) / 2, 0, 1)
    t = torch.clamp((target + 1) / 2, 0, 1)
    mu_p = p.mean()
    mu_t = t.mean()
    sp = ((p - mu_p) ** 2).mean().sqrt()
    st = ((t - mu_t) ** 2).mean().sqrt()
    spt = ((p - mu_p) * (t - mu_t)).mean()
    C1, C2 = 0.01**2, 0.03**2
    return (
        ((2 * mu_p * mu_t + C1) * (2 * spt + C2))
        / ((mu_p**2 + mu_t**2 + C1) * (sp**2 + st**2 + C2))
    ).item()


def compute_vari_rmse(pred, target, mask):
    p = torch.clamp((pred + 1) / 2, 0, 1)
    t = torch.clamp((target + 1) / 2, 0, 1)
    Rp, Gp, Bp = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    Rt, Gt, Bt = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    dp = torch.clamp(Gp + Rp - Bp, min=0.1)
    dt = torch.clamp(Gt + Rt - Bt, min=0.1)
    vp = torch.clamp((Gp - Rp) / dp, -1, 1)
    vt = torch.clamp((Gt - Rt) / dt, -1, 1)
    return (((vp - vt) ** 2 * mask).sum() / (mask.sum() + 1e-8)).sqrt().item()


# ── Config ──────────────────────────────────────────────────
BASE_CKPT = "outputs/checkpoints/gan_liss4_finetune/liss4_ep08_psnr18.63.pt"
CKPT_DIR = "outputs/checkpoints/gan_liss4_g_only"
RES_DIR = "outputs/results/gan_liss4_g_only"
cfg = {
    "in_channels": 5,
    "features_g": 64,
    "learning_rate_g": 0.00002,
    "max_epochs": 35,
    "batch_size": 2,
    "patience": 10,
    "save_every": 2,
    "lambda_l1": 100.0,
    "lambda_vari": 0.5,
    "lambda_spectral": 1.0,
    "lambda_edge": 5.0,
    "patches_dir": "data/processed/patches_liss4",
}


def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    train_ds = GANDataset(cfg["patches_dir"], "train", augment=True)
    val_ds = GANDataset(cfg["patches_dir"], "val", augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
    )

    G = Generator(in_channels=cfg["in_channels"], features=cfg["features_g"]).to(DEVICE)

    existing = sorted(glob.glob(f"{CKPT_DIR}/g_only_ep*.pt"))
    if existing:
        ckpt = torch.load(existing[-1], map_location=DEVICE, weights_only=False)
        G.load_state_dict(ckpt["G_state"])
        start_epoch = ckpt["epoch"]
        best_psnr = ckpt["best_psnr"]
        print(f"Resuming from {os.path.basename(existing[-1])}")
    else:
        ckpt = torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False)
        G.load_state_dict(ckpt["G_state"])
        start_epoch = 0
        best_psnr = 0.0
        print(f"Loaded base: {BASE_CKPT} (was PSNR {ckpt.get('val_psnr','?')})")

    opt_G = torch.optim.Adam(
        G.parameters(), lr=cfg["learning_rate_g"], betas=(0.5, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=cfg["max_epochs"], eta_min=1e-6
    )
    criterion_L1 = nn.L1Loss()
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    no_improve = 0
    history = {"loss": [], "psnr": [], "ssim": [], "vari": []}

    print(f"\n{'='*60}\nGenerator-Only Fine-Tuning (no adversarial term)\n{'='*60}\n")

    for epoch in range(start_epoch, cfg["max_epochs"]):
        G.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            gi = batch["gen_input"].to(DEVICE)
            cf = batch["cloudfree"].to(DEVICE)
            mk = batch["mask"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake = G(gi)
                loss_l1 = criterion_L1(fake, cf) * cfg["lambda_l1"]
                phys = physics_loss(fake, cf, mk, cfg)
                loss = loss_l1 + phys["total"]
            opt_G.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt_G)
            nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler.step(opt_G)
            scaler.update()
            total_loss += loss.item()
            n += 1

        scheduler.step()

        G.eval()
        val_psnr, val_ssim, val_vari, nv = 0.0, 0.0, 0.0, 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for batch in val_loader:
                gi = batch["gen_input"].to(DEVICE)
                cf = batch["cloudfree"].to(DEVICE)
                mk = batch["mask"].to(DEVICE)
                fk = G(gi)
                for i in range(fk.shape[0]):
                    val_psnr += compute_psnr(fk[i : i + 1], cf[i : i + 1])
                    val_ssim += compute_ssim(fk[i : i + 1], cf[i : i + 1])
                    val_vari += compute_vari_rmse(
                        fk[i : i + 1], cf[i : i + 1], mk[i : i + 1]
                    )
                    nv += 1
        val_psnr /= nv
        val_ssim /= nv
        val_vari /= nv
        avg_loss = total_loss / n
        current_lr = opt_G.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:02d}/{cfg['max_epochs']} | loss:{avg_loss:.4f} | LR:{current_lr:.2e} | PSNR:{val_psnr:.2f}dB SSIM:{val_ssim:.4f} VARI-RMSE:{val_vari:.4f}"
        )
        history["loss"].append(avg_loss)
        history["psnr"].append(val_psnr)
        history["ssim"].append(val_ssim)
        history["vari"].append(val_vari)

        if (epoch + 1) % cfg["save_every"] == 0:
            path = f"{CKPT_DIR}/g_only_ep{epoch+1:02d}_psnr{val_psnr:.2f}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "G_state": G.state_dict(),
                    "val_psnr": val_psnr,
                    "best_psnr": best_psnr,
                },
                path,
            )
            print(f"  Saved: {os.path.basename(path)}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            no_improve = 0
            torch.save(G.state_dict(), f"{CKPT_DIR}/best_generator_g_only.pt")
            print(f"  New best (PSNR: {best_psnr:.2f} dB)")
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nDone. Best PSNR: {best_psnr:.2f} dB")
    if history["psnr"]:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history["psnr"], color="green")
        ax[0].set_title("Val PSNR")
        ax[0].grid(alpha=0.3)
        ax[1].plot(history["vari"], color="teal")
        ax[1].set_title("Val VARI-RMSE")
        ax[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RES_DIR}/curves.png", dpi=120)
        print(f"Curves: {RES_DIR}/curves.png")


if __name__ == "__main__":
    train()
