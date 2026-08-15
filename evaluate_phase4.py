"""
evaluate_phase4.py
Complete evaluation framework for the Phase 3 (physics-informed cGAN) results.

What it does:
  - Loads the trained best_generator.pt (CPU-safe; no CUDA required).
  - Runs inference over the FULL test set (536 patches).
  - Reports PSNR, SSIM, VARI-RMSE overall AND stratified by cloud coverage:
        light  : cloud_coverage < 0.30
        medium : 0.30 <= cloud_coverage < 0.60
        heavy  : cloud_coverage >= 0.60
  - Prints a comparison table (our model vs. literature baselines).
  - Saves raw metric arrays (.npy) and (if matplotlib is available) a
    distribution plot.

SELF-CONTAINED: GANDataset + metric functions are copied locally so there is
NO circular import from run_gan_training.py. Normalisation is done manually
(torch + cv2 + numpy only) to avoid the albumentations/skimage NumPy-2.x crash.

Run:
    python evaluate_phase4.py
"""
import os, sys, json, cv2, torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, "D:\\CloudRemoval_Project")

from src.models.generator import Generator


def _norm(img):
    """Replicate A.Normalize(mean=0.5,std=0.5,max_pixel=255): (x/255-0.5)/0.5."""
    return img.astype(np.float32) / 127.5 - 1.0


def _to_tensor(arr):
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()  # HWC -> CHW


class GANDataset(Dataset):
    def __init__(self, patches_dir, split):
        self.split_dir = Path(patches_dir) / split
        with open(self.split_dir / "patch_manifest.json") as f:
            self.manifest = json.load(f)
        self.patch_ids = sorted(self.manifest.keys())

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

        cloudy_t = _to_tensor(_norm(cloudy))
        cloudfree_t = _to_tensor(_norm(cloudfree))
        mask_t = torch.from_numpy(mask_bin).unsqueeze(0).float()
        edge_t = torch.from_numpy(self._edge(mask_bin)).unsqueeze(0)
        gen_input = torch.cat([cloudy_t, mask_t, edge_t], dim=0)
        return {"gen_input": gen_input, "cloudfree": cloudfree_t, "mask": mask_t}


def compute_psnr(pred, tgt):
    p = torch.clamp((pred + 1) / 2, 0, 1)
    t = torch.clamp((tgt + 1) / 2, 0, 1)
    mse = ((p - t) ** 2).mean().item()
    return 100.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def compute_ssim(pred, tgt):
    p = torch.clamp((pred + 1) / 2, 0, 1)
    t = torch.clamp((tgt + 1) / 2, 0, 1)
    mu_p, mu_t = p.mean(), t.mean()
    sp = ((p - mu_p) ** 2).mean().sqrt()
    st = ((t - mu_t) ** 2).mean().sqrt()
    spt = ((p - mu_p) * (t - mu_t)).mean()
    C1, C2 = 0.01**2, 0.03**2
    return (((2 * mu_p * mu_t + C1) * (2 * spt + C2)) /
            ((mu_p**2 + mu_t**2 + C1) * (sp**2 + st**2 + C2))).item()


def compute_vari_rmse(pred, tgt, mask):
    p = torch.clamp((pred + 1) / 2, 0, 1)
    t = torch.clamp((tgt + 1) / 2, 0, 1)
    Rp, Gp, Bp = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    Rt, Gt, Bt = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    dp = torch.clamp(Gp + Rp - Bp, min=0.1)
    dt = torch.clamp(Gt + Rt - Bt, min=0.1)
    vp = torch.clamp((Gp - Rp) / dp, -1.0, 1.0)
    vt = torch.clamp((Gt - Rt) / dt, -1.0, 1.0)
    ms = mask.sum()
    if ms < 1e-5:
        return 0.0
    mse = ((vp - vt) ** 2 * mask).sum() / ms
    return torch.sqrt(torch.clamp(mse, min=0.0, max=1.0)).item()


def main():
    with open("configs/gan_config.yaml") as f:
        config = yaml.safe_load(f)["gan"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    G = Generator(in_channels=config["in_channels"], features=config["features_g"]).to(DEVICE)
    G.load_state_dict(torch.load(
        "outputs/checkpoints/gan/best_generator.pt",
        map_location=DEVICE, weights_only=False))
    G.eval()
    print("Generator loaded from outputs/checkpoints/gan/best_generator.pt")

    test_ds = GANDataset(config["patches_dir"], "test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test patches: {len(test_ds)}")

    with open(f"{config['patches_dir']}/test/patch_manifest.json") as f:
        manifest = json.load(f)

    results = {
        "all":    {"psnr": [], "ssim": [], "vari": []},
        "light":  {"psnr": [], "ssim": [], "vari": []},
        "medium": {"psnr": [], "ssim": [], "vari": []},
        "heavy":  {"psnr": [], "ssim": [], "vari": []},
    }

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            gi = batch["gen_input"].to(DEVICE)
            cf = batch["cloudfree"].to(DEVICE)
            mk = batch["mask"].to(DEVICE)
            fake = G(gi)

            p = compute_psnr(fake, cf)
            s = compute_ssim(fake, cf)
            v = compute_vari_rmse(fake, cf, mk)

            results["all"]["psnr"].append(p)
            results["all"]["ssim"].append(s)
            results["all"]["vari"].append(v)

            pid = test_ds.patch_ids[idx]
            coverage = manifest[pid]["cloud_coverage"]
            if coverage < 0.30:
                key = "light"
            elif coverage < 0.60:
                key = "medium"
            else:
                key = "heavy"
            results[key]["psnr"].append(p)
            results[key]["ssim"].append(s)
            results[key]["vari"].append(v)

            if idx % 100 == 0:
                print(f"  Evaluated {idx}/{len(test_ds)} patches...")

    print(f"\n{'='*65}")
    print("PHASE 4 EVALUATION - STRATIFIED BY CLOUD COVERAGE")
    print(f"{'='*65}")
    print(f"{'Category':<14} {'N':>5} {'PSNR':>10} {'SSIM':>9} {'VARI-RMSE':>11}")
    print("-" * 55)
    labels = [("all", "ALL"), ("light", "Light<30%"),
              ("medium", "Med 30-60%"), ("heavy", "Heavy>60%")]
    for key, label in labels:
        r = results[key]
        if len(r["psnr"]) == 0:
            continue
        print(f"{label:<14} {len(r['psnr']):>5} "
              f"{np.mean(r['psnr']):>9.2f}dB "
              f"{np.mean(r['ssim']):>9.4f} "
              f"{np.mean(r['vari']):>11.4f}")
    print("=" * 65)

    our_psnr = np.mean(results["all"]["psnr"])
    our_ssim = np.mean(results["all"]["ssim"])
    our_vari = np.mean(results["all"]["vari"])
    print(f"\n{'='*65}")
    print("COMPARISON: Yours (physics-informed cGAN) vs. Literature")
    print(f"{'='*65}")
    print(f"{'Model':<30} {'PSNR':>10} {'SSIM':>9} {'VARI-RMSE':>11}")
    print("-" * 62)
    print(f"{'Pix2Pix (literature ref)':<30} {'~24.50 dB':>10} {'~0.720':>9} {'N/A':>11}")
    print(f"{'SpAGAN (literature ref)':<30} {'~25.80 dB':>10} {'~0.790':>9} {'N/A':>11}")
    print(f"{'Yours (physics-informed)':<30} {our_psnr:>9.2f}dB {our_ssim:>9.4f} {our_vari:>11.4f}")
    print("=" * 65)
    print("Note: baseline rows are reported literature values, NOT re-measured here.")
    print("      VARI-RMSE is your unique spectral-correctness metric (lower = better).")

    os.makedirs("outputs/results/gan", exist_ok=True)
    np.save("outputs/results/gan/test_psnr.npy", np.array(results["all"]["psnr"]))
    np.save("outputs/results/gan/test_ssim.npy", np.array(results["all"]["ssim"]))
    np.save("outputs/results/gan/test_vari.npy", np.array(results["all"]["vari"]))
    print("\nSaved metric arrays to outputs/results/gan/test_*.npy")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Phase 4 - Test Set Metric Distributions",
                     fontsize=13, fontweight="bold")
        for ax, data, title, xlabel, color in [
            (axes[0], results["all"]["psnr"], "PSNR", "dB", "steelblue"),
            (axes[1], results["all"]["ssim"], "SSIM", "", "seagreen"),
            (axes[2], results["all"]["vari"], "VARI-RMSE", "", "purple"),
        ]:
            ax.hist(data, bins=30, color=color, edgecolor="white")
            ax.axvline(np.mean(data), color="red", linestyle="--",
                       label=f"Mean: {np.mean(data):.3f}")
            ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("outputs/results/gan/phase4_metrics.png", dpi=120,
                    bbox_inches="tight")
        plt.close()
        print("Saved plot: outputs/results/gan/phase4_metrics.png")
    except Exception as e:
        print(f"[plot skipped] matplotlib unavailable: {e}")
        print("Metrics above are still complete and saved as .npy.")

    print("\nPhase 4 evaluation complete.")


if __name__ == "__main__":
    main()
