import os, sys, glob, torch, yaml
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, "D:\\CloudRemoval_Project")

from src.data.dataset import CloudSegmentationDataset
from src.models.segmentation import AttentionUNet
from torch.utils.data import DataLoader

# ── Load checkpoint ──────────────────────────────────
CKPT_DIR = "outputs/checkpoints/segmentation"
ckpts = glob.glob(f"{CKPT_DIR}/best_*.pt")

if not ckpts:
    print(f"No checkpoint found in {CKPT_DIR}")
    sys.exit(1)

best = max(ckpts)
print(f"Loading: {os.path.basename(best)}")
checkpoint = torch.load(best, map_location="cpu")
print(f"Epoch: {checkpoint['epoch']}  |  val/IoU: {checkpoint['val_iou']:.4f}")

# ── Load model ───────────────────────────────────────
with open("configs/seg_config.yaml") as f:
    config = yaml.safe_load(f)["segmentation"]

model = AttentionUNet(config)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print("Model loaded")

# ── Load test data (num_workers=0 required on Windows) ──
test_dataset = CloudSegmentationDataset(
    patches_dir="data/processed/patches",
    split="test",
    config_path="configs/data_config.yaml",
)
test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,  # Windows fix — no multiprocessing
)
print(f"Test patches: {len(test_dataset)}")

# ── Run inference ────────────────────────────────────
samples = []
with torch.no_grad():
    for batch in test_loader:
        images = batch["image"]
        masks = batch["mask"]
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        for i in range(images.shape[0]):
            inter = (preds[i] * masks[i]).sum()
            union = (preds[i] + masks[i]).clamp(0, 1).sum()
            iou = (inter / (union + 1e-8)).item()
            samples.append(
                {
                    "image": images[i].permute(1, 2, 0).numpy(),
                    "mask": masks[i, 0].numpy(),
                    "pred": preds[i, 0].numpy(),
                    "prob": probs[i, 0].numpy(),
                    "iou": iou,
                }
            )

        if len(samples) >= 6:
            break

ious = [f"{s['iou']:.3f}" for s in samples]
print(f"Sample IoUs: {ious}")

# ── Plot ─────────────────────────────────────────────
fig, axes = plt.subplots(6, 4, figsize=(16, 24))
fig.suptitle(
    f"Phase 2 — Attention U-Net Cloud Segmentation\n"
    f"val/IoU: {checkpoint['val_iou']:.4f}  |  "
    f"ResNet34 + SCSE  |  Epoch: {checkpoint['epoch']}",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)

for ax, title in zip(
    axes[0], ["Cloudy Input", "Ground Truth Mask", "Predicted Mask", "Cloud Overlay"]
):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

for idx, s in enumerate(samples[:6]):
    img = np.clip(s["image"], 0, 1)

    axes[idx, 0].imshow(img)
    axes[idx, 0].set_ylabel(
        f"Sample {idx+1}\nIoU: {s['iou']:.3f}",
        fontsize=9,
        rotation=0,
        labelpad=55,
        va="center",
    )
    axes[idx, 0].axis("off")

    axes[idx, 1].imshow(s["mask"], cmap="Blues", vmin=0, vmax=1)
    axes[idx, 1].axis("off")

    axes[idx, 2].imshow(s["prob"], cmap="Reds", vmin=0, vmax=1)
    axes[idx, 2].axis("off")

    overlay = img.copy()
    cloud_px = s["pred"] > 0.5
    overlay[cloud_px, 0] = 1.0
    overlay[cloud_px, 1] *= 0.2
    overlay[cloud_px, 2] *= 0.2
    axes[idx, 3].imshow(np.clip(overlay, 0, 1))
    axes[idx, 3].axis("off")

plt.tight_layout()
os.makedirs("outputs/results", exist_ok=True)
save_path = "outputs/results/phase2_results.png"
plt.savefig(save_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"\nSaved: {save_path}")
print("Done")
