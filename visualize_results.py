# Save as visualize_segmentation.py in D:\CloudRemoval_Project\

import os, glob, torch, yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, "D:\\CloudRemoval_Project")

from src.data.dataset import create_dataloaders
from src.models.segmentation import AttentionUNet

# ── Load checkpoint ──────────────────────────────────────
CKPT_DIR = "outputs/checkpoints/segmentation"
ckpts = glob.glob(f"{CKPT_DIR}/best_*.pt")
best = max(ckpts)
print(f"Loading: {os.path.basename(best)}")

checkpoint = torch.load(best, map_location="cpu")
print(f"Epoch: {checkpoint['epoch']}  |  val/IoU: {checkpoint['val_iou']:.4f}")

# ── Load model ───────────────────────────────────────────
with open("configs/seg_config.yaml") as f:
    config = yaml.safe_load(f)["segmentation"]

model = AttentionUNet(config)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print("Model loaded")

# ── Load test data ───────────────────────────────────────
loaders = create_dataloaders(
    patches_dir="data/processed/patches",
    mode="segmentation",
    config_path="configs/data_config.yaml",
)

# ── Visualize 6 test samples ─────────────────────────────
samples = []
with torch.no_grad():
    for batch in loaders["test"]:
        images = batch["image"]
        masks = batch["mask"]
        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).float()

        for i in range(min(2, images.shape[0])):
            samples.append(
                {
                    "image": images[i].permute(1, 2, 0).numpy(),
                    "mask": masks[i, 0].numpy(),
                    "pred": preds[i, 0].numpy(),
                    "iou": (
                        (preds[i] * masks[i]).sum()
                        / ((preds[i] + masks[i]).clamp(0, 1).sum() + 1e-8)
                    ).item(),
                }
            )
        if len(samples) >= 6:
            break

# ── Plot ─────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 12))
fig.suptitle(
    f"Phase 2 Results — Attention U-Net Cloud Segmentation\n"
    f'Best val/IoU: {checkpoint["val_iou"]:.4f} | Epoch: {checkpoint["epoch"]}',
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

for idx, s in enumerate(samples):
    row = idx // 2
    col = (idx % 2) * 4

    # Input image
    ax = fig.add_subplot(3, 8, row * 8 + col + 1)
    ax.imshow(np.clip(s["image"], 0, 1))
    ax.set_title("Cloudy Input", fontsize=8)
    ax.axis("off")

    # Ground truth mask
    ax = fig.add_subplot(3, 8, row * 8 + col + 2)
    ax.imshow(s["mask"], cmap="Blues", vmin=0, vmax=1)
    ax.set_title("True Mask", fontsize=8)
    ax.axis("off")

    # Predicted mask
    ax = fig.add_subplot(3, 8, row * 8 + col + 3)
    ax.imshow(s["pred"], cmap="Reds", vmin=0, vmax=1)
    ax.set_title(f'Predicted\nIoU:{s["iou"]:.3f}', fontsize=8)
    ax.axis("off")

    # Overlay: image + predicted mask
    ax = fig.add_subplot(3, 8, row * 8 + col + 4)
    overlay = np.clip(s["image"].copy(), 0, 1)
    overlay[s["pred"] > 0.5, 0] = 1.0  # red channel where cloud predicted
    overlay[s["pred"] > 0.5, 1] *= 0.3
    overlay[s["pred"] > 0.5, 2] *= 0.3
    ax.imshow(overlay)
    ax.set_title("Cloud Overlay", fontsize=8)
    ax.axis("off")

plt.tight_layout()
os.makedirs("outputs/results", exist_ok=True)
plt.savefig(
    "outputs/results/phase2_segmentation_results.png", dpi=150, bbox_inches="tight"
)
plt.show()
print("Saved: outputs/results/phase2_segmentation_results.png")
