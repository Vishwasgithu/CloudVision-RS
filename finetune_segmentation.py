"""
Run from D:\\CloudRemoval_Project:  conda activate cloudremoval
python finetune_segmentation.py

Loads your existing trained checkpoint (best_iou0.7727_ep22.pt) as the starting
point, then fine-tunes on real LISS-IV patches (patches_liss4/) at a low
learning rate for a small number of epochs -- adapting the model's understanding
of "what a cloud looks like" from RICE2's RGB statistics to your real Red-Green-NIR
band substitution, without forgetting what it already learned.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(r"D:\CloudRemoval_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CloudSegmentationDataset
from src.models.segmentation import AttentionUNet

# ---- CONFIG ----
CHECKPOINT_IN = PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation" / "best_iou0.7727_ep22.pt"
CHECKPOINT_OUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation_liss4_finetune"
PATCHES_DIR = PROJECT_ROOT / "data" / "processed" / "patches_liss4"
CONFIG_PATH = "configs/data_config.yaml"  # reused as-is, resolved relative to project root

MODEL_CONFIG = dict(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=3, num_classes=1)

FINETUNE_LR = 1e-5   # ~10-20x lower than typical from-scratch LR -- small nudge, not a rewrite
EPOCHS = 8           # small dataset (864 patches) + low LR -- more epochs risks overfitting
BATCH_SIZE = 2       # matches your original hardware-constrained batch size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint_flexible(model, path):
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {path}")


def iou_score(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    return (intersection / union.clamp(min=1e-6)).mean().item()


def main():
    print(f"Device: {DEVICE}")
    CHECKPOINT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = AttentionUNet(MODEL_CONFIG).to(DEVICE)
    load_checkpoint_flexible(model, CHECKPOINT_IN)

    train_ds = CloudSegmentationDataset(str(PATCHES_DIR), "train", CONFIG_PATH)
    val_ds = CloudSegmentationDataset(str(PATCHES_DIR), "val", CONFIG_PATH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, drop_last=True)  # num_workers=0: Windows constraint
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train patches: {len(train_ds)}, Val patches: {len(val_ds)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
    criterion = nn.BCEWithLogitsLoss()

    best_iou = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_ious = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                logits = model(images)
                val_ious.append(iou_score(logits, masks))

        avg_train_loss = train_loss / len(train_loader)
        avg_val_iou = sum(val_ious) / len(val_ious)
        print(f"Epoch {epoch}/{EPOCHS} — train_loss: {avg_train_loss:.4f}, val_IoU: {avg_val_iou:.4f}")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            out_path = CHECKPOINT_OUT_DIR / f"best_iou{avg_val_iou:.4f}_liss4_ep{epoch}.pt"
            torch.save(model.state_dict(), out_path)
            print(f"  New best -> saved {out_path}")

    print(f"\nFine-tuning done. Best val IoU on real LISS-IV data: {best_iou:.4f}")
    print(f"Compare against original RICE2-only IoU of 0.7727 to see if adaptation helped.")


if __name__ == "__main__":
    main()
