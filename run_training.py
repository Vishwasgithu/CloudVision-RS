import os
import sys
import yaml
import glob
import torch
import shutil

# 💡 FIX 1: Overrides the SSL error you had on the first attempt
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def compute_iou(logits, masks):
    preds = (torch.sigmoid(logits) > 0.5).float()
    inter = (preds * masks).sum()
    union = (preds + masks).clamp(0, 1).sum()
    return (inter / (union + 1e-8)).item()


# 💡 FIX 2: The Critical Windows Multiprocessing Shield
if __name__ == "__main__":
    DEVICE = torch.device("cuda")
    CKPT_DIR = "outputs/checkpoints/segmentation"
    os.makedirs(CKPT_DIR, exist_ok=True)

    from src.data.dataset import create_dataloaders
    from src.models.segmentation import AttentionUNet
    from src.losses.seg_loss import SegmentationLoss

    with open("configs/seg_config.yaml") as f:
        config = yaml.safe_load(f)["segmentation"]

    config["patches_dir"] = "data/processed/patches"
    config["seg_batch_size"] = 4  # 4GB VRAM limit for RTX 3050
    config["num_workers"] = 2  # Now safe to use background workers!

    loaders = create_dataloaders(
        patches_dir=config["patches_dir"],
        mode="segmentation",
        config_path="configs/data_config.yaml",
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches:   {len(loaders['val'])}")

    model = AttentionUNet(config).to(DEVICE)
    criterion = SegmentationLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=60, eta_min=1e-6
    )

    best_iou, no_improve = 0.0, 0
    PATIENCE, MAX_EPOCHS = 12, 60

    print("\nStarting training...")
    print("Each epoch ~5-7 min on RTX 3050")
    print("Let it run overnight\n")

    for epoch in range(MAX_EPOCHS):
        model.train()
        t_loss, t_iou, n = 0, 0, 0
        for batch in loaders["train"]:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            t_iou += compute_iou(logits, masks)
            n += 1

        model.eval()
        v_loss, v_iou, m = 0, 0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                logits = model(images)
                v_loss += criterion(logits, masks)["loss"].item()
                v_iou += compute_iou(logits, masks)
                m += 1

        scheduler.step()
        tl, ti = t_loss / n, t_iou / n
        vl, vi = v_loss / m, v_iou / m

        print(
            f"Epoch {epoch+1:02d}/{MAX_EPOCHS} | "
            f"train_loss:{tl:.4f} train_iou:{ti:.4f} | "
            f"val_loss:{vl:.4f} val_iou:{vi:.4f}"
        )

        if vi > best_iou:
            best_iou = vi
            no_improve = 0
            for old in glob.glob(f"{CKPT_DIR}/best_*.pt"):
                os.remove(old)
            ckpt = f"{CKPT_DIR}/best_iou{best_iou:.4f}_ep{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "val_iou": best_iou,
                    "config": config,
                },
                ckpt,
            )
            print(f"  ✓ Best: {best_iou:.4f} saved")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nDone. Best val/IoU: {best_iou:.4f}")
    print(f"Checkpoint at: {CKPT_DIR}/")
