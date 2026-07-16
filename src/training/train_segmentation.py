"""
train_segmentation.py

PyTorch Lightning module for segmentation training.

WHY PYTORCH LIGHTNING:
Raw training loops have common bugs:
- Forgetting optimizer.zero_grad() before backward()
- Forgetting model.eval() during validation (dropout behaves differently)
- Saving wrong checkpoint (last instead of best)
- Incorrect gradient accumulation
Lightning handles all of these correctly by design.

You write: what happens in one training step, one validation step,
and how to configure the optimizer.
Lightning handles: the loop, mixed precision, checkpointing,
early stopping, GPU transfer, and logging.

COLAB SESSION MANAGEMENT:
Free Colab disconnects after ~90 min idle. We save checkpoints to
Google Drive every epoch. If session disconnects, restart and the
trainer resumes from the last Drive checkpoint automatically.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
import yaml
import wandb

# Repo root = parent of src/ (this file is src/training/train_segmentation.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.models.segmentation import AttentionUNet
from src.losses.seg_loss import SegmentationLoss
from src.data.dataset import create_dataloaders


class SegmentationModule(pl.LightningModule):
    """
    Lightning module wrapping model + loss + optimizer + metrics.

    torchmetrics handles IoU and Dice correctly:
    - Accumulates predictions across the entire validation set
    - Computes final metric over all accumulated predictions
    This is more accurate than averaging per-batch metrics,
    especially when batches have varying cloud coverage.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config    = config
        self.model     = AttentionUNet(config)
        self.criterion = SegmentationLoss(
            bce_weight=config['bce_weight'],
            dice_weight=config['dice_weight']
        )

        # torchmetrics — accumulates across batches for accurate epoch metrics
        # task='binary' because we have 2 classes: cloud and no-cloud
        self.train_iou = torchmetrics.JaccardIndex(task='binary', threshold=0.5)
        self.val_iou   = torchmetrics.JaccardIndex(task='binary', threshold=0.5)
        self.val_dice  = torchmetrics.F1Score(task='binary', threshold=0.5)

        # Save hyperparameters for checkpoint loading
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        One training step = one batch.
        Lightning calls this automatically inside the training loop.

        batch['image']: [B, 3, 256, 256] float [0, 1]
        batch['mask']:  [B, 1, 256, 256] float {0, 1}
        """
        images = batch['image']   # [B, 3, H, W]
        masks  = batch['mask']    # [B, 1, H, W]

        logits = self(images)     # [B, 1, H, W] raw logits
        losses = self.criterion(logits, masks)

        # Log all loss components to wandb
        self.log('train/loss',      losses['loss'],      on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/bce_loss',  losses['bce_loss'],  on_step=False, on_epoch=True)
        self.log('train/dice_loss', losses['dice_loss'], on_step=False, on_epoch=True)

        # Update training IoU (uses probabilities, not logits)
        probs = torch.sigmoid(logits)
        self.train_iou(probs, masks.int())

        return losses['loss']

    def on_train_epoch_end(self):
        """Called at end of each training epoch."""
        iou = self.train_iou.compute()
        self.log('train/iou', iou, prog_bar=True)
        self.train_iou.reset()

    def validation_step(self, batch, batch_idx):
        """
        One validation step.
        Model is automatically in eval() mode during validation.
        Gradients are automatically disabled.
        Lightning handles both — you do not need model.eval() or torch.no_grad().
        """
        images = batch['image']
        masks  = batch['mask']

        logits = self(images)
        losses = self.criterion(logits, masks)

        probs = torch.sigmoid(logits)
        self.val_iou(probs, masks.int())
        self.val_dice(probs, masks.int())

        self.log('val/loss', losses['loss'], on_step=False, on_epoch=True, prog_bar=True)

        return losses['loss']

    def on_validation_epoch_end(self):
        """Called at end of each validation epoch. Compute and log final metrics."""
        iou  = self.val_iou.compute()
        dice = self.val_dice.compute()

        self.log('val/iou',  iou,  prog_bar=True)
        self.log('val/dice', dice, prog_bar=True)

        self.val_iou.reset()
        self.val_dice.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',        # maximising IoU
            factor=0.5,        # halve LR when plateau detected
            patience=5,        # wait 5 epochs before reducing
            min_lr=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/iou',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def train(config: dict, is_colab: bool = False):
    """
    Main training function.

    is_colab=True: saves checkpoints to Google Drive path
    is_colab=False: saves locally to outputs/checkpoints/segmentation
    """

    pl.seed_everything(42, workers=True)

    # ── DataLoaders ─────────────────────────────────────
    patches_dir = config['patches_dir']
    loaders = create_dataloaders(
        patches_dir=patches_dir,
        mode='segmentation',
        config_path=str(PROJECT_ROOT / 'configs' / 'data_config.yaml')
    )

    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches:   {len(loaders['val'])}")

    # ── Model ────────────────────────────────────────────
    module = SegmentationModule(config)

    # ── Checkpoint directory ─────────────────────────────
    if is_colab:
        ckpt_dir = config['drive_checkpoint_dir']
    else:
        ckpt_dir = config['checkpoint_dir']

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ── Callbacks ────────────────────────────────────────
    callbacks = [

        # Save best model (highest val IoU)
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='seg-best-{epoch:02d}-iou{val/iou:.4f}',
            monitor='val/iou',
            mode='max',           # higher IoU = better
            save_top_k=1,         # keep only the best
            save_last=True,       # also save last (for resuming)
            verbose=True
        ),

        # Stop training if val IoU does not improve for patience epochs
        EarlyStopping(
            monitor='val/iou',
            patience=config['early_stopping_patience'],
            mode='max',
            verbose=True
        ),

        # Log learning rate to wandb every epoch
        LearningRateMonitor(logging_interval='epoch'),

        # Nice progress bar
        RichProgressBar()
    ]

    # ── Logger ───────────────────────────────────────────
    logger = WandbLogger(
        project=config['wandb_project'],
        name=config['wandb_run_name'],
        save_dir='outputs/logs'
    )

    # ── Trainer ──────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu',
        devices=1,
        precision='16-mixed',     # fp16 mixed precision — halves VRAM
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=False,      # True is slower but fully reproducible
        gradient_clip_val=1.0,    # Prevents exploding gradients
    )

    # ── Train ────────────────────────────────────────────
    # ckpt_path='last' automatically resumes from last checkpoint
    # if it exists. If no checkpoint exists, trains from scratch.
    last_ckpt = Path(ckpt_dir) / 'last.ckpt'
    resume_from = str(last_ckpt) if last_ckpt.exists() else None

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    else:
        print("Starting fresh training")

    trainer.fit(
        module,
        train_dataloaders=loaders['train'],
        val_dataloaders=loaders['val'],
        ckpt_path=resume_from
    )

    # ── Test evaluation ──────────────────────────────────
    print("\nRunning test set evaluation...")
    trainer.test(module, dataloaders=loaders['test'], ckpt_path='best')

    print(f"\nBest checkpoint saved at: {ckpt_dir}")
    wandb.finish()
    return trainer


if __name__ == '__main__':
    with open(str(PROJECT_ROOT / 'configs' / 'seg_config.yaml')) as f:
        config = yaml.safe_load(f)['segmentation']

    # Detect if running on Colab
    try:
        import google.colab
        IS_COLAB = True
        print("Running on Google Colab")
    except ImportError:
        IS_COLAB = False
        print("Running locally")

    train(config, is_colab=IS_COLAB)
