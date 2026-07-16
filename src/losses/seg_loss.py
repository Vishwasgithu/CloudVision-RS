"""
seg_loss.py

Combined BCE + Dice loss for cloud segmentation.

WHY TWO LOSSES TOGETHER:

Binary Cross Entropy (BCE):
- Treats each pixel independently
- Computes: -[y*log(p) + (1-y)*log(1-p)] per pixel
- Problem: if 80% of pixels are cloud-free, predicting ALL zeros
  gives 80% accuracy and low BCE. Model learns to ignore clouds.
- It IS needed because it calibrates the probability output correctly.

Dice Loss:
- Computes: 1 - (2 * intersection) / (prediction + target)
- If prediction is all zeros and target has any cloud pixels,
  intersection = 0, so Dice = 1.0 (maximum loss)
- Cannot be cheated by predicting majority class
- Directly optimises for overlap between predicted and actual mask

Together: BCE calibrates probabilities, Dice forces actual cloud detection.
Starting weight: 50/50. If cloud coverage is very low (<15%), increase
Dice weight to 0.7.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Formula: 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)

    smooth = 1.0 prevents division by zero when both prediction
    and target are all zeros (no cloud in patch).
    Without smooth, loss would be undefined for clear-sky patches.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        predictions: [B, 1, H, W] — raw logits (before sigmoid)
        targets:     [B, 1, H, W] — binary float {0.0, 1.0}
        """
        # Apply sigmoid to convert logits to probabilities [0, 1]
        pred_prob = torch.sigmoid(predictions)

        # Flatten spatial dimensions for computation
        pred_flat = pred_prob.view(-1)
        tgt_flat  = targets.view(-1)

        intersection = (pred_flat * tgt_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + tgt_flat.sum() + self.smooth
        )

        return 1.0 - dice


class SegmentationLoss(nn.Module):
    """
    Combined BCE + Dice loss.

    BCEWithLogitsLoss is numerically more stable than
    applying sigmoid then BCELoss separately.
    It computes: sigmoid + BCE in one operation using
    the log-sum-exp trick to prevent overflow.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=1.0)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> dict:
        """
        Returns a dict with total loss and individual components.
        Returning components separately lets you log them to wandb
        so you can see whether BCE or Dice is driving the training.
        """
        bce_loss  = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)

        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return {
            'loss':      total,
            'bce_loss':  bce_loss,
            'dice_loss': dice_loss
        }
