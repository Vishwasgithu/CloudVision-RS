"""
segmentation.py

Attention U-Net for cloud segmentation using
segmentation-models-pytorch (smp).

WHY USE smp INSTEAD OF WRITING FROM SCRATCH:
smp connects pretrained encoders (ResNet34, EfficientNet etc.)
to U-Net decoders with correct skip connection dimensions.
Writing this manually requires careful channel dimension matching
that has subtle bugs. smp is tested across thousands of use cases.
Your contribution is the physics loss (Phase 3), not the U-Net code.

WHY ResNet34 ENCODER:
- Pretrained on ImageNet — low-level edge detectors transfer to
  cloud boundaries immediately, even before fine-tuning
- Small enough for Colab T4 at batch size 16 with fp16
- Outperforms ResNet50 on small datasets (less overfitting)
- You can swap to EfficientNet-B2 with one line if you want to try

WHY ATTENTION U-NET:
- Attention gates on skip connections learn to suppress irrelevant
  regions (open ocean, uniform desert) and amplify cloud edges
- +3 to 8% IoU over plain U-Net at only ~2% more parameters
- No increase in training time

HOW ATTENTION GATES WORK:
Gate signal (from decoder) + skip signal (from encoder) are both
projected to a shared dimension, added, passed through ReLU then
sigmoid → produces spatial attention map [0, 1] at each location.
This map multiplies the skip features — zeroing out irrelevant areas,
amplifying cloud-relevant areas — before they reach the decoder.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import yaml
from pathlib import Path


class AttentionUNet(nn.Module):
    """
    Wraps smp.Unet with attention decoder.

    smp provides UnetPlusPlus which has attention-like mechanisms,
    but for explicit attention gates we use the standard Unet
    with the 'scse' decoder attention type — Spatial and Channel
    Squeeze-and-Excitation. This is more flexible than fixed
    attention gates and works better in practice on small datasets.

    SCSE attention:
    - Spatial SE: learns which spatial locations are important
    - Channel SE: learns which feature channels are important
    - Applied at each decoder stage on the skip connection features
    """

    def __init__(self, config: dict):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=config['encoder_name'],       # 'resnet34'
            encoder_weights=config['encoder_weights'], # 'imagenet'
            in_channels=config['in_channels'],         # 3 (RGB)
            classes=config['num_classes'],             # 1 (binary mask)
            activation=None,     # No activation here — we apply sigmoid
                                 # inside the loss (BCEWithLogitsLoss)
                                 # and explicitly during inference
            decoder_attention_type='scse',  # spatial+channel attention
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      [B, 3, H, W]  — normalised RGB image in [0, 1]
        return: [B, 1, H, W]  — raw logits (not sigmoid-activated)

        Raw logits are returned (not probabilities) because:
        - BCEWithLogitsLoss expects logits (more numerically stable)
        - During inference: apply sigmoid then threshold at 0.5
        """
        return self.model(x)

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Inference method. Returns binary mask {0, 1}.
        Use this during inference, NOT during training.

        threshold=0.5: pixels with probability > 0.5 are classified as cloud.
        You can tune this threshold — higher = more conservative (fewer false positives),
        lower = more aggressive (catches thin clouds, more false positives).
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.sigmoid(logits)
            mask   = (probs > threshold).float()
        return mask
