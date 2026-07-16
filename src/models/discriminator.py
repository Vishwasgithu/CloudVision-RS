"""
discriminator.py

PatchGAN Discriminator — classifies 70×70 patches as real or fake.

WHY PATCHGAN INSTEAD OF FULL-IMAGE DISCRIMINATOR:
A full-image discriminator outputs one real/fake score for the
whole image. The generator can fool it by making most of the image
realistic and hiding artifacts in small regions.

PatchGAN classifies overlapping N×N patches independently.
The generator must be realistic at the LOCAL texture level
across the ENTIRE image — it cannot hide artifacts anywhere.
70×70 is standard (from Pix2Pix paper, Isola et al. 2017).
It captures enough context to judge local texture realism.

WHY CONDITION ON BOTH INPUT AND OUTPUT:
The discriminator sees (input_image, output_image) concatenated.
It learns: given this cloudy input, does this cloud-free output
look realistic? Without the input, it only judges if the output
looks like a satellite image generally — not whether it is a
plausible reconstruction for this specific input.
"""

import torch
import torch.nn as nn


class DiscBlock(nn.Module):
    """Conv → BatchNorm → LeakyReLU block for discriminator."""
    def __init__(self, in_ch, out_ch, stride=2, use_bn=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4,
                      stride=stride, padding=1, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.

    Input: concatenation of condition (cloudy image) + target/generated image
           Shape: [B, 6, 256, 256]  (3 + 3 channels)

    Output: [B, 1, 30, 30] — each value is real/fake score for one patch
            The 30×30 grid covers 70×70 receptive fields in the input.

    WHY spectral normalization is NOT used here:
    We rely on two-timescale update rule instead (train D once, G twice).
    Spectral norm is an alternative stability technique but adds complexity.
    The two-timescale approach is simpler and equally effective here.
    """
    def __init__(self, in_channels=6, features=64):
        super().__init__()

        self.model = nn.Sequential(
            # No BatchNorm on first layer (input layer convention)
            DiscBlock(in_channels, features,   stride=2, use_bn=False),
            DiscBlock(features,    features*2, stride=2, use_bn=True),
            DiscBlock(features*2,  features*4, stride=2, use_bn=True),
            DiscBlock(features*4,  features*8, stride=1, use_bn=True),
            # Final conv → 1 channel, no activation (raw logits for BCEWithLogitsLoss)
            nn.Conv2d(features*8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, input_img, target_img):
        """
        input_img:  [B, 3, 256, 256] — cloudy satellite image
        target_img: [B, 3, 256, 256] — real cloud-free OR generated image
        returns:    [B, 1, 30, 30]   — patch-level real/fake scores (logits)
        """
        x = torch.cat([input_img, target_img], dim=1)  # [B, 6, 256, 256]
        return self.model(x)
