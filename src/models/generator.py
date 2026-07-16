"""
generator.py

U-Net Generator for cloud removal.

INPUT:  5-channel tensor = RGB cloudy image (3) + cloud mask (1) + Sobel edge map (1)
OUTPUT: 3-channel RGB cloud-free image in range [-1, 1] via tanh activation

WHY U-NET FOR GENERATOR:
Skip connections preserve spatial detail of land cover boundaries.
Without them the generator loses fine structure (field edges, river banks)
during the encoding bottleneck. With them, fine detail is preserved.

WHY 5-CHANNEL INPUT:
- Channels 0-2: RGB cloudy image — what we have
- Channel 3:    cloud mask — tells generator WHICH pixels to reconstruct
- Channel 4:    Sobel edge map of mask — tells generator WHERE the
                cloud boundary is, so transitions are smooth

WHY tanh OUTPUT (not sigmoid):
tanh maps to [-1, 1] which is symmetric around zero.
Generator learns both positive and negative corrections equally.
The target images are also normalised to [-1, 1] during GAN training.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic building block: Conv → BatchNorm → LeakyReLU
    Used in the encoder (downsampling path).

    WHY LeakyReLU not ReLU:
    ReLU sets all negative values to zero — this kills gradients
    for negative activations (dying ReLU problem).
    LeakyReLU allows a small gradient (0.2x) for negatives,
    keeping gradient flow alive throughout deep networks.
    """
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Decoder block: Upsample → Conv → BatchNorm → ReLU
    Used in the decoder (upsampling path).

    WHY BILINEAR UPSAMPLE + CONV instead of ConvTranspose2d:
    ConvTranspose2d (transposed convolution) produces checkerboard
    artifacts — a well-known failure mode where the output has a
    regular grid pattern of brighter/darker pixels.
    Bilinear upsampling followed by a regular convolution avoids this.

    WHY dropout=True on first decoder blocks:
    Dropout during training creates stochasticity in the generator output.
    This prevents the generator from memorising exact pixel values
    and forces it to learn more general reconstruction strategies.
    Only applied during training (nn.Dropout handles train/eval mode).
    """
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    U-Net Generator with skip connections.

    Architecture:
    Encoder: 5ch → 64 → 128 → 256 → 512 → 512 → 512 (bottleneck)
    Decoder: 512 → 512 → 256 → 128 → 64 → 3ch (output)

    Skip connections concatenate encoder feature maps into decoder.
    This doubles the channel count at each decoder stage (hence in_ch*2).
    The decoder must fuse both the upsampled signal AND the skip features.
    """
    def __init__(self, in_channels=5, features=64):
        super().__init__()

        # ── Encoder ────────────────────────────────────────
        # First encoder: no BatchNorm (input layer convention)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )                                           # → [B, 64, 128, 128]
        self.enc2 = ConvBlock(features,   features*2)  # → [B, 128, 64, 64]
        self.enc3 = ConvBlock(features*2, features*4)  # → [B, 256, 32, 32]
        self.enc4 = ConvBlock(features*4, features*8)  # → [B, 512, 16, 16]
        self.enc5 = ConvBlock(features*8, features*8)  # → [B, 512, 8, 8]

        # Bottleneck — no BatchNorm at the very bottom
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )                                           # → [B, 512, 4, 4]

        # ── Decoder ────────────────────────────────────────
        # in_ch is doubled because of skip connection concatenation
        self.dec1 = UpBlock(features*8,   features*8, dropout=True)   # → [B, 512, 8, 8]
        self.dec2 = UpBlock(features*8*2, features*8, dropout=True)   # → [B, 512, 16, 16]
        self.dec3 = UpBlock(features*8*2, features*4, dropout=False)  # → [B, 256, 32, 32]
        self.dec4 = UpBlock(features*4*2, features*2, dropout=False)  # → [B, 128, 64, 64]
        self.dec5 = UpBlock(features*2*2, features,   dropout=False)  # → [B, 64, 128, 128]

        # Final output: upsample to full resolution
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features*2, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()   # output range [-1, 1]
        )

    def forward(self, x):
        """
        x: [B, 5, 256, 256] — RGB + mask + edge
        returns: [B, 3, 256, 256] — cloud-free RGB in [-1, 1]
        """
        # Encoder — save outputs for skip connections
        e1 = self.enc1(x)       # [B, 64, 128, 128]
        e2 = self.enc2(e1)      # [B, 128, 64, 64]
        e3 = self.enc3(e2)      # [B, 256, 32, 32]
        e4 = self.enc4(e3)      # [B, 512, 16, 16]
        e5 = self.enc5(e4)      # [B, 512, 8, 8]
        bn = self.bottleneck(e5) # [B, 512, 4, 4]

        # Decoder — concatenate skip connections
        d1 = self.dec1(bn)                     # [B, 512, 8, 8]
        d2 = self.dec2(torch.cat([d1, e5], 1)) # [B, 512, 16, 16]
        d3 = self.dec3(torch.cat([d2, e4], 1)) # [B, 256, 32, 32]
        d4 = self.dec4(torch.cat([d3, e3], 1)) # [B, 128, 64, 64]
        d5 = self.dec5(torch.cat([d4, e2], 1)) # [B, 64, 128, 128]
        return self.final(torch.cat([d5, e1], 1)) # [B, 3, 256, 256]
