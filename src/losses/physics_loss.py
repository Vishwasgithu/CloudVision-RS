"""
physics_loss.py

Physics-informed loss functions — the scientific contribution of this project.

Standard GANs optimize for visual realism (SSIM, PSNR).
Satellite images carry scientific information: vegetation indices,
spectral band ratios, surface reflectance.

A visually convincing but spectrally wrong reconstruction is
scientifically useless for ISRO land cover mapping.

These losses directly penalize scientifically incorrect outputs
DURING TRAINING — forcing the model to learn spectral correctness,
not just visual plausibility.

All physics losses are computed ONLY over cloud-masked pixels
(the region the model actually reconstructed). Applying them
to unchanged clear-sky pixels would penalize the model for
perfectly correct outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def denorm(tensor):
    """
    Convert from GAN normalisation [-1, 1] back to [0, 1].
    Required before computing vegetation indices and spectral ratios
    which are defined for reflectance values in [0, 1].

    Formula: x_01 = (x_11 + 1) / 2
    At x=-1: (-1+1)/2 = 0  ✓
    At x=+1: (+1+1)/2 = 1  ✓
    """
    return (tensor + 1.0) / 2.0


class VARILoss(nn.Module):
    """
    VARI Preservation Loss — Vegetation Index Consistency.

    VARI = Visible Atmospherically Resistant Index
    Formula: (Green - Red) / (Green + Red - Blue + ε)

    Published: Gitelson et al. 2002
    Validated against NDVI with correlation r=0.92.
    Standard RGB-accessible vegetation index (NDVI requires NIR band
    which RICE2 RGB dataset does not have).

    WHY THIS MATTERS SCIENTIFICALLY:
    A forest patch reconstructed with wrong spectral values will have
    wrong VARI even if it looks green to the human eye.
    ISRO scientists use vegetation indices directly for land classification.
    A visually green but spectrally flat patch (R≈G≈B) would be
    misclassified as urban or bare soil in downstream analysis.

    Loss is computed only over cloud-masked region (mask=1 pixels).
    These are the pixels the generator actually reconstructed.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predicted, target, cloud_mask):
        # Denormalise from [-1,1] to [0,1]
        pred01 = torch.clamp(denorm(predicted), 0, 1)
        tgt01  = torch.clamp(denorm(target),    0, 1)

        Rp, Gp, Bp = pred01[:,0:1], pred01[:,1:2], pred01[:,2:3]
        Rt, Gt, Bt = tgt01[:,0:1],  tgt01[:,1:2],  tgt01[:,2:3]

        # denominator clamped to minimum 0.1 — prevents any division issues
        denom_p = torch.clamp(Gp + Rp - Bp, min=0.1)
        denom_t = torch.clamp(Gt + Rt - Bt, min=0.1)

        vari_pred = torch.clamp((Gp - Rp) / denom_p, -1.0, 1.0)
        vari_tgt  = torch.clamp((Gt - Rt) / denom_t, -1.0, 1.0)

        diff = (vari_pred - vari_tgt) ** 2
        loss = (diff * cloud_mask).sum() / (cloud_mask.sum() + 1e-8)

        # Safety clamp — if loss is still unreasonably large, cap it
        return torch.clamp(loss, max=10.0)


class SpectralRatioLoss(nn.Module):
    """
    Spectral Ratio Consistency Loss.

    Computes per-pixel channel ratios: R/G, B/G, R/B
    Penalizes deviations in these ratios over the cloud-masked region.

    WHY RATIOS AND NOT ABSOLUTE CHANNEL VALUES:
    A forest at noon and the same forest at different sun angle have
    different absolute brightness but the SAME spectral ratios.
    R/G, B/G, R/B encode the spectral signature of a land cover type
    independent of illumination conditions.

    Physically meaningful signatures:
    Water:      high B/G, low R/G  (strong blue absorption)
    Vegetation: low R/G, low B/G   (chlorophyll absorbs red and blue)
    Urban:      R/G ≈ B/G ≈ 1     (spectrally flat)
    Bare soil:  high R/G           (strong red reflectance)

    A model producing wrong ratios makes downstream classification wrong
    regardless of how visually convincing the output looks.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predicted, target, cloud_mask):
        pred01 = torch.clamp(denorm(predicted), 0, 1)
        tgt01  = torch.clamp(denorm(target),    0, 1)

        eps = 0.1  # large enough to prevent overflow

        Rp, Gp, Bp = pred01[:,0:1], pred01[:,1:2], pred01[:,2:3]
        Rt, Gt, Bt = tgt01[:,0:1],  tgt01[:,1:2],  tgt01[:,2:3]

        loss = 0
        for rp, rt in [
            (Rp/(Gp+eps), Rt/(Gt+eps)),
            (Bp/(Gp+eps), Bt/(Gt+eps)),
            (Rp/(Bp+eps), Rt/(Bt+eps))
        ]:
            diff  = torch.clamp((rp - rt)**2, max=10.0)
            loss += (diff * cloud_mask).sum() / (cloud_mask.sum() + 1e-8)

        return loss / 3.0


class EdgeCoherenceLoss(nn.Module):
    """
    Edge Coherence Loss — Eliminates Cloud Boundary Seams.

    The most visible artifact in GAN-based cloud removal is a seam
    at the cloud boundary — a visible discontinuity where the
    reconstructed region meets the original clear-sky region.

    This loss penalizes gradient magnitude differences at cloud
    boundary pixels specifically.

    HOW IT WORKS:
    1. Dilate the cloud mask to get a boundary region (pixels near the edge)
    2. Compute Sobel gradient magnitude of predicted and target at those pixels
    3. L1 loss on gradient magnitudes at boundary pixels only

    WHY GRADIENT MAGNITUDE NOT PIXEL VALUES:
    The surface reflectance should be spatially smooth at the cloud boundary
    (it was smooth before the cloud arrived — land surface does not change).
    Matching gradient magnitude enforces this physical continuity constraint.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels — detect horizontal and vertical edges
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )

    def _gradient_magnitude(self, img):
        """Compute Sobel gradient magnitude per channel."""
        B, C, H, W = img.shape
        sobel_x = self.sobel_x.to(img.device)
        sobel_y = self.sobel_y.to(img.device)
        mag = torch.zeros(B, 1, H, W, device=img.device)
        for c in range(C):
            ch = img[:, c : c + 1, :, :]
            gx = F.conv2d(ch, sobel_x, padding=1)
            gy = F.conv2d(ch, sobel_y, padding=1)
            mag += torch.sqrt(gx**2 + gy**2 + 1e-8)
        return mag / C  # average over channels

    def _get_boundary(self, mask):
        """Dilate mask and find boundary region (dilated - eroded)."""
        kernel = torch.ones(1, 1, 5, 5, device=mask.device)
        # Dilation: any pixel within 2px of cloud is in dilated mask
        dilated = F.conv2d(mask.float(), kernel, padding=2)
        dilated = (dilated > 0).float()
        # Erosion: only pixels fully surrounded by cloud
        eroded = F.conv2d(mask.float(), kernel, padding=2)
        eroded = (eroded == 25).float()
        # Boundary = dilated minus eroded interior
        return (dilated - eroded).clamp(0, 1)

    def forward(self, predicted, target, cloud_mask):
        """
        predicted:   [B, 3, H, W]
        target:      [B, 3, H, W]
        cloud_mask:  [B, 1, H, W]
        """
        # 1. Convert the mask to float immediately (This fixes the CUDA error!)
        cloud_mask = cloud_mask.float().to(predicted.device)
        predicted = predicted.float()
        target = target.float()

        # 2. Compute boundary by dilating the mask
        dilated = F.max_pool2d(cloud_mask, kernel_size=5, stride=1, padding=2)
        boundary = dilated - cloud_mask

        # 3. Use .item() to safely check if the boundary is empty
        if boundary.sum().item() < 1.0:
            return torch.tensor(0.0, device=predicted.device)

        # 4. Compute gradients
        pred_grad = self._gradient_magnitude(predicted)
        tgt_grad = self._gradient_magnitude(target)

        # 5. L1 Loss on boundary pixels only
        loss = F.l1_loss(pred_grad * boundary, tgt_grad * boundary, reduction="sum")

        return loss / (boundary.sum() + 1e-8)


class PerceptualLoss(nn.Module):
    """
    VGG Perceptual Loss — Semantic Texture Matching.

    Extracts intermediate feature maps from pretrained VGG-16.
    L1 loss between feature maps of predicted and target images.

    WHY VGG FEATURES INSTEAD OF PIXEL L1:
    Pixel L1 penalizes any pixel difference equally regardless of
    semantic meaning. Two patches can look identical to a human but
    differ at pixel level (textures shifted by 1 pixel).
    VGG features encode texture, structure, and semantic similarity
    at multiple scales — much closer to how humans and scientists
    judge image quality.

    WHY relu2_2 AND relu3_3 SPECIFICALLY:
    relu2_2: encodes textures (vegetation texture, water texture, urban texture)
    relu3_3: encodes structural patterns (field boundaries, river meanders)
    Higher layers (relu4, relu5) encode semantic object identity — too
    abstract for surface reconstruction.
    """

    def __init__(self, device):
        super().__init__()
        import torchvision.models as models

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Extract features up to relu2_2 (layer index 9)
        # and relu3_3 (layer index 16)
        self.slice1 = nn.Sequential(*list(vgg.features)[:9]).to(device)
        self.slice2 = nn.Sequential(*list(vgg.features)[9:16]).to(device)

        # VGG is frozen — we never update its weights
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, predicted, target, cloud_mask=None):
        """
        cloud_mask is unused here — perceptual loss applies to full image
        because VGG features integrate over large receptive fields.
        """
        # VGG was trained on ImageNet with this normalisation
        # predicted and target are in [-1,1], convert to VGG input range
        pred01 = denorm(predicted)
        tgt01 = denorm(target)

        # VGG ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406], device=predicted.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=predicted.device).view(
            1, 3, 1, 1
        )
        pred_vgg = (pred01 - mean) / std
        tgt_vgg = (tgt01 - mean) / std

        # Extract features and compute L1 loss
        p1 = self.slice1(pred_vgg)
        t1 = self.slice1(tgt_vgg)
        p2 = self.slice2(p1)
        t2 = self.slice2(t1)

        loss = F.l1_loss(p1, t1) + F.l1_loss(p2, t2)
        return loss


class PhysicsLoss(nn.Module):
    """
    Combined physics-informed loss module.
    Wraps all four physics loss components with configurable weights.
    """

    def __init__(self, config, device):
        super().__init__()
        self.lambda_vari = config["lambda_vari"]
        self.lambda_spectral = config["lambda_spectral"]
        self.lambda_edge = config["lambda_edge"]
        self.lambda_perceptual = config["lambda_perceptual"]

        self.vari_loss = VARILoss()
        self.spectral_loss = SpectralRatioLoss()
        self.edge_loss = EdgeCoherenceLoss()
        self.perceptual_loss = PerceptualLoss(device)

    def forward(self, predicted, target, cloud_mask):
        """
        Returns dict of individual losses and weighted total.
        Individual components are logged separately to wandb.
        """
        l_vari = self.vari_loss(predicted, target, cloud_mask)
        l_spec = self.spectral_loss(predicted, target, cloud_mask)
        l_edge = self.edge_loss(predicted, target, cloud_mask)
        l_perc = self.perceptual_loss(predicted, target, cloud_mask)

        total = (
            self.lambda_vari * l_vari
            + self.lambda_spectral * l_spec
            + self.lambda_edge * l_edge
            + self.lambda_perceptual * l_perc
        )

        return {
            "physics_total": total,
            "vari_loss": l_vari,
            "spectral_loss": l_spec,
            "edge_loss": l_edge,
            "perceptual_loss": l_perc,
        }
