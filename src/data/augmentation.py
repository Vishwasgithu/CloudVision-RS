"""
augmentation.py

Research-grade augmentation pipeline for cloud segmentation over agricultural
optical satellite imagery.

Scientific constraints
-----------------------
1. Radiometric fidelity
   Optical bands encode surface reflectance; transforms must simulate physically
   plausible acquisition variations (illumination, sensor noise, atmospheric
   scattering) without altering band-to-band spectral ratios.

2. Geometric fidelity
   Nadir-viewing overhead imagery is orthorectified to an affine projection.
   Only rigid transforms (flips, 90-degree rotations) and scale-preserving
   photometric ops are valid. Elastic/grid/perspective warps are forbidden.

3. Mask consistency
   Every spatial transform is applied identically to image and mask via
   Albumentations' ``additional_targets`` mechanism.

4. No random crop
   Verified that stored patches are exactly 256×256. RandomCrop is excluded
   to avoid redundant cropping and to preserve full-field context.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# ImageNet normalization constants for ResNet34 pretrained encoder
# ---------------------------------------------------------------------------
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_augmentation(
    split: str,
    aug_cfg: Dict,
    image_size: int = 256,
    always_apply_normalize: bool = True,
) -> A.Compose:
    """
    Build an Albumentations pipeline for the given split.

    Parameters
    ----------
    split : str
        ``'train'`` enables stochastic augmentations.
        ``'val'`` or ``'test'`` returns deterministic normalization only.
    aug_cfg : dict
        Configuration dict from ``data_config.yaml``.
        Expected keys (all optional, bool unless noted):
          ``random_rotate_90``, ``horizontal_flip``, ``vertical_flip``,
          ``random_brightness_contrast``, ``brightness_limit`` (float),
          ``contrast_limit`` (float), ``random_gamma``, ``gaussian_noise``,
          ``random_shadow``, ``solarize``.
    image_size : int
        Target spatial size. Used only for documentation; patches are already
        this size so no cropping is performed.
    always_apply_normalize : bool
        If True, Normalize + ToTensorV2 are always appended.

    Returns
    -------
    A.Compose
        Albumentations composition ready for ``transform(image=..., mask=...)``.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

    transforms: List[A.BasicTransform] = []

    if split == "train":
        transforms.extend(_build_train_transforms(aug_cfg))

    if always_apply_normalize:
        transforms.extend([
            _build_normalize(),
            ToTensorV2(),
        ])

    return A.Compose(
        transforms,
        additional_targets={"mask": "mask"},
    )


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _build_train_transforms(cfg: Dict) -> List[A.BasicTransform]:
    """
    Stochastic augmentations applied ONLY during training.

    Ordering principle
    -------------------
    Spatial transforms first (flips, rotations), then photometric transforms
    (brightness, gamma, noise, shadow, solarize). This avoids interpolating
    noise/shadow masks and keeps geometric operations clean.
    """
    t: List[A.BasicTransform] = []

    # ---- Spatial (isotropic) ------------------------------------------------
    if cfg.get("horizontal_flip", True):
        t.append(A.HorizontalFlip(p=0.5))

    if cfg.get("vertical_flip", True):
        t.append(A.VerticalFlip(p=0.5))

    if cfg.get("random_rotate_90", True):
        t.append(A.RandomRotate90(p=0.5))

    # ---- Photometric (radiometrically plausible) ---------------------------
    if cfg.get("random_brightness_contrast", True):
        t.append(
            A.RandomBrightnessContrast(
                brightness_limit=cfg.get("brightness_limit", 0.15),
                contrast_limit=cfg.get("contrast_limit", 0.15),
                p=0.5,
            )
        )

    if cfg.get("random_gamma", True):
        t.append(A.RandomGamma(gamma_limit=(80, 120), p=0.3))

    if cfg.get("gaussian_noise", False):
        t.append(
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                mean=0.0,
                p=0.25,
            )
        )

    if cfg.get("random_shadow", False):
        t.append(
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=0.3,
            )
        )

    if cfg.get("solarize", False):
        t.append(
            A.Solarize(
                thresholds=(128, 255),
                p=0.2,
            )
        )

    return t


def _build_normalize() -> A.Normalize:
    """
    ImageNet normalization for ResNet34 pretrained encoder.

    Why ImageNet stats, not per-dataset stats:
    The encoder was pretrained on ImageNet with these exact mean/std values.
    Feeding inputs normalized to a different distribution shifts the feature
    space and forces the first convolution to re-learn a whitening transform
    from scratch, destabilizing training and slowing convergence.
    """
    return A.Normalize(mean=_MEAN, std=_STD, max_pixel_value=255.0)


# ---------------------------------------------------------------------------
# Convenience presets
# ---------------------------------------------------------------------------

def get_segmentation_transforms(
    split: str,
    aug_cfg: Optional[Dict] = None,
) -> A.Compose:
    """
    Convenience wrapper for segmentation training.

    Uses sensible defaults when ``aug_cfg`` is None.
    """
    defaults: Dict = {
        "horizontal_flip": True,
        "vertical_flip": True,
        "random_rotate_90": True,
        "random_brightness_contrast": True,
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "random_gamma": True,
        "gaussian_noise": True,
        "random_shadow": False,
        "solarize": False,
    }
    if aug_cfg:
        defaults.update(aug_cfg)
    return build_augmentation(split, defaults)


def get_gan_transforms(
    split: str,
    aug_cfg: Optional[Dict] = None,
) -> A.Compose:
    """
    Convenience wrapper for GAN (cloud removal) training.

    GAN pipelines need an extra ``'label': 'image'`` target so that the
    cloud-free reference image receives the same photometric transforms as
    the cloudy input.
    """
    defaults: Dict = {
        "horizontal_flip": True,
        "vertical_flip": True,
        "random_rotate_90": True,
        "random_brightness_contrast": True,
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "random_gamma": True,
        "gaussian_noise": True,
        "random_shadow": False,
        "solarize": False,
    }
    if aug_cfg:
        defaults.update(aug_cfg)

    transforms: List[A.BasicTransform] = []

    if split == "train":
        transforms.extend(_build_train_transforms(defaults))

    transforms.extend([
        _build_normalize(),
        ToTensorV2(),
    ])

    return A.Compose(
        transforms,
        additional_targets={
            "label": "image",
            "mask": "mask",
        },
    )
