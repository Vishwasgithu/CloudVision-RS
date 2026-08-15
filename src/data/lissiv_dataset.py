"""
lissiv_dataset.py

Loads LISS-IV multispectral imagery from Bhoonidhi downloads.
Handles 4-band TIF files: Green(B2), Red(B3), NIR(B4), SWIR(B5).

Key difference from RICE2:
- RICE2 was RGB (3 channels)
- LISS-IV is 4-band (Green, Red, NIR, SWIR)
- NIR enables true NDVI instead of VARI approximation
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio
import cv2


def load_lissiv_scene(scene_dir: str, bands=[2,3,4]):
    """
    Load LISS-IV scene from Bhoonidhi download directory.

    Bhoonidhi gives one TIF per band:
    *_B2.tif = Green
    *_B3.tif = Red
    *_B4.tif = NIR   ← most important for NDVI
    *_B5.tif = SWIR

    bands parameter: list of band numbers to load
    Returns: numpy array [H, W, n_bands] in uint16
             rasterio metadata for CRS preservation
    """
    scene_dir = Path(scene_dir)
    band_arrays = []
    meta = None

    for b in bands:
        # Bhoonidhi naming: typically *_B{n}.tif or *band{n}*.tif
        candidates = list(scene_dir.glob(f'*B{b}*.tif')) + \
                     list(scene_dir.glob(f'*band{b}*.tif')) + \
                     list(scene_dir.glob(f'*_b{b}*.tif'))

        if not candidates:
            raise FileNotFoundError(
                f"Band {b} TIF not found in {scene_dir}. "
                f"Files: {list(scene_dir.glob('*.tif'))}"
            )

        with rasterio.open(candidates[0]) as src:
            arr = src.read(1).astype(np.float32)
            if meta is None:
                meta = src.meta.copy()
            band_arrays.append(arr)

    # Stack bands: [H, W, n_bands]
    image = np.stack(band_arrays, axis=2)
    return image, meta


def normalise_lissiv(image: np.ndarray, percentile=2) -> np.ndarray:
    """
    Normalise LISS-IV imagery to [0, 255] uint8.

    LISS-IV pixel values are typically 0-1023 (10-bit) or 0-4095 (12-bit).
    We use percentile stretch (2-98%) to handle outliers.

    WHY PERCENTILE NOT MIN-MAX:
    Satellite images have occasional very bright pixels (specular reflection,
    cloud tops) that would compress the entire image range if we used min-max.
    2-98 percentile stretch ignores these outliers.
    """
    result = np.zeros_like(image, dtype=np.uint8)
    for c in range(image.shape[2]):
        band = image[:,:,c]
        lo   = np.percentile(band[band > 0], percentile)
        hi   = np.percentile(band[band > 0], 100 - percentile)
        stretched = np.clip((band - lo) / (hi - lo + 1e-8), 0, 1) * 255
        result[:,:,c] = stretched.astype(np.uint8)
    return result


def compute_ndvi(image_01: np.ndarray, nir_idx=2, red_idx=1) -> np.ndarray:
    """
    Compute NDVI from normalised [0,1] multispectral image.
    NDVI = (NIR - Red) / (NIR + Red)
    Range: [-1, 1]  Vegetation typically 0.2-0.8
    """
    NIR = image_01[:,:,nir_idx].astype(np.float32)
    Red = image_01[:,:,red_idx].astype(np.float32)
    return (NIR - Red) / (NIR + Red + 1e-8)
