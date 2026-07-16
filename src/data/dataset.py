"""
dataset.py

TWO dataset classes:
1. CloudSegmentationDataset → for Attention U-Net training
2. CloudRemovalDataset      → for cGAN training

KEY NORMALISATION RULE (memorise this):
Segmentation model uses sigmoid output → targets must be in [0, 1]
GAN generator uses tanh output       → targets must be in [-1, 1]

Getting this wrong is the most common silent failure in GAN training.
The loss goes down but the model learns nothing useful.

ALBUMENTATIONS SYNCHRONISATION:
When you pass image=cloudy AND additional_targets={'label':'image', 'mask':'mask'}
to a Compose transform, the SAME random parameters are applied to all.
If it decides to flip horizontally, ALL THREE flip identically.
This is guaranteed by albumentations internally.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Repo root = parent of src/ (this file is src/data/dataset.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_config_path(config_path: str) -> str:
    """
    Resolve a config path relative to the repo root so code works
    regardless of the current working directory (local or Colab).
    Absolute paths are returned unchanged.
    """
    p = Path(config_path)
    return str(p) if p.is_absolute() else str(PROJECT_ROOT / p)


# ─────────────────────────────────────────────
# TRANSFORM FACTORIES
# ─────────────────────────────────────────────

def segmentation_transforms(split: str, aug_cfg: Dict) -> A.Compose:
    """
    Transforms for segmentation dataset.
    
    Output image range: [0, 1]  (divided by 255, no mean/std shift)
    Output mask range:  {0, 1}  (binarised before this, ToTensorV2 preserves)
    
    Augmentations applied ONLY to 'train' split.
    Val and test get normalisation only — no random ops.
    This is important: augmenting validation adds noise to your metrics.
    """
    spatial_augs = []
    if split == 'train':
        if aug_cfg.get('horizontal_flip'):
            spatial_augs.append(A.HorizontalFlip(p=0.5))
        if aug_cfg.get('vertical_flip'):
            spatial_augs.append(A.VerticalFlip(p=0.5))
        if aug_cfg.get('random_rotate_90'):
            spatial_augs.append(A.RandomRotate90(p=0.5))
        spatial_augs.append(
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get('brightness_limit', 0.2),
                contrast_limit=aug_cfg.get('contrast_limit', 0.2),
                p=0.5
            )
        )
        if aug_cfg.get('random_gamma'):
            spatial_augs.append(A.RandomGamma(p=0.5))
        if aug_cfg.get('hue_saturation'):
            spatial_augs.append(
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                )
            )
        if aug_cfg.get('elastic_transform'):
            spatial_augs.append(A.ElasticTransform(p=0.3))
        if aug_cfg.get('grid_distortion'):
            spatial_augs.append(A.GridDistortion(p=0.3))

    return A.Compose(
        spatial_augs + [
            # Divide by 255 → [0, 1]. No mean/std normalisation.
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ],
        additional_targets={'mask': 'mask'}
    )


def gan_transforms(split: str, aug_cfg: Dict) -> A.Compose:
    """
    Transforms for GAN dataset.
    
    Output cloudy range:    [-1, 1]
    Output cloudfree range: [-1, 1]   ← target for generator, must match tanh
    Output mask range:      {0, 1}    ← used as conditioning input, not target
    
    How [-1,1] normalisation works:
    A.Normalize with mean=0.5, std=0.5 on [0,1] float:
    (x - 0.5) / 0.5  =  2x - 1
    At x=0: (0-0.5)/0.5 = -1 ✓
    At x=1: (1-0.5)/0.5 = +1 ✓
    But albumentations applies to uint8, so max_pixel_value=255.0 first.
    """
    spatial_augs = []
    if split == 'train':
        if aug_cfg.get('horizontal_flip'):
            spatial_augs.append(A.HorizontalFlip(p=0.5))
        if aug_cfg.get('vertical_flip'):
            spatial_augs.append(A.VerticalFlip(p=0.5))
        if aug_cfg.get('random_rotate_90'):
            spatial_augs.append(A.RandomRotate90(p=0.5))
        spatial_augs.append(
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get('brightness_limit', 0.2),
                contrast_limit=aug_cfg.get('contrast_limit', 0.2),
                p=0.5
            )
        )
        if aug_cfg.get('random_gamma'):
            spatial_augs.append(A.RandomGamma(p=0.5))
        if aug_cfg.get('hue_saturation'):
            spatial_augs.append(
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                )
            )
        if aug_cfg.get('elastic_transform'):
            spatial_augs.append(A.ElasticTransform(p=0.3))
        if aug_cfg.get('grid_distortion'):
            spatial_augs.append(A.GridDistortion(p=0.3))

    return A.Compose(
        spatial_augs + [
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ],
        # 'label' gets SAME image transform (normalised to [-1,1])
        # 'mask'  gets ONLY spatial transforms, NOT normalisation
        additional_targets={
            'label': 'image',
            'mask':  'mask'
        }
    )


# ─────────────────────────────────────────────
# DATASET CLASSES
# ─────────────────────────────────────────────

class CloudSegmentationDataset(Dataset):
    """
    Returns image-mask pairs for segmentation training.
    
    __getitem__ returns:
    {
      'image': FloatTensor [3, 256, 256]  range [0, 1]
      'mask':  FloatTensor [1, 256, 256]  values {0.0, 1.0}
      'patch_id': str
      'cloud_coverage': float
    }
    """

    def __init__(
        self,
        patches_dir: str,
        split: str,
        config_path: str = 'configs/data_config.yaml'
    ):
        with open(_resolve_config_path(config_path)) as f:
            cfg = yaml.safe_load(f)['data']

        self.split_dir = Path(patches_dir) / split
        self.transform = segmentation_transforms(split, cfg.get('augmentation', {}))

        with open(self.split_dir / 'patch_manifest.json') as f:
            self.manifest = json.load(f)

        self.patch_ids = sorted(self.manifest.keys())

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> Dict:
        pid = self.patch_ids[idx]

        image = cv2.imread(str(self.split_dir / 'cloud' / f'{pid}.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # → RGB

        mask = cv2.imread(
            str(self.split_dir / 'mask' / f'{pid}.png'),
            cv2.IMREAD_GRAYSCALE
        )
        # Binarise: RICE2 masks use 255 for cloud
        mask_binary = (mask > 127).astype(np.uint8)  # {0, 1}

        out = self.transform(image=image, mask=mask_binary)

        # mask from albumentations is [H, W], we need [1, H, W]
        mask_tensor = out['mask'].unsqueeze(0).float()

        return {
            'image':          out['image'],   # [3, H, W] float [0,1]
            'mask':           mask_tensor,    # [1, H, W] float {0,1}
            'patch_id':       pid,
            'cloud_coverage': self.manifest[pid]['cloud_coverage']
        }


class CloudRemovalDataset(Dataset):
    """
    Returns (cloudy, cloudfree, mask, edge_map) for GAN training.
    
    __getitem__ returns:
    {
      'cloudy':    FloatTensor [3, 256, 256]  range [-1, 1]
      'cloudfree': FloatTensor [3, 256, 256]  range [-1, 1]
      'mask':      FloatTensor [1, 256, 256]  values {0.0, 1.0}
      'edge_map':  FloatTensor [1, 256, 256]  range [0, 1]
      'patch_id':  str
      'cloud_coverage': float
    }
    
    WHAT IS edge_map:
    Sobel gradient magnitude of the cloud mask.
    High values = cloud boundaries.
    Used as 5th input channel to the GAN generator.
    
    WHY edge_map:
    The generator's hardest task is creating a realistic transition
    from reconstructed surface (under cloud) to original surface
    (always visible). Giving it explicit boundary locations makes
    this easier and produces fewer seam artifacts.
    """

    def __init__(
        self,
        patches_dir: str,
        split: str,
        config_path: str = 'configs/data_config.yaml'
    ):
        with open(_resolve_config_path(config_path)) as f:
            cfg = yaml.safe_load(f)['data']

        self.split_dir = Path(patches_dir) / split
        self.transform = gan_transforms(split, cfg.get('augmentation', {}))

        with open(self.split_dir / 'patch_manifest.json') as f:
            self.manifest = json.load(f)

        self.patch_ids = sorted(self.manifest.keys())

    def _edge_map(self, mask_255: np.ndarray) -> np.ndarray:
        """
        Compute Sobel edge magnitude of cloud mask.
        
        Sobel operator:
        Gx = [-1 0 +1; -2 0 +2; -1 0 +1] applied horizontally
        Gy = [-1 -2 -1; 0 0 0; +1 +2 +1] applied vertically
        G = sqrt(Gx^2 + Gy^2)
        
        Applied to the binary mask (values 0 or 255):
        Interior cloud pixels → G ≈ 0 (uniform, no gradient)
        Cloud edge pixels     → G is large (sharp transition 0→255)
        
        Output normalised to [0, 1] float32.
        """
        m = mask_255.astype(np.float32)
        gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
        G  = np.sqrt(gx**2 + gy**2)

        if G.max() > 0:
            G = G / G.max()  # normalise to [0, 1]
        return G.astype(np.float32)

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> Dict:
        pid = self.patch_ids[idx]

        cloudy = cv2.imread(str(self.split_dir / 'cloud' / f'{pid}.png'))
        cloudy = cv2.cvtColor(cloudy, cv2.COLOR_BGR2RGB)

        label = cv2.imread(str(self.split_dir / 'label' / f'{pid}.png'))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            str(self.split_dir / 'mask' / f'{pid}.png'),
            cv2.IMREAD_GRAYSCALE
        )
        mask_binary = (mask > 127).astype(np.uint8)  # {0, 1}

        # Edge map from mask (before augmentation — computed on binary mask)
        edge = self._edge_map(mask_binary * 255)      # [H, W] float32 [0,1]

        # Synchronized augmentation
        out = self.transform(
            image=cloudy,
            label=label,
            mask=mask_binary
        )

        # edge_map: apply only spatial transforms manually
        # (albumentations does not support float mask as additional target)
        # We apply the same flip/rotate by tracking the random state
        # Simple fix: recompute edge after mask augmentation
        aug_mask = out['mask'].numpy()        # [H, W] after spatial aug
        edge_aug = self._edge_map(aug_mask * 255)

        edge_tensor = torch.from_numpy(edge_aug).unsqueeze(0)  # [1, H, W]
        mask_tensor = out['mask'].unsqueeze(0).float()          # [1, H, W]

        return {
            'cloudy':         out['image'],   # [3, H, W] [-1, 1]
            'cloudfree':      out['label'],   # [3, H, W] [-1, 1]
            'mask':           mask_tensor,    # [1, H, W] {0, 1}
            'edge_map':       edge_tensor,    # [1, H, W] [0, 1]
            'patch_id':       pid,
            'cloud_coverage': self.manifest[pid]['cloud_coverage']
        }


# ─────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────

def create_dataloaders(
    patches_dir: str,
    mode: str = 'segmentation',
    config_path: str = 'configs/data_config.yaml'
) -> Dict[str, DataLoader]:
    """
    Returns {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    
    mode = 'segmentation' → CloudSegmentationDataset
    mode = 'removal'      → CloudRemovalDataset
    
    DataLoader settings explained:
    
    shuffle=True for train only:
    Shuffling ensures the model does not memorise the order of batches.
    Val and test are NOT shuffled — deterministic order for reproducible metrics.
    
    drop_last=True for train only:
    If dataset has 1001 samples with batch_size=8, the last batch has 1 sample.
    BatchNorm layers in the model break with batch_size=1.
    drop_last discards this incomplete batch. Val/test keep all samples.
    
    pin_memory=True:
    Allocates host memory in pinned (page-locked) memory.
    GPU DMA can transfer from pinned memory faster than pageable memory.
    Only use when you have enough RAM — on your 16GB system, this is fine.
    """
    with open(_resolve_config_path(config_path)) as f:
        cfg = yaml.safe_load(f)['data']

    batch_size = (
        cfg['seg_batch_size'] if mode == 'segmentation'
        else cfg['gan_batch_size']
    )
    num_workers = cfg['num_workers']
    pin_memory  = cfg['pin_memory']

    DatasetClass = (
        CloudSegmentationDataset if mode == 'segmentation'
        else CloudRemovalDataset
    )

    loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = DatasetClass(patches_dir, split, config_path)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train'),
            persistent_workers=(num_workers > 0)
        )

    return loaders