"""
dataset_builder.py

PURPOSE:
Builds a reproducible train/val/test split at the IMAGE level.

WHY IMAGE-LEVEL SPLIT MATTERS:
A 512x512 RICE2 image produces approximately 9 patches at 256x256 
with 50% overlap. If you split at the PATCH level, patches from 

import cv2
import yaml
image_001 appear in both train and test. The model has seen the 
exact scene during training. Metrics inflate by 15-25%. 
Your results become scientifically meaningless.

This class:
- Reads all 736 image IDs from RICE2
- Computes cloud coverage per image (used for stratified split)
- Splits at image level with stratification
- Saves the split as JSON (committed to Git for reproducibility)
"""
import yaml
import cv2
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np

class DatasetBuilder:

    def __init__(self, config_path: str = 'configs/data_config.yaml'):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)['data']
        
        self.rice2_path = Path(cfg['rice2_path'])
        self.processed_path = Path(cfg['processed_path'])
        self.seed = cfg['random_seed']
        self.split_ratios = cfg['split_ratios']
        
        # Use actual folder names from config
        self.mask_folder = cfg['mask_folder']  # 'mask'
        
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_all_image_ids(self) -> List[str]:
        """
        Scan the cloud folder and collect all image IDs.
        Image ID = filename without extension.
        Example: '0.png' → image_id = '0'
        """
        cloud_dir = self.rice2_path / 'cloud'
        
        # Support both .png and .jpg
        ids = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            ids.extend([f.stem for f in cloud_dir.glob(ext)])
        
        ids = sorted(ids)
        self.logger.info(f"Found {len(ids)} images in RICE2")
        return ids

    def _compute_cloud_coverage(self, image_id: str) -> float:
        """
        Compute what fraction of the image is cloud.
        
        RICE2 mask convention: 255 = cloud, 0 = clear sky / ground
        Some versions use 1 = cloud, so we threshold at 127.
        
        This value is used for STRATIFIED splitting — we ensure 
        train/val/test have similar proportions of light, medium, 
        and heavy cloud images.
        """
        mask_path = self.rice2_path / self.mask_folder / f'{image_id}.png'
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")
        
        # Threshold: anything above 127 is cloud
        binary = (mask > 127).astype(np.float32)
        return float(binary.mean())

    def _stratified_split(
        self,
        image_ids: List[str],
        coverages: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """
        Split images into train/val/test ensuring balanced cloud coverage.
        
        WHY STRATIFIED:
        Without stratification, random splitting might put most
        heavy-cloud images in test and light-cloud in train.
        Your model appears weak on test not because it is weak,
        but because the test set is systematically harder.
        Stratification prevents this.
        
        BINS:
        Light  : coverage < 0.30
        Medium : 0.30 <= coverage < 0.60
        Heavy  : coverage >= 0.60
        
        Each bin is split independently in the correct ratio.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        light  = [id for id in image_ids if coverages[id] < 0.30]
        medium = [id for id in image_ids if 0.30 <= coverages[id] < 0.60]
        heavy  = [id for id in image_ids if coverages[id] >= 0.60]

        self.logger.info(
            f"Coverage bins — Light: {len(light)}, "
            f"Medium: {len(medium)}, Heavy: {len(heavy)}"
        )

        train_ids, val_ids, test_ids = [], [], []

        for bin_ids, name in [(light,'light'),(medium,'medium'),(heavy,'heavy')]:
            random.shuffle(bin_ids)
            n = len(bin_ids)
            n_train = int(n * self.split_ratios['train'])
            n_val   = int(n * self.split_ratios['val'])

            train_ids.extend(bin_ids[:n_train])
            val_ids.extend(bin_ids[n_train : n_train + n_val])
            test_ids.extend(bin_ids[n_train + n_val:])

            self.logger.info(
                f"  {name}: {n_train} train | "
                f"{n_val} val | {n - n_train - n_val} test"
            )

        return {
            'train': sorted(train_ids),
            'val':   sorted(val_ids),
            'test':  sorted(test_ids)
        }

    def build_manifest(self, force_rebuild: bool = False) -> Dict:
        """
        Build and save the split manifest JSON file.
        
        WHAT IS THE MANIFEST:
        A JSON file that records which image IDs belong to train/val/test.
        This file is committed to Git. Anyone who clones the repo
        uses the exact same split. Results are reproducible.
        
        If the manifest already exists and force_rebuild=False,
        it loads the existing one. The split never changes between
        runs unless you explicitly pass force_rebuild=True.
        """
        manifest_path = (
            self.processed_path / 'splits' / 'split_manifest.json'
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        if manifest_path.exists() and not force_rebuild:
            self.logger.info(f"Loading existing manifest: {manifest_path}")
            with open(manifest_path) as f:
                return json.load(f)

        self.logger.info("Building new manifest...")

        image_ids = self._get_all_image_ids()

        # Compute cloud coverage for all images
        self.logger.info("Computing cloud coverage (runs once)...")
        coverages = {}
        for i, image_id in enumerate(image_ids):
            if i % 100 == 0:
                self.logger.info(f"  {i}/{len(image_ids)}")
            coverages[image_id] = self._compute_cloud_coverage(image_id)

        splits = self._stratified_split(image_ids, coverages)

        # Compute per-split statistics
        stats = {}
        for sname, sids in splits.items():
            covs = [coverages[id] for id in sids]
            stats[sname] = {
                'count': len(sids),
                'mean_cloud_coverage': float(np.mean(covs)),
                'std_cloud_coverage':  float(np.std(covs)),
                'min_cloud_coverage':  float(np.min(covs)),
                'max_cloud_coverage':  float(np.max(covs))
            }

        manifest = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'random_seed': self.seed,
                'total_images': len(image_ids),
                'split_ratios': self.split_ratios
            },
            'splits': splits,
            'coverages': coverages,
            'statistics': stats
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(
            f"Manifest saved → "
            f"Train: {stats['train']['count']} | "
            f"Val: {stats['val']['count']} | "
            f"Test: {stats['test']['count']}"
        )
        return manifest