"""
build_dataset.py

Run this once after downloading RICE2:
    python scripts/build_dataset.py

What it does:
1. Builds image-level split manifest
2. Extracts 256x256 patches for all splits
3. Validates the pipeline
4. Prints a summary

Expected output at the end:
    ✅ ALL CHECKS PASSED
    Phase 1 complete. Ready for Phase 2.
"""

import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import json
import logging

from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_builder import DatasetBuilder
from src.data.patch_extractor import PatchExtractor
from src.data.dataset import create_dataloaders

import yaml


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)-25s | %(message)s',
        handlers=[
            logging.FileHandler('outputs/logs/phase1.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('build_dataset')

    Path('outputs/logs').mkdir(parents=True, exist_ok=True)

    config_path = 'configs/data_config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)['data']

    rice2_path   = Path(cfg['rice2_path'])
    patches_path = Path(cfg['processed_path']) / 'patches'

    # ── Step 1: Build manifest ──────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1: Building split manifest")
    logger.info("=" * 55)

    builder  = DatasetBuilder(config_path)
    manifest = builder.build_manifest(force_rebuild=False)

    for split in ['train', 'val', 'test']:
        s = manifest['statistics'][split]
        logger.info(
            f"  {split:5s}: {s['count']:3d} images | "
            f"mean cloud coverage: {s['mean_cloud_coverage']:.3f}"
        )

    # ── Step 2: Extract patches ─────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 2: Extracting patches")
    logger.info("=" * 55)

    extractor = PatchExtractor(config_path)
    total_patches = {}

    for split in ['train', 'val', 'test']:
        image_ids = manifest['splits'][split]
        count = extractor.extract_and_save_split(
            rice2_path=rice2_path,
            output_path=patches_path,
            image_ids=image_ids,
            split_name=split
        )
        total_patches[split] = count

    # ── Step 3: Validate ────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 3: Validation")
    logger.info("=" * 55)

    errors = []

    # Check 1: no data leakage at image level
    train_set = set(manifest['splits']['train'])
    val_set   = set(manifest['splits']['val'])
    test_set  = set(manifest['splits']['test'])

    if train_set & test_set:
        errors.append(
            f"LEAKAGE: {len(train_set & test_set)} images in train AND test"
        )
    if train_set & val_set:
        errors.append(
            f"LEAKAGE: {len(train_set & val_set)} images in train AND val"
        )
    if not errors:
        logger.info("  ✓ No image-level data leakage")

    # Check 2: patch manifest files exist
    for split in ['train', 'val', 'test']:
        mp = patches_path / split / 'patch_manifest.json'
        if not mp.exists():
            errors.append(f"Missing patch manifest for {split}")
        else:
            logger.info(f"  ✓ {split} patch manifest exists")

    # Check 3: DataLoader loads correctly
    try:
        seg_loaders = create_dataloaders(
            str(patches_path), mode='segmentation', config_path=config_path
        )
        batch = next(iter(seg_loaders['train']))

        img_min = batch['image'].min().item()
        img_max = batch['image'].max().item()

        if img_min < -0.01 or img_max > 1.01:
            errors.append(
                f"Normalisation error: seg image range "
                f"[{img_min:.3f}, {img_max:.3f}], expected [0,1]"
            )
        else:
            logger.info(
                f"  ✓ Segmentation normalisation: "
                f"[{img_min:.4f}, {img_max:.4f}]"
            )

        mask_vals = batch['mask'].unique().tolist()
        if any(v not in [0.0, 1.0] for v in mask_vals):
            errors.append(f"Mask values not binary: {mask_vals}")
        else:
            logger.info(f"  ✓ Mask values binary: {mask_vals}")

    except Exception as e:
        errors.append(f"DataLoader failed: {e}")

    try:
        gan_loaders = create_dataloaders(
            str(patches_path), mode='removal', config_path=config_path
        )
        batch = next(iter(gan_loaders['train']))

        c_min = batch['cloudy'].min().item()
        c_max = batch['cloudy'].max().item()

        if c_min < -1.01 or c_max > 1.01:
            errors.append(
                f"GAN normalisation error: "
                f"[{c_min:.3f}, {c_max:.3f}], expected [-1,1]"
            )
        else:
            logger.info(
                f"  ✓ GAN normalisation: [{c_min:.4f}, {c_max:.4f}]"
            )

    except Exception as e:
        errors.append(f"GAN DataLoader failed: {e}")

    # ── Summary ─────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("PHASE 1 SUMMARY")
    logger.info("=" * 55)
    logger.info(f"  Train patches : {total_patches.get('train', 0)}")
    logger.info(f"  Val patches   : {total_patches.get('val', 0)}")
    logger.info(f"  Test patches  : {total_patches.get('test', 0)}")
    logger.info(
        f"  Total patches : "
        f"{sum(total_patches.values())}"
    )

    if errors:
        logger.error("PHASE 1 FAILED:")
        for e in errors:
            logger.error(f"  ✗ {e}")
        sys.exit(1)
    else:
        logger.info("✅ ALL CHECKS PASSED")
        logger.info("Phase 1 complete. Ready for Phase 2.")


if __name__ == '__main__':
    main()