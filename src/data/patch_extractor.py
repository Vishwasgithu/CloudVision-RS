import cv2
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


class PatchExtractor:

    def __init__(self, config_path: str = 'configs/data_config.yaml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)['data']

        self.patch_size       = cfg['patch_size']
        self.overlap          = cfg['overlap']
        self.min_coverage     = cfg['min_cloud_coverage']
        self.max_coverage     = cfg['max_cloud_coverage']
        self.cloudy_folder    = cfg['cloudy_folder']
        self.cloudfree_folder = cfg['cloudfree_folder']
        self.mask_folder      = cfg['mask_folder']
        self.stride           = int(self.patch_size * (1 - self.overlap))

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _read_triple(
        self,
        rice2_path: Path,
        image_id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read cloudy, cloudfree, and mask for one image ID.

        OpenCV reads BGR by default.
        We convert images to RGB immediately.
        Mask is grayscale — no colour conversion needed.
        """
        def read_rgb(path):
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Cannot read: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        def read_gray(path):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Cannot read: {path}")
            return img

        cloudy    = read_rgb(
            rice2_path / self.cloudy_folder / f'{image_id}.png'
        )
        cloudfree = read_rgb(
            rice2_path / self.cloudfree_folder / f'{image_id}.png'
        )
        mask      = read_gray(
            rice2_path / self.mask_folder / f'{image_id}.png'
        )

        return cloudy, cloudfree, mask

    def _pad_to_fit(
        self,
        arr: np.ndarray,
        is_mask: bool = False
    ) -> np.ndarray:
        """
        Pad image or mask so sliding window covers the full image.

        WHY:
        With stride=128 on a 512x512 image, the last window
        starts at position 384 and ends at 640 — out of bounds.
        Reflect padding mirrors pixels at the boundary.
        Better than zero-padding because it produces realistic
        pixel values the model won't treat as anomalies.
        """
        H, W = arr.shape[:2]

        pad_h = (
            self.stride - (H - self.patch_size) % self.stride
        ) % self.stride
        pad_w = (
            self.stride - (W - self.patch_size) % self.stride
        ) % self.stride

        if pad_h == 0 and pad_w == 0:
            return arr

        if is_mask:
            return np.pad(
                arr,
                ((0, pad_h), (0, pad_w)),
                mode='reflect'
            )
        else:
            return np.pad(
                arr,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='reflect'
            )

    def extract_from_image(
        self,
        rice2_path: Path,
        image_id: str
    ) -> List[Dict]:
        """
        Extract all valid patches from one image triple.

        COVERAGE FILTER:
        < min_coverage (5%):  skip — too little cloud, useless for training
        > max_coverage (95%): skip — too much cloud, no surface context

        Returns list of dicts with patch arrays and metadata.
        """
        cloudy, cloudfree, mask = self._read_triple(rice2_path, image_id)

        cloudy    = self._pad_to_fit(cloudy,    is_mask=False)
        cloudfree = self._pad_to_fit(cloudfree, is_mask=False)
        mask      = self._pad_to_fit(mask,      is_mask=True)

        H, W = cloudy.shape[:2]
        patches = []
        skipped_low  = 0
        skipped_high = 0

        for r in range(0, H - self.patch_size + 1, self.stride):
            for c in range(0, W - self.patch_size + 1, self.stride):

                p_cloudy    = cloudy[r:r+self.patch_size, c:c+self.patch_size]
                p_cloudfree = cloudfree[r:r+self.patch_size, c:c+self.patch_size]
                p_mask      = mask[r:r+self.patch_size, c:c+self.patch_size]

                coverage = float((p_mask > 127).mean())

                if coverage < self.min_coverage:
                    skipped_low += 1
                    continue
                if coverage > self.max_coverage:
                    skipped_high += 1
                    continue

                patch_id = f"{image_id}_r{r:04d}_c{c:04d}"

                patches.append({
                    'patch_id':       patch_id,
                    'source_id':      image_id,
                    'row':            r,
                    'col':            c,
                    'cloud_coverage': coverage,
                    'cloudy':         p_cloudy,
                    'cloudfree':      p_cloudfree,
                    'mask':           p_mask
                })

        self.logger.debug(
            f"{image_id}: {len(patches)} kept | "
            f"{skipped_low} too clear | {skipped_high} too cloudy"
        )
        return patches

    def extract_and_save_split(
        self,
        rice2_path: Path,
        output_path: Path,
        image_ids: List[str],
        split_name: str
    ) -> int:
        """
        Extract and save all patches for one split to disk.

        FILE STRUCTURE:
        output_path/split_name/cloud/imageid_r0000_c0000.png
        output_path/split_name/label/imageid_r0000_c0000.png
        output_path/split_name/mask/imageid_r0000_c0000.png

        Also saves patch_manifest.json recording source image,
        position, and cloud coverage for every patch.

        WHY PNG NOT JPEG:
        PNG is lossless. JPEG compression alters pixel values.
        A mask saved as JPEG will have intermediate values (not 0/255).
        Binarisation at load time would then be inconsistent.
        """
        out = output_path / split_name
        for folder in ['cloud', 'label', 'mask']:
            (out / folder).mkdir(parents=True, exist_ok=True)

        patch_manifest = {}
        total = 0

        for i, image_id in enumerate(image_ids):
            if i % 25 == 0:
                self.logger.info(
                    f"[{split_name}] {i}/{len(image_ids)} images | "
                    f"{total} patches so far"
                )

            try:
                patches = self.extract_from_image(rice2_path, image_id)
            except ValueError as e:
                self.logger.error(f"Skipping {image_id}: {e}")
                continue

            for p in patches:
                pid = p['patch_id']

                cv2.imwrite(
                    str(out / 'cloud' / f'{pid}.png'),
                    cv2.cvtColor(p['cloudy'], cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    str(out / 'label' / f'{pid}.png'),
                    cv2.cvtColor(p['cloudfree'], cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    str(out / 'mask' / f'{pid}.png'),
                    p['mask']
                )

                patch_manifest[pid] = {
                    'source_id':      p['source_id'],
                    'row':            p['row'],
                    'col':            p['col'],
                    'cloud_coverage': p['cloud_coverage']
                }
                total += 1

        manifest_path = out / 'patch_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(patch_manifest, f, indent=2)

        self.logger.info(
            f"[{split_name}] Complete: {total} patches from "
            f"{len(image_ids)} images"
        )
        return total