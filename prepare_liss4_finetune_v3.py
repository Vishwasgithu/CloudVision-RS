"""
Run: conda activate cloudremoval
python prepare_liss4_finetune_v3.py

Fixes the "cloudy looks identical to clean" problem: the previous version's cloud
mask was scattered speckles, and Poisson blending on a speckled mask just pulls
every pixel back toward the destination color (no visible cloud transfers).
This version keeps only solid, sizeable connected cloud blobs, and verifies each
blend actually changed the image before accepting it.
"""

import os, json, random, zipfile
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds

JUN_ZIP = r"C:\Users\vishw\Downloads\R2F05JUN2026078508009300049SSANSTUC00GTDD.zip"
JUL_ZIP = r"C:\Users\vishw\Downloads\R2F04JUL2026078914009400050SSANSTUC00GTDA.zip"
OUT_DIR = r"D:\CloudRemoval_Project\data\processed\patches_liss4_realcloud_v3"
OVERLAP_BOUNDS = dict(left=493656.44, bottom=3343565.0, right=561996.44, top=3378255.0)
PATCH = 256
BLACK_FRACTION_SKIP = 0.05
MIN_CLOUD_FRAC, MAX_CLOUD_FRAC = 0.05, 0.65
MIN_COMPONENT_AREA = 400          # discard connected blobs smaller than this (pixels) -- removes speckle noise
MIN_LARGEST_COMPONENT_SHARE = 0.5  # the single largest blob must be at least half the total masked area --
                                    # ensures one real solid cloud, not many scattered small ones
MIN_MEAN_BLEND_DIFF = 22.0         # raised from 6.0 -- that was technically-passing but visually
                                    # invisible; 22/255 (~8.6%) is a genuinely perceptible cloud presence
MAX_BLEND_ATTEMPTS = 6
SPLIT_RATIOS = dict(train=0.8, val=0.1, test=0.1)


def find_internal(zip_path, band_filename):
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.lower().endswith(band_filename.lower()):
                return name
    raise FileNotFoundError(band_filename)


def read_full_res_overlap(zip_path, band_filename):
    internal = find_internal(zip_path, band_filename)
    vsi = "/vsizip/" + zip_path.replace("\\", "/") + "/" + internal
    with rasterio.open(vsi) as src:
        window = from_bounds(**OVERLAP_BOUNDS, transform=src.transform)
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        return src.read(1, window=window).astype(np.float32)


def normalize_u8(arr, lo_pct=2, hi_pct=98):
    valid = arr[arr > 0]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(valid, [lo_pct, hi_pct])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def solid_cloud_mask(rgb_u8):
    """Find bright+spectrally-flat pixels (candidate cloud), then keep only
    sizeable CONNECTED blobs -- discards scattered speckle noise that Poisson
    blending can't meaningfully transfer."""
    f = rgb_u8.astype(np.float32)
    brightness = f.mean(axis=2)
    band_std = f.std(axis=2)
    bright_thresh = np.percentile(brightness, 75)
    flat_thresh = np.percentile(band_std, 35)
    raw_mask = ((brightness > bright_thresh) & (band_std < flat_thresh)).astype(np.uint8)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(raw_mask), False

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
    total_component_area = areas.sum()
    if total_component_area == 0:
        return np.zeros_like(raw_mask), False
    largest_area = areas.max()
    largest_label = 1 + int(np.argmax(areas))

    solid_mask = np.zeros_like(raw_mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            solid_mask[labels == lbl] = 1

    is_solid_enough = (largest_area / total_component_area) >= MIN_LARGEST_COMPONENT_SHARE
    # slight dilation gives the blob a solid core, further helping Poisson blending
    solid_mask = cv2.dilate(solid_mask, np.ones((5, 5), np.uint8), iterations=1)
    return solid_mask, is_solid_enough


def tile_image(rgb_u8, black_skip=True):
    H, W = rgb_u8.shape[:2]
    tiles = []
    for y in range(0, H - PATCH, PATCH):
        for x in range(0, W - PATCH, PATCH):
            tile = rgb_u8[y:y+PATCH, x:x+PATCH]
            if black_skip:
                black_frac = (tile.sum(axis=2) == 0).mean()
                if black_frac > BLACK_FRACTION_SKIP:
                    continue
            tiles.append(tile)
    return tiles


def blend_with_verification(clean_tile, donor_pool):
    """Try up to MAX_BLEND_ATTEMPTS donors, each with both clone modes; accept the
    first blend that actually changed the image meaningfully inside the mask.
    Returns None if all attempts produced a too-weak blend (patch gets skipped)."""
    attempts = random.sample(donor_pool, min(MAX_BLEND_ATTEMPTS, len(donor_pool)))
    best = None  # (mean_diff, cloudy, mask) -- keep the strongest attempt as fallback info
    for donor_img, donor_mask in attempts:
        mask_255 = (donor_mask * 255).astype(np.uint8)
        if mask_255.sum() == 0:
            continue
        x, y, w, h = cv2.boundingRect(mask_255)
        center = (x + w // 2, y + h // 2)
        mask_bool = donor_mask.astype(bool)
        if mask_bool.sum() == 0:
            continue

        for clone_mode in (cv2.MIXED_CLONE, cv2.NORMAL_CLONE):
            try:
                cloudy = cv2.seamlessClone(donor_img, clean_tile, mask_255, center, clone_mode)
            except cv2.error:
                continue
            diff = np.abs(cloudy.astype(np.float32) - clean_tile.astype(np.float32))
            mean_diff_in_mask = diff[mask_bool].mean()
            if best is None or mean_diff_in_mask > best[0]:
                best = (mean_diff_in_mask, cloudy, donor_mask)
            if mean_diff_in_mask >= MIN_MEAN_BLEND_DIFF:
                return cloudy, donor_mask, mean_diff_in_mask
    return None


def main():
    print("Reading JUN (clean source) and JUL (cloud donor source), full resolution...")
    jun_rgb = np.dstack([
        normalize_u8(read_full_res_overlap(JUN_ZIP, "BAND3.tif")),
        normalize_u8(read_full_res_overlap(JUN_ZIP, "BAND2.tif")),
        normalize_u8(read_full_res_overlap(JUN_ZIP, "BAND4.tif")),
    ])
    jul_rgb = np.dstack([
        normalize_u8(read_full_res_overlap(JUL_ZIP, "BAND3.tif")),
        normalize_u8(read_full_res_overlap(JUL_ZIP, "BAND2.tif")),
        normalize_u8(read_full_res_overlap(JUL_ZIP, "BAND4.tif")),
    ])
    print(f"JUN shape: {jun_rgb.shape}, JUL shape: {jul_rgb.shape}")

    clean_tiles = tile_image(jun_rgb)
    print(f"Clean (JUN) tiles: {len(clean_tiles)}")

    jul_tiles = tile_image(jul_rgb)
    donor_pool = []
    rejected_scattered = 0
    for t in jul_tiles:
        m, is_solid = solid_cloud_mask(t)
        frac = m.mean()
        if not (MIN_CLOUD_FRAC <= frac <= MAX_CLOUD_FRAC):
            continue
        if not is_solid:
            rejected_scattered += 1
            continue
        donor_pool.append((t, m))

    print(f"JUL cloud-donor tiles found (solid blobs only): {len(donor_pool)}")
    print(f"Rejected as too-scattered (would've caused the invisible-blend bug): {rejected_scattered}")
    if len(donor_pool) < 20:
        print("WARNING: very few solid donor tiles found. Consider lowering "
              "MIN_LARGEST_COMPONENT_SHARE slightly (e.g. 0.4) and re-running.")
        return

    patches = {}
    patch_ids = []
    skipped_weak = 0
    accepted_diffs = []

    for i, clean_tile in enumerate(clean_tiles):
        result = blend_with_verification(clean_tile, donor_pool)
        if result is None:
            skipped_weak += 1
            continue
        cloudy, mask, mean_diff = result
        accepted_diffs.append(mean_diff)
        pid = f"liss4rc_{i}"
        patches[pid] = (cloudy, clean_tile, mask)
        patch_ids.append(pid)

    print(f"Built {len(patch_ids)} verified real-cloud-blended patches "
          f"(skipped {skipped_weak} where all attempted blends were too weak).")
    if accepted_diffs:
        arr = np.array(accepted_diffs)
        print(f"Accepted blend strength (mean abs diff, 0-255 scale): "
              f"min={arr.min():.1f}, median={np.median(arr):.1f}, max={arr.max():.1f}")
        print("If median is only barely above the 22.0 threshold, most patches are still "
              "borderline -- consider raising MIN_MEAN_BLEND_DIFF further.")

    random.shuffle(patch_ids)
    n = len(patch_ids)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    splits = {"train": patch_ids[:n_train], "val": patch_ids[n_train:n_train+n_val],
              "test": patch_ids[n_train+n_val:]}

    for split, ids in splits.items():
        for sub in ["cloud", "label", "mask"]:
            os.makedirs(os.path.join(OUT_DIR, split, sub), exist_ok=True)
        manifest = {}
        for pid in ids:
            cloudy, clean_tile, mask = patches[pid]
            cv2.imwrite(os.path.join(OUT_DIR, split, "label", f"{pid}.png"),
                        cv2.cvtColor(clean_tile, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUT_DIR, split, "cloud", f"{pid}.png"),
                        cv2.cvtColor(cloudy, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUT_DIR, split, "mask", f"{pid}.png"), mask * 255)
            manifest[pid] = {"cloud_coverage": float(mask.mean())}
        with open(os.path.join(OUT_DIR, split, "patch_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"{split}: {len(ids)} patches -> {os.path.join(OUT_DIR, split)}")

    print(f"\nDone. Verified real-cloud fine-tuning dataset ready at: {OUT_DIR}")


if __name__ == "__main__":
    main()