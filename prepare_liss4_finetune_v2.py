"""
Run: conda activate cloudremoval
python prepare_liss4_finetune_v2.py

Replaces the flat-gray synthetic clouds with REAL cloud pixels cut from the
JUL scene (confirmed cloudy) and Poisson-blended (cv2.seamlessClone) onto
JUN's clean tiles -- no flat color, no hard seam, real cloud texture/brightness.
"""

import os, json, random, zipfile
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds

JUN_ZIP = r"C:\Users\vishw\Downloads\R2F05JUN2026078508009300049SSANSTUC00GTDD.zip"
JUL_ZIP = r"C:\Users\vishw\Downloads\R2F04JUL2026078914009400050SSANSTUC00GTDA.zip"
OUT_DIR = r"D:\CloudRemoval_Project\data\processed\patches_liss4_realcloud"
OVERLAP_BOUNDS = dict(left=493656.44, bottom=3343565.0, right=561996.44, top=3378255.0)
PATCH = 256
BLACK_FRACTION_SKIP = 0.05
MIN_CLOUD_FRAC, MAX_CLOUD_FRAC = 0.05, 0.65  # donor tile must have a usable, not-total, cloud area
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


def heuristic_cloud_mask(rgb_u8):
    """Clouds: bright AND spectrally flat (low variation across bands).
    Real vegetation/soil/water have much more inter-band contrast."""
    f = rgb_u8.astype(np.float32)
    brightness = f.mean(axis=2)
    band_std = f.std(axis=2)  # low when R,G,B(NIR) are all similar -- "whiteness"
    bright_thresh = np.percentile(brightness, 75)
    flat_thresh = np.percentile(band_std, 35)
    mask = ((brightness > bright_thresh) & (band_std < flat_thresh)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


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


def main():
    print("Reading JUN (clean source) and JUL (cloud donor source), full resolution...")
    jun_rgb = np.dstack([
        normalize_u8(read_full_res_overlap(JUN_ZIP, "BAND3.tif")),   # R <- Red
        normalize_u8(read_full_res_overlap(JUN_ZIP, "BAND2.tif")),   # G <- Green
        normalize_u8(read_full_res_overlap(JUN_ZIP, "BAND4.tif")),   # B <- NIR
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
    for t in jul_tiles:
        m = heuristic_cloud_mask(t)
        frac = m.mean()
        if MIN_CLOUD_FRAC <= frac <= MAX_CLOUD_FRAC:
            donor_pool.append((t, m))
    print(f"JUL cloud-donor tiles found (usable range {MIN_CLOUD_FRAC}-{MAX_CLOUD_FRAC} coverage): {len(donor_pool)}")

    if len(donor_pool) < 20:
        print("WARNING: very few donor tiles found -- heuristic thresholds may need loosening. "
              "Proceeding anyway, but check the ratio above.")

    patches = {}
    patch_ids = []
    center = (PATCH // 2, PATCH // 2)

    for i, clean_tile in enumerate(clean_tiles):
        donor_img, donor_mask = random.choice(donor_pool)
        mask_255 = (donor_mask * 255).astype(np.uint8)
        if mask_255.sum() == 0:
            continue
        try:
            cloudy = cv2.seamlessClone(donor_img, clean_tile, mask_255, center, cv2.NORMAL_CLONE)
        except cv2.error:
            continue  # skip rare seamlessClone failures (e.g. mask touching border)

        pid = f"liss4rc_{i}"
        patches[pid] = (cloudy, clean_tile, donor_mask)
        patch_ids.append(pid)

    print(f"Built {len(patch_ids)} real-cloud-blended patches.")

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

    print(f"\nDone. Real-cloud fine-tuning dataset ready at: {OUT_DIR}")


if __name__ == "__main__":
    main()
