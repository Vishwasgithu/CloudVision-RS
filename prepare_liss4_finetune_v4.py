"""
prepare_liss4_finetune_v4.py

FIXES THE REAL ROOT CAUSE of the invisible/weak blend problem: the cloud mask
itself was low quality, not the blending method. Proof: v3's own diagnostics
showed "cloud" pixels with Std=75-90 and Min as low as 17 -- a real cloud should
be high-mean, LOW-std. The mask was contaminated with non-cloud material.

Root cause: solid_cloud_mask() compared each tile against ITS OWN percentiles
(top 25% brightest of just that one 256x256 tile). A tile of bright dry soil
can rank "bright" relative to itself without being anywhere near true cloud DN
values. Fix: compute brightness/flatness thresholds ONCE across the entire JUL
scene, and add a hard purity gate (reject any mask whose own internal std is
too high) so we only ever donate real, spectrally-clean cloud material.

Keeps the alpha-blend method and diagnostics from the previous version --
those parts were fine, only the mask detection needed fixing.
"""

import os, json, random, zipfile
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds

JUN_ZIP = r"C:\Users\vishw\Downloads\R2F05JUN2026078508009300049SSANSTUC00GTDD.zip"
JUL_ZIP = r"C:\Users\vishw\Downloads\R2F04JUL2026078914009400050SSANSTUC00GTDA.zip"
OUT_DIR = r"D:\CloudRemoval_Project\data\processed\patches_liss4_v4"
OVERLAP_BOUNDS = dict(left=493656.44, bottom=3343565.0, right=561996.44, top=3378255.0)

PATCH = 256
BLACK_FRACTION_SKIP = 0.05
MIN_CLOUD_FRAC, MAX_CLOUD_FRAC = 0.05, 0.65
MIN_COMPONENT_AREA = 400
MIN_LARGEST_COMPONENT_SHARE = 0.5
MAX_MASK_INTERNAL_STD = 20.0     # NEW: hard purity gate -- if pixels inside the accepted
                                  # mask still vary this much, it's contaminated, reject the tile
SCENE_BRIGHT_PERCENTILE = 92     # NEW: computed ONCE across the whole JUL scene, not per-tile
SCENE_FLAT_PERCENTILE = 25       # NEW: same -- absolute, scene-wide, not tile-relative
MIN_MEAN_BLEND_DIFF = 25.0
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


def compute_scene_wide_thresholds(jul_rgb):
    """Computed ONCE across the entire scene -- this is the actual fix.
    A pixel must be bright/flat relative to the WHOLE scene to count as cloud,
    not just relative to whatever happens to be in its own small tile."""
    f = jul_rgb.astype(np.float32)
    brightness = f.mean(axis=2)
    band_std = f.std(axis=2)
    valid = brightness > 0
    bright_thresh = np.percentile(brightness[valid], SCENE_BRIGHT_PERCENTILE)
    flat_thresh = np.percentile(band_std[valid], SCENE_FLAT_PERCENTILE)
    print(f"Scene-wide cloud thresholds: brightness > {bright_thresh:.1f}, band_std < {flat_thresh:.1f}")
    return bright_thresh, flat_thresh


def solid_cloud_mask(rgb_u8, bright_thresh, flat_thresh):
    f = rgb_u8.astype(np.float32)
    brightness = f.mean(axis=2)
    band_std = f.std(axis=2)
    raw_mask = ((brightness > bright_thresh) & (band_std < flat_thresh)).astype(np.uint8)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(raw_mask, dtype=np.float32), False

    areas = stats[1:, cv2.CC_STAT_AREA]
    total_area = areas.sum()
    if total_area == 0:
        return np.zeros_like(raw_mask, dtype=np.float32), False
    largest_area = areas.max()

    solid_mask = np.zeros_like(raw_mask, dtype=np.float32)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            solid_mask[labels == lbl] = 1.0

    is_solid_enough = (largest_area / total_area) >= MIN_LARGEST_COMPONENT_SHARE

    # NEW: purity gate -- measure how much the ACTUAL pixels under this mask vary.
    # A real cloud should be near-uniformly bright; high internal std means the
    # blob still swallowed non-cloud material via the closing/dilation steps.
    mask_bool = solid_mask > 0.5
    if mask_bool.sum() > 0:
        internal_std = f[mask_bool].std()
        is_pure_enough = internal_std <= MAX_MASK_INTERNAL_STD
    else:
        is_pure_enough = False

    solid_mask = cv2.dilate(solid_mask, np.ones((5, 5), np.uint8), iterations=1).astype(np.float32)
    return solid_mask, (is_solid_enough and is_pure_enough)


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


def blend_alpha_radiative(clean_tile, donor_pool):
    attempts = random.sample(donor_pool, min(6, len(donor_pool)))
    G = clean_tile.astype(np.float32)
    for C_tile, mask in attempts:
        M_feathered = cv2.GaussianBlur(mask, (5, 5), 0)[..., np.newaxis]
        alpha = random.uniform(0.6, 0.95)
        C = C_tile.astype(np.float32)
        I_cloudy = np.clip((1.0 - alpha * M_feathered) * G + (alpha * M_feathered) * C, 0, 255).astype(np.uint8)
        mask_bool = mask > 0.5
        if not np.any(mask_bool):
            continue
        diff = np.abs(I_cloudy.astype(np.float32) - G)
        mean_diff = diff[mask_bool].mean()
        if mean_diff >= MIN_MEAN_BLEND_DIFF:
            return I_cloudy, mask, mean_diff
    return None


def main():
    print("Reading JUN (clean) and JUL (cloud donor), full resolution...")
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

    bright_thresh, flat_thresh = compute_scene_wide_thresholds(jul_rgb)

    clean_tiles = tile_image(jun_rgb)
    jul_tiles = tile_image(jul_rgb)

    donor_pool = []
    rejected_impure = 0
    for t in jul_tiles:
        m, is_valid = solid_cloud_mask(t, bright_thresh, flat_thresh)
        frac = m.mean()
        if not (MIN_CLOUD_FRAC <= frac <= MAX_CLOUD_FRAC):
            continue
        if not is_valid:
            rejected_impure += 1
            continue
        donor_pool.append((t, m))

    print(f"Clean (JUN) tiles: {len(clean_tiles)}")
    print(f"Valid PURE cloud donor tiles: {len(donor_pool)} (rejected {rejected_impure} as contaminated/scattered)")
    if len(donor_pool) < 15:
        print("WARNING: very few pure donors. Try SCENE_BRIGHT_PERCENTILE=88 or MAX_MASK_INTERNAL_STD=25.")
        return

    patches, patch_ids = {}, []
    skipped_weak = 0
    accepted_diffs = []
    for i, clean_tile in enumerate(clean_tiles):
        result = blend_alpha_radiative(clean_tile, donor_pool)
        if result is None:
            skipped_weak += 1
            continue
        cloudy, mask, mean_diff = result
        accepted_diffs.append(mean_diff)
        pid = f"liss4v4_{i}"
        patches[pid] = (cloudy, clean_tile, mask)
        patch_ids.append(pid)

    print(f"\nBuilt {len(patch_ids)} patches (skipped {skipped_weak} weak blends).")
    if accepted_diffs:
        arr = np.array(accepted_diffs)
        print(f"Blend strength: min={arr.min():.1f}, median={np.median(arr):.1f}, max={arr.max():.1f}")

    random.shuffle(patch_ids)
    n = len(patch_ids)
    n_train, n_val = int(n*0.8), int(n*0.1)
    splits = {"train": patch_ids[:n_train], "val": patch_ids[n_train:n_train+n_val],
              "test": patch_ids[n_train+n_val:]}

    for split, ids in splits.items():
        for sub in ["cloud", "label", "mask"]:
            os.makedirs(os.path.join(OUT_DIR, split, sub), exist_ok=True)
        manifest = {}
        for pid in ids:
            cloudy, clean_tile, mask = patches[pid]
            cv2.imwrite(os.path.join(OUT_DIR, split, "label", f"{pid}.png"), cv2.cvtColor(clean_tile, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUT_DIR, split, "cloud", f"{pid}.png"), cv2.cvtColor(cloudy, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUT_DIR, split, "mask", f"{pid}.png"), (mask*255).astype(np.uint8))
            manifest[pid] = {"cloud_coverage": float(mask.mean())}
        with open(os.path.join(OUT_DIR, split, "patch_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"{split}: {len(ids)} -> {os.path.join(OUT_DIR, split)}")

    print(f"\nDone: {OUT_DIR}")


if __name__ == "__main__":
    main()
