"""
Run: conda activate cloudremoval, then:
python prepare_liss4_finetune.py

Builds a new patches_liss4/ dataset (same train/cloud/label/mask structure your
dataset.py already reads) from real JUN-scene clean pixels + real RICE2 cloud
shapes overlaid synthetically. Band mapping: model R<-Real Red, G<-Real Green,
B<-Real NIR (substituting Blue, per our earlier decision).
"""

import os
import json
import random
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds

# ---- CONFIG ----
ZIP_PATH = r"C:\Users\vishw\Downloads\R2F05JUN2026078508009300049SSANSTUC00GTDD.zip"
RICE2_PATCHES_DIR = r"D:\CloudRemoval_Project\data\processed\patches"
OUT_DIR = r"D:\CloudRemoval_Project\data\processed\patches_liss4"
OVERLAP_BOUNDS = dict(left=493656.44, bottom=3343565.0, right=561996.44, top=3378255.0)
PATCH = 256
BLACK_FRACTION_SKIP = 0.05  # skip patch if >5% pixels are no-data (black)
SPLIT_RATIOS = dict(train=0.8, val=0.1, test=0.1)


def find_internal(zip_path, band_filename):
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.lower().endswith(band_filename.lower()):
                return name
    raise FileNotFoundError(band_filename)


def read_full_res_band(zip_path, band_filename):
    internal = find_internal(zip_path, band_filename)
    vsi = "/vsizip/" + zip_path.replace("\\", "/") + "/" + internal
    with rasterio.open(vsi) as src:
        window = from_bounds(**OVERLAP_BOUNDS, transform=src.transform)
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=window)  # full resolution, no downsampling
    return data.astype(np.float32)


def normalize_u8(arr, lo_pct=2, hi_pct=98):
    lo, hi = np.percentile(arr[arr > 0], [lo_pct, hi_pct])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def load_rice2_masks():
    mask_dir = os.path.join(RICE2_PATCHES_DIR, "train", "mask")
    files = [f for f in os.listdir(mask_dir) if f.lower().endswith(".png")]
    print(f"Found {len(files)} RICE2 masks to sample cloud shapes from.")
    return [os.path.join(mask_dir, f) for f in files]


def synth_cloud(clean_patch, mask_binary):
    """Overlay a synthetic cloud where mask==1, with soft edges + brightness noise."""
    mask_f = mask_binary.astype(np.float32)
    mask_f = cv2.GaussianBlur(mask_f, (9, 9), 0)  # soften edges, avoid hard cutout look
    cloud_color = 200 + np.random.randint(-15, 15)  # bright grayish-white
    noise = np.random.normal(0, 8, clean_patch.shape).astype(np.float32)
    cloudy = clean_patch.astype(np.float32) * (1 - mask_f[..., None]) + \
             (cloud_color + noise) * mask_f[..., None]
    return np.clip(cloudy, 0, 255).astype(np.uint8)


def main():
    print("Reading full-res clean crop from JUN scene (this may take a minute)...")
    g = read_full_res_band(ZIP_PATH, "BAND2.tif")
    r = read_full_res_band(ZIP_PATH, "BAND3.tif")
    nir = read_full_res_band(ZIP_PATH, "BAND4.tif")
    print(f"Full-res crop shape: {r.shape}")

    r_n, g_n, nir_n = normalize_u8(r), normalize_u8(g), normalize_u8(nir)
    # Model channel order R,G,B <- Red, Green, NIR
    clean_rgb = np.dstack([r_n, g_n, nir_n])

    rice2_masks = load_rice2_masks()

    H, W = clean_rgb.shape[:2]
    patch_ids = []
    patches = {}  # pid -> (clean_patch, mask_binary)

    for y in range(0, H - PATCH, PATCH):
        for x in range(0, W - PATCH, PATCH):
            tile = clean_rgb[y:y+PATCH, x:x+PATCH]
            black_frac = (tile.sum(axis=2) == 0).mean()
            if black_frac > BLACK_FRACTION_SKIP:
                continue  # skip no-data corner tiles

            mask_path = random.choice(rice2_masks)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (PATCH, PATCH))
            mask_binary = (mask_img > 127).astype(np.uint8)

            pid = f"liss4_{y}_{x}"
            patches[pid] = (tile, mask_binary)
            patch_ids.append(pid)

    print(f"Built {len(patch_ids)} usable clean 256x256 patches (no-data tiles skipped).")

    random.shuffle(patch_ids)
    n = len(patch_ids)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    splits = {
        "train": patch_ids[:n_train],
        "val": patch_ids[n_train:n_train+n_val],
        "test": patch_ids[n_train+n_val:]
    }

    for split, ids in splits.items():
        for sub in ["cloud", "label", "mask"]:
            os.makedirs(os.path.join(OUT_DIR, split, sub), exist_ok=True)

        manifest = {}
        for pid in ids:
            clean_patch, mask_binary = patches[pid]
            cloudy_patch = synth_cloud(clean_patch, mask_binary)
            coverage = float(mask_binary.mean())

            cv2.imwrite(os.path.join(OUT_DIR, split, "label", f"{pid}.png"),
                        cv2.cvtColor(clean_patch, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUT_DIR, split, "cloud", f"{pid}.png"),
                        cv2.cvtColor(cloudy_patch, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUT_DIR, split, "mask", f"{pid}.png"),
                        mask_binary * 255)
            manifest[pid] = {"cloud_coverage": coverage}

        with open(os.path.join(OUT_DIR, split, "patch_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"{split}: {len(ids)} patches written to {os.path.join(OUT_DIR, split)}")

    print(f"\nDone. Fine-tuning dataset ready at: {OUT_DIR}")
    print("Point your training script's patches_dir to this folder to fine-tune.")


if __name__ == "__main__":
    main()
