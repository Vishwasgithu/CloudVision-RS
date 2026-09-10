import json
import os
import random
import zipfile
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

# --- PATH CONFIGURATION ---
JUN_ZIP = r"C:\Users\vishw\Downloads\R2F05JUN2026078508009300049SSANSTUC00GTDD.zip"
JUL_ZIP = r"C:\Users\vishw\Downloads\R2F04JUL2026078914009400050SSANSTUC00GTDA.zip"

OUTPUT_DIR = r"D:\CloudRemoval_Project\data\processed\patches_liss4_realcloud"

PATCH_SIZE = 256
STRIDE = 128


def get_band_paths(zip_path):
  """Finds BAND2, BAND3, BAND4 paths inside the zip."""
  with zipfile.ZipFile(zip_path, "r") as z:
    files = z.namelist()

  b2 = next(f for f in files if "BAND2" in f and f.endswith(".tif"))
  b3 = next(f for f in files if "BAND3" in f and f.endswith(".tif"))
  b4 = next(f for f in files if "BAND4" in f and f.endswith(".tif"))

  vsi = f"/vsizip/{zip_path}/"
  return vsi + b4, vsi + b3, vsi + b2  # NIR, Red, Green


def read_tile_8bit(src_nir, src_r, src_g, y, x, patch_size):
  """Reads a small 256x256 tile directly from disk into 8-bit FCC format."""
  win = Window(x, y, patch_size, patch_size)

  nir = src_nir.read(1, window=win)
  r = src_r.read(1, window=win)
  g = src_g.read(1, window=win)

  # Check if window is empty or out of bounds
  if nir.shape[0] != patch_size or nir.shape[1] != patch_size:
    return None

  fcc = np.stack([nir, r, g], axis=-1).astype(np.float32)

  # Lightweight local tile normalization (0-255)
  fcc_min, fcc_max = np.percentile(fcc, (2, 98))
  if fcc_max - fcc_min == 0:
    return None

  fcc_8bit = np.clip((fcc - fcc_min) / (fcc_max - fcc_min) * 255.0, 0, 255)
  return fcc_8bit.astype(np.uint8)


def generate_dataset():
  print("Initializing streaming patch generation from GeoTIFF ZIPs...")

  jun_b4, jun_b3, jun_b2 = get_band_paths(JUN_ZIP)
  jul_b4, jul_b3, jul_b2 = get_band_paths(JUL_ZIP)

  with (
      rasterio.open(jun_b4) as jun_nir,
      rasterio.open(jun_b3) as jun_r,
      rasterio.open(jun_b2) as jun_g,
      rasterio.open(jul_b4) as jul_nir,
      rasterio.open(jul_b3) as jul_r,
      rasterio.open(jul_b2) as jul_g,
  ):

    h, w = jun_nir.height, jun_nir.width
    print(f"Scene Dimensions: {h} x {w}")

    # Step 1: Collect cloud donor patches from July scene
    print("Collecting cloud donor patches from July scene...")
    cloud_donors = []

    for y in range(0, h - PATCH_SIZE, STRIDE * 2):  # Subsampled for speed
      for x in range(0, w - PATCH_SIZE, STRIDE * 2):
        jul_patch = read_tile_8bit(
            jul_nir, jul_r, jul_g, y, x, PATCH_SIZE
        )
        if jul_patch is None:
          continue

        gray = cv2.cvtColor(jul_patch, cv2.COLOR_BGR2GRAY)
        _, bin_mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

        coverage = np.sum(bin_mask > 0) / (PATCH_SIZE * PATCH_SIZE)
        if 0.05 <= coverage <= 0.65:
          cloud_donors.append((jul_patch, bin_mask))

    print(f"JUL cloud-donor tiles collected: {len(cloud_donors)}")

    if not cloud_donors:
      raise ValueError(
          "No valid cloud donor tiles found! Adjust threshold or coverage"
          " criteria."
      )

    # Step 2: Extract June clean patches & apply aligned cloud blending
    print("Generating aligned synthetic patches from June scene...")
    patches = []
    patch_id = 0

    for y in range(0, h - PATCH_SIZE, STRIDE):
      for x in range(0, w - PATCH_SIZE, STRIDE):
        clean_patch = read_tile_8bit(
            jun_nir, jun_r, jun_g, y, x, PATCH_SIZE
        )
        if clean_patch is None:
          continue

        # Pick random cloud donor
        jul_donor, mask = random.choice(cloud_donors)

        # Softened Alpha Map
        alpha = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=-1)

        # Spatial Aligned Blending
        cloudy_synthetic = (
            clean_patch.astype(np.float32) * (1.0 - alpha)
            + jul_donor.astype(np.float32) * alpha
        ).astype(np.uint8)

        mask_binary = (alpha * 255).astype(np.uint8)

        patches.append({
            "id": f"liss4rc_{patch_id:04d}",
            "cloudy": cloudy_synthetic,
            "clean": clean_patch,
            "mask": mask_binary,
            "coverage": float(np.mean(alpha)),
        })
        patch_id += 1

    # Step 3: Train / Val / Test Split & Save
    random.shuffle(patches)
    total = len(patches)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)

    splits = {
        "train": patches[:n_train],
        "val": patches[n_train : n_train + n_val],
        "test": patches[n_train + n_val :],
    }

    for split, split_patches in splits.items():
      manifest = {}
      dir_cloudy = os.path.join(OUTPUT_DIR, split, "cloud")
      dir_label = os.path.join(OUTPUT_DIR, split, "label")
      dir_mask = os.path.join(OUTPUT_DIR, split, "mask")

      os.makedirs(dir_cloudy, exist_ok=True)
      os.makedirs(dir_label, exist_ok=True)
      os.makedirs(dir_mask, exist_ok=True)

      for item in split_patches:
        pid = item["id"]

        cv2.imwrite(
            os.path.join(dir_cloudy, f"{pid}.png"),
            cv2.cvtColor(item["cloudy"], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(dir_label, f"{pid}.png"),
            cv2.cvtColor(item["clean"], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(dir_mask, f"{pid}.png"),
            cv2.cvtColor(item["mask"], cv2.COLOR_RGB2BGR),
        )

        manifest[pid] = {"cloud_coverage": item["coverage"]}

      manifest_path = os.path.join(OUTPUT_DIR, split, "patch_manifest.json")
      with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

      print(
          f"{split}: {len(split_patches)} patches saved to"
          f" {os.path.join(OUTPUT_DIR, split)}"
      )

    print(f"\nDone. Dataset successfully generated at: {OUTPUT_DIR}")


if __name__ == "__main__":
  generate_dataset()