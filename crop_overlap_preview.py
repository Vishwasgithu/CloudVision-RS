"""
Run locally: conda activate cloudremoval
Requires rasterio (pip install rasterio --break-system-packages if missing) and pillow/numpy.

Usage:
    python crop_overlap_preview.py "C:\\Users\\vishw\\Downloads\\R2F05JUN2026078508009300049SSANSTUC00GTDD.zip" JUN
    python crop_overlap_preview.py "C:\\Users\\vishw\\Downloads\\R2F04JUL2026078914009400050SSANSTUC00GTDA.zip" JUL

Reads ONLY the overlapping window directly from inside the zip (no full extraction),
builds a NIR-Red-Green false color preview (clouds appear bright white),
computes an NDVI preview, and prints a rough cloud-fraction estimate.
"""

import sys
import os
import zipfile
import numpy as np

try:
    import rasterio
    from rasterio.windows import from_bounds
except ImportError:
    print("Missing rasterio. Run: pip install rasterio --break-system-packages")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Missing pillow. Run: pip install pillow --break-system-packages")
    sys.exit(1)

# Overlap bounds computed from both scenes' BAND_META.txt (UTM Zone 43N, both scenes)
OVERLAP_BOUNDS = dict(left=493656.44, bottom=3343565.0, right=561996.44, top=3378255.0)
TARGET_WIDTH_PX = 1400  # preview size, keeps PNG small for upload


def find_internal_name(zip_path, band_filename):
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.lower().endswith(band_filename.lower()):
                return name
    raise FileNotFoundError(f"{band_filename} not found in {zip_path}")


def read_band_window(zip_path, band_filename):
    internal = find_internal_name(zip_path, band_filename)
    vsi_path = "/vsizip/" + zip_path.replace("\\", "/") + "/" + internal
    with rasterio.open(vsi_path) as src:
        window = from_bounds(**OVERLAP_BOUNDS, transform=src.transform)
        # clip window to what's actually available in this scene
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=window, out_shape=(
            int(TARGET_WIDTH_PX * window.height / window.width), TARGET_WIDTH_PX
        ))
    return data.astype(np.float32)


def normalize_u8(arr, lo_pct=2, hi_pct=98):
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def main(zip_path, label):
    print(f"\nReading overlap window from {label} scene...")
    g = read_band_window(zip_path, "BAND2.tif")
    r = read_band_window(zip_path, "BAND3.tif")
    nir = read_band_window(zip_path, "BAND4.tif")

    print(f"Window shape: {g.shape}, DN ranges -> G:[{g.min():.0f},{g.max():.0f}] "
          f"R:[{r.min():.0f},{r.max():.0f}] NIR:[{nir.min():.0f},{nir.max():.0f}]")

    # False color composite: NIR-R-G (standard vegetation false color)
    composite = np.dstack([normalize_u8(nir), normalize_u8(r), normalize_u8(g)])
    out_dir = os.path.dirname(zip_path)
    fc_path = os.path.join(out_dir, f"overlap_{label}_falsecolor.png")
    Image.fromarray(composite).save(fc_path)
    print(f"Saved false color preview: {fc_path}")

    # NDVI preview (true NDVI since we have real NIR)
    denom = np.clip(nir + r, 1, None)
    ndvi = (nir - r) / denom
    ndvi_u8 = ((ndvi + 1) / 2 * 255).astype(np.uint8)  # map [-1,1] -> [0,255]
    ndvi_path = os.path.join(out_dir, f"overlap_{label}_ndvi.png")
    Image.fromarray(ndvi_u8).save(ndvi_path)
    print(f"Saved NDVI preview: {ndvi_path}")

    # Rough cloud-fraction heuristic: pixels bright in ALL three bands simultaneously
    g_n, r_n, nir_n = normalize_u8(g), normalize_u8(r), normalize_u8(nir)
    bright_mask = (g_n > 180) & (r_n > 180) & (nir_n > 150)
    cloud_frac = 100 * bright_mask.mean()
    print(f"Rough bright-pixel (likely cloud) fraction in overlap zone: {cloud_frac:.1f}%")
    print("(heuristic only — upload the PNGs so we can visually confirm)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python crop_overlap_preview.py "<zip path>" <JUN|JUL>')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
