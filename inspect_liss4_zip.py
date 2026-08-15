"""
Run this locally on your Windows machine (D:\\CloudRemoval_Project) — NOT in Claude's sandbox.
It does NOT extract full band GeoTIFFs. It only:
  1. Lists zip contents + file sizes (so we know what product you actually got)
  2. Pulls out the metadata file (tiny, text) and prints it
  3. Pulls out the quicklook/browse image (tiny jpg) and saves it separately

Usage:
    conda activate cloudremoval
    python inspect_liss4_zip.py "D:\path\to\scene1.zip"
    python inspect_liss4_zip.py "D:\path\to\scene2.zip"
"""

import sys
import zipfile
import os

def inspect(zip_path):
    print(f"\n{'='*70}\nINSPECTING: {zip_path}\n{'='*70}")

    if not os.path.exists(zip_path):
        print(f"ERROR: file not found: {zip_path}")
        return

    out_dir = os.path.splitext(zip_path)[0] + "_preview"
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        infos = z.infolist()
        print(f"\nTotal files: {len(infos)}")
        print(f"{'File':<60}{'Size (MB)':>12}")
        print("-" * 72)
        for info in infos:
            size_mb = info.file_size / (1024 * 1024)
            print(f"{info.filename:<60}{size_mb:>12.2f}")

        # Extract small text/xml metadata files
        for info in infos:
            name_lower = info.filename.lower()
            if name_lower.endswith(('.txt', '.xml', '.hdr')) and info.file_size < 2_000_000:
                target = os.path.join(out_dir, os.path.basename(info.filename))
                with z.open(info) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
                print(f"\n--- Extracted metadata: {target} ---")
                try:
                    with open(target, 'r', errors='ignore') as f:
                        content = f.read()
                        print(content[:3000])
                except Exception as e:
                    print(f"(could not print: {e})")

        # Extract quicklook/browse image
        for info in infos:
            name_lower = info.filename.lower()
            if any(k in name_lower for k in ['quicklook', 'browse', 'thumb', 'preview']) and \
               name_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                target = os.path.join(out_dir, os.path.basename(info.filename))
                with z.open(info) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
                print(f"\n--- Extracted quicklook: {target} ---")

    print(f"\nPreview files saved to: {out_dir}")
    print("Upload the metadata text output above + the quicklook image to chat.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_liss4_zip.py <path_to_zip>")
        sys.exit(1)
    inspect(sys.argv[1])
