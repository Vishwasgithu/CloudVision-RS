"""
inference_production.py

Production inference: takes any satellite image (PNG or GeoTIFF),
runs cloud segmentation + removal, outputs georeferenced GeoTIFF.
This is what makes the project operationally usable by ISRO scientists.
"""
import os, sys, cv2, torch, yaml
import numpy as np
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, 'D:\\CloudRemoval_Project')

from src.models.segmentation import AttentionUNet
from src.models.generator    import Generator
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter

# ── Config ───────────────────────────────────────────
with open('configs/seg_config.yaml') as f:
    seg_config = yaml.safe_load(f)['segmentation']
with open('configs/gan_config.yaml') as f:
    gan_config = yaml.safe_load(f)['gan']

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 256
STRIDE     = 128   # 50% overlap for seamless stitching

# ── Load models ───────────────────────────────────────
print("Loading segmentation model...")
seg_model = AttentionUNet(seg_config).to(DEVICE)

# Dynamically pick the best checkpoint (highest IoU in filename)
import glob, re
seg_cks = sorted(glob.glob('outputs/checkpoints/segmentation/best_iou*.pt'))
def _ck_iou(p):
    m = re.search(r'best_iou([\d.]+)', os.path.basename(p))
    return float(m.group(1)) if m else 0.0
seg_best = max(seg_cks, key=_ck_iou) if seg_cks else None
if seg_best is None:
    raise FileNotFoundError("No segmentation checkpoint found in outputs/checkpoints/segmentation/")
print(f"  Best segmentation checkpoint: {os.path.basename(seg_best)}")
seg_ckpt  = torch.load(
    seg_best,
    map_location=DEVICE, weights_only=False)
seg_model.load_state_dict(seg_ckpt['model_state'])
seg_model.eval()

print("Loading GAN generator...")
gen_model = Generator(
    in_channels=gan_config['in_channels'],
    features=gan_config['features_g']).to(DEVICE)
gen_model.load_state_dict(torch.load(
    'outputs/checkpoints/gan/best_generator.pt',
    map_location=DEVICE, weights_only=False))
gen_model.eval()
print("Both models loaded\n")

def get_uncertainty_map(gen_model, gen_input, n_samples=10):
    """
    Monte Carlo Dropout uncertainty estimation.
    Runs generator 10 times with dropout enabled.
    Pixel variance across runs = reconstruction uncertainty.
    High uncertainty = model is unsure = scientist should verify manually.
    """
    # Enable dropout layers (training mode activates them)
    gen_model.train()
    outputs = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = gen_model(gen_input)
            outputs.append(((out + 1) / 2).cpu().numpy())
    gen_model.eval()

    stack       = np.stack(outputs, axis=0)   # [10, B, 3, H, W]
    mean_output = stack.mean(axis=0)          # [B, 3, H, W]
    uncertainty = stack.var(axis=0).mean(axis=1)  # [B, H, W]
    return mean_output, uncertainty

# ── Gaussian weight for feathering ───────────────────
def gaussian_weight(size=256):
    """
    Creates a 2D Gaussian weight matrix.
    Peak weight at centre, tapers to near-zero at edges.
    When overlapping patches are accumulated with these weights,
    boundaries blend smoothly — no visible seams.
    """
    center = size // 2
    w = np.zeros((size, size))
    w[center, center] = 1.0
    w = gaussian_filter(w, sigma=center * 0.35)
    return (w / w.max()).astype(np.float32)

GAUSS_W = gaussian_weight(PATCH_SIZE)

# ── Segmentation transform ────────────────────────────
seg_transform = A.Compose([
    A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0),
    ToTensorV2()
])

# ── GAN transform ─────────────────────────────────────
gan_transform = A.Compose([
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=255.0),
    ToTensorV2()
])

def pad_image(img, is_mask=False):
    """Pad image so sliding window covers all pixels."""
    H, W = img.shape[:2]
    pad_h = (STRIDE - (H - PATCH_SIZE) % STRIDE) % STRIDE
    pad_w = (STRIDE - (W - PATCH_SIZE) % STRIDE) % STRIDE
    if pad_h == 0 and pad_w == 0:
        return img, H, W
    if is_mask:
        padded = np.pad(img, ((0,pad_h),(0,pad_w)), mode='reflect')
    else:
        padded = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect')
    return padded, H, W

def run_inference(image_path: str, output_path: str = None):
    """
    Full pipeline inference on one image.

    Steps:
    1. Load image (PNG or GeoTIFF)
    2. Run segmentation to get cloud mask
    3. Run GAN to get cloud-free reconstruction
    4. Stitch patches with Gaussian feathering
    5. Save as GeoTIFF with original CRS (if input was GeoTIFF)
    """
    image_path = Path(image_path)
    print(f"Processing: {image_path.name}")

    # ── Step 1: Load image ────────────────────────────
    geo_meta = None
    if image_path.suffix.lower() in ['.tif', '.tiff']:
        try:
            import rasterio
            with rasterio.open(image_path) as src:
                img_data = src.read([1,2,3]).transpose(1,2,0)
                geo_meta = src.meta.copy()
                geo_meta.update({'count':3, 'dtype':'uint8'})
            print(f"  GeoTIFF loaded: CRS={geo_meta.get('crs','unknown')}")
        except ImportError:
            img_data = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            print("  rasterio not available, loaded as plain image")
    else:
        img_data = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    if img_data.dtype != np.uint8:
        img_data = ((img_data - img_data.min()) /
                    (img_data.max() - img_data.min()) * 255).astype(np.uint8)

    H_orig, W_orig = img_data.shape[:2]
    print(f"  Image size: {H_orig}×{W_orig}")

    # Pad to fit sliding window
    img_padded, H_orig, W_orig = pad_image(img_data)
    H_pad, W_pad = img_padded.shape[:2]

    # ── Step 2: Segmentation ──────────────────────────
    print("  Running cloud segmentation...")
    mask_accum  = np.zeros((H_pad, W_pad), dtype=np.float32)
    weight_accum = np.zeros((H_pad, W_pad), dtype=np.float32)

    with torch.no_grad():
        for r in range(0, H_pad - PATCH_SIZE + 1, STRIDE):
            for c in range(0, W_pad - PATCH_SIZE + 1, STRIDE):
                patch = img_padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                t = seg_transform(image=patch)['image'].unsqueeze(0).to(DEVICE)
                logit = seg_model(t)
                prob  = torch.sigmoid(logit)[0,0].cpu().numpy()
                mask_accum[r:r+PATCH_SIZE, c:c+PATCH_SIZE]   += prob * GAUSS_W
                weight_accum[r:r+PATCH_SIZE, c:c+PATCH_SIZE] += GAUSS_W

    cloud_mask = (mask_accum / (weight_accum + 1e-8))[:H_orig, :W_orig]
    cloud_mask_bin = (cloud_mask > 0.5).astype(np.uint8)

    coverage = cloud_mask_bin.mean() * 100
    print(f"  Cloud coverage: {coverage:.1f}%")

    # ── Step 3: GAN cloud removal ─────────────────────
    print("  Running cloud removal...")
    output_accum = np.zeros((H_pad, W_pad, 3), dtype=np.float32)
    weight_out   = np.zeros((H_pad, W_pad),    dtype=np.float32)

    # Pad mask too
    mask_padded = np.pad(cloud_mask_bin,
                         ((0, H_pad-H_orig),(0, W_pad-W_orig)), mode='reflect')

    with torch.no_grad():
        for r in range(0, H_pad - PATCH_SIZE + 1, STRIDE):
            for c in range(0, W_pad - PATCH_SIZE + 1, STRIDE):
                patch_img  = img_padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                patch_mask = mask_padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE]

                # GAN input: 5 channels
                img_t  = gan_transform(image=patch_img)['image']
                mask_t = torch.from_numpy(patch_mask.astype(np.float32)).unsqueeze(0)

                # Sobel edge map
                m  = (patch_mask * 255).astype(np.float32)
                gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
                G  = np.sqrt(gx**2 + gy**2)
                edge_t = torch.from_numpy(
                    (G/G.max() if G.max()>0 else G).astype(np.float32)
                ).unsqueeze(0)

                gen_input = torch.cat([img_t, mask_t, edge_t], dim=0)
                gen_input = gen_input.unsqueeze(0).to(DEVICE)

                fake = gen_model(gen_input)
                # Denorm [-1,1] → [0,255]
                out_patch = np.clip(
                    ((fake[0].cpu().permute(1,2,0).numpy() + 1) / 2 * 255),
                    0, 255).astype(np.float32)

                w3d = GAUSS_W[:,:,np.newaxis]
                output_accum[r:r+PATCH_SIZE, c:c+PATCH_SIZE] += out_patch * w3d
                weight_out[r:r+PATCH_SIZE, c:c+PATCH_SIZE]   += GAUSS_W

    # ── Step 4: Stitch and crop ───────────────────────
    output_full = (output_accum / (weight_out[:,:,np.newaxis] + 1e-8))
    output_crop = np.clip(output_full[:H_orig, :W_orig], 0, 255).astype(np.uint8)
    print("  Stitching complete (Gaussian feathering applied)")

    # Generate uncertainty map for the full image
    print("  Generating uncertainty map...")
    # Run on a representative patch to get uncertainty
    sample_patch = img_padded[H_pad//2-128:H_pad//2+128,
                              W_pad//2-128:W_pad//2+128]
    img_t   = gan_transform(image=sample_patch)['image']
    mask_t  = torch.from_numpy(
        mask_padded[H_pad//2-128:H_pad//2+128,
                    W_pad//2-128:W_pad//2+128].astype(np.float32)).unsqueeze(0)
    m       = (mask_padded[H_pad//2-128:H_pad//2+128,
                           W_pad//2-128:W_pad//2+128]*255).astype(np.float32)
    gx      = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    gy      = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    G_edge  = np.sqrt(gx**2 + gy**2)
    edge_t  = torch.from_numpy(
        (G_edge/G_edge.max() if G_edge.max()>0 else G_edge).astype(np.float32)
    ).unsqueeze(0)
    gi_sample = torch.cat([img_t, mask_t, edge_t], dim=0).unsqueeze(0).to(DEVICE)

    _, uncertainty_patch = get_uncertainty_map(gen_model, gi_sample)
    print(f"  Mean uncertainty: {uncertainty_patch.mean():.4f} "
          f"(lower = more confident)")

    # ── Step 5: Save output ───────────────────────────
    if output_path is None:
        stem = image_path.stem
        output_path = f"outputs/results/gan/{stem}_cloudfree.tif"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if geo_meta is not None:
        # Save as GeoTIFF with original CRS preserved
        import rasterio
        with rasterio.open(output_path, 'w', **geo_meta) as dst:
            dst.write(output_crop.transpose(2,0,1))
            dst.update_tags(
                PROCESSING='CloudVision-RS physics-informed cloud removal',
                CLOUD_COVERAGE_REMOVED=f'{coverage:.1f}%',
                MODEL='Attention-UNet + Physics-Informed cGAN',
                TEAM='NirmalDrishti'
            )
        print(f"  GeoTIFF saved with CRS: {output_path}")
    else:
        # Save as PNG
        output_path = output_path.replace('.tif', '.png')
        cv2.imwrite(output_path,
                    cv2.cvtColor(output_crop, cv2.COLOR_RGB2BGR))
        print(f"  PNG saved: {output_path}")

    # ── Save side-by-side comparison ──────────────────
    comparison = np.hstack([
        img_data[:H_orig, :W_orig],
        output_crop
    ])
    comp_path = output_path.replace('.tif', '_comparison.png').replace('.png', '_comparison.png')
    cv2.imwrite(comp_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"  Comparison saved: {comp_path}")
    print(f"  Done. Cloud coverage removed: {coverage:.1f}%\n")

    return output_crop, cloud_mask_bin, coverage


# ── Run on a test image ───────────────────────────────
if __name__ == '__main__':
    import glob

    # Test on one of your actual test patches first
    test_patches = glob.glob('data/processed/patches/test/cloud/*.png')
    if test_patches:
        test_img = test_patches[0]
        print(f"Testing on: {test_img}\n")
        result, mask, coverage = run_inference(
            test_img,
            'outputs/results/gan/test_inference.tif'
        )
        print("Production inference test complete.")
        print(f"Output shape: {result.shape}")
        print(f"Cloud coverage processed: {coverage:.1f}%")
    else:
        print("No test patches found. Check data/processed/patches/test/cloud/")
