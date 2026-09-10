import os

os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "TRUE"  # fixes OMP Error #15 (duplicate libiomp5md.dll)
)

import sys
import uuid
import yaml
import glob
import re
import json
import cv2
import torch
import zipfile
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.segmentation import AttentionUNet
from src.models.generator import Generator

app = FastAPI(title="CloudVision-RS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 256
STRIDE = 128
MAX_AOI_SIZE = 2048  # cap AOI processing size -- protects the 4GB GPU from an oversized region request

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"
SCENES_DIR = STATIC_DIR / "scenes"
for d in (UPLOAD_DIR, RESULT_DIR, SCENES_DIR):
    d.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

with open(CONFIG_DIR / "seg_config.yaml") as f:
    seg_config = yaml.safe_load(f)["segmentation"]

with open(CONFIG_DIR / "gan_config.yaml") as f:
    gan_config = yaml.safe_load(f)["gan"]

# ═══════════════════════════════════════════════════════════════
# CHECKPOINT SELECTION -- always prefer the most-adapted real
# checkpoint available, and expose which one is active so the
# frontend can be honest about what produced the results
# ═══════════════════════════════════════════════════════════════


def pick_generator_checkpoints():
    """Returns (baseline_path, baseline_label, finetuned_path_or_None, finetuned_label_or_None)."""
    baseline = (
        PROJECT_ROOT / "outputs" / "checkpoints" / "gan_baseline" / "best_generator_baseline.pt",
        "RICE2 baseline",
    )
    finetuned_candidates = [
        (
            PROJECT_ROOT / "outputs" / "checkpoints" / "gan" / "best_generator.pt",
            "LISS-IV fine-tuned (generator-only)",
        ),
        (
            PROJECT_ROOT / "outputs" / "checkpoints" / "gan_liss4_g_only" / "best_generator_g_only.pt",
            "LISS-IV fine-tuned (adversarial)",
        ),
    ]
    ft_path, ft_label = None, None
    for path, label in finetuned_candidates:
        if os.path.exists(path):
            ft_path, ft_label = path, label
            break
    if not os.path.exists(baseline[0]):
        raise FileNotFoundError(
            "No RICE2 baseline generator found -- this should always exist."
        )
    return baseline[0], baseline[1], ft_path, ft_label


def pick_segmentation_checkpoint():
    # Explicitly use V1 candidate checkpoint for production
    v1_ckpt = PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation" / "v1_brightness_aug_model_state.pt"
    if v1_ckpt.exists():
        return v1_ckpt, "V1 brightness/contrast augmentation"
    
    # Fallback to dynamic selection if V1 not available
    candidates_dirs = [
        (
            PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation_liss4_finetune",
            "LISS-IV fine-tuned",
        ),
        (
            PROJECT_ROOT / "outputs" / "checkpoints" / "segmentation",
            "RICE2 baseline (not LISS-IV adapted)",
        ),
    ]

    def _iou(path):
        m = re.search(r"iou([\d.]+)", path.name)
        return float(m.group(1)) if m else 0.0

    for dir_path, label in candidates_dirs:
        checkpoints = list(dir_path.glob("best_*.pt")) + list(dir_path.glob("*.pt"))

        if checkpoints:
            best_ckpt = max(checkpoints, key=_iou)
            return best_ckpt, label

    raise FileNotFoundError(
        f"No segmentation checkpoint found under {PROJECT_ROOT / 'outputs' / 'checkpoints'}"
    )


def load_state(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if isinstance(ckpt, dict) and "G_state" in ckpt:
        return ckpt["G_state"]
    return ckpt


print("Loading segmentation model...")
seg_model = AttentionUNet(seg_config).to(DEVICE)
seg_ckpt_path, seg_label = pick_segmentation_checkpoint()
seg_model.load_state_dict(load_state(seg_ckpt_path))
seg_model.eval()
print(f"  -> {seg_ckpt_path} ({seg_label})")

print("Loading generator models (both, selected per-request by data domain)...")
gen_ckpt_baseline, gen_label_baseline, gen_ckpt_finetuned, gen_label_finetuned = (
    pick_generator_checkpoints()
)

gen_model_rice2 = Generator(
    in_channels=gan_config["in_channels"], features=gan_config["features_g"]
).to(DEVICE)
gen_model_rice2.load_state_dict(load_state(gen_ckpt_baseline))
gen_model_rice2.eval()
print(f"  -> RICE2 baseline: {gen_ckpt_baseline} (used for RICE2 sample grid)")

if gen_ckpt_finetuned:
    gen_model_liss4 = Generator(
        in_channels=gan_config["in_channels"], features=gan_config["features_g"]
    ).to(DEVICE)
    gen_model_liss4.load_state_dict(load_state(gen_ckpt_finetuned))
    gen_model_liss4.eval()
    print(
        f"  -> LISS-IV fine-tuned: {gen_ckpt_finetuned} ({gen_label_finetuned}) (used for real GeoTIFF/AOI scenes)"
    )
else:
    gen_model_liss4 = (
        gen_model_rice2  # fall back to baseline until a fine-tuned checkpoint exists
    )
    print(
        "  -> No LISS-IV fine-tuned checkpoint found yet; real-scene mode will fall back to RICE2 baseline"
    )

MODEL_INFO = {
    "segmentation_checkpoint": seg_ckpt_path,
    "segmentation_label": seg_label,
    "generator_rice2_checkpoint": gen_ckpt_baseline,
    "generator_rice2_label": "RICE2 baseline (used for RICE2 sample grid)",
    "generator_liss4_checkpoint": gen_ckpt_finetuned or gen_ckpt_baseline,
    "generator_liss4_label": (
        f"{gen_label_finetuned} (used for real scenes)"
        if gen_ckpt_finetuned
        else "RICE2 baseline -- no LISS-IV fine-tune available yet, real scenes will look worse than expected"
    ),
}


@app.get("/api/model_info")
def model_info():
    """Frontend should display this so results are never shown without saying which model produced them."""
    return MODEL_INFO


# ═══════════════════════════════════════════════════════════════
# CORE INFERENCE (shared by PNG-sample mode and GeoTIFF-AOI mode)
# Exactly your original sliding-window + Gaussian feathering logic,
# refactored into a reusable function instead of duplicated.
# ═══════════════════════════════════════════════════════════════

seg_transform = A.Compose(
    [A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0), ToTensorV2()]
)


def gaussian_weight(size=256):
    center = size // 2
    w = np.zeros((size, size))
    w[center, center] = 1.0
    w = gaussian_filter(w, sigma=center * 0.35)
    return (w / w.max()).astype(np.float32)


GAUSS_W = gaussian_weight(PATCH_SIZE)


def pad_image(img):
    H, W = img.shape[:2]
    pad_h = (STRIDE - (H - PATCH_SIZE) % STRIDE) % STRIDE
    pad_w = (STRIDE - (W - PATCH_SIZE) % STRIDE) % STRIDE
    if pad_h == 0 and pad_w == 0:
        return img, H, W
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"), H, W


def get_uncertainty_map(gen_model, gen_input, n_samples=5):
    gen_model.train()
    outputs = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = gen_model(gen_input)
            outputs.append(((out + 1) / 2).cpu().numpy())
    gen_model.eval()
    stack = np.stack(outputs, axis=0)
    return stack.mean(axis=0), stack.var(axis=0).mean(axis=1)


def run_pipeline(img_rgb_u8, gen_model, threshold=0.5, mc_samples=5):
    """
    img_rgb_u8: HxWx3 uint8 array, R/G/B channel order already resolved by caller
                (real RGB for RICE2 PNGs, or Red/Green/NIR-as-Blue for LISS-IV).
    gen_model: gen_model_rice2 for RICE2 samples, gen_model_liss4 for real scenes --
               caller MUST pass the domain-matched model, this function does not choose.
    Returns: cloud_mask_bin, clean_uint8, uncertainty_heatmap_bgr, coverage_pct
    """
    H_orig, W_orig = img_rgb_u8.shape[:2]
    img_padded, H_orig, W_orig = pad_image(img_rgb_u8)
    H_pad, W_pad = img_padded.shape[:2]

    mask_accum = np.zeros((H_pad, W_pad), dtype=np.float32)
    weight_accum = np.zeros((H_pad, W_pad), dtype=np.float32)
    with torch.no_grad():
        for r in range(0, H_pad - PATCH_SIZE + 1, STRIDE):
            for c in range(0, W_pad - PATCH_SIZE + 1, STRIDE):
                patch = img_padded[r : r + PATCH_SIZE, c : c + PATCH_SIZE]
                t = seg_transform(image=patch)["image"].unsqueeze(0).to(DEVICE)
                prob = torch.sigmoid(seg_model(t))[0, 0].cpu().numpy()
                mask_accum[r : r + PATCH_SIZE, c : c + PATCH_SIZE] += prob * GAUSS_W
                weight_accum[r : r + PATCH_SIZE, c : c + PATCH_SIZE] += GAUSS_W

    cloud_mask = (mask_accum / (weight_accum + 1e-8))[:H_orig, :W_orig]
    cloud_mask_bin = (cloud_mask > threshold).astype(np.uint8)
    coverage = float(cloud_mask_bin.mean() * 100)

    output_accum = np.zeros((H_pad, W_pad, 3), dtype=np.float32)
    unc_accum = np.zeros((H_pad, W_pad), dtype=np.float32)
    edge_accum = np.zeros((H_pad, W_pad), dtype=np.float32)
    weight_out = np.zeros((H_pad, W_pad), dtype=np.float32)
    mask_padded = np.pad(
        cloud_mask_bin, ((0, H_pad - H_orig), (0, W_pad - W_orig)), mode="reflect"
    )

    for r in range(0, H_pad - PATCH_SIZE + 1, STRIDE):
        for c in range(0, W_pad - PATCH_SIZE + 1, STRIDE):
            patch_img = img_padded[r : r + PATCH_SIZE, c : c + PATCH_SIZE]
            patch_mask = mask_padded[r : r + PATCH_SIZE, c : c + PATCH_SIZE]
            sx = cv2.Sobel(patch_mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(patch_mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            edge_map = np.sqrt(sx**2 + sy**2)
            edge_map /= edge_map.max() + 1e-8
            edge_accum[r : r + PATCH_SIZE, c : c + PATCH_SIZE] += edge_map * GAUSS_W

            img_norm = (patch_img.astype(np.float32) / 127.5) - 1.0
            x_input = np.concatenate(
                [img_norm, patch_mask[:, :, None], edge_map[:, :, None]], axis=2
            )
            x_tensor = (
                torch.from_numpy(x_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            )

            recons, unc = get_uncertainty_map(gen_model, x_tensor, n_samples=mc_samples)
            recons_np, unc_np = recons[0].transpose(1, 2, 0), unc[0]
            original_part = patch_img.astype(np.float32) / 255.0
            mask_3d = patch_mask[:, :, None]
            blended = recons_np * mask_3d + original_part * (1.0 - mask_3d)

            for ch in range(3):
                output_accum[r : r + PATCH_SIZE, c : c + PATCH_SIZE, ch] += (
                    blended[:, :, ch] * GAUSS_W
                )
            unc_accum[r : r + PATCH_SIZE, c : c + PATCH_SIZE] += unc_np * GAUSS_W
            weight_out[r : r + PATCH_SIZE, c : c + PATCH_SIZE] += GAUSS_W

    output_img = (output_accum / (weight_out[:, :, None] + 1e-8))[:H_orig, :W_orig]
    output_img = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)
    unc_map = (unc_accum / (weight_out + 1e-8))[:H_orig, :W_orig]
    unc_map = (np.clip(unc_map / (unc_map.max() + 1e-8), 0, 1) * 255).astype(np.uint8)
    unc_heatmap = cv2.applyColorMap(unc_map, cv2.COLORMAP_JET)

    edge_map_full = (edge_accum / (weight_out + 1e-8))[:H_orig, :W_orig]
    edge_map_u8 = (np.clip(edge_map_full, 0, 1) * 255).astype(np.uint8)
    edge_heatmap = cv2.applyColorMap(
        edge_map_u8, cv2.COLORMAP_BONE
    )  # boundary map, distinct look from uncertainty

    orig_cropped = img_rgb_u8[:H_orig, :W_orig]
    diff = cv2.absdiff(orig_cropped, output_img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_HOT)

    return cloud_mask_bin, output_img, unc_heatmap, coverage, edge_heatmap, diff_heatmap


# ═══════════════════════════════════════════════════════════════
# HONEST METRICS -- only computed where ground truth genuinely
# exists (RICE2 samples with a matching label file). Real math,
# reused verbatim from the training/eval scripts, not fabricated.
# ═══════════════════════════════════════════════════════════════


def compute_real_metrics(clean_u8, label_u8):
    p = clean_u8.astype(np.float32) / 255.0
    t = label_u8.astype(np.float32) / 255.0
    mse = ((p - t) ** 2).mean()
    psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
    mu_p, mu_t = p.mean(), t.mean()
    sp, st = p.std(), t.std()
    spt = ((p - mu_p) * (t - mu_t)).mean()
    C1, C2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_p * mu_t + C1) * (2 * spt + C2)) / (
        (mu_p**2 + mu_t**2 + C1) * (sp**2 + st**2 + C2)
    )
    Rp, Gp, Bp = p[..., 0], p[..., 1], p[..., 2]
    Rt, Gt, Bt = t[..., 0], t[..., 1], t[..., 2]
    dp = np.clip(Gp + Rp - Bp, 0.1, None)
    dt = np.clip(Gt + Rt - Bt, 0.1, None)
    vari_p = np.clip((Gp - Rp) / dp, -1, 1)
    vari_t = np.clip((Gt - Rt) / dt, -1, 1)
    vari_rmse = float(np.sqrt(((vari_p - vari_t) ** 2).mean()))
    return {
        "psnr_db": round(float(psnr), 2),
        "ssim": round(float(ssim), 4),
        "vari_rmse": round(vari_rmse, 4),
    }


def compute_mask_accuracy(pred_mask_bin, true_mask_bin):
    """How good the CLOUD DETECTION itself is -- separate from reconstruction quality.
    IoU: overlap / union. Dice: 2x overlap / (pred area + true area). Both in [0,1], higher=better.
    """
    intersection = (pred_mask_bin & true_mask_bin).sum()
    union = (pred_mask_bin | true_mask_bin).sum()
    iou = float(intersection / union) if union > 0 else 1.0
    dice = (
        float(2 * intersection / (pred_mask_bin.sum() + true_mask_bin.sum()))
        if (pred_mask_bin.sum() + true_mask_bin.sum()) > 0
        else 1.0
    )
    return {"iou": round(iou, 4), "dice": round(dice, 4)}


def compute_ndvi_stats(nir, red):
    ndvi = (nir.astype(np.float32) - red.astype(np.float32)) / np.clip(
        nir.astype(np.float32) + red.astype(np.float32), 1, None
    )
    return {
        "ndvi_mean": round(float(ndvi.mean()), 4),
        "ndvi_std": round(float(ndvi.std()), 4),
    }


# ═══════════════════════════════════════════════════════════════
# SAMPLE MODE (RICE2 PNGs, real ground truth available)
# ═══════════════════════════════════════════════════════════════


@app.get("/api/samples")
def get_samples():
    samples = []
    rice1_dir = PROJECT_ROOT / "datasets" / "RICE2" / "RICE1"
    rice1_cloud_dir = rice1_dir / "cloud"
    rice1_label_dir = rice1_dir / "label"
    if rice1_cloud_dir.exists():
        for cloud_path in sorted(rice1_cloud_dir.glob("*.png")):
            pid = cloud_path.stem
            label_path = rice1_label_dir / f"{pid}.png"
            dest_name = f"sample_{pid}.png"
            import shutil

            shutil.copy(str(cloud_path), str(UPLOAD_DIR / dest_name))
            samples.append(
                {
                    "id": pid,
                    "name": f"{pid}.png",
                    "url": f"/static/uploads/{dest_name}",
                    "cloud_coverage_pct": None,
                    "bin": "thin",
                    "dataset": "RICE1",
                    "has_ground_truth": label_path.exists(),
                    "has_ground_truth_mask": False,
                }
            )
    manifest_path = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "patch_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for pid, meta in manifest.items():
            coverage = meta.get("cloud_coverage", 0.0) * 100
            if coverage < 30:
                bin_label = "light"
            elif coverage < 60:
                bin_label = "medium"
            else:
                bin_label = "heavy"
            cloud_path = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "cloud" / f"{pid}.png"
            label_path = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "label" / f"{pid}.png"
            mask_path = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "mask" / f"{pid}.png"
            if not cloud_path.exists():
                continue
            dest_name = f"sample_{pid}.png"
            import shutil

            shutil.copy(str(cloud_path), str(UPLOAD_DIR / dest_name))
            samples.append(
                {
                    "id": pid,
                    "name": f"{pid}.png",
                    "url": f"/static/uploads/{dest_name}",
                    "cloud_coverage_pct": round(coverage, 1),
                    "bin": bin_label,
                    "dataset": "RICE2",
                    "source_id": int(meta.get("source_id", -1)),
                    "has_ground_truth": label_path.exists(),
                    "has_ground_truth_mask": mask_path.exists(),
                }
            )
    AGRICULTURAL_SOURCE_IDS = {
        154, 163, 168, 190, 197, 216, 223,
        386, 414, 421, 428, 433, 439, 471,
        498, 504, 523, 530, 533,
    }
    rice1_samples = [s for s in samples if s["dataset"] == "RICE1"]
    rice2_samples = [
        s for s in samples
        if s["dataset"] == "RICE2" and s.get("source_id", -1) in AGRICULTURAL_SOURCE_IDS
    ]
    return rice1_samples + rice2_samples


@app.post("/api/process")
async def process_sample(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    dataset: str = Form("RICE2"),
    threshold: float = Form(0.5),
    mc_samples: int = Form(5),
):
    file_id = str(uuid.uuid4())
    label_path, true_mask_path = None, None
    if file:
        filename = f"{file_id}_{file.filename}"
        filepath = UPLOAD_DIR / filename
        with open(filepath, "wb") as buf:
            buf.write(await file.read())
    elif sample_id:
        if dataset == "RICE1":
            cloud_path = PROJECT_ROOT / "datasets" / "RICE2" / "RICE1" / "cloud" / f"{sample_id}.png"
            maybe_label = PROJECT_ROOT / "datasets" / "RICE2" / "RICE1" / "label" / f"{sample_id}.png"
            maybe_mask = None
        else:
            cloud_path = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "cloud" / f"{sample_id}.png"
            maybe_label = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "label" / f"{sample_id}.png"
            maybe_mask = PROJECT_ROOT / "data" / "processed" / "patches" / "test" / "mask" / f"{sample_id}.png"
        if not cloud_path.exists():
            raise HTTPException(404, "Sample not found")
        filename = f"sample_{sample_id}.png"
        filepath = UPLOAD_DIR / filename
        import shutil

        shutil.copy(
            str(cloud_path), str(filepath)
        )  # actually serve it, don't just read the original
        if maybe_label and maybe_label.exists():
            label_path = maybe_label
        if maybe_mask and maybe_mask.exists():
            true_mask_path = maybe_mask
    else:
        raise HTTPException(400, "file or sample_id required")

    img = cv2.cvtColor(cv2.imread(str(filepath)), cv2.COLOR_BGR2RGB)
    mask, clean, unc_heatmap, coverage, edge_heatmap, diff_heatmap = run_pipeline(
        img, gen_model_rice2, threshold, mc_samples
    )

    mask_name, clean_name, unc_name = (
        f"mask_{file_id}.png",
        f"clean_{file_id}.png",
        f"unc_{file_id}.png",
    )
    edge_name, diff_name = f"edge_{file_id}.png", f"diff_{file_id}.png"
    cv2.imwrite(str(RESULT_DIR / mask_name), (mask * 255).astype(np.uint8))
    cv2.imwrite(str(RESULT_DIR / clean_name), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(RESULT_DIR / unc_name), unc_heatmap)
    cv2.imwrite(str(RESULT_DIR / edge_name), edge_heatmap)
    cv2.imwrite(str(RESULT_DIR / diff_name), diff_heatmap)

    result = {
        "cloud_coverage": coverage,
        "cloudy_url": f"/static/uploads/{filename}",
        "mask_url": f"/static/results/{mask_name}",
        "clean_url": f"/static/results/{clean_name}",
        "uncertainty_url": f"/static/results/{unc_name}",
        "edge_url": f"/static/results/{edge_name}",
        "diff_url": f"/static/results/{diff_name}",
        "ground_truth_url": None,  # true clean image, for visual comparison -- filled below if it exists
        "ground_truth_mask_url": None,  # true cloud mask, for visual comparison
        "mask_accuracy": None,  # how good the DETECTION is (IoU/Dice vs true mask) -- separate from reconstruction quality
        "metrics": None,  # reconstruction accuracy (PSNR/SSIM/VARI) -- only if ground truth exists, never fabricated
        "generator_used": "RICE2 baseline",
        "model_info": MODEL_INFO,
    }
    if label_path is not None:
        gt_name = f"gt_clean_{file_id}.png"
        import shutil

        shutil.copy(str(label_path), str(RESULT_DIR / gt_name))
        result["ground_truth_url"] = f"/static/results/{gt_name}"
        label_img = cv2.cvtColor(cv2.imread(str(label_path)), cv2.COLOR_BGR2RGB)
        result["metrics"] = compute_real_metrics(clean, label_img)
    if true_mask_path is not None:
        gt_mask_name = f"gt_mask_{file_id}.png"
        import shutil

        shutil.copy(str(true_mask_path), str(RESULT_DIR / gt_mask_name))
        result["ground_truth_mask_url"] = f"/static/results/{gt_mask_name}"
        true_mask_raw = cv2.imread(str(true_mask_path), cv2.IMREAD_GRAYSCALE)
        true_mask_bin = (true_mask_raw > 127).astype(np.uint8)
        result["mask_accuracy"] = compute_mask_accuracy(mask, true_mask_bin)
    return result


# ═══════════════════════════════════════════════════════════════
# GEOTIFF / ZIP MODE (real LISS-IV or Sentinel scenes)
# Upload -> quicklook preview -> AOI selection -> windowed inference
# ═══════════════════════════════════════════════════════════════

scene_registry = (
    {}
)  # scene_id -> {path, band_files/internal_names, width, height, transform, crs}


def find_band_in_zip(zip_path, keyword):
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if keyword.lower() in name.lower() and name.lower().endswith(
                (".tif", ".tiff")
            ):
                return name
    return None


@app.post("/api/upload_scene")
async def upload_scene(file: UploadFile = File(...)):
    """Accepts a LISS-IV zip (BAND2/3/4 GeoTIFFs) or a plain multi-band GeoTIFF.
    Returns a downsampled false-color quicklook + scene dimensions for AOI selection."""
    scene_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    save_path = SCENES_DIR / f"{scene_id}{ext}"
    with open(save_path, "wb") as buf:
        buf.write(await file.read())

    if ext == ".zip":
        g_name = find_band_in_zip(save_path, "BAND2")
        r_name = find_band_in_zip(save_path, "BAND3")
        nir_name = find_band_in_zip(save_path, "BAND4")
        if not (g_name and r_name and nir_name):
            raise HTTPException(
                400, "Could not find BAND2/BAND3/BAND4 GeoTIFFs inside zip."
            )
        vsi = lambda n: "/vsizip/" + str(save_path).replace("\\", "/") + "/" + n
        band_paths = {"green": vsi(g_name), "red": vsi(r_name), "nir": vsi(nir_name)}
    elif ext in (".tif", ".tiff"):
        # assumes a 3+ band geotiff, band order Green,Red,NIR -- adjust if your source differs
        band_paths = {
            "green": str(save_path),
            "red": str(save_path),
            "nir": str(save_path),
        }
    else:
        raise HTTPException(400, "Only .zip (LISS-IV) or .tif/.tiff accepted.")

    with rasterio.open(band_paths["red"]) as src:
        width, height = src.width, src.height
        transform, crs = src.transform, src.crs

    scene_registry[scene_id] = {
        "band_paths": band_paths,
        "width": width,
        "height": height,
        "transform": transform,
        "crs": str(crs),
        "is_zip": ext == ".zip",
    }

    # Build a small downsampled false-color quicklook so the browser never handles full res
    preview_w = 1200
    scale = preview_w / width
    preview_h = int(height * scale)

    def read_downsampled(path):
        with rasterio.open(path) as src:
            return src.read(1, out_shape=(preview_h, preview_w)).astype(np.float32)

    if band_paths["is_zip"]:
        g = read_downsampled(band_paths["green"])
        r = read_downsampled(band_paths["red"])
        n = read_downsampled(band_paths["nir"])
    else:
        g = read_downsampled(band_paths["green"])
        r = read_downsampled(band_paths["red"])
        n = read_downsampled(band_paths["nir"])

    def norm_u8(a):
        lo, hi = np.percentile(a[a > 0], [2, 98]) if (a > 0).any() else (0, 1)
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255

    preview = np.dstack([norm_u8(n), norm_u8(r), norm_u8(g)]).astype(
        np.uint8
    )  # false color NIR-R-G
    preview_name = f"preview_{scene_id}.png"
    cv2.imwrite(
        str(SCENES_DIR / preview_name), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    )

    return {
        "scene_id": scene_id,
        "width": width,
        "height": height,
        "preview_url": f"/static/scenes/{preview_name}",
        "preview_width": preview_w,
        "preview_height": preview_h,
        "max_aoi_size": MAX_AOI_SIZE,
    }


@app.post("/api/process_region")
async def process_region(
    scene_id: str = Form(...),
    x: int = Form(...),
    y: int = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    threshold: float = Form(0.5),
    mc_samples: int = Form(5),
):
    """x,y,width,height are in FULL-RESOLUTION pixel coordinates of the original scene
    (frontend must scale from preview coordinates before calling this)."""
    if scene_id not in scene_registry:
        raise HTTPException(404, "Scene not found -- re-upload.")
    scene = scene_registry[scene_id]
    width = min(width, MAX_AOI_SIZE, scene["width"] - x)
    height = min(height, MAX_AOI_SIZE, scene["height"] - y)
    if width <= 0 or height <= 0:
        raise HTTPException(400, "Invalid region.")

    window = Window(x, y, width, height)

    def read_window(path):
        with rasterio.open(path) as src:
            return src.read(1, window=window).astype(np.float32)

    g = read_window(scene["band_paths"]["green"])
    r = read_window(scene["band_paths"]["red"])
    n = read_window(scene["band_paths"]["nir"])

    def norm_u8(a):
        lo, hi = np.percentile(a[a > 0], [2, 98]) if (a > 0).any() else (0, 1)
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255

    # Band mapping decision: model R/G/B <- real Red/Green/NIR (NIR substitutes Blue)
    img_rgb = np.dstack([norm_u8(r), norm_u8(g), norm_u8(n)]).astype(np.uint8)

    mask, clean, unc_heatmap, coverage, edge_heatmap, diff_heatmap = run_pipeline(
        img_rgb, gen_model_liss4, threshold, mc_samples
    )
    ndvi_before = compute_ndvi_stats(n, r)

    file_id = str(uuid.uuid4())
    cloudy_name, mask_name, clean_name, unc_name = (
        f"aoi_cloudy_{file_id}.png",
        f"aoi_mask_{file_id}.png",
        f"aoi_clean_{file_id}.png",
        f"aoi_unc_{file_id}.png",
    )
    edge_name, diff_name = f"aoi_edge_{file_id}.png", f"aoi_diff_{file_id}.png"
    cv2.imwrite(str(RESULT_DIR / cloudy_name), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(RESULT_DIR / mask_name), (mask * 255).astype(np.uint8))
    cv2.imwrite(str(RESULT_DIR / clean_name), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(RESULT_DIR / unc_name), unc_heatmap)
    cv2.imwrite(str(RESULT_DIR / edge_name), edge_heatmap)
    cv2.imwrite(str(RESULT_DIR / diff_name), diff_heatmap)

    return {
        "cloud_coverage": coverage,
        "cloudy_url": f"/static/results/{cloudy_name}",
        "mask_url": f"/static/results/{mask_name}",
        "clean_url": f"/static/results/{clean_name}",
        "uncertainty_url": f"/static/results/{unc_name}",
        "edge_url": f"/static/results/{edge_name}",
        "diff_url": f"/static/results/{diff_name}",
        "ndvi_before": ndvi_before,
        "metrics": None,  # no ground truth exists for a real scene -- intentionally honest, not fabricated
        "generator_used": MODEL_INFO["generator_liss4_label"],
        "model_info": MODEL_INFO,
        "note": "No ground truth available for real scenes -- accuracy metrics (PSNR/SSIM) cannot be computed here. Cloud coverage, NDVI, and the uncertainty heatmap are real, directly computed values.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
