import os
import json
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

# Use the EXACT production GAN, transform, stitching configuration
from inference_production import (
    gen_model,
    gan_transform,
    DEVICE,
    PATCH_SIZE,
    STRIDE,
    GAUSS_W,
    pad_image,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("data/processed/patches/test")
CLOUD_DIR = ROOT / "cloud"
GT_MASK_DIR = ROOT / "mask"
GT_TARGET_DIR = ROOT / "label"
MANIFEST_PATH = ROOT / "patch_manifest.json"

OUT_DIR = Path("outputs/diagnostics/gt_mask_reconstruction")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGRICULTURAL_SOURCE_IDS = {
    154, 163, 168, 190, 197, 216, 223,
    386, 414, 421, 428, 433, 439, 471,
    498, 504, 523, 530, 533
}


# ============================================================
# METRICS
# ============================================================

def compute_psnr(pred, target):
    p = pred.astype(np.float32) / 255.0
    t = target.astype(np.float32) / 255.0
    mse = np.mean((p - t) ** 2)
    return 100.0 if mse < 1e-10 else float(10.0 * np.log10(1.0 / mse))


def compute_ssim(pred, target):
    p = pred.astype(np.float32) / 255.0
    t = target.astype(np.float32) / 255.0
    mu_p, mu_t = p.mean(), t.mean()
    sp, st = p.std(), t.std()
    spt = ((p - mu_p) * (t - mu_t)).mean()
    C1, C2 = 0.01**2, 0.03**2
    ssim_val = ((2 * mu_p * mu_t + C1) * (2 * spt + C2)) / (
        (mu_p**2 + mu_t**2 + C1) * (sp**2 + st**2 + C2)
    )
    return float(ssim_val)


def compute_vari_rmse(pred, target, mask):
    """
    VARI-RMSE over GT cloud-mask pixels.

    VARI = (G - R) / (G + R - B)
    Denominator clamped to min=0.1.
    VARI clamped to [-1, 1].
    RMSE computed only over GT cloud-mask pixels.
    """
    p = pred.astype(np.float32) / 255.0
    t = target.astype(np.float32) / 255.0

    Rp, Gp, Bp = p[..., 0], p[..., 1], p[..., 2]
    Rt, Gt, Bt = t[..., 0], t[..., 1], t[..., 2]

    dp = np.clip(Gp + Rp - Bp, 0.1, None)
    dt = np.clip(Gt + Rt - Bt, 0.1, None)

    vp = np.clip((Gp - Rp) / dp, -1, 1)
    vt = np.clip((Gt - Rt) / dt, -1, 1)

    valid = mask.astype(bool)

    if valid.sum() == 0:
        return 0.0

    diff = (vp[valid] - vt[valid]) ** 2
    return float(np.sqrt(np.mean(diff)))


# ============================================================
# SOBEL EDGE
# ============================================================

def mask_to_edge(mask):
    """
    EXACT same Sobel procedure used by production inference.
    Input: binary mask [H,W], values 0/1.
    Output: normalized edge map [H,W], float32.
    """

    m = (mask.astype(np.float32) * 255.0)

    gx = cv2.Sobel(
        m,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        m,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    G = np.sqrt(gx ** 2 + gy ** 2)

    if G.max() > 0:
        G = G / G.max()

    return G.astype(np.float32)


# ============================================================
# GT-MASK RECONSTRUCTION
# ============================================================

def reconstruct_with_gt_mask(img, gt_mask):
    """
    Same reconstruction mechanism as production inference,
    but replaces predicted mask with GT mask.

    GAN input:
        RGB + GT mask + GT Sobel edge
        = 5 channels
    """

    img_padded, H_orig, W_orig = pad_image(img)
    mask_padded, _, _ = pad_image(gt_mask, is_mask=True)

    H_pad, W_pad = img_padded.shape[:2]

    output_accum = np.zeros(
        (H_pad, W_pad, 3),
        dtype=np.float32
    )

    weight_out = np.zeros(
        (H_pad, W_pad),
        dtype=np.float32
    )

    with torch.no_grad():

        for r in range(
            0,
            H_pad - PATCH_SIZE + 1,
            STRIDE
        ):

            for c in range(
                0,
                W_pad - PATCH_SIZE + 1,
                STRIDE
            ):

                patch_img = img_padded[
                    r:r + PATCH_SIZE,
                    c:c + PATCH_SIZE
                ]

                patch_mask = mask_padded[
                    r:r + PATCH_SIZE,
                    c:c + PATCH_SIZE
                ]

                # -----------------------------
                # RGB -> 3 channels
                # -----------------------------

                img_t = gan_transform(
                    image=patch_img
                )["image"]

                # -----------------------------
                # GT mask -> 1 channel
                # -----------------------------

                mask_t = torch.from_numpy(
                    patch_mask.astype(np.float32)
                ).unsqueeze(0)

                # -----------------------------
                # GT mask -> Sobel edge
                # -----------------------------

                edge = mask_to_edge(patch_mask)

                edge_t = torch.from_numpy(
                    edge
                ).unsqueeze(0)

                # -----------------------------
                # FINAL 5 CHANNEL INPUT
                # -----------------------------

                gen_input = torch.cat(
                    [
                        img_t,
                        mask_t,
                        edge_t
                    ],
                    dim=0
                )

                # Safety check
                if gen_input.shape != (
                    5,
                    PATCH_SIZE,
                    PATCH_SIZE
                ):
                    raise RuntimeError(
                        f"Wrong GAN input shape: "
                        f"{tuple(gen_input.shape)}"
                    )

                gen_input = (
                    gen_input
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                # -----------------------------
                # GAN reconstruction
                # -----------------------------

                fake = gen_model(gen_input)

                out_patch = np.clip(
                    (
                        (
                            fake[0]
                            .cpu()
                            .permute(1, 2, 0)
                            .numpy()
                            + 1
                        )
                        / 2
                        * 255
                    ),
                    0,
                    255
                ).astype(np.float32)

                # -----------------------------
                # Gaussian stitching
                # -----------------------------

                w3d = GAUSS_W[:, :, np.newaxis]

                output_accum[
                    r:r + PATCH_SIZE,
                    c:c + PATCH_SIZE
                ] += out_patch * w3d

                weight_out[
                    r:r + PATCH_SIZE,
                    c:c + PATCH_SIZE
                ] += GAUSS_W

    output_full = (
        output_accum /
        (weight_out[:, :, np.newaxis] + 1e-8)
    )

    output_crop = np.clip(
        output_full[:H_orig, :W_orig],
        0,
        255
    ).astype(np.uint8)

    return output_crop


# ============================================================
# LOAD AGRICULTURAL PATCHES
# ============================================================

def load_agricultural_patches():

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    selected = []

    for pid, meta in manifest.items():

        source_id = meta.get("source_id")

        if source_id is None:
            try:
                source_id = int(pid.split("_")[0])
            except Exception:
                continue

        if int(source_id) not in AGRICULTURAL_SOURCE_IDS:
            continue

        cloud_path = CLOUD_DIR / f"{pid}.png"
        mask_path = GT_MASK_DIR / f"{pid}.png"
        target_path = GT_TARGET_DIR / f"{pid}.png"

        if not cloud_path.exists():
            continue

        if not mask_path.exists():
            continue

        coverage = float(
            meta.get("cloud_coverage", 0.0)
        ) * 100.0

        if coverage < 30:
            category = "light"
        elif coverage < 60:
            category = "medium"
        else:
            category = "heavy"

        selected.append(
            {
                "id": pid,
                "source_id": int(source_id),
                "coverage": coverage,
                "category": category,
                "cloud_path": cloud_path,
                "mask_path": mask_path,
                "label_path": target_path,
            }
        )

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("GT-MASK RECONSTRUCTION DIAGNOSTIC")
    print("=" * 70)

    print("\nLoading agricultural test patches...")

    samples = load_agricultural_patches()

    print(f"Selected patches: {len(samples)}")

    counts = {
        "light": 0,
        "medium": 0,
        "heavy": 0
    }

    for s in samples:
        counts[s["category"]] += 1

    print(
        f"Light  : {counts['light']}"
    )
    print(
        f"Medium : {counts['medium']}"
    )
    print(
        f"Heavy  : {counts['heavy']}"
    )

    expected = 116

    if len(samples) != expected:
        raise RuntimeError(
            f"Expected {expected} agricultural patches, "
            f"but found {len(samples)}."
        )

    # --------------------------------------------------------
    # Smoke test
    # --------------------------------------------------------

    print("\nRunning 1-patch smoke test...")

    smoke = samples[0]

    img = cv2.imread(
        str(smoke["cloud_path"]),
        cv2.IMREAD_COLOR
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    gt_mask_raw = cv2.imread(
        str(smoke["mask_path"]),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise RuntimeError(
            f"Could not read {smoke['cloud_path']}"
        )

    if gt_mask_raw is None:
        raise RuntimeError(
            f"Could not read {smoke['mask_path']}"
        )

    gt_mask = (
        gt_mask_raw > 127
    ).astype(np.uint8)

    print("\nImage shape:", img.shape)
    print("GT mask shape:", gt_mask.shape)

    edge = mask_to_edge(gt_mask)

    print("Edge shape:", edge.shape)

    # Verify one 5-channel tensor manually
    test_img_t = gan_transform(
        image=img
    )["image"]

    test_mask_t = torch.from_numpy(
        gt_mask.astype(np.float32)
    ).unsqueeze(0)

    test_edge_t = torch.from_numpy(
        edge
    ).unsqueeze(0)

    test_input = torch.cat(
        [
            test_img_t,
            test_mask_t,
            test_edge_t
        ],
        dim=0
    )

    print("GAN input shape:", tuple(test_input.shape))

    assert test_input.shape == (
        5,
        PATCH_SIZE,
        PATCH_SIZE
    )

    smoke_output = reconstruct_with_gt_mask(
        img,
        gt_mask
    )

    print("Reconstruction shape:", smoke_output.shape)

    # Save smoke artifacts
    smoke_dir = OUT_DIR / "smoke_test"
    smoke_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(smoke_dir / "cloudy.png"),
        cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )
    )

    cv2.imwrite(
        str(smoke_dir / "gt_mask.png"),
        gt_mask * 255
    )

    cv2.imwrite(
        str(smoke_dir / "gt_edge.png"),
        (edge * 255).astype(np.uint8)
    )

    cv2.imwrite(
        str(smoke_dir / "gt_mask_reconstruction.png"),
        cv2.cvtColor(
            smoke_output,
            cv2.COLOR_RGB2BGR
        )
    )

    print("\nSmoke test PASSED.")

    # --------------------------------------------------------
    # Full evaluation
    # --------------------------------------------------------

    print("\nStarting full 116-patch evaluation...")
    print("This may take some time on the RTX 3050.\n")

    rows = []

    for idx, sample in enumerate(
        samples,
        start=1
    ):

        pid = sample["id"]

        print(
            f"[{idx:3d}/{len(samples)}] "
            f"{pid:<25} "
            f"{sample['category']:<7} "
            f"{sample['coverage']:.1f}%"
        )

        img = cv2.imread(
            str(sample["cloud_path"]),
            cv2.IMREAD_COLOR
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        gt_mask_raw = cv2.imread(
            str(sample["mask_path"]),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None or gt_mask_raw is None:
            print("    SKIPPED: image/mask read failure")
            continue

        gt_mask = (
            gt_mask_raw > 127
        ).astype(np.uint8)

        reconstruction = reconstruct_with_gt_mask(
            img,
            gt_mask
        )

        target = cv2.imread(
            str(sample["label_path"]),
            cv2.IMREAD_COLOR
        )

        target = cv2.cvtColor(
            target,
            cv2.COLOR_BGR2RGB
        )

        if target.shape != reconstruction.shape:
            raise RuntimeError(
                f"Shape mismatch for {pid}: "
                f"reconstruction={reconstruction.shape}, "
                f"target={target.shape}"
            )

        psnr = compute_psnr(
            reconstruction,
            target
        )

        ssim_value = compute_ssim(
            reconstruction,
            target
        )

        vari = compute_vari_rmse(
            reconstruction,
            target,
            gt_mask
        )

        rows.append(
            {
                "patch_id": pid,
                "source_id": sample["source_id"],
                "category": sample["category"],
                "cloud_coverage_pct": round(
                    sample["coverage"],
                    4
                ),
                "psnr_db_gt_mask": round(
                    psnr,
                    4
                ),
                "ssim_gt_mask": round(
                    ssim_value,
                    4
                ),
                "vari_rmse_gt_mask": round(
                    vari,
                    4
                ),
            }
        )

        # Save only a small set of diagnostic cases
        if idx <= 5:
            sample_dir = (
                OUT_DIR /
                "examples" /
                sample["category"]
            )

            sample_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            cv2.imwrite(
                str(
                    sample_dir /
                    f"{pid}_reconstruction.png"
                ),
                cv2.cvtColor(
                    reconstruction,
                    cv2.COLOR_RGB2BGR
                )
            )

            cv2.imwrite(
                str(
                    sample_dir /
                    f"{pid}_gt_mask.png"
                ),
                gt_mask * 255
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = OUT_DIR / "gt_mask_reconstruction_per_patch.csv"

    if not rows:
        raise RuntimeError(
            "No evaluation results were produced."
        )

    fieldnames = list(rows[0].keys())

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_path = (
        OUT_DIR /
        "gt_mask_reconstruction_summary.txt"
    )

    lines = []

    lines.append(
        "GT-MASK RECONSTRUCTION DIAGNOSTIC"
    )
    lines.append("=" * 60)
    lines.append(
        f"Evaluated patches: {len(rows)}"
    )
    lines.append("")

    for category in [
        "light",
        "medium",
        "heavy"
    ]:

        subset = [
            r for r in rows
            if r["category"] == category
        ]

        if not subset:
            continue

        psnrs = np.array(
            [r["psnr_db_gt_mask"] for r in subset]
        )

        ssims = np.array(
            [r["ssim_gt_mask"] for r in subset]
        )

        varis = np.array(
            [r["vari_rmse_gt_mask"] for r in subset]
        )

        lines.append(
            f"{category.upper()} "
            f"(N={len(subset)})"
        )

        lines.append(
            f"  PSNR      : "
            f"{psnrs.mean():.4f} ± {psnrs.std():.4f} dB"
        )

        lines.append(
            f"  SSIM      : "
            f"{ssims.mean():.4f} ± {ssims.std():.4f}"
        )

        lines.append(
            f"  VARI-RMSE : "
            f"{varis.mean():.4f} ± {varis.std():.4f}"
        )

        lines.append("")

    # Overall

    psnrs = np.array(
        [r["psnr_db_gt_mask"] for r in rows]
    )

    ssims = np.array(
        [r["ssim_gt_mask"] for r in rows]
    )

    varis = np.array(
        [r["vari_rmse_gt_mask"] for r in rows]
    )

    lines.append(
        f"OVERALL (N={len(rows)})"
    )

    lines.append(
        f"  PSNR      : "
        f"{psnrs.mean():.4f} ± {psnrs.std():.4f} dB"
    )

    lines.append(
        f"  SSIM      : "
        f"{ssims.mean():.4f} ± {ssims.std():.4f}"
    )

    lines.append(
        f"  VARI-RMSE : "
        f"{varis.mean():.4f} ± {varis.std():.4f}"
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nPer-patch CSV:\n{csv_path}"
    )

    print(
        f"\nSummary:\n{summary_path}"
    )

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()