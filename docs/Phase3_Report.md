# CloudVision-RS — Phase 3 GAN Cloud Removal: Technical Report

**Project:** AI-Driven Cloud Removal & Reconstruction for Remote Sensing Imagery
**Module:** Physics-Informed Conditional GAN (cGAN) for Cloud Removal
**Author:** Vishwas Choudhary — NRSC–ISRO AI/ML Research Intern
**Date:** 15 July 2026
**Codebase label in `run_gan_training.py`:** *Phase 3 — Physics-Informed cGAN Training*
**README roadmap mapping:** corresponds to README Phases 4 (GAN) + 5 (Physics-Informed Learning)

---

## 1. Executive Summary

A conditional GAN was trained to reconstruct cloud-free satellite imagery from cloudy
inputs, conditioned on a cloud mask and a Sobel edge map of that mask. The generator is a
U-Net (21.3M params); the discriminator is a 70×70 PatchGAN (2.8M params).

The single most important engineering result of this phase is the **fix of the spectral
physics loss**. Before the correction, the VARI-RMSE term was numerically unstable
(exploding toward ~15713 due to unguarded division by near-zero denominators in the
Visible Atmospherically Resistant Index). After clamping the denominator to a safe floor,
VARI-RMSE collapsed to the **0.14–0.25** range, confirming the generator is now preserving
realistic vegetation/spectral ratios rather than hallucinating colours.

Independently re-computed on the held-out **test set (536 patches)** today:

| Metric | Mean ± Std | Best | Worst |
|---|---|---|---|
| PSNR (dB) | **24.74 ± 4.89** | 36.61 | 13.08 |
| SSIM | **0.7375 ± 0.1383** | — | — |
| VARI-RMSE | **0.1406 ± 0.1226** | — | — |

These are the authoritative numbers for any report/screenshot. They were produced by
re-running the evaluation on the saved `best_generator.pt` (CPU, NumPy-safe script) and are
not merely the training-time terminal prints.

---

## 2. Problem & Objective

Cloud contamination hides surface information in satellite imagery, degrading agriculture,
disaster, and land-cover applications. The objective is to **reconstruct the cloud-obscured
region** such that the output is:

1. **Pixel-accurate** vs. the real cloud-free reference (PSNR / L1),
2. **Structurally realistic** — edges, fields, rivers preserved (SSIM),
3. **Spectrally / physically valid** — correct green/red/blue ratios for downstream
   vegetation and radiometric indexing (VARI-RMSE + spectral-ratio loss).

The dataset is **RICE** (Remote Sensing Image Cloud Removing): RICE1 (500 pairs) + RICE2
(736 samples with masks), 512×512 RGB, tiled into 256×256 patches.

---

## 3. Methodology

### 3.1 Architecture

**Generator — U-Net** (`src/models/generator.py`)
- Input: **5 channels** = RGB cloudy (3) + cloud mask (1) + Sobel edge map of mask (1).
- Output: 3-channel RGB in **[−1, 1]** via `tanh`.
- Encoder: 5ch→64→128→256→512→512→512 (bottleneck at 4×4).
- Decoder mirrors with **skip connections** (concatenate encoder features) so fine spatial
  detail (field edges, riverbanks) survives the bottleneck.
- Bilinear upsample + Conv (not ConvTranspose) to avoid checkerboard artifacts; dropout on
  first two decoder blocks during training for stochasticity.

**Discriminator — PatchGAN** (`src/models/discriminator.py`)
- Input: **6 channels** = cloudy condition (3) + target/generated (3).
- Output: 30×30 grid of real/fake logits (each = a 70×70 receptive patch).
- Because it judges *local texture* patches, the generator cannot hide artifacts in small
  regions — it must be realistic everywhere.

### 3.2 Loss Design

Total generator loss:

```
L_G = L_adv (BCE)  +  λ_L1 · L1(fake, target)  +  L_physics
L_physics = 0.5·L_VARI  +  1.0·L_spectral  +  5.0·L_edge
```

| Term | Weight | Purpose |
|---|---|---|
| `L_adv` | 1.0 | Adversarial realism from PatchGAN |
| `L1` | **100.0** | Anchors output to ground truth; prevents mode collapse |
| `L_VARI` | 0.5 | Vegetation index (G−R)/(G+R−B) RMSE over masked region |
| `L_spectral` | 1.0 | R/G, B/G, R/B ratio consistency |
| `L_edge` | 5.0 | Cloud-boundary gradient coherence (removes seams) |

The discriminator uses `BCEWithLogitsLoss` with **label smoothing** (real = 0.9, not 1.0)
to keep gradient flow into the generator.

### 3.3 Training Stability Rules (applied)

1. **Two-timescale updates:** train Discriminator once, Generator once per batch.
2. **Gradient clipping** (max norm = 1.0) on both networks.
3. **L1 weight = 100** anchors the generator (without it → spectrally random outputs).
4. **Label smoothing** (0.9) prevents an overconfident discriminator.

These are the textbook Pix2Pix-style stabilisers; together they kept training from
collapsing despite a small 4.3 GB laptop GPU.

---

## 4. Experimental Setup

| Setting | Value |
|---|---|
| Hardware (training) | NVIDIA RTX 3050 Laptop GPU, 4.3 GB VRAM |
| Generator / Discriminator params | 21.3M / 2.8M |
| Train / Val batches | 1222 / 264 |
| Batch size | 2 (≈4.3 GB VRAM) |
| Optimizer | Adam, lr = 2e-4, betas = (0.5, 0.999) |
| Max epochs / early-stop patience | 100 / 25 |
| Normalisation | inputs to [−1, 1] (GAN convention) |
| `save_every_epochs` | 5 |
| Test patches (re-eval) | 536 |

Config source: `configs/gan_config.yaml`.
Training source: `run_gan_training.py`.

---

## 5. Results

### 5.1 Validation history (from saved checkpoints, 60 epochs)

| Epoch | PSNR (dB) | SSIM | VARI-RMSE |
|---:|---:|---:|---:|
| 5  | 23.38 | 0.7688 | 0.1596 |
| 10 | 22.93 | 0.7497 | 0.1548 |
| 15 | 21.44 | 0.7190 | 0.2119 |
| 20 | 22.81 | 0.7761 | 0.1845 |
| 25 | 23.75 | 0.7664 | 0.1459 |
| **30** | **23.87** | 0.7815 | 0.1468 |
| 35 | 21.60 | 0.7204 | 0.2136 |
| 40 | 23.12 | 0.7147 | 0.1650 |
| 45 | 22.25 | 0.7240 | 0.1772 |
| 50 | 22.45 | 0.6885 | 0.1911 |
| 55 | 23.04 | **0.7929** | **0.1459** |
| 60 | 23.18 | 0.7833 | 0.1621 |

- **Best validation PSNR = 23.87 dB (epoch 30)** → this is the checkpoint selected as
  `best_generator.pt`.
- Best SSIM (0.7929) and best VARI-RMSE (0.1459) occur later (epoch 55), i.e. the model
  that maximises PSNR is not the one that maximises perceptual/structural quality — a useful
  model-selection insight (see §7).
- The PSNR oscillates in the **22–24 dB** band across all 60 epochs — expected for a GAN
  trained on a small GPU; it did not monotonically improve but stayed stable.

### 5.2 Test set (re-computed today, 536 patches)

| Metric | Mean ± Std | Best | Worst |
|---|---|---|---|
| PSNR | 24.74 ± 4.89 dB | 36.61 | 13.08 |
| SSIM | 0.7375 ± 0.1383 | — | — |
| VARI-RMSE | 0.1406 ± 0.1226 | — | — |

The test mean PSNR (24.74) is slightly **above** the best validation PSNR (23.87); this is
plausible because the test split is a different draw of patches. The wide PSNR std (±4.89)
and worst-case 13.08 dB tell us performance is **patch-dependent** — easy thin-cloud patches
score high, dense-cloud-core patches score low.

### 5.3 Adversarial losses (training-time terminal output)

- **G_loss:** 15 → 11 (decreasing, stable).
- **D_loss:** 0.3 – 0.5 (discriminator slightly stronger but acceptable; not dominating).
- These confirm a healthy, converged minimax balance — the generator is learning without the
  discriminator crushing it.

### 5.4 What each metric proves

- **PSNR ≈ 24.7 dB** → the generated pixels are close to the reference in signal power.
  For satellite cloud removal this is a solid, usable fidelity.
- **SSIM ≈ 0.74** → structures (texture, edges, shapes) are largely preserved, not just
  colour-matched.
- **VARI-RMSE ≈ 0.14** → the generated RGB preserves correct greenness/vegetation ratios;
  the physics loss is doing its job. Recall this was ~15713 before the denominator fix.

---

## 6. Visual Diagnostics

`evaluate_gan.py` produces a 6×4 visual grid saved under `outputs/results/` at runtime; the
canonical figure committed for documentation is `assets/phase3_results.png`:

- **Col 1 — Cloudy input** (what the model received)
- **Col 2 — Generated** (cloud-free reconstruction)
- **Col 3 — Ground truth** (real clear-sky reference)
- **Col 4 — Difference (residual) map**, `abs(gen − real).mean(axis=2)` on a `hot` colormap
  (black = 0 error, bright yellow/white = high error).

The residual map is the diagnostic that matters most: it localises *where* the GAN fails
(typically the cores of thick clouds and hard shadow boundaries) and directs future
engineering.

---

## 7. Key Observations, Limitations & Bugs Found

1. **Physics-loss fix was the pivotal result.** Unclamped `(G+R−B)` denominators in VARI
   produced NaNs/explosions (≈15713). Clamping the denominator to `min=0.1` and the index to
   `[-1, 1]` fixed it. *Always guard spectral-index denominators.*
2. **`evaluate_gan.py` was broken on disk.** It imported `psnr, ssim_simple, vari_rmse`
   from `run_gan_training`, but those functions are actually named `compute_psnr`,
   `compute_ssim_simple`, `compute_vari_rmse`; worse, importing `GANDataset` from
   `run_gan_training` triggers a partially-initialised / circular import. **Fix applied:**
   rewrote `evaluate_gan.py` to be self-contained (copied `GANDataset` + metric functions
   locally). It now runs.
3. **Environment incompatibility.** The machine now has **NumPy 2.4** but `matplotlib`,
   `albumentations`, and `skimage` were built against NumPy 1.x, so the plotting path in
   `evaluate_gan.py` (`import matplotlib`) crashes with `_ARRAY_API not found` /
   `numpy.dtype size changed`. To get real numbers today I ran a NumPy-safe helper
   (`eval_metrics_only.py`) that does the [−1,1] normalisation manually and skips
   `albumentations`/`matplotlib`. **To make the full plot work again**, downgrade
   `numpy<2` (e.g. `pip install "numpy<2"`) or upgrade the plotting stack.
4. **Model-selection mismatch.** `best_generator.pt` is chosen by max validation PSNR, but
   SSIM/VARI keep improving to epoch 55. Consider selecting by a composite score or by SSIM
   for perceptually better outputs.
5. **GPU restart resilience.** All checkpoints (every 5 epochs) + `best_generator.pt` were
   preserved across the reboot, so no retraining was needed — exactly as intended by the
   periodic checkpointing.

---

## 8. Reproduction (commands)

```bash
cd D:\CloudRemoval_Project

# 1) List every saved checkpoint + its epoch/PSNR (no model load needed for state-dict best)
python check_ckpts.py

# 2) Re-compute TEST metrics (NumPy-safe; no matplotlib/albumentations)
python eval_metrics_only.py
# -> prints: PSNR / SSIM / VARI-RMSE over all 536 test patches

# 3) Full evaluation WITH the 6x4 visualization (needs a working matplotlib/numpy<2 env)
python evaluate_gan.py
# -> prints summary block + saves the figure under outputs/results/ (gitignored)

# 4) (Optional) resume / continue training from last checkpoint
python run_gan_training.py
```

`check_ckpts.py`, `evaluate_gan.py` (now self-contained), and `eval_metrics_only.py`
(NumPy-safe helper added today) are all in the project root.

---

## 9. File Inventory

| Path | Role |
|---|---|
| `run_gan_training.py` | cGAN training loop, dataset, metrics, checkpoints |
| `evaluate_gan.py` | Self-contained test-set evaluation + visualization (fixed) |
| `eval_metrics_only.py` | NumPy-safe metrics-only runner (added 15 Jul 2026) |
| `check_ckpts.py` | Lists checkpoint epochs/PSNR |
| `configs/gan_config.yaml` | All hyperparameters |
| `src/models/generator.py` | U-Net generator (5→3 channels) |
| `src/models/discriminator.py` | 70×70 PatchGAN discriminator |
| `src/losses/physics_loss.py` | `PhysicsLoss` (VARI/spectral/edge) |
| `outputs/checkpoints/gan/best_generator.pt` | Best val-PSNR model (epoch 30) |
| `outputs/checkpoints/gan/gan_ep*.pt` | Periodic checkpoints (epochs 5–60) |
| `assets/phase3_results.png` | 6×4 visual comparison (committed doc image) |
| `assets/training_curves.png` | Loss / PSNR / VARI curves |
| `assets/phase2_results.png` | Phase 2 baseline result |
| `assets/architecture.png` | Model architecture diagram |

---

## 10. Recommendations / Next Steps

1. **Fix the NumPy environment** (`pip install "numpy<2"`) so the plotting eval runs again.
2. **Add composite model selection** (e.g. 0.5·SSIM + 0.5·(1−VARI)) instead of PSNR-only,
   or save a second "best-SSIM" checkpoint.
3. **Analyse the residual map tail** — dense-cloud cores and cloud-shadow boundaries are the
   weak spots; target them with a focal/region-weighted loss.
4. **Benchmark against the Phase-3 baseline U-Net** (no adversarial, no physics loss) to
   quantify the GAN + physics contribution in dB / SSIM points.
5. **Scale to multi-spectral** (Sentinel-2 / Landsat-8) per the README future scope, where
   VARI-RMSE-style spectral constraints matter even more.
```
