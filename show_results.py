# Save as show_results.py
import os, sys, cv2, glob, torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, "D:\\CloudRemoval_Project")

from src.models.generator import Generator
import yaml, json

with open("configs/gan_config.yaml") as f:
    config = yaml.safe_load(f)["gan"]

DEVICE = torch.device("cuda")
G = Generator(in_channels=config["in_channels"], features=config["features_g"]).to(
    DEVICE
)
G.load_state_dict(
    torch.load(
        "outputs/checkpoints/gan/best_generator.pt",
        map_location=DEVICE,
        weights_only=False,
    )
)
G.eval()

# Read patches directly from disk — no normalisation confusion
patches_dir = Path("data/processed/patches/test")
with open(patches_dir / "patch_manifest.json") as f:
    manifest = json.load(f)

patch_ids = sorted(manifest.keys())

import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose(
    [
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
        ToTensorV2(),
    ],
    additional_targets={"label": "image", "mask": "mask"},
)

samples = []
for pid in patch_ids[:12]:
    cloudy = cv2.cvtColor(
        cv2.imread(str(patches_dir / "cloud" / f"{pid}.png")), cv2.COLOR_BGR2RGB
    )
    label = cv2.cvtColor(
        cv2.imread(str(patches_dir / "label" / f"{pid}.png")), cv2.COLOR_BGR2RGB
    )
    mask_raw = cv2.imread(
        str(patches_dir / "mask" / f"{pid}.png"), cv2.IMREAD_GRAYSCALE
    )
    mask_bin = (mask_raw > 127).astype(np.uint8)

    # Print raw stats for first sample
    if len(samples) == 0:
        print(
            f"Raw cloudy  — min:{cloudy.min()} max:{cloudy.max()} mean:{cloudy.mean():.1f}"
        )
        print(
            f"Raw label   — min:{label.min()} max:{label.max()} mean:{label.mean():.1f}"
        )

    out = transform(image=cloudy, label=label, mask=mask_bin)
    cloudy_t = out["image"].unsqueeze(0).to(DEVICE)
    label_t = out["label"].unsqueeze(0).to(DEVICE)
    mask_t = out["mask"].unsqueeze(0).float().unsqueeze(0).to(DEVICE)

    # Edge map
    import cv2 as cv

    m = (mask_bin * 255).astype(np.float32)
    gx = cv.Sobel(m, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(m, cv.CV_32F, 0, 1, ksize=3)
    G_ = np.sqrt(gx**2 + gy**2)
    edge = (
        torch.from_numpy((G_ / G_.max() if G_.max() > 0 else G_).astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    gen_input = torch.cat([cloudy_t, mask_t, edge], dim=1)

    with torch.no_grad():
        fake = G(gen_input)

    # Denorm: [-1,1] → [0,1] → [0,255] uint8
    def to_uint8(t):
        arr = t[0].cpu().permute(1, 2, 0).numpy()
        arr = np.clip((arr + 1.0) / 2.0, 0, 1)
        return (arr * 255).astype(np.uint8)

    samples.append(
        {
            "cloudy": cloudy,  # raw uint8 from disk
            "generated": to_uint8(fake),
            "real": label,  # raw uint8 from disk
        }
    )

print(f"\nGenerated sample stats:")
print(
    f"  min:{samples[0]['generated'].min()}  max:{samples[0]['generated'].max()}  mean:{samples[0]['generated'].mean():.1f}"
)
print(f"Real label stats:")
print(
    f"  min:{samples[0]['real'].min()}  max:{samples[0]['real'].max()}  mean:{samples[0]['real'].mean():.1f}"
)

# Plot with auto-stretch for visibility
fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
fig.suptitle(
    "Phase 3 — cGAN Cloud Removal Results\n(Auto-brightness for visibility)",
    fontsize=13,
    fontweight="bold",
)

cols = ["Cloudy Input", "Generated (Cloud-Free)", "Ground Truth"]
for ax, c in zip(axes[0], cols):
    ax.set_title(c, fontsize=10, fontweight="bold")

for idx, s in enumerate(samples):
    for j, (key, img) in enumerate(
        [
            ("cloudy", s["cloudy"]),
            ("generated", s["generated"]),
            ("real", s["real"]),
        ]
    ):
        # Auto-stretch: scale to full range for visibility
        stretched = img.astype(float)
        mn, mx = stretched.min(), stretched.max()
        if mx > mn:
            stretched = ((stretched - mn) / (mx - mn) * 255).astype(np.uint8)

        axes[idx, j].imshow(stretched)
        if j == 0:
            axes[idx, j].set_ylabel(f"Sample {idx+1}", fontsize=8)
        axes[idx, j].axis("off")

plt.tight_layout()
os.makedirs("outputs/results/gan", exist_ok=True)
plt.savefig("outputs/results/gan/phase3_final.png", dpi=120, bbox_inches="tight")
plt.show()
print("\nSaved: outputs/results/gan/phase3_final.png")
