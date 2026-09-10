import os
import sys
import re
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, r"D:\CloudRemoval_Project")

from src.models.segmentation import AttentionUNet


# ============================================================
# CONFIG
# ============================================================

BASE = Path("data/processed/patches/test")
CONFIG_PATH = Path("configs/seg_config.yaml")
CHECKPOINT_DIR = Path("outputs/checkpoints/segmentation")

THRESHOLDS = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD EXACT PRODUCTION CONFIG
# ============================================================

with open(CONFIG_PATH, "r") as f:
    seg_config = yaml.safe_load(f)["segmentation"]


print("=" * 75)
print("FULL TEST-SET SEGMENTATION THRESHOLD ANALYSIS")
print("=" * 75)

print(f"Device:              {DEVICE}")
print(f"Config:              {CONFIG_PATH}")


# ============================================================
# FIND BEST CHECKPOINT
# ============================================================

seg_cks = glob.glob(
    "outputs/checkpoints/segmentation/best_iou*.pt"
)

if not seg_cks:
    raise FileNotFoundError(
        "No best_iou*.pt checkpoint found."
    )


def checkpoint_iou(path):

    name = os.path.basename(path)

    match = re.search(
        r"best_iou([\d.]+)",
        name
    )

    return float(match.group(1)) if match else 0.0


checkpoint = max(
    seg_cks,
    key=checkpoint_iou
)


print(
    f"Checkpoint:          {checkpoint}"
)

print(
    f"Checkpoint IoU:      "
    f"{checkpoint_iou(checkpoint):.4f}"
)


# ============================================================
# LOAD EXACT PRODUCTION MODEL
# ============================================================

print()
print("Loading segmentation model...")

seg_model = AttentionUNet(
    seg_config
).to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint_data = torch.load(
    checkpoint,
    map_location=DEVICE,
    weights_only=False
)

print(
    f"Checkpoint type:     "
    f"{type(checkpoint_data).__name__}"
)


# Your checkpoint uses "model_state"
if (
    isinstance(checkpoint_data, dict)
    and "model_state" in checkpoint_data
):

    state_dict = checkpoint_data["model_state"]

elif (
    isinstance(checkpoint_data, dict)
    and "model_state_dict" in checkpoint_data
):

    state_dict = checkpoint_data["model_state_dict"]

else:

    state_dict = checkpoint_data


seg_model.load_state_dict(
    state_dict
)

seg_model.eval()

print(
    "Segmentation model loaded successfully."
)


# ============================================================
# EXACT PRODUCTION SEGMENTATION TRANSFORM
# ============================================================

seg_transform = A.Compose([
    A.Normalize(
        mean=(0, 0, 0),
        std=(1, 1, 1),
        max_pixel_value=255.0
    ),
    ToTensorV2()
])


# ============================================================
# METRIC STORAGE
# ============================================================

stats = {}

for threshold in THRESHOLDS:

    stats[threshold] = {

        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,

        "gt_cloud_pixels": 0,
        "pred_cloud_pixels": 0,

        "images": 0
    }


# ============================================================
# TEST FILES
# ============================================================

cloud_files = sorted(
    (BASE / "cloud").glob("*.png")
)

if not cloud_files:

    raise FileNotFoundError(
        f"No test images found in {BASE / 'cloud'}"
    )


print()
print(
    f"Test images:          {len(cloud_files)}"
)

if len(cloud_files) != 536:

    print(
        f"WARNING: Expected 536 images, "
        f"found {len(cloud_files)}."
    )


# ============================================================
# RUN FULL TEST SET
# ============================================================

print()
print(
    "Running exact production segmentation "
    "on complete test set..."
)
print()


with torch.no_grad():

    for index, cloud_path in enumerate(
        cloud_files,
        start=1
    ):

        # ----------------------------------------------------
        # Corresponding ground-truth mask
        # ----------------------------------------------------

        mask_path = (
            BASE /
            "mask" /
            cloud_path.name
        )

        if not mask_path.exists():

            print(
                f"WARNING: Missing mask: "
                f"{cloud_path.name}"
            )

            continue


        # ----------------------------------------------------
        # Read cloudy image
        # ----------------------------------------------------

        image = cv2.imread(
            str(cloud_path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            print(
                f"WARNING: Could not read: "
                f"{cloud_path.name}"
            )

            continue


        # ----------------------------------------------------
        # Production uses OpenCV BGR image directly
        # with seg_transform.
        # ----------------------------------------------------

        transformed = seg_transform(
            image=image
        )


        tensor = transformed[
            "image"
        ].unsqueeze(0).to(DEVICE)


        # ----------------------------------------------------
        # Segmentation model
        # ----------------------------------------------------

        logits = seg_model(
            tensor
        )


        # ----------------------------------------------------
        # Probability map
        # ----------------------------------------------------

        probability = (
            torch.sigmoid(logits)
            [0, 0]
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Ground truth
        #
        # RICE2:
        # 255 = cloud
        # <=127 = non-cloud
        # ----------------------------------------------------

        gt_raw = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if gt_raw is None:

            print(
                f"WARNING: Could not read mask: "
                f"{mask_path.name}"
            )

            continue


        gt = (
            gt_raw > 127
        )


        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if probability.shape != gt.shape:

            probability = cv2.resize(
                probability,
                (
                    gt.shape[1],
                    gt.shape[0]
                ),
                interpolation=cv2.INTER_LINEAR
            )


        # ----------------------------------------------------
        # Evaluate all thresholds
        # ----------------------------------------------------

        for threshold in THRESHOLDS:

            pred = (
                probability >= threshold
            )


            tp = np.logical_and(
                pred,
                gt
            ).sum()

            fp = np.logical_and(
                pred,
                ~gt
            ).sum()

            fn = np.logical_and(
                ~pred,
                gt
            ).sum()

            tn = np.logical_and(
                ~pred,
                ~gt
            ).sum()


            stats[threshold]["tp"] += int(tp)
            stats[threshold]["fp"] += int(fp)
            stats[threshold]["fn"] += int(fn)
            stats[threshold]["tn"] += int(tn)

            stats[threshold][
                "gt_cloud_pixels"
            ] += int(gt.sum())

            stats[threshold][
                "pred_cloud_pixels"
            ] += int(pred.sum())

            stats[threshold][
                "images"
            ] += 1


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index % 50 == 0
            or index == len(cloud_files)
        ):

            print(
                f"Processed "
                f"{index}/{len(cloud_files)} images"
            )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 100)

print(
    f"{'Threshold':<12}"
    f"{'IoU':<12}"
    f"{'Dice':<12}"
    f"{'Precision':<14}"
    f"{'Recall':<12}"
    f"{'GT Cov.':<12}"
    f"{'Pred Cov.':<12}"
)

print("=" * 100)


results = []


for threshold in THRESHOLDS:

    s = stats[threshold]

    tp = s["tp"]
    fp = s["fp"]
    fn = s["fn"]
    tn = s["tn"]


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    union = (
        tp +
        fp +
        fn
    )

    iou = (
        tp / union
        if union > 0
        else 0.0
    )


    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice_den = (
        2 * tp +
        fp +
        fn
    )

    dice = (
        (2 * tp) / dice_den
        if dice_den > 0
        else 0.0
    )


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )


    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    total_pixels = (
        tp +
        fp +
        fn +
        tn
    )


    gt_coverage = (
        (tp + fn)
        / total_pixels
        * 100
    )


    pred_coverage = (
        (tp + fp)
        / total_pixels
        * 100
    )


    results.append({
        "threshold": threshold,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "gt_coverage": gt_coverage,
        "pred_coverage": pred_coverage
    })


    print(
        f"{threshold:<12.2f}"
        f"{iou:<12.4f}"
        f"{dice:<12.4f}"
        f"{precision:<14.4f}"
        f"{recall:<12.4f}"
        f"{gt_coverage:<12.2f}"
        f"{pred_coverage:<12.2f}"
    )


print("=" * 100)


# ============================================================
# BEST BY IoU
# ============================================================

best_iou = max(
    results,
    key=lambda x: x["iou"]
)


print()
print("BEST BY IoU")
print("-" * 50)

print(
    f"Threshold : {best_iou['threshold']:.2f}"
)

print(
    f"IoU       : {best_iou['iou']:.4f}"
)

print(
    f"Dice      : {best_iou['dice']:.4f}"
)

print(
    f"Precision : {best_iou['precision']:.4f}"
)

print(
    f"Recall    : {best_iou['recall']:.4f}"
)


# ============================================================
# BEST BY DICE
# ============================================================

best_dice = max(
    results,
    key=lambda x: x["dice"]
)


print()
print("BEST BY DICE")
print("-" * 50)

print(
    f"Threshold : {best_dice['threshold']:.2f}"
)

print(
    f"IoU       : {best_dice['iou']:.4f}"
)

print(
    f"Dice      : {best_dice['dice']:.4f}"
)

print(
    f"Precision : {best_dice['precision']:.4f}"
)

print(
    f"Recall    : {best_dice['recall']:.4f}"
)


# ============================================================
# CONCLUSION
# ============================================================

print()
print("=" * 75)
print("WHAT THIS EXPERIMENT TELLS US")
print("=" * 75)

print(
    "This experiment isolates the effect of the "
    "binary probability threshold."
)

print(
    "It does NOT prove that threshold is the root "
    "cause of light-cloud under-segmentation."
)

print(
    "If lower thresholds substantially improve recall "
    "and IoU while maintaining acceptable precision, "
    "thresholding contributes to the problem."
)

print(
    "If recall remains poor even at lower thresholds, "
    "the problem is likely deeper: model confidence, "
    "training data, spectral/radiometric ambiguity, "
    "texture, spatial context, loss function, "
    "or architecture."
)

print()
print("Analysis complete.")