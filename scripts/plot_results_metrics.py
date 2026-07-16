import os
import matplotlib.pyplot as plt

# 1. Authentic data from your successful training run
epochs = list(range(1, 35))
train_loss = [
    0.4498,
    0.3836,
    0.3627,
    0.3579,
    0.3390,
    0.3374,
    0.3294,
    0.3171,
    0.3117,
    0.3155,
    0.3030,
    0.2956,
    0.2911,
    0.2882,
    0.2879,
    0.2812,
    0.2790,
    0.2771,
    0.2735,
    0.2713,
    0.2690,
    0.2634,
    0.2647,
    0.2606,
    0.2552,
    0.2558,
    0.2555,
    0.2520,
    0.2496,
    0.2440,
    0.2416,
    0.2408,
    0.2376,
    0.2372,
]
val_loss = [
    0.3276,
    0.2877,
    0.2636,
    0.2599,
    0.2738,
    0.2688,
    0.2364,
    0.2369,
    0.2427,
    0.2311,
    0.2461,
    0.2241,
    0.2192,
    0.2153,
    0.2186,
    0.2046,
    0.2197,
    0.2076,
    0.2099,
    0.2196,
    0.2170,
    0.1994,
    0.2138,
    0.2248,
    0.2151,
    0.2040,
    0.2047,
    0.2055,
    0.2047,
    0.2024,
    0.2042,
    0.1985,
    0.2087,
    0.1993,
]
train_iou = [
    0.5153,
    0.5793,
    0.5966,
    0.6009,
    0.6187,
    0.6224,
    0.6296,
    0.6405,
    0.6458,
    0.6443,
    0.6557,
    0.6647,
    0.6694,
    0.6739,
    0.6675,
    0.6764,
    0.6779,
    0.6787,
    0.6849,
    0.6874,
    0.6902,
    0.6940,
    0.6937,
    0.6947,
    0.7037,
    0.7016,
    0.7033,
    0.7077,
    0.7066,
    0.7135,
    0.7140,
    0.7192,
    0.7232,
    0.7215,
]
val_iou = [
    0.6418,
    0.6887,
    0.7094,
    0.7076,
    0.7155,
    0.7067,
    0.7374,
    0.7240,
    0.7248,
    0.7349,
    0.7197,
    0.7509,
    0.7530,
    0.7618,
    0.7554,
    0.7659,
    0.7521,
    0.7640,
    0.7627,
    0.7578,
    0.7561,
    0.7727,
    0.7571,
    0.7445,
    0.7506,
    0.7630,
    0.7659,
    0.7651,
    0.7672,
    0.7651,
    0.7656,
    0.7696,
    0.7603,
    0.7712,
]

# 2. Configure a clean, academic style layout
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Plot A: BCE + Dice Loss
ax1.plot(
    epochs,
    train_loss,
    label="Train Loss",
    color="#1f77b4",
    linewidth=2.5,
    marker="o",
    markersize=4,
)
ax1.plot(
    epochs,
    val_loss,
    label="Validation Loss",
    color="#ff7f0e",
    linewidth=2.5,
    marker="s",
    markersize=4,
)
ax1.set_title(
    "Segmentation Loss History (BCE + Dice Loss)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax1.set_xlabel("Epochs", fontsize=11)
ax1.set_ylabel("Loss Metric Value", fontsize=11)
ax1.legend(fontsize=10, loc="upper right", frameon=True)
ax1.grid(True, linestyle="--", alpha=0.6)

# Plot B: Intersection over Union (IoU)
ax2.plot(
    epochs,
    train_iou,
    label="Train IoU",
    color="#2ca02c",
    linewidth=2.5,
    marker="o",
    markersize=4,
)
ax2.plot(
    epochs,
    val_iou,
    label="Validation IoU",
    color="#d62728",
    linewidth=2.5,
    marker="s",
    markersize=4,
)
# Highlight the maximum validation milestone achieved at epoch 22
ax2.axhline(y=0.7727, color="#7f7f7f", linestyle=":", linewidth=1.5)
ax2.annotate(
    "Best Val IoU: 0.7727 (Ep 22)",
    xy=(22, 0.7727),
    xytext=(8, 0.74),
    arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
    fontsize=10,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
)

ax2.set_title(
    "Semantic Accuracy Metric (IoU History)", fontsize=13, fontweight="bold", pad=12
)
ax2.set_xlabel("Epochs", fontsize=11)
ax2.set_ylabel("IoU Score", fontsize=11)
ax2.legend(fontsize=10, loc="lower right", frameon=True)
ax2.grid(True, linestyle="--", alpha=0.6)

# Final Polish & File Export
plt.tight_layout()
output_name = "training_performance_curves.png"
plt.savefig(output_name, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n[SUCCESS] Presentation graph exported smoothly to disk!")
print(f"Target location: {os.path.abspath(output_name)}")
print("You can now drop this image directly into your presentation slide deck.")
