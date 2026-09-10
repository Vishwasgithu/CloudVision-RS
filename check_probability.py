import cv2
import torch
import numpy as np
from pathlib import Path
from inference_production import seg_model, seg_transform, DEVICE, PATCH_SIZE

img_path = "data/processed/patches/test/cloud/0_r0000_c0000.png"
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

H, W = img.shape[:2]
patch = img[:PATCH_SIZE, :PATCH_SIZE]

t = seg_transform(image=patch)["image"].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logit = seg_model(t)
    prob = torch.sigmoid(logit)[0,0].cpu().numpy()

prob_img = (prob * 255).clip(0,255).astype(np.uint8)
cv2.imwrite("outputs/results/gan/test_inference_probability.png", prob_img)

print("Probability map saved: outputs/results/gan/test_inference_probability.png")
print(f"Probability min:  {prob.min():.4f}")
print(f"Probability max:  {prob.max():.4f}")
print(f"Probability mean: {prob.mean():.4f}")
print(f">0.5 coverage:    {(prob > 0.5).mean()*100:.2f}%")
