# AI-Driven Cloud Analysis and Removal from Remote Sensing Satellite Imagery

**A Comprehensive Technical Report**

---

## 1. Executive Summary

Cloud contamination represents one of the most significant barriers to the effective utilization of satellite imagery in remote sensing applications. Optical satellite sensors are frequently obstructed by clouds, rendering large portions of acquired data unusable for critical applications such as agriculture monitoring, disaster management, and environmental surveillance.

This project presents a novel, two-stage Physics-Informed AI framework for automated cloud analysis and removal from remote sensing satellite imagery. Unlike conventional approaches that attempt to reconstruct entire images indiscriminately, our framework first performs precise semantic segmentation to identify cloud-covered regions, followed by targeted reconstruction of only those regions using a Physics-Informed Conditional Generative Adversarial Network (cGAN).

**Key Results:**
- **Cloud Segmentation:** Attention U-Net with ResNet34 encoder achieves **77.27% IoU** for pixel-accurate cloud mask generation
- **Cloud Reconstruction:** Physics-Informed cGAN achieves **24.74 dB PSNR**, **0.7375 SSIM**, and **0.1406 VARI-RMSE** on the test set (536 patches)
- **End-to-End Pipeline:** Complete workflow from raw satellite imagery to cloud-free output, validated on the RICE2 dataset

The integration of physics-informed constraints ensures that reconstructed images maintain spectral consistency and radiometric fidelity, making the framework suitable for scientific applications where visual realism alone is insufficient.

---

## 2. Introduction

### 2.1 Problem Statement

Satellite remote sensing has become an indispensable tool for monitoring Earth's surface. However, cloud cover remains a persistent and pervasive challenge. Clouds obscure surface features, rendering satellite imagery unsuitable for time-critical applications such as crop health monitoring, flood assessment, wildfire detection, and urban planning.

Traditional cloud removal approaches suffer from several limitations:
- **Indiscriminate reconstruction:** Methods that process entire images often modify cloud-free regions unnecessarily
- **Lack of physical accuracy:** Purely visual reconstruction may produce spectrally invalid results
- **Poor boundary delineation:** Without precise segmentation, reconstruction artifacts appear at cloud boundaries

The problem is analogous to repairing a damaged road: one would repair only the damaged portion, not the entire road. Similarly, cloud removal systems should reconstruct only cloud-covered regions while preserving the integrity of cloud-free areas.

### 2.2 Project Objective

The objective of this project is to develop an end-to-end AI-driven framework that:

1. **Accurately identifies** cloud-covered regions through pixel-level semantic segmentation
2. **Precisely reconstructs** only the cloud-obscured areas using generative AI
3. **Preserves scientific validity** by incorporating physics-informed constraints that maintain spectral and reflectance properties

The framework follows a systematic two-stage architecture:

```
Satellite Image
      │
      ▼
Dataset Preparation
      │
      ▼
Cloud Analysis & Segmentation
    (Attention U-Net)
      │
      ▼
    Cloud Mask
      │
      ▼
Physics-Informed Conditional GAN
      │
      ▼
Cloud-Free Satellite Image
      │
      ▼
Performance Evaluation
 (PSNR • SSIM • VARI RMSE)
      │
      ▼
   Final Cloud-Free Output
```

---

## 3. Methodology

### 3.1 Phase 1: Cloud Analysis & Semantic Segmentation

#### 3.1.1 What is Cloud Analysis?

Cloud analysis is the process of studying a satellite image to determine which regions are covered by clouds. The goal at this stage is **not** to remove clouds, but to detect their exact location with pixel-level precision.

**Why do we need cloud analysis?**
If the reconstruction model receives the entire image without knowing where clouds are located, it may unnecessarily modify cloud-free regions. Cloud analysis enables the model to focus exclusively on regions that actually need reconstruction, preserving the original information in clear areas.

#### 3.1.2 What is Semantic Segmentation?

Semantic segmentation is a computer vision technique where every pixel in an image is assigned a class label. In our project, each pixel is classified as either:

- **Cloud** (foreground)
- **Non-Cloud** (background)

Unlike image classification, which produces a single label for an entire image, semantic segmentation generates a detailed pixel-level map. This granularity is essential for cloud removal because the reconstruction model requires exact coordinates of cloud-covered regions to function effectively.

**Why Semantic Segmentation?**
Cloud removal requires precise spatial information about cloud locations. Pixel-level segmentation provides an accurate cloud mask, allowing the reconstruction model to work exclusively on cloud-covered regions while leaving cloud-free areas untouched.

#### 3.1.3 Architecture: Attention U-Net with ResNet34 Encoder

**What is Attention U-Net?**

Attention U-Net is an enhanced variant of the original U-Net architecture, specifically designed for medical and remote sensing segmentation tasks. It incorporates **Attention Gates** that automatically focus on relevant regions while suppressing irrelevant background information.

**Why Attention U-Net?**

We selected Attention U-Net based on the following technical advantages:

- **Superior segmentation performance:** Demonstrated state-of-the-art results across multiple segmentation benchmarks
- **Precise boundary detection:** Accurately captures cloud edges and irregular cloud boundaries
- **Targeted feature focus:** Attention mechanisms prioritize cloud regions over background
- **Reduced false positives:** Minimizes misclassification of bright surfaces (snow, sand) as clouds
- **Robustness to complexity:** Handles diverse cloud patterns—from thin cirrus clouds to thick cumulus formations

Since the reconstruction model's performance directly depends on mask quality, achieving high segmentation accuracy is paramount.

**What is ResNet34 Encoder?**

The encoder serves as the feature extraction backbone of the network. Instead of training from scratch, we leverage **ResNet34**, a pre-trained convolutional neural network originally trained on ImageNet (14 million images).

**Why ResNet34?**

ResNet34 provides:

- **Transfer learning benefits:** Pre-trained weights capture general image features (edges, textures, patterns) that accelerate convergence
- **Reduced training time:** Fewer epochs required to reach optimal performance
- **Improved feature representation:** Deep residual connections enable learning of hierarchical features
- **Parameter efficiency:** 34-layer architecture balances depth with computational feasibility

**What are Skip Connections?**

During the encoding process, fine spatial details are progressively lost as the network abstracts features into deeper layers. Skip connections are direct pathways that transfer feature maps from the encoder to the corresponding decoder layers.

**Why Skip Connections?**

Skip connections preserve:

- **Cloud boundaries:** Fine edge information survives the bottleneck
- **Textural details:** Small-scale features critical for accurate segmentation
- **Spatial relationships:** Positional information that would otherwise be lost

These preserved features enable the decoder to reconstruct high-resolution segmentation maps with precise boundary delineation.

#### 3.1.4 Attention Mechanisms: SCSE Module

**What is SCSE (Spatial and Channel Squeeze-and-Excitation)?**

SCSE is a dual-attention mechanism that enhances feature representations through two complementary pathways:

- **Channel Attention:** Identifies *which* feature channels are most important for the current prediction
- **Spatial Attention:** Identifies *where* in the spatial dimension the important features are located

**Why SCSE?**

Clouds exhibit significant variability in size, shape, opacity, and brightness. The SCSE module enables the model to:

- **Focus on relevant cloud features:** Amplify informative channels (e.g., cloud texture, brightness) while suppressing irrelevant ones
- **Localize cloud regions:** Spatial attention highlights cloud locations in the feature map
- **Adapt to diverse patterns:** The mechanism dynamically adjusts to different cloud morphologies

This results in more accurate segmentation masks with fewer false detections.

#### 3.1.5 Loss Function: BCE + Dice Loss

**What is a Loss Function?**

A loss function quantifies the discrepancy between the predicted segmentation mask and the ground truth mask. Minimizing this loss during training guides the model toward accurate predictions.

**Why BCE + Dice Loss?**

We combined two complementary loss functions because each addresses a different aspect of segmentation quality:

- **Binary Cross Entropy (BCE):** Treats each pixel independently, ensuring correct classification at each spatial location. It is effective for learning the general shape and distribution of cloud regions.
- **Dice Loss:** Measures the overlap between predicted and ground truth masks by computing the Dice coefficient. It is insensitive to class imbalance and focuses on region-level similarity.

**Synergy:** BCE handles pixel-wise accuracy, while Dice Loss ensures region-wise overlap. Together, they produce segmentation masks that are both locally accurate and globally consistent.

#### 3.1.6 Optimizer & Learning Rate Strategy

**What is an Optimizer?**

An optimizer updates the model's learnable parameters (weights) after each training step to minimize the loss function. The choice of optimizer significantly impacts training stability and convergence speed.

**Why AdamW?**

AdamW is a variant of the Adam optimizer with decoupled weight decay. It provides:

- **Faster convergence:** Adaptive learning rates for each parameter
- **Training stability:** Prevents gradient explosions common in segmentation networks
- **Better regularization:** Weight decay promotes generalization without explicit regularization layers

**What is Cosine Annealing Learning Rate Scheduling?**

Cosine Annealing is a learning rate scheduler that gradually reduces the learning rate following a cosine curve from the initial value to near-zero.

**Why Cosine Annealing?**

- **Phase 1 (High Learning Rate):** Enables rapid initial learning and quick escape from poor local minima
- **Phase 2 (Low Learning Rate):** Allows fine-grained parameter adjustments for convergence to an optimal solution
- **Smooth decay:** Prevents sudden learning rate jumps that could destabilize training

This scheduling strategy balances exploration (early training) with exploitation (late training).

#### 3.1.7 Evaluation Metric: Intersection over Union (IoU)

**What is IoU?**

IoU (Intersection over Union), also known as the Jaccard Index, measures the overlap between the predicted cloud mask and the ground truth cloud mask.

**Formula:**
```
IoU = (Area of Intersection) / (Area of Union)
```

**Why IoU?**

IoU is the standard metric for segmentation tasks because it:

- **Measures region overlap:** Rewards predictions that closely match ground truth boundaries
- **Handles class imbalance:** Robust when cloud pixels are sparse or dense
- **Threshold-independent:** Provides a single scalar value that directly reflects segmentation quality

An IoU of ~77.27% indicates strong segmentation performance, meaning the model correctly identifies approximately 77% of cloud pixels while minimizing false positives.

---

### 3.2 Phase 2: Cloud Reconstruction (Physics-Informed cGAN)

#### 3.2.1 What is Conditional GAN (cGAN)?

A Conditional Generative Adversarial Network (cGAN) is a generative model that learns to produce realistic outputs conditioned on additional input information. In our case, the generator receives a cloudy image and a cloud mask as inputs and generates a cloud-free image.

**Architecture Components:**

**Generator — U-Net Architecture:**
- **Input:** 5 channels = RGB cloudy image (3) + cloud mask (1) + Sobel edge map of mask (1)
- **Output:** 3-channel RGB cloud-free image in range [-1, 1]
- **Encoder-Decoder Structure:** Progressive downsampling to capture context, followed by upsampling to restore spatial resolution
- **Skip Connections:** Concatenate encoder features to decoder layers to preserve fine details
- **Bilinear Upsampling:** Used instead of transposed convolutions to avoid checkerboard artifacts

**Discriminator — PatchGAN Architecture:**
- **Input:** 6 channels = cloudy condition (3) + generated/real target (3)
- **Output:** 30×30 grid of real/fake logits (each representing a 70×70 receptive field patch)
- **Purpose:** Judges local texture patches, ensuring the generator produces realistic outputs everywhere, not just globally

#### 3.2.2 Why Physics-Informed Learning?

Standard GANs optimize for visual realism using adversarial and pixel-wise losses. However, in remote sensing applications, **visual realism is insufficient**. The reconstructed images must preserve:

- **Spectral signatures:** Correct reflectance properties across RGB channels
- **Vegetation indices:** Accurate greenness ratios for agricultural analysis
- **Radiometric consistency:** Physically meaningful pixel values

**How is Physics Integrated?**

The physics-informed loss comprises three components:

1. **VARI Loss (Vegetation Atmospherically Resistant Index):**
   - Formula: `VARI = (G - R) / (G + R - B)`
   - Measures vegetation greenness while compensating for atmospheric effects
   - Ensures reconstructed vegetation has correct spectral ratios

2. **Spectral Ratio Loss:**
   - Enforces consistency in R/G, B/G, and R/B ratios
   - Prevents color shifts that would make the image scientifically invalid

3. **Edge Loss:**
   - Preserves sharp boundaries at cloud edges
   - Uses gradient-based coherence to prevent seam artifacts

**Loss Weighting:**
```
L_G = L_adv (BCE) + λ_L1 · L1(fake, target) + L_physics
L_physics = 0.5·L_VARI + 1.0·L_spectral + 5.0·L_edge
```

The physics loss is weighted alongside adversarial and L1 losses to balance realism with scientific validity.

#### 3.2.3 How the Cloud Mask Guides Reconstruction

The cloud mask from Phase 1 serves as a **spatial prior** for the GAN:

- **Conditional Input:** The mask is concatenated with the cloudy image and fed to the generator
- **Selective Reconstruction:** The GAN learns to modify only cloud-covered regions while preserving cloud-free areas
- **Boundary Preservation:** The Sobel edge map derived from the mask helps the generator maintain sharp transitions at cloud boundaries

**Synergy Between Phases:**
The segmentation mask directly optimizes the GAN's reconstruction task by:
1. Reducing the search space — the generator only needs to inpaint masked regions
2. Providing structural guidance — the mask architecture informs the generator where fine details matter most
3. Enabling physics loss targeting — spectral constraints are applied specifically to reconstructed regions

#### 3.2.4 Training Stability Mechanisms

Training GANs is notoriously challenging due to the adversarial minimax game between Generator and Discriminator. We employed several stabilization techniques:

1. **Two-Timescale Updates:** Train Discriminator once, Generator once per batch
2. **Gradient Clipping:** Maximum gradient norm of 1.0 on both networks to prevent explosions
3. **L1 Loss Weight (λ=100):** Strong pixel-level supervision prevents mode collapse and spectrally random outputs
4. **Label Smoothing:** Real labels set to 0.9 instead of 1.0 to prevent discriminator overconfidence
5. **Adam Optimizer:** lr=2e-4, betas=(0.5, 0.999) — standard configuration for GAN training

These stabilizers ensure the generator learns meaningful representations without being dominated by the discriminator.

---

## 4. Experimental Setup & Implementation

### 4.1 Dataset: RICE (Remote Sensing Image Cloud Removing)

The framework was trained and evaluated on the **RICE dataset**, specifically:

- **RICE1:** 500 image pairs of cloudy and cloud-free satellite imagery
- **RICE2:** 736 samples with paired masks, providing cloud ground truth for supervised training
- **Image Specifications:** 512×512 RGB patches, tiled into 256×256 manageable patches for deep learning

The dataset contains diverse cloud patterns including thin cirrus clouds, thick cumulus clouds, and cloud shadows, providing a robust testbed for cloud removal algorithms.

### 4.2 Data Pipeline Architecture

The data pipeline was implemented using PyTorch's `Dataset` and `DataLoader` abstractions for efficient, reproducible training:

- **Dataset Class:** Handles patch loading, normalization, and augmentation
- **Normalization:** Images normalized to [-1, 1] range (standard GAN convention)
- **DataLoader:** Batched loading with shuffling for training, sequential loading for validation
- **Augmentation:** Applied to improve generalization given the limited dataset size

### 4.3 Evaluation Metrics

#### 4.3.1 Segmentation Metrics: IoU (Intersection over Union)

IoU measures the overlap between predicted and ground truth segmentation masks. It is computed as:

```
IoU = (TP) / (TP + FP + FN)
```

Where TP = True Positives, FP = False Positives, FN = False Negatives.

An IoU of 77.27% indicates that approximately 77% of the predicted cloud regions correctly overlap with ground truth cloud regions.

#### 4.3.2 Reconstruction Metrics

**PSNR (Peak Signal-to-Noise Ratio):**
- Measures pixel-level fidelity between generated and ground truth images
- Higher values indicate closer match
- **24.74 dB** indicates good reconstruction quality for satellite imagery

**SSIM (Structural Similarity Index):**
- Measures structural similarity in terms of luminance, contrast, and structure
- Values closer to 1 indicate better structural preservation
- **0.7375** indicates good preservation of edges, textures, and spatial relationships

**VARI-RMSE (Visible Atmospherically Resistant Index Root Mean Square Error):**
- Measures spectral accuracy in vegetation regions
- Lower values indicate better preservation of vegetation spectral signatures
- **0.1406** confirms the physics-informed loss successfully maintains spectral consistency

---

## 5. Results & Discussion

### 5.1 Key Achievements

The project successfully delivered a complete end-to-end cloud removal framework with the following milestones:

- **Dataset Preparation:** Successfully processed and prepared the RICE2 dataset for deep learning, including patch extraction, normalization, and mask generation
- **Data Pipeline:** Built an efficient PyTorch-based data pipeline with `Dataset` and `DataLoader` for seamless training and evaluation
- **Cloud Segmentation:** Developed Attention U-Net with ResNet34 backbone, achieving **77.27% IoU** for pixel-accurate cloud mask generation
- **Cloud Reconstruction:** Designed and trained a Physics-Informed cGAN achieving:
  - **24.74 ± 4.89 dB PSNR**
  - **0.7375 ± 0.1383 SSIM**
  - **0.1406 ± 0.1226 VARI-RMSE**
- **Physics Integration:** Successfully incorporated spectral and reflectance constraints, reducing VARI-RMSE from ~15713 (pre-fix) to 0.14 (post-fix)

### 5.2 Challenges Encountered & Solutions

**Challenge 1: Limited Dataset Size**
- **Problem:** The RICE2 dataset, while appropriate, is relatively small for training deep generative models, risking poor generalization
- **Solution:** Applied data augmentation techniques including random flips, rotations, and color jittering to artificially expand the training distribution

**Challenge 2: Dense Cloud Regions**
- **Problem:** Thick cloud regions contain minimal underlying surface information, making reconstruction extremely difficult
- **Solution:** Physics-informed loss functions guide the generator toward spectrally plausible reconstructions, while the adversarial loss encourages visual realism

**Challenge 3: GAN Training Stability**
- **Problem:** GANs are inherently unstable due to the adversarial minimax game between generator and discriminator
- **Solution:** Applied comprehensive stabilization strategies: two-timescale updates, gradient clipping, label smoothing, L1 loss anchoring (λ=100), and cosine annealing learning rate scheduling

**Challenge 4: Scientific Accuracy**
- **Problem:** Visual realism alone is insufficient for remote sensing; reconstructed images must preserve spectral information
- **Solution:** Introduced physics-informed loss components (VARI, spectral ratios, edge coherence) that enforce spectral consistency, ensuring outputs are suitable for downstream vegetation analysis and radiometric indexing

---

## 6. Real-World Impact & Applications

The developed framework has significant practical implications across multiple domains:

### 6.1 Agriculture
Cloud-free satellite images enable accurate crop monitoring, vegetation health assessment, and precision agriculture. Farmers and agronomists can analyze crop patterns, detect stress, and optimize irrigation without waiting for clear-sky windows.

### 6.2 Disaster Management
During floods, wildfires, or cyclones, timely satellite imagery is critical. Clouds often obstruct disaster zones when monitoring is most needed. Our framework enables analysis during cloudy conditions, supporting emergency response teams with current surface information.

### 6.3 Urban Planning
Urban planners rely on satellite imagery for infrastructure development, land-use analysis, and population monitoring. Cloud-free images provide reliable base data for zoning decisions, transportation planning, and smart city initiatives.

### 6.4 Environmental Monitoring
Long-term observation of forests, rivers, glaciers, and ecosystems requires consistent satellite data. Cloud removal enables reliable time-series analysis, supporting climate change research, deforestation tracking, and water resource management.

### 6.5 Defense and Surveillance
Military and intelligence applications demand continuous terrain monitoring. Improved satellite imagery assists in terrain analysis, strategic monitoring, and border surveillance, where cloud cover should not compromise operational awareness.

---

## 7. Conclusion & Future Work

### 7.1 Summary of Contributions

This project successfully demonstrates the integration of semantic segmentation and generative AI for practical cloud removal in remote sensing. The two-stage framework—Attention U-Net for cloud segmentation followed by Physics-Informed cGAN for reconstruction—addresses the fundamental challenge of preserving both visual quality and scientific validity in reconstructed satellite imagery.

The key technical contributions include:

1. **Precise Cloud Segmentation:** Attention U-Net with ResNet34 encoder achieves 77.27% IoU, providing accurate spatial priors for the reconstruction stage
2. **Physics-Informed Reconstruction:** Novel integration of spectral constraints (VARI, spectral ratios) into the GAN framework, reducing VARI-RMSE from 15713 to 0.14
3. **End-to-End Pipeline:** Complete workflow from dataset preparation through evaluation, with reproducible training and testing protocols
4. **Robust Training Framework:** Comprehensive stabilization strategies enabling stable GAN training on limited hardware

### 7.2 Limitations

Despite strong performance, several limitations warrant acknowledgment:

- **Dense cloud cores:** The most challenging areas remain scenes with thick, opaque clouds where underlying features are almost entirely obscured
- **Dataset scale:** Performance could improve with larger, more geographically diverse training data
- **Computational resources:** Training was conducted on consumer-grade GPU (RTX 3050, 4.3 GB VRAM), limiting batch size and model capacity

### 7.3 Future Work

Several promising directions exist for extending this framework:

#### Diffusion Models
Modern diffusion models have demonstrated superior generative quality compared to GANs. Replacing the cGAN with a diffusion-based approach could yield higher-fidelity reconstructions with better preservation of fine image details and more stable training dynamics.

#### Vision Transformers (ViTs)
Transformer-based architectures capture long-range spatial relationships more effectively than convolutional networks. Integrating Vision Transformers or hybrid CNN-Transformer architectures could improve the model's understanding of global context in both segmentation and reconstruction tasks.

#### Multi-Temporal Data Fusion
Instead of relying solely on a single cloudy image, future models could fuse information from multiple satellite captures taken at different times. Temporal information provides complementary views of cloud-covered regions, potentially enabling more accurate reconstruction of heavily obscured areas.

#### SAR (Synthetic Aperture Radar) Integration
SAR sensors can penetrate clouds, providing complementary information to optical imagery. Fusing optical satellite images with SAR data could significantly improve reconstruction accuracy, particularly under thick cloud cover where optical data is completely obscured.

#### Larger and More Diverse Datasets
Training on larger, geographically diverse datasets would improve the model's ability to generalize across different regions, seasons, and cloud types. Including multi-spectral and hyper-spectral data would further enhance the framework's utility for specialized remote sensing applications.

#### Foundation Models for Remote Sensing
Recent foundation models pre-trained on massive satellite imagery corpora offer powerful transfer learning capabilities. Fine-tuning such models for cloud removal could leverage learned representations to achieve superior performance with reduced training data requirements.

---

## 8. Conclusion

This project presents a comprehensive AI-driven framework for cloud analysis and removal in remote sensing satellite imagery. By integrating Attention U-Net for precise cloud segmentation with a Physics-Informed Conditional GAN for image reconstruction, the framework generates cloud-free satellite images that are both visually realistic and scientifically valid.

The combination of semantic segmentation, generative adversarial networks, and physics-informed learning creates a balanced approach that addresses the unique challenges of remote sensing applications. The evaluation results—77.27% IoU for segmentation and 24.74 dB PSNR with 0.1406 VARI-RMSE for reconstruction—demonstrate the framework's effectiveness and its potential to support real-world remote sensing operations where cloud contamination is a pervasive challenge.

This work establishes a foundation for future research in physics-aware generative models for scientific imaging applications, bridging the gap between computer vision and remote sensing science.

---

## Appendix: Project Journey & Learnings

### From Concept to Implementation

This project began with a fundamental problem: clouds hide valuable information in satellite images, limiting their utility for remote sensing applications. The solution required not just removing clouds, but understanding where they are and how to reconstruct obscured regions while preserving scientific integrity.

The development followed a systematic research and engineering workflow:

1. **Literature Review:** Studied semantic segmentation architectures (U-Net, Attention U-Net) and generative models (GANs, cGANs) for image restoration
2. **Dataset Analysis:** Analyzed the RICE dataset structure, identifying challenges such as class imbalance and variable cloud density
3. **Segmentation Model Development:** Implemented and trained Attention U-Net, iterating on attention mechanisms and loss functions to achieve 77.27% IoU
4. **Physics-Informed Design:** Integrated spectral constraints into the GAN framework, requiring deep understanding of vegetation indices and radiometric properties
5. **End-to-End Integration:** Connected segmentation outputs to reconstruction inputs, optimizing the pipeline for practical deployment

### Technical Lessons Learned

- **Architecture Selection Matters:** Attention mechanisms significantly improve segmentation of irregular objects like clouds
- **Physics-Informed Constraints Are Essential:** In scientific applications, visual realism must be complemented by spectral validity
- **GAN Training Requires Patience:** Stabilization techniques are not optional—they are prerequisites for successful training
- **Evaluation Must Be Multi-Faceted:** PSNR, SSIM, and spectral metrics together provide a complete picture of reconstruction quality

### Engineering Insights

- **Modular Design:** Separating segmentation and reconstruction into distinct phases enables independent optimization and debugging
- **Checkpointing Strategy:** Regular model snapshots (every 5 epochs) enable recovery and model selection without retraining
- **Environment Management:** Dependency versions (particularly NumPy compatibility) can silently break evaluation pipelines
- **Documentation:** Inline code documentation and structured notebooks accelerate knowledge transfer and reproducibility

This project has been instrumental in developing expertise at the intersection of deep learning, computer vision, and remote sensing science, providing a strong foundation for future research in AI-driven Earth observation systems.
