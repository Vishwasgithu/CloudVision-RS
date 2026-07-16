# CloudVision-RS

## AI-Driven Cloud Analysis, Removal and Reconstruction for Remote Sensing Satellite Imagery

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![Research](https://img.shields.io/badge/Research-Remote%20Sensing-green)]()
[![Status](https://img.shields.io/badge/Status-Active-orange)]()

---

## Project Overview

Cloud contamination significantly reduces the usability of satellite imagery by obscuring critical surface information required for Earth Observation applications.

CloudVision-RS is an AI-driven research project focused on developing an advanced framework for:

- Cloud Analysis
- Cloud Detection
- Cloud Coverage Estimation
- Cloud Removal
- Satellite Image Reconstruction
- Quantitative Performance Evaluation

The project explores Computer Vision, Deep Learning, Generative AI, and Physics-Informed Learning approaches to generate scientifically reliable cloud-free satellite imagery.

---

## Problem Statement

Satellite images are extensively used in:

- Agriculture Monitoring
- Disaster Management
- Environmental Assessment
- Urban Planning
- Land Cover Mapping
- Climate Studies

However, cloud coverage often hides important geographical information, reducing the effectiveness of downstream analysis.

The objective of this project is to reconstruct cloud-obscured regions while preserving structural, spatial, and radiometric consistency.

---

## Research Objectives

1. Detect cloud-covered regions from remote sensing imagery.
2. Generate accurate cloud-free reconstructions.
3. Preserve physical and radiometric properties of the scene.
4. Evaluate reconstruction quality using standard remote sensing metrics.
5. Investigate Physics-Informed AI approaches for scientifically consistent image restoration.

---

## Dataset

### Primary Dataset

**RICE (Remote Sensing Image Cloud Removing Dataset)**

- RICE1
  - 500 image pairs
  - Cloudy Image
  - Cloud-Free Image

- RICE2
  - 736 image samples
  - Cloudy Image
  - Cloud-Free Image
  - Cloud Mask

Image Resolution:

```
512 × 512 RGB
```

---

## Project Roadmap

### Phase 1: Infrastructure Setup

- Environment Configuration
- Project Structure Creation
- GitHub Repository Setup
- Dataset Preparation
- Metrics Implementation

### Phase 2: Cloud Detection

- Attention U-Net
- Binary Cloud Segmentation
- IoU Evaluation

### Phase 3: Baseline Cloud Reconstruction

- U-Net Reconstruction Model
- Performance Benchmarking

### Phase 4: GAN-Based Cloud Removal

- SpA-GAN
- Spatial Attention Mechanisms
- Adversarial Training

### Phase 5: Physics-Informed Learning

- Spectral Consistency Loss
- Radiometric Consistency Loss
- Physics-Constrained Optimization

### Phase 6: Diffusion Refinement (Optional)

- DDIM-Based Refinement
- Texture Enhancement
- Error Correction

### Phase 7: Deployment

- Interactive Streamlit Application
- Visualization Dashboard
- Result Analysis Interface

---

## Evaluation Metrics

The project uses standard image restoration and remote sensing evaluation metrics:

- PSNR
- SSIM
- RMSE
- IoU
- Dice Score
- LPIPS
- SAM
- ERGAS

---

## Technology Stack

### Programming

- Python

### Computer Vision

- OpenCV
- NumPy

### Deep Learning

- PyTorch
- CUDA

### Data Analysis

- Pandas
- SciPy

### Visualization

- Matplotlib

### Development

- Git
- GitHub
- VS Code
- Anaconda

---

## Current Status

### Completed

- Environment Setup
- Project Structure
- Git Repository Initialization
- GitHub Repository Setup
- Scientific Library Installation
- Research Planning

### In Progress

- PyTorch Setup
- Dataset Integration
- Cloud Detection Pipeline

### Upcoming

- Attention U-Net Implementation
- SpA-GAN Development
- Physics-Informed Loss Functions

---

## Repository Structure

```text
CloudVision-RS

├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── configs/          # model & data hyperparameters (gan_config.yaml, ...)
├── src/              # source code (models, losses, data, training)
├── scripts/          # utility scripts (dataset build, plotting)
├── notebooks/        # learning / exploration notebooks
├── docs/             # reports & documentation (Phase3_Report.md, ...)
│
├── assets/           # committed images used in README/docs
│   ├── architecture.png
│   ├── phase2_results.png
│   ├── phase3_results.png
│   └── training_curves.png
│
├── outputs/          # runtime-generated, git-ignored
│   ├── checkpoints/
│   ├── logs/
│   └── results/
│
├── datasets/         # raw RICE data, git-ignored
└── tests/
```

---

## Future Scope

- Multi-Spectral Cloud Removal
- Sentinel-2 Adaptation
- Landsat-8 Adaptation
- Physics-Informed Diffusion Models
- Large-Scale Earth Observation Applications

---

## Author

Vishwas Choudhary

AI & Data Science

NRSC–ISRO AI/ML Research Intern
