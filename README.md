# P2WNet: Homography Estimation for Part-to-Whole and Cross-Modality Scenarios (ICME 2025 Modified Version)

## Introduction

This GitHub repository provides the official implementation of **P2WNet: Homography Estimation for Part-to-Whole and Cross-Modality Scenarios** (ICME 2025 modified version).

P2WNet focuses on solving homography estimation problems in **Part-to-Whole** and **Cross-Modality** scenarios, which are highly valuable for image registration, cross-modal matching, and related applications.

The repository includes:
- Complete training and inference code
- Model definitions and training scripts
- Evaluation tools and visualization scripts
- Preprocessing and data loading modules

---

## 📥 Dataset Download

The dataset constructed for P2WNet (containing various Part-to-Whole and cross-modality image pairs) can be downloaded from Hugging Face:

**Download Link**:  
[https://huggingface.co/datasets/ckkkk333000/P2WNet_dataset](https://huggingface.co/datasets/ckkkk333000/P2WNet_dataset)

After downloading, please extract it to the `./dataset/` directory (or update the path in your configuration file).

---

## 📦 Pretrained Models Download

Pre-trained model weights (including versions for different modalities and resolutions) are available for download:

**Download Link**:  
[https://huggingface.co/datasets/ckkkk333000/P2WNet](https://huggingface.co/datasets/ckkkk333000/P2WNet)

After downloading, place the models in the `./checkpoints/` directory for immediate inference or fine-tuning.

---

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
