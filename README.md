# SRCC: Semantic-Reliability Co-Calibration for Noisy Multi-Label Chest X-Ray Image Classification

This repository provides a PyTorch implementation of our method for noisy multi-label chest X-ray (CXR) classification.

---

## 🔍 Overview

We propose a **semantic-reliability co-calibration framework** that jointly addresses:

- Representation bias caused by noisy labels
- Supervision reliability heterogeneity across classes and samples
- Biased label dependency modeling

Our method integrates:

- Semantic alignment between image features and label embeddings
- Dynamic graph convolution for label dependency modeling
- Reliability-aware supervision adjustment under noise and imbalance

---

## 📁 Repository Structure
SRCC/
├── data/ # dataset loaders
├── utils/ # loss functions and utilities
├── gcn_ours.py # model definition
├── train_lsrs_clean.py # training script
├── eval_lsrs_clean.py # evaluation script
├── metrics.py # evaluation metrics
├── voc.py # VOC dataset utilities


---

## ⚙️ Environment

- Python 3.8
- PyTorch 1.11
- CUDA (recommended)

Install dependencies:

```bash
pip install torch torchvision numpy

## 📊 Datasets
We conduct experiments on:

NIH ChestX-ray14
CheXpert (CXP)
OpenI
PadChest
Pascal VOC (for generalization)


