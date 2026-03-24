# CSV-2026 Challenge

[![CSV 2026 Challenge Site](https://img.shields.io/badge/Official-CSV%202026%20Challenge-red?style=for-the-badge)](http://www.csv-isbi.net/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](./LICENSE)

For official rules, visit the [CSV 2026 challenge page](http://119.29.231.17/index.html)
Our training code for the CSV 2026 challenge, including:

- classification view pretraining
- offline crop generation for classification
- classification training
- segmentation training

## Overview

This repository is organized into two major components:

- `classification/`: view pretraining, offline crop generation, and classification model training
- `segmentation/`: segmentation model training

The classification pipeline follows three stages:

1. view pretraining
2. offline crop generation
3. classification training

## Repository Structure

```text
CSV2026_Challenge/
├── classification/
│   ├── cls_dataset.py
│   ├── cls_models.py
│   ├── cls_pretrain.py
│   ├── cls_train.py
│   ├── cls_utils.py
│   └── offline_crop.py
├── segmentation/
│   └── seg_train.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Classification Pipeline

### 1. View Pretraining

Pretrain a ResNet-based model to classify longitudinal and transverse views using a 3-channel input: ultrasound image + plaque mask + vessel mask.

Code:

`classification/cls_pretrain.py`

Run:

```bash
python classification/cls_pretrain.py pretrain_view \
  --images_dir data/images \
  --pseudo_dir data/pseudo \
  --out_dir runs/view_pretrain_resnet \
  --imagenet_pretrained \
  --amp
```

Expected input:

- `images_dir/{case_id}.h5`
- `long_img`
- `trans_img`
- `pseudo_dir/{case_id}_pseudo.h5`
- `long_mask`
- `trans_mask`

Expected output:

- pretrained checkpoint such as `view_best.pth`
- training curves
- split file
- training history

### 2. Offline Crop Generation

Build offline cropped samples for classification training and save the corresponding manifest and group mapping for leakage-safe cross-validation.

Code:

`classification/offline_crop.py`

Run:

```bash
python classification/offline_crop.py \
  --splits_json data/splits42.json \
  --images_dir data/train/images \
  --labels_dir data/train/labels \
  --out_root data/offline_crops012
```

Expected input:

- `images_dir/{case_id}.h5`
- `long_img`
- `trans_img`
- `labels_dir/{case_id}_label.h5`
- `long_mask`
- `trans_mask`
- `cls`
- `splits_json`
- `train`
- `val`

Expected output:

```text
data/offline_crops012/
├── images/
├── labels/
├── manifest.json
├── groups.json
└── global_summary.json
```

Important:

- `manifest.json` stores sample-level records such as `new_id`, `source_id`, `cls`, `is_copy`, and `aug_idx`
- `groups.json` stores the mapping from `source_id` to generated sample IDs
- cropped samples must be split by `source_id` rather than by `new_id`

### 3. Classification Training

Train the final dual-view classification model on offline crops with supervised contrastive learning, memory bank updates, and weighted class centers.

Code:

`classification/cls_train.py`

Run:

```bash
python classification/cls_train.py \
  --manifest_json data/offline_crops012/manifest.json \
  --tv_images_dir data/offline_crops012/images \
  --tv_labels_dir data/offline_crops012/labels \
  --test_splits_json data/splits42.json \
  --test_images_dir data/test/images \
  --test_labels_dir data/test/labels \
  --out_dir runs/cls_train \
  --pretrain_ckpt runs/view_pretrain_resnet/view_best.pth \
  --center_weighted \
  --amp
```

Training and validation input:

- `manifest_json`
- `tv_images_dir/{new_id}.h5`
- `tv_labels_dir/{new_id}_label.h5`

Test input:

- `test_splits_json`
- `test`
- `test_images_dir/{case_id}.h5`
- `test_labels_dir/{case_id}_label.h5`

## Segmentation Training

Train the segmentation model using the script below.


## Notes

- Classification pretraining, offline crop generation, and classification training are designed to be used as a pipeline.
- Offline cropped samples must be split by `source_id` to avoid leakage.
- Update all paths according to your local environment before running the scripts.
