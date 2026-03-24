# CSV-2026 

This repository is organized around three core training code:

- classification pretraining
- classification training
- segmentation training


## Directory Layout

```text
CSV2026_Challenge/
├── classification/
│   ├── cls_pretrain.py
│   └── cls_train.py
├── segmentation/
│   └── seg_train.py
├── requirements.txt
├── .gitignore
└── README.md
```


## Training

### View pretraining
```bash
python classification/pretrain_view.py \
  --images_dir data/images \
  --pseudo_dir data/pseudo \
  --out_dir runs/view_pretrain_resnet \
  --id_min 200 --id_max 999 \
  --epochs 80 --batch_size 8 --num_workers 4 \
  --imagenet_pretrained --amp
```

### 2. Classification Training

`classification/cls_train.py`

Trains the classification model on offline crops with supervised contrastive learning and center weighting.

```bash
python classification/resnet_offlinetrain_supconB_center_weighted_v2.py \
  --manifest_json data/offline_crops012/manifest.json \
  --tv_images_dir data/offline_crops012/images \
  --tv_labels_dir data/offline_crops012/labels \
  --test_splits_json data/splits42.json \
  --test_images_dir data/test/images \
  --test_labels_dir data/test/labels \
  --out_dir runs/resnet_supconB_center_weighted \
  --pretrain_ckpt runs/view_pretrain_resnet/view_best.pth \
  --epochs 50 \
  --batch_size 8 \
  --num_workers 4 \
  --amp
```

### 3. Segmentation Training

`segmentation/seg_train.py`

Trains a 2D UNet-style segmentation model based on an nnU-Net plan file.

```bash
python segmentation/train_unet2d_h5_by_nnunet_plan.py \
  --h5_dir data/seg_h5 \
  --plans_json data/nnunet_plans.json \
  --config 2d \
  --out_dir runs/unet2d_plan \
  --epochs 200 \
  --batch_size 16
```

## Environment

Recommended:

- Python 3.10
- PyTorch 2.x
- CUDA-capable GPU for training

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

