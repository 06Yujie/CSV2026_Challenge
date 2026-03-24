import os
import json
import time
import random
import argparse
from typing import Dict, List, Tuple

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tvm


def format_case_id(x) -> str:
    return str(x).zfill(4)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_image_path(images_dir: str, case_id: str) -> str:
    return os.path.join(images_dir, f"{case_id}.h5")


def get_pseudo_path(pseudo_dir: str, case_id: str) -> str:
    return os.path.join(pseudo_dir, f"{case_id}_pseudo.h5")


def read_image_h5(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        if "long_img" not in keys or "trans_img" not in keys:
            raise KeyError(f"Missing long_img/trans_img in {path}. Keys={list(keys)}")
        long_img = f["long_img"][()]
        trans_img = f["trans_img"][()]
    return long_img.astype(np.float32), trans_img.astype(np.float32)


def read_pseudo_h5(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        if "long_mask" not in keys or "trans_mask" not in keys:
            raise KeyError(f"Missing long_mask/trans_mask in pseudo {path}. Keys={list(keys)}")
        long_mask = f["long_mask"][()]
        trans_mask = f["trans_mask"][()]
    return long_mask.astype(np.float32), trans_mask.astype(np.float32)


def list_available_case_ids(
    images_dir: str,
    pseudo_dir: str,
    id_min: int,
    id_max: int,
) -> List[str]:
    case_ids = []
    for n in range(id_min, id_max + 1):
        case_id = format_case_id(n)
        image_path = get_image_path(images_dir, case_id)
        pseudo_path = get_pseudo_path(pseudo_dir, case_id)
        if os.path.exists(image_path) and os.path.exists(pseudo_path):
            case_ids.append(case_id)
    return case_ids


def build_input_3ch(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.ndim != 2 or mask.ndim != 2:
        raise ValueError(f"Expect 2D arrays, got image={image.shape}, mask={mask.shape}")

    mean = float(image.mean())
    std = float(image.std())
    image_z = (image - mean) / (std + 1e-8)

    if np.isclose(mask, 128.0).any() or np.isclose(mask, 255.0).any():
        plaque = np.isclose(mask, 128.0).astype(np.float32)
        vessel = np.isclose(mask, 255.0).astype(np.float32)
    else:
        unique_values = np.unique(mask.astype(np.int32))
        unique_set = set(unique_values.tolist())
        if unique_set.issubset({0, 1, 2}):
            plaque = (mask == 1).astype(np.float32)
            vessel = (mask == 2).astype(np.float32)
        else:
            raise ValueError(f"Unknown pseudo mask coding. unique={np.unique(mask)[:20]} ...")

    return np.stack([image_z.astype(np.float32), plaque, vessel], axis=0)


def save_curves(output_path: str, history: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(8, 5))
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    if "val_acc" in history:
        plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("view pretraining")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


class ViewPretrainNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        imagenet_pretrained: bool = True,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = tvm.ResNet18_Weights.DEFAULT if imagenet_pretrained else None
        backbone = tvm.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ViewPretrainDataset(Dataset):
    def __init__(self, case_ids: List[str], images_dir: str, pseudo_dir: str):
        self.images_dir = images_dir
        self.pseudo_dir = pseudo_dir
        self.samples = []
        for case_id in case_ids:
            self.samples.append((case_id, 0))
            self.samples.append((case_id, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        case_id, view_label = self.samples[index]
        image_path = get_image_path(self.images_dir, case_id)
        pseudo_path = get_pseudo_path(self.pseudo_dir, case_id)

        long_img, trans_img = read_image_h5(image_path)
        long_mask, trans_mask = read_pseudo_h5(pseudo_path)

        if view_label == 0:
            image, mask = long_img, long_mask
        else:
            image, mask = trans_img, trans_mask

        x = torch.from_numpy(build_input_3ch(image, mask)).float()
        y = torch.tensor(view_label, dtype=torch.long)
        return case_id, x, y


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    num_samples = 0
    num_correct = 0

    for _, x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        batch_size = x.size(0)
        loss_sum += float(loss.item()) * batch_size
        num_correct += int((pred == y).sum().item())
        num_samples += batch_size

    acc = num_correct / max(num_samples, 1)
    avg_loss = loss_sum / max(num_samples, 1)
    return avg_loss, float(acc)


def train(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    case_ids = list_available_case_ids(
        images_dir=args.images_dir,
        pseudo_dir=args.pseudo_dir,
        id_min=args.id_min,
        id_max=args.id_max,
    )
    if len(case_ids) == 0:
        raise RuntimeError(
            f"No existing ids in range {args.id_min}-{args.id_max} "
            f"under images={args.images_dir} pseudo={args.pseudo_dir}"
        )

    rng = np.random.RandomState(args.seed)
    case_ids = np.array(sorted(case_ids))
    rng.shuffle(case_ids)

    num_val = int(round(len(case_ids) * args.val_frac))
    val_ids = sorted(case_ids[:num_val].tolist())
    train_ids = sorted(case_ids[num_val:].tolist())

    with open(os.path.join(args.out_dir, "split_ids_view.json"), "w") as f:
        json.dump({"train": train_ids, "val": val_ids}, f, indent=2)

    train_dataset = ViewPretrainDataset(train_ids, args.images_dir, args.pseudo_dir)
    val_dataset = ViewPretrainDataset(val_ids, args.images_dir, args.pseudo_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_device = "cuda" if device == "cuda" else "cpu"

    model = ViewPretrainNet(
        num_classes=2,
        imagenet_pretrained=args.imagenet_pretrained,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(amp_device, enabled=args.amp)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_ckpt_path = os.path.join(args.out_dir, "view_best.pth")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(
        f"[DATA] n_cases={len(case_ids)} "
        f"train_cases={len(train_ids)} val_cases={len(val_ids)} "
        f"train_samples={len(train_dataset)} val_samples={len(val_dataset)}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        train_loss_sum = 0.0
        num_samples = 0

        for _, x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_device, enabled=args.amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = x.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            num_samples += batch_size

        train_loss = train_loss_sum / max(num_samples, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        save_curves(os.path.join(args.out_dir, "curves.png"), history)

        print(
            f"[VIEW] epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"time={time.time() - start_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "cfg": {
                    "imagenet_pretrained": bool(args.imagenet_pretrained),
                    "hidden_dim1": args.hidden_dim1,
                    "hidden_dim2": args.hidden_dim2,
                    "dropout": args.dropout,
                    "id_min": args.id_min,
                    "id_max": args.id_max,
                    "val_frac": args.val_frac,
                    "input": "img_z+plaque01+vessel01_from_pseudo",
                    "pseudo_dir": args.pseudo_dir,
                },
            }
            torch.save(checkpoint, best_ckpt_path)
            print(f"[SAVE] {best_ckpt_path} best_val_acc={best_val_acc:.4f}")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--pseudo_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--id_min", type=int, default=200)
    parser.add_argument("--id_max", type=int, default=999)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_frac", type=float, default=0.10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--imagenet_pretrained", action="store_true", default=True)

    parser.add_argument("--hidden_dim1", type=int, default=512)
    parser.add_argument("--hidden_dim2", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)

    return parser


def main():
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
