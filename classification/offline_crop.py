import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F


PLAQUE_VALUE = 128
VESSEL_VALUE = 255


def format_case_id(x: Any) -> str:
    s = str(x)
    return s.zfill(4) if s.isdigit() and len(s) < 4 else s


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def summarize_quantiles(
    x: np.ndarray,
    quantile_points: Tuple[float, ...] = (0.0, 0.5, 0.95, 1.0),
) -> Dict[str, float]:
    if x.size == 0:
        return {f"q{int(q * 100):02d}": float("nan") for q in quantile_points}

    values = np.quantile(x.astype(np.float64), quantile_points)
    out: Dict[str, float] = {}
    for q, v in zip(quantile_points, values):
        out[f"q{int(q * 100):02d}"] = float(v)

    out["mean"] = float(x.mean())
    out["min"] = float(x.min())
    out["max"] = float(x.max())
    return out


def build_case_rng(seed: int, case_id: str) -> np.random.RandomState:
    h = 2166136261
    for ch in case_id.encode("utf-8"):
        h = (h ^ ch) * 16777619
        h &= 0xFFFFFFFF
    return np.random.RandomState((seed ^ h) & 0xFFFFFFFF)


def read_h5_array(path: str, key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f[key][:]


def read_h5_scalar(path: str, key: str) -> int:
    with h5py.File(path, "r") as f:
        return int(f[key][()])


def write_image_h5(path: str, long_img: np.ndarray, trans_img: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("long_img", data=long_img.astype(np.float32), compression="gzip")
        f.create_dataset("trans_img", data=trans_img.astype(np.float32), compression="gzip")


def write_label_h5(
    path: str,
    long_mask: np.ndarray,
    trans_mask: np.ndarray,
    cls: int,
) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("long_mask", data=long_mask.astype(np.uint8), compression="gzip")
        f.create_dataset("trans_mask", data=trans_mask.astype(np.uint8), compression="gzip")
        f.create_dataset("cls", data=np.uint64(int(cls)))


def map_raw_mask_to_012(raw_mask: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(raw_mask, dtype=np.uint8)
    mask[raw_mask == VESSEL_VALUE] = 2
    mask[raw_mask == PLAQUE_VALUE] = 1
    return mask


def get_plaque_mask(mask_012: np.ndarray) -> np.ndarray:
    return mask_012 == 1


def get_foreground_mask(mask_012: np.ndarray) -> np.ndarray:
    return mask_012 > 0


def compute_centroid(binary_mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(binary_mask)
    if ys.size == 0:
        h, w = binary_mask.shape
        return float(h / 2.0), float(w / 2.0)
    return float(ys.mean()), float(xs.mean())


def resize_to_original_shape(
    image_crop: np.ndarray,
    mask_crop: np.ndarray,
    out_h: int,
    out_w: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    image_tensor = torch.from_numpy(image_crop.astype(np.float32))[None, None].to(device)
    mask_tensor = torch.from_numpy(mask_crop.astype(np.float32))[None, None].to(device)

    image_tensor = F.interpolate(
        image_tensor,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )
    mask_tensor = F.interpolate(
        mask_tensor,
        size=(out_h, out_w),
        mode="nearest-exact",
    )

    image_out = image_tensor[0, 0].detach().cpu().numpy().astype(np.float32)
    mask_out = mask_tensor[0, 0].detach().cpu().numpy()
    mask_out = np.rint(mask_out).astype(np.int32)
    mask_out = np.clip(mask_out, 0, 2).astype(np.uint8)
    return image_out, mask_out


def crop_single_view(
    image: np.ndarray,
    mask_012: np.ndarray,
    rng: np.random.RandomState,
    fg_ratio_min: float,
    plaque_keep_ratio: float,
    scale_min: float,
    scale_max: float,
    jitter: float,
    max_tries: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], bool]:
    if image.ndim != 2 or mask_012.ndim != 2:
        raise ValueError(f"Expected 2D inputs, got image={image.shape}, mask={mask_012.shape}")

    h, w = image.shape

    plaque = get_plaque_mask(mask_012)
    foreground_union = get_foreground_mask(mask_012)

    plaque_total = float(plaque.sum())
    if plaque_total > 0:
        foreground = plaque
        foreground_total = plaque_total
    else:
        foreground = foreground_union
        foreground_total = float(foreground_union.sum())

    if foreground_total <= 0:
        stats = {"fg_cov": 1.0, "pl_cov": 1.0, "scale": 1.0}
        return image.astype(np.float32), mask_012.astype(np.uint8), stats, True

    cy, cx = compute_centroid(plaque if plaque_total > 0 else foreground)

    best_score = None
    best_result: Optional[Tuple[np.ndarray, np.ndarray, Dict[str, float]]] = None
    accepted = False

    for _ in range(int(max_tries)):
        scale = float(rng.uniform(scale_min, scale_max))
        crop_h = int(max(2, round(h * scale)))
        crop_w = int(max(2, round(w * scale)))

        jitter_y = float(rng.uniform(-jitter, jitter)) * crop_h
        jitter_x = float(rng.uniform(-jitter, jitter)) * crop_w
        center_y = cy + jitter_y
        center_x = cx + jitter_x

        top = int(round(center_y - crop_h / 2.0))
        left = int(round(center_x - crop_w / 2.0))
        top = max(0, min(top, h - crop_h))
        left = max(0, min(left, w - crop_w))

        image_crop = image[top:top + crop_h, left:left + crop_w].astype(np.float32)
        mask_crop = mask_012[top:top + crop_h, left:left + crop_w].astype(np.uint8)

        plaque_crop = mask_crop == 1
        foreground_crop = plaque_crop if plaque_total > 0 else (mask_crop > 0)

        fg_cov = float(foreground_crop.sum()) / max(1.0, foreground_total)
        pl_cov = float(plaque_crop.sum()) / max(1.0, plaque_total) if plaque_total > 0 else 1.0

        ok = (pl_cov >= plaque_keep_ratio) and (fg_cov >= fg_ratio_min)
        score = (pl_cov, fg_cov, -scale)

        if best_score is None or score > best_score:
            best_score = score
            best_result = (
                image_crop,
                mask_crop,
                {"fg_cov": fg_cov, "pl_cov": pl_cov, "scale": scale},
            )

        if ok:
            accepted = True
            break

    assert best_result is not None
    best_image_crop, best_mask_crop, best_stats = best_result
    return best_image_crop, best_mask_crop, best_stats, accepted


def crop_two_views(
    long_img: np.ndarray,
    trans_img: np.ndarray,
    long_mask: np.ndarray,
    trans_mask: np.ndarray,
    rng: np.random.RandomState,
    device: torch.device,
    fg_ratio_min: float,
    plaque_keep_ratio: float,
    scale_min: float,
    scale_max: float,
    jitter: float,
    max_tries: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    long_img_crop, long_mask_crop, long_stats, long_accepted = crop_single_view(
        long_img,
        long_mask,
        rng,
        fg_ratio_min,
        plaque_keep_ratio,
        scale_min,
        scale_max,
        jitter,
        max_tries,
    )
    trans_img_crop, trans_mask_crop, trans_stats, trans_accepted = crop_single_view(
        trans_img,
        trans_mask,
        rng,
        fg_ratio_min,
        plaque_keep_ratio,
        scale_min,
        scale_max,
        jitter,
        max_tries,
    )

    long_h, long_w = long_img.shape
    trans_h, trans_w = trans_img.shape

    long_img_out, long_mask_out = resize_to_original_shape(
        long_img_crop,
        long_mask_crop,
        long_h,
        long_w,
        device,
    )
    trans_img_out, trans_mask_out = resize_to_original_shape(
        trans_img_crop,
        trans_mask_crop,
        trans_h,
        trans_w,
        device,
    )

    stats = {
        "long_fg_cov": long_stats["fg_cov"],
        "long_pl_cov": long_stats["pl_cov"],
        "long_scale": long_stats["scale"],
        "long_accepted": 1.0 if long_accepted else 0.0,
        "trans_fg_cov": trans_stats["fg_cov"],
        "trans_pl_cov": trans_stats["pl_cov"],
        "trans_scale": trans_stats["scale"],
        "trans_accepted": 1.0 if trans_accepted else 0.0,
    }

    return long_img_out, trans_img_out, long_mask_out, trans_mask_out, stats


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_json", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--out_root", required=True)

    parser.add_argument("--label_suffix", default="_label.h5")
    parser.add_argument("--n_aug_per_case", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--fg_ratio_min", type=float, default=0.90)
    parser.add_argument("--plaque_keep_ratio", type=float, default=0.999999)
    parser.add_argument("--scale_min", type=float, default=0.70)
    parser.add_argument("--scale_max", type=float, default=1.00)
    parser.add_argument("--jitter", type=float, default=0.10)
    parser.add_argument("--max_tries", type=int, default=30)

    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_copy", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    out_images_dir = os.path.join(args.out_root, "images")
    out_labels_dir = os.path.join(args.out_root, "labels")

    ensure_dir(args.out_root)

    if (os.path.isdir(out_images_dir) or os.path.isdir(out_labels_dir)) and not args.overwrite:
        raise RuntimeError(
            f"Output appears to exist: {args.out_root}. "
            f"Use --overwrite to allow writing into existing directories."
        )

    ensure_dir(out_images_dir)
    ensure_dir(out_labels_dir)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")

    with open(args.splits_json, "r") as f:
        splits = json.load(f)

    if "train" not in splits or "val" not in splits:
        raise KeyError("splits_json must contain keys: 'train' and 'val'")

    source_ids = [format_case_id(x) for x in (splits["train"] + splits["val"])]

    seen = set()
    unique_source_ids = []
    for case_id in source_ids:
        if case_id not in seen:
            seen.add(case_id)
            unique_source_ids.append(case_id)
    source_ids = unique_source_ids

    print(
        f"[INFO] #src(train+val)={len(source_ids)} "
        f"n_aug_per_case={args.n_aug_per_case} include_copy={not args.no_copy}"
    )

    next_id = 0

    def allocate_new_id() -> str:
        nonlocal next_id
        new_id = str(next_id).zfill(4) if next_id < 10000 else str(next_id)
        next_id += 1
        return new_id

    manifest: List[Dict[str, Any]] = []
    groups: Dict[str, List[str]] = {}

    crop_stats: Dict[str, List[float]] = {
        "long_fg_cov": [],
        "long_pl_cov": [],
        "long_scale": [],
        "long_accepted": [],
        "trans_fg_cov": [],
        "trans_pl_cov": [],
        "trans_scale": [],
        "trans_accepted": [],
    }

    num_copy = 0
    num_aug = 0

    for idx, case_id in enumerate(source_ids):
        image_path = os.path.join(args.images_dir, f"{case_id}.h5")
        label_path = os.path.join(args.labels_dir, f"{case_id}{args.label_suffix}")

        long_img = read_h5_array(image_path, "long_img").astype(np.float32)
        trans_img = read_h5_array(image_path, "trans_img").astype(np.float32)
        cls = read_h5_scalar(label_path, "cls")

        long_raw = read_h5_array(label_path, "long_mask").astype(np.uint8)
        trans_raw = read_h5_array(label_path, "trans_mask").astype(np.uint8)

        long_mask = map_raw_mask_to_012(long_raw)
        trans_mask = map_raw_mask_to_012(trans_raw)

        groups.setdefault(case_id, [])
        rng = build_case_rng(args.seed, case_id)

        if not args.no_copy:
            new_id = allocate_new_id()
            write_image_h5(os.path.join(out_images_dir, f"{new_id}.h5"), long_img, trans_img)
            write_label_h5(
                os.path.join(out_labels_dir, f"{new_id}{args.label_suffix}"),
                long_mask,
                trans_mask,
                cls,
            )
            groups[case_id].append(new_id)
            manifest.append(
                {
                    "new_id": new_id,
                    "source_id": case_id,
                    "cls": int(cls),
                    "is_copy": 1,
                    "aug_idx": -1,
                }
            )
            num_copy += 1

        for aug_idx in range(int(max(0, args.n_aug_per_case))):
            long_img_out, trans_img_out, long_mask_out, trans_mask_out, stats = crop_two_views(
                long_img,
                trans_img,
                long_mask,
                trans_mask,
                rng=rng,
                device=device,
                fg_ratio_min=args.fg_ratio_min,
                plaque_keep_ratio=args.plaque_keep_ratio,
                scale_min=args.scale_min,
                scale_max=args.scale_max,
                jitter=args.jitter,
                max_tries=args.max_tries,
            )

            new_id = allocate_new_id()
            write_image_h5(
                os.path.join(out_images_dir, f"{new_id}.h5"),
                long_img_out,
                trans_img_out,
            )
            write_label_h5(
                os.path.join(out_labels_dir, f"{new_id}{args.label_suffix}"),
                long_mask_out,
                trans_mask_out,
                cls,
            )

            groups[case_id].append(new_id)
            manifest.append(
                {
                    "new_id": new_id,
                    "source_id": case_id,
                    "cls": int(cls),
                    "is_copy": 0,
                    "aug_idx": int(aug_idx),
                }
            )
            num_aug += 1

            for key in crop_stats:
                crop_stats[key].append(float(stats[key]))

        if (idx + 1) % 20 == 0 or (idx + 1) == len(source_ids):
            print(f"[PROGRESS] {idx + 1}/{len(source_ids)} src cases processed. new_total={len(manifest)}")

    manifest_path = os.path.join(args.out_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    groups_path = os.path.join(args.out_root, "groups.json")
    with open(groups_path, "w") as f:
        json.dump(groups, f, indent=2)

    crop_stats_np = {k: np.array(v, dtype=np.float64) for k, v in crop_stats.items()}
    long_accept_rate = (
        float(crop_stats_np["long_accepted"].mean()) if crop_stats_np["long_accepted"].size else float("nan")
    )
    trans_accept_rate = (
        float(crop_stats_np["trans_accepted"].mean()) if crop_stats_np["trans_accepted"].size else float("nan")
    )

    group_labels: Dict[str, int] = {}
    for record in manifest:
        group_labels[record["source_id"]] = int(record["cls"])

    num_groups = len(group_labels)
    num_positive_groups = int(sum(group_labels.values()))

    group_sizes = np.array([len(v) for v in groups.values()], dtype=np.int64)

    summary = {
        "source": {
            "splits_json": args.splits_json,
            "images_dir": args.images_dir,
            "labels_dir": args.labels_dir,
            "n_src_trainval_unique": int(len(source_ids)),
        },
        "generation": {
            "include_copy": bool(not args.no_copy),
            "n_aug_per_case": int(args.n_aug_per_case),
            "n_copy_samples": int(num_copy),
            "n_aug_samples": int(num_aug),
            "n_total_new": int(len(manifest)),
            "n_groups": int(num_groups),
            "n_groups_pos": int(num_positive_groups),
            "group_size": {
                "min": int(group_sizes.min()) if group_sizes.size else 0,
                "max": int(group_sizes.max()) if group_sizes.size else 0,
                "mean": float(group_sizes.mean()) if group_sizes.size else float("nan"),
            },
            "id_start": "0000",
            "id_end": str(next_id - 1).zfill(4) if (next_id - 1) < 10000 else str(next_id - 1),
        },
        "crop_cfg": {
            "fg_ratio_min": float(args.fg_ratio_min),
            "plaque_keep_ratio": float(args.plaque_keep_ratio),
            "scale_min": float(args.scale_min),
            "scale_max": float(args.scale_max),
            "jitter": float(args.jitter),
            "max_tries": int(args.max_tries),
            "device": str(device),
            "seed": int(args.seed),
        },
        "crop_global_stats_aug_only": {
            "long_accept_rate": long_accept_rate,
            "trans_accept_rate": trans_accept_rate,
            "long_fg_cov": summarize_quantiles(crop_stats_np["long_fg_cov"]),
            "long_pl_cov": summarize_quantiles(crop_stats_np["long_pl_cov"]),
            "long_scale": summarize_quantiles(crop_stats_np["long_scale"]),
            "trans_fg_cov": summarize_quantiles(crop_stats_np["trans_fg_cov"]),
            "trans_pl_cov": summarize_quantiles(crop_stats_np["trans_pl_cov"]),
            "trans_scale": summarize_quantiles(crop_stats_np["trans_scale"]),
        },
        "outputs": {
            "images_dir": out_images_dir,
            "labels_dir": out_labels_dir,
            "manifest_json": manifest_path,
            "groups_json": groups_path,
        },
        "leakage_guard": {
            "note": "Do not random-split new_ids. Use group-based CV with groups=source_id."
        },
    }

    summary_path = os.path.join(args.out_root, "global_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] wrote manifest: {manifest_path}")
    print(f"[OK] wrote groups:   {groups_path}")
    print(f"[DONE] wrote summary: {summary_path}")
    print(f"[DONE] new dataset root: {args.out_root}")


if __name__ == "__main__":
    main()
