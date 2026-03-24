import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .dataset import TwoViewH5Dataset, read_h5_scalar, TVF
    from .models import (
        RSNet,
        UnifiedMemoryBank,
        load_view_pretrain_resnet18,
        supcon_eq2_single_view_with_bank,
        visualize_memory_bank,
    )
    from .utils import (
        average_precision_binary,
        binary_metrics,
        format_case_id,
        is_finite,
        load_manifest_items,
        set_seed,
        stratified_group_kfold_by_source,
        sweep_best_f1_threshold,
        worker_init_fn,
    )
except ImportError:
    from dataset import TwoViewH5Dataset, read_h5_scalar, TVF
    from models import (
        RSNet,
        UnifiedMemoryBank,
        load_view_pretrain_resnet18,
        supcon_eq2_single_view_with_bank,
        visualize_memory_bank,
    )
    from utils import (
        average_precision_binary,
        binary_metrics,
        format_case_id,
        is_finite,
        load_manifest_items,
        set_seed,
        stratified_group_kfold_by_source,
        sweep_best_f1_threshold,
        worker_init_fn,
    )


@torch.no_grad()
def infer_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bank: UnifiedMemoryBank,
    tau_gate: float,
):
    model.eval()
    muL, muT = bank.centers()

    all_p, all_y = [], []
    preds = []
    for batch in loader:
        xL = batch["xL"].to(device, non_blocking=True)
        xT = batch["xT"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).long()
        cid = batch["cid"]

        out = model(xL, xT, muL=muL, muT=muT, tau_gate=tau_gate)
        p1 = torch.softmax(out["logitF"], dim=1)[:, 1].detach().cpu().numpy()

        all_p.append(p1)
        all_y.append(y.detach().cpu().numpy())
        for j in range(len(cid)):
            preds.append(dict(case_id=str(cid[j]), y=int(y[j].item()), p1=float(p1[j])))

    p = np.concatenate(all_p) if all_p else np.array([], dtype=np.float32)
    y = np.concatenate(all_y) if all_y else np.array([], dtype=np.int64)
    return p, y, preds


@torch.no_grad()
def evaluate_val_loss_and_pr_auc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bank: UnifiedMemoryBank,
    w_view: float,
    lambda_cmcl: float,
    tau_cmcl: float,
    tau_gate: float,
    warmup_done: bool,
) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="mean")
    muL, muT = bank.centers()

    total_loss = 0.0
    n = 0
    all_p, all_y = [], []

    for batch in loader:
        xL = batch["xL"].to(device, non_blocking=True)
        xT = batch["xT"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).long()

        out = model(xL, xT, muL=muL, muT=muT, tau_gate=tau_gate)

        loss_ceF = ce(out["logitF"], y)
        loss_ceV = ce(out["logitL"], y) + ce(out["logitT"], y)

        if not warmup_done:
            loss_cmcl = torch.zeros((), device=device)
        else:
            loss_cmcl_L = supcon_eq2_single_view_with_bank(
                out["zL"],
                y,
                bank.mL,
                bank.y,
                tau=tau_cmcl,
                use_bank_neg=True,
                use_batch_neg=True,
                mask_bank_same_label=True,
            )
            loss_cmcl_T = supcon_eq2_single_view_with_bank(
                out["zT"],
                y,
                bank.mT,
                bank.y,
                tau=tau_cmcl,
                use_bank_neg=True,
                use_batch_neg=True,
                mask_bank_same_label=True,
            )
            loss_cmcl = 0.5 * (loss_cmcl_L + loss_cmcl_T)

        loss = loss_ceF + w_view * loss_ceV + lambda_cmcl * loss_cmcl

        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        n += bs

        p1 = torch.softmax(out["logitF"], dim=1)[:, 1].detach().cpu().numpy()
        all_p.append(p1)
        all_y.append(y.detach().cpu().numpy())

    val_loss = total_loss / max(1, n)
    p = np.concatenate(all_p) if all_p else np.array([], dtype=np.float32)
    yy = np.concatenate(all_y) if all_y else np.array([], dtype=np.int64)
    pr_auc = average_precision_binary(p, yy) if yy.size else float("nan")

    if not is_finite(val_loss):
        val_loss = float("inf")
    return float(val_loss), float(pr_auc)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bank: UnifiedMemoryBank,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    w_view: float,
    lambda_cmcl: float,
    tau_cmcl: float,
    tau_gate: float,
    bank_alpha: float,
    warmup_epochs: int,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    muL, muT = bank.centers()

    total_loss = 0.0
    n = 0
    amp_device = device.type

    for batch in loader:
        xL = batch["xL"].to(device, non_blocking=True)
        xT = batch["xT"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).long()
        idx = batch["idx"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None

        with torch.amp.autocast(amp_device, enabled=use_amp):
            out = model(xL, xT, muL=muL, muT=muT, tau_gate=tau_gate)
            loss_ceF = ce(out["logitF"], y)
            loss_ceV = ce(out["logitL"], y) + ce(out["logitT"], y)

            if epoch < warmup_epochs:
                loss_cmcl = torch.zeros((), device=device)
            else:
                loss_cmcl_L = supcon_eq2_single_view_with_bank(
                    out["zL"],
                    y,
                    bank.mL,
                    bank.y,
                    tau=tau_cmcl,
                    use_bank_neg=True,
                    use_batch_neg=True,
                    mask_bank_same_label=True,
                )
                loss_cmcl_T = supcon_eq2_single_view_with_bank(
                    out["zT"],
                    y,
                    bank.mT,
                    bank.y,
                    tau=tau_cmcl,
                    use_bank_neg=True,
                    use_batch_neg=True,
                    mask_bank_same_label=True,
                )
                loss_cmcl = 0.5 * (loss_cmcl_L + loss_cmcl_T)

            loss = loss_ceF + w_view * loss_ceV + lambda_cmcl * loss_cmcl

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        valid = idx >= 0
        if valid.any():
            bank.ema_update(idx[valid], out["zL"][valid], out["zT"][valid], alpha=bank_alpha)

        muL, muT = bank.centers()

        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        n += bs

    if n == 0:
        return dict(loss=float("inf"))
    return dict(loss=total_loss / n)


def run_fold(
    fold_id: int,
    train_ids: List[str],
    val_ids: List[str],
    args,
    device: torch.device,
) -> Tuple[str, List[Dict[str, Any]]]:
    fold_dir = os.path.join(args.out_dir, f"fold{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, "preds"), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, "metrics"), exist_ok=True)

    train_index_map = {cid: i for i, cid in enumerate(train_ids)}

    y_train = []
    for cid in train_ids:
        lab_path = os.path.join(args.tv_labels_dir, f"{cid}{args.label_suffix}")
        y_train.append(read_h5_scalar(lab_path, "cls"))
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)

    aug_cfg = dict(
        p_blur=args.p_blur,
        blur_sigma_min=args.blur_sigma_min,
        blur_sigma_max=args.blur_sigma_max,
        brightness_min=args.brightness_min,
        brightness_max=args.brightness_max,
        contrast_min=args.contrast_min,
        contrast_max=args.contrast_max,
    )

    ds_train = TwoViewH5Dataset(
        train_ids,
        args.tv_images_dir,
        args.tv_labels_dir,
        label_suffix=args.label_suffix,
        train_index_map=train_index_map,
        aug=True,
        aug_cfg=aug_cfg,
    )
    ds_val = TwoViewH5Dataset(
        val_ids,
        args.tv_images_dir,
        args.tv_labels_dir,
        label_suffix=args.label_suffix,
        train_index_map=None,
        aug=False,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    model = RSNet(rep_dim=args.rep_dim, MSA_out=args.MSA_out, moe_hidden=args.moe_hidden).to(device)
    if args.pretrain_ckpt:
        load_view_pretrain_resnet18(model.backbone, args.pretrain_ckpt, strict=args.pretrain_strict)

    bank = UnifiedMemoryBank(n_samples=len(train_ids), dim=args.rep_dim, device=device)
    bank.set_labels(y_train_t)
    bank.configure_center_weighting(args.center_weighted, q=args.center_outlier_q, weight=args.center_outlier_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=(args.amp and device.type == "cuda"))

    best_pr = -1e18
    best_path = os.path.join(fold_dir, "checkpoints", "best.pt")

    best_val_loss = float("inf")
    patience_ctr = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        tr = train_one_epoch(
            model,
            dl_train,
            device,
            bank,
            optimizer,
            scaler,
            epoch=epoch,
            w_view=args.w_view,
            lambda_cmcl=args.lambda_cmcl,
            tau_cmcl=args.tau_cmcl,
            tau_gate=args.tau_gate,
            bank_alpha=args.bank_alpha,
            warmup_epochs=args.warmup_epochs,
            grad_clip=args.grad_clip,
        )

        warmup_done = epoch >= args.warmup_epochs
        val_loss, val_pr = evaluate_val_loss_and_pr_auc(
            model,
            dl_val,
            device,
            bank,
            w_view=args.w_view,
            lambda_cmcl=args.lambda_cmcl,
            tau_cmcl=args.tau_cmcl,
            tau_gate=args.tau_gate,
            warmup_done=warmup_done,
        )

        dt = time.time() - t0
        print(
            f"[fold{fold_id}][E{epoch:03d}] {dt:6.1f}s  "
            f"train_loss={tr['loss']:.4f}  val_loss={val_loss:.4f}  val_PR-AUC={val_pr:.4f}"
        )

        pr_for_cmp = val_pr if is_finite(val_pr) else -1e18
        if pr_for_cmp > best_pr:
            best_pr = float(pr_for_cmp)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "bank_mL": bank.mL.detach().cpu(),
                    "bank_mT": bank.mT.detach().cpu(),
                    "bank_y": bank.y.detach().cpu(),
                    "args": vars(args),
                    "fold_id": fold_id,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "val_loss": float(val_loss),
                    "val_pr_auc": (float(val_pr) if is_finite(val_pr) else None),
                    "best_pr_auc": float(best_pr),
                },
                best_path,
            )
            with open(os.path.join(fold_dir, "metrics", "best.json"), "w") as f:
                json.dump(
                    dict(
                        epoch=int(epoch),
                        val_loss=float(val_loss),
                        val_pr_auc=(float(val_pr) if is_finite(val_pr) else None),
                        best_pr_auc=float(best_pr),
                    ),
                    f,
                    indent=2,
                )

        if epoch < args.warmup_epochs:
            pass
        else:
            if val_loss + args.es_min_delta < best_val_loss:
                best_val_loss = float(val_loss)
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"[fold{fold_id}] Early stop at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
                    break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    with torch.no_grad():
        bank.mL.copy_(ckpt["bank_mL"].to(device))
        bank.mT.copy_(ckpt["bank_mT"].to(device))
        bank.y.copy_(ckpt["bank_y"].to(device))
        bank.initialized = True

    mb_dir = os.path.join(fold_dir, "memory_bank_viz")
    visualize_memory_bank(bank, out_dir=mb_dir, tag="best")
    print(f"[fold{fold_id}] Saved memory bank visualizations to: {mb_dir}")

    _, _, preds = infer_probs(model, dl_val, device, bank, tau_gate=args.tau_gate)

    out_oof = os.path.join(fold_dir, "preds", "oof_val_fold.json")
    with open(out_oof, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"[fold{fold_id}] Saved OOF(val) preds: {out_oof}  (n={len(preds)})")
    return best_path, preds


@torch.no_grad()
def eval_test_probs_for_fold(best_ckpt_path: str, test_ids: List[str], args, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(best_ckpt_path, map_location=device)

    model = RSNet(rep_dim=args.rep_dim, MSA_out=args.MSA_out, moe_hidden=args.moe_hidden).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    bank = UnifiedMemoryBank(n_samples=ckpt["bank_y"].shape[0], dim=args.rep_dim, device=device)
    with torch.no_grad():
        bank.mL.copy_(ckpt["bank_mL"].to(device))
        bank.mT.copy_(ckpt["bank_mT"].to(device))
        bank.y.copy_(ckpt["bank_y"].to(device))
        bank.initialized = True

    ds_test = TwoViewH5Dataset(
        test_ids,
        args.test_images_dir,
        args.test_labels_dir,
        label_suffix=args.label_suffix,
        train_index_map=None,
        aug=False,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    p, y, _ = infer_probs(model, dl_test, device, bank, tau_gate=args.tau_gate)
    return p, y


def build_parser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest_json", required=True)
    ap.add_argument("--tv_images_dir", required=True)
    ap.add_argument("--tv_labels_dir", required=True)

    ap.add_argument("--test_splits_json", required=True)
    ap.add_argument("--test_images_dir", required=True)
    ap.add_argument("--test_labels_dir", required=True)

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--pretrain_ckpt", default="")
    ap.add_argument("--pretrain_strict", action="store_true")

    ap.add_argument("--label_suffix", default="_label.h5")

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--rep_dim", type=int, default=128)
    ap.add_argument("--MSA_out", type=int, default=256)
    ap.add_argument("--moe_hidden", type=int, default=256)

    ap.add_argument("--w_view", type=float, default=0.5)
    ap.add_argument("--lambda_cmcl", type=float, default=0.2)
    ap.add_argument("--tau_cmcl", type=float, default=0.07)
    ap.add_argument("--tau_gate", type=float, default=0.05)
    ap.add_argument("--bank_alpha", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--p_blur", type=float, default=0.35)
    ap.add_argument("--blur_sigma_min", type=float, default=0.10)
    ap.add_argument("--blur_sigma_max", type=float, default=0.60)
    ap.add_argument("--brightness_min", type=float, default=0.95)
    ap.add_argument("--brightness_max", type=float, default=1.05)
    ap.add_argument("--contrast_min", type=float, default=0.95)
    ap.add_argument("--contrast_max", type=float, default=1.05)

    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--es_min_delta", type=float, default=1e-4)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--center_weighted", action="store_true")
    ap.add_argument("--center_outlier_q", type=float, default=0.10)
    ap.add_argument("--center_outlier_weight", type=float, default=2.0)

    return ap


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "oof"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    if TVF is None:
        print("[WARN] torchvision.transforms.functional unavailable; blur will be skipped.")

    items = load_manifest_items(args.manifest_json)

    pool_ids = [format_case_id(d["new_id"]) for d in items]
    pool_y = [int(d["cls"]) for d in items]
    pool_groups = [format_case_id(d["source_id"]) for d in items]

    id2src = {format_case_id(d["new_id"]): format_case_id(d["source_id"]) for d in items}

    print(f"[INFO] offline pool: n_samples={len(pool_ids)}  n_sources={len(set(pool_groups))}")
    print(f"[INFO] offline labels: pos={int(sum(pool_y))}  neg={int(len(pool_y) - sum(pool_y))}")

    folds = stratified_group_kfold_by_source(pool_ids, pool_y, pool_groups, k=5, seed=args.seed)
    print("[INFO] CV5 folds built by SOURCE groups (no leakage).")
    for i, (tr, va) in enumerate(folds):
        pos_tr = sum(read_h5_scalar(os.path.join(args.tv_labels_dir, f"{c}{args.label_suffix}"), "cls") for c in tr)
        pos_va = sum(read_h5_scalar(os.path.join(args.tv_labels_dir, f"{c}{args.label_suffix}"), "cls") for c in va)
        src_tr = len(set(id2src[c] for c in tr))
        src_va = len(set(id2src[c] for c in va))
        print(f"  fold{i}: train={len(tr)} (pos={pos_tr}, src={src_tr})  val={len(va)} (pos={pos_va}, src={src_va})")

    with open(args.test_splits_json, "r") as f:
        test_splits = json.load(f)
    if "test" not in test_splits:
        raise KeyError("test_splits_json must contain 'test'")
    test_ids = [format_case_id(x) for x in test_splits["test"]]
    print(f"[INFO] test cases from splits42: n={len(test_ids)}")

    best_ckpts = []
    oof_all = []

    for fold_id, (train_ids, val_ids) in enumerate(folds):
        set_seed(args.seed + 1000 * fold_id)
        ckpt_path, oof_preds = run_fold(fold_id, train_ids, val_ids, args, device)
        best_ckpts.append(ckpt_path)
        oof_all.extend(oof_preds)

    oof_path = os.path.join(args.out_dir, "oof", "oof_val_all.json")
    with open(oof_path, "w") as f:
        json.dump(oof_all, f, indent=2)
    print(f"[OOF] Saved merged OOF(val): {oof_path}  (n={len(oof_all)})")

    p_oof = np.array([d["p1"] for d in oof_all], dtype=np.float32)
    y_oof = np.array([d["y"] for d in oof_all], dtype=np.int64)

    thr_info = sweep_best_f1_threshold(p_oof, y_oof, n_steps=500)
    best_thr = float(thr_info["best_thr"])
    best_f1 = float(thr_info["best_f1"])
    thr_out = dict(best_thr=best_thr, best_f1=best_f1, n=int(y_oof.size), pos=int(y_oof.sum()))
    thr_json = os.path.join(args.out_dir, "oof", "best_threshold.json")
    with open(thr_json, "w") as f:
        json.dump(thr_out, f, indent=2)
    print(f"[OOF] best_F1={best_f1:.4f} @ thr={best_thr:.4f}  saved: {thr_json}")

    fold_test_metrics = []
    for fold_id, ckpt_path in enumerate(best_ckpts):
        p, y = eval_test_probs_for_fold(ckpt_path, test_ids, args, device)
        m = binary_metrics(p, y, thr=best_thr)
        m["fold"] = int(fold_id)
        m["_n"] = int(y.size)
        m["_pos"] = int(y.sum())
        fold_test_metrics.append(m)
        print(
            f"[TEST][fold{fold_id}] "
            f"acc={m['acc']:.4f} f1={m['f1']:.4f} prec={m['prec']:.4f} rec={m['rec']:.4f} bacc={m['bacc']:.4f}"
        )

    def mean_std(key: str) -> Tuple[float, float]:
        vals = np.array([m[key] for m in fold_test_metrics], dtype=np.float64)
        return float(vals.mean()), float(vals.std(ddof=0))

    summary = {
        "global_threshold_from_oof": thr_out,
        "fold_test_metrics_at_global_thr": fold_test_metrics,
        "mean_std": {
            "acc": dict(mean=mean_std("acc")[0], std=mean_std("acc")[1]),
            "f1": dict(mean=mean_std("f1")[0], std=mean_std("f1")[1]),
            "prec": dict(mean=mean_std("prec")[0], std=mean_std("prec")[1]),
            "rec": dict(mean=mean_std("rec")[0], std=mean_std("rec")[1]),
            "bacc": dict(mean=mean_std("bacc")[0], std=mean_std("bacc")[1]),
        },
    }
    summ_path = os.path.join(args.out_dir, "summary_test_at_global_thr.json")
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] Summary saved: {summ_path}")


if __name__ == "__main__":
    main()
