import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def format_case_id(x: Any) -> str:
    s = str(x)
    return s.zfill(4) if s.isdigit() and len(s) < 4 else s


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)


def is_finite(x: float) -> bool:
    return (x is not None) and (not (isinstance(x, float) and (math.isnan(x) or math.isinf(x))))


def binary_metrics(p: np.ndarray, y: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    pred = (p >= thr).astype(np.int64)
    y = y.astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
    spec = tn / max(1, (tn + fp))
    bacc = 0.5 * (rec + spec)
    return dict(acc=acc, f1=f1, prec=prec, rec=rec, spec=spec, bacc=bacc, tp=tp, tn=tn, fp=fp, fn=fn)


def average_precision_binary(p: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(np.int64)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(1, (tp + fp))
    ap = precision[y_sorted == 1].sum() / max(1, n_pos)
    return float(ap)


def sweep_best_f1_threshold(p: np.ndarray, y: np.ndarray, n_steps: int = 500) -> Dict[str, float]:
    if p.size == 0:
        return dict(best_thr=0.5, best_f1=float("nan"))
    qs = np.linspace(0.0, 1.0, num=101)
    cand = np.unique(np.clip(np.quantile(p, qs), 0.0, 1.0))
    grid = np.linspace(0.0, 1.0, num=max(50, n_steps))
    thrs = np.unique(np.concatenate([cand, grid]))

    best_thr, best_f1 = 0.5, -1.0
    for t in thrs:
        f1 = binary_metrics(p, y, thr=float(t))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
    return dict(best_thr=best_thr, best_f1=float(best_f1))


def load_manifest_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        out = []
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            d = dict(v)
            d["new_id"] = d.get("new_id", k)
            out.append(d)
        data = out

    if not isinstance(data, list):
        raise RuntimeError(f"manifest must be list or dict, got {type(data)}")

    items = []
    for d in data:
        if not isinstance(d, dict):
            continue

        new_id = format_case_id(d.get("new_id", d.get("case_id", d.get("id", ""))))
        src = d.get("source_id", d.get("source_case_id", d.get("source", d.get("group", ""))))
        src = format_case_id(src)
        cls = d.get("cls", d.get("y", None))

        if (new_id == "") or (src == "") or (cls is None):
            continue

        cls = int(cls)
        if cls not in (0, 1):
            continue

        items.append(
            dict(
                new_id=new_id,
                source_id=src,
                cls=cls,
                is_copy=int(d.get("is_copy", 0)),
                aug_idx=int(d.get("aug_idx", -1)),
            )
        )

    if len(items) == 0:
        raise RuntimeError("manifest parsed 0 valid items. Need keys: new_id + source_id + cls.")
    return items


def stratified_group_kfold_by_source(
    sample_ids: List[str],
    sample_y: List[int],
    sample_groups: List[str],
    k: int,
    seed: int,
) -> List[Tuple[List[str], List[str]]]:
    rng = np.random.RandomState(seed)

    sample_ids = [format_case_id(s) for s in sample_ids]
    y = np.asarray(sample_y, dtype=np.int64)
    g = np.asarray([str(x) for x in sample_groups], dtype=object)

    group_to_idx: Dict[str, List[int]] = {}
    for i, gg in enumerate(g):
        group_to_idx.setdefault(gg, []).append(i)

    groups = sorted(group_to_idx.keys())
    group_y = []
    for gg in groups:
        ys = y[group_to_idx[gg]]
        if ys.min() != ys.max():
            raise RuntimeError(f"[LEAK CHECK FAIL] group {gg} has mixed labels: {ys.tolist()}")
        group_y.append(int(ys[0]))
    group_y = np.asarray(group_y, dtype=np.int64)

    g0 = np.where(group_y == 0)[0].tolist()
    g1 = np.where(group_y == 1)[0].tolist()
    rng.shuffle(g0)
    rng.shuffle(g1)

    folds_g = [[] for _ in range(k)]
    for j, gi in enumerate(g0):
        folds_g[j % k].append(groups[gi])
    for j, gi in enumerate(g1):
        folds_g[j % k].append(groups[gi])

    folds = []
    for f in range(k):
        val_groups = set(folds_g[f])
        tr_idx, va_idx = [], []
        for gg in groups:
            idxs = group_to_idx[gg]
            (va_idx if gg in val_groups else tr_idx).extend(idxs)

        tr_ids = [sample_ids[i] for i in tr_idx]
        va_ids = [sample_ids[i] for i in va_idx]
        folds.append((tr_ids, va_ids))
    return folds
