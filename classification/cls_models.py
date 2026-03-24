import json
import os
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class UnifiedMemoryBank(nn.Module):
    def __init__(self, n_samples: int, dim: int, device: torch.device):
        super().__init__()
        self.n = n_samples
        self.dim = dim
        self.register_buffer("mL", F.normalize(torch.randn(n_samples, dim, device=device), dim=1))
        self.register_buffer("mT", F.normalize(torch.randn(n_samples, dim, device=device), dim=1))
        self.register_buffer("y", torch.zeros(n_samples, dtype=torch.long, device=device))
        self.initialized = False
        self.center_weighted = False
        self.center_outlier_q = 0.10
        self.center_outlier_weight = 2.0

    @torch.no_grad()
    def configure_center_weighting(self, enabled: bool, q: float = 0.10, weight: float = 2.0):
        self.center_weighted = bool(enabled)
        self.center_outlier_q = float(q)
        self.center_outlier_weight = float(weight)

    @torch.no_grad()
    def set_labels(self, y_all: torch.Tensor):
        assert y_all.shape[0] == self.n
        self.y.copy_(y_all.long())
        self.initialized = True

    @torch.no_grad()
    def ema_update(self, idx: torch.Tensor, zL: torch.Tensor, zT: torch.Tensor, alpha: float):
        if idx.numel() == 0:
            return
        idx = idx.long()
        zL = F.normalize(zL.detach(), dim=1)
        zT = F.normalize(zT.detach(), dim=1)
        self.mL[idx] = F.normalize(alpha * self.mL[idx] + (1.0 - alpha) * zL, dim=1)
        self.mT[idx] = F.normalize(alpha * self.mT[idx] + (1.0 - alpha) * zT, dim=1)

    @torch.no_grad()
    def centers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        muL0 = torch.zeros(2, self.dim, device=self.mL.device)
        muT0 = torch.zeros(2, self.dim, device=self.mT.device)
        for k in (0, 1):
            mask = (self.y == k)
            if mask.any():
                muL0[k] = self.mL[mask].mean(dim=0)
                muT0[k] = self.mT[mask].mean(dim=0)
        muL0 = F.normalize(muL0, dim=1)
        muT0 = F.normalize(muT0, dim=1)

        if not self.center_weighted:
            return muL0, muT0

        muL = torch.zeros_like(muL0)
        muT = torch.zeros_like(muT0)
        q = float(np.clip(self.center_outlier_q, 0.0, 0.5))
        w_out = float(max(1.0, self.center_outlier_weight))

        for k in (0, 1):
            mask = (self.y == k)
            if not mask.any():
                continue
            mLk = self.mL[mask]
            mTk = self.mT[mask]
            sL = (mLk * muL0[k]).sum(dim=1)
            sT = (mTk * muT0[k]).sum(dim=1)
            s = torch.minimum(sL, sT)
            s_np = s.detach().cpu().float().numpy()
            thr = float(np.quantile(s_np, q)) if s_np.size > 0 else float("nan")
            w = torch.ones_like(s)
            w = w + (s <= thr).float() * (w_out - 1.0)
            w_sum = w.sum().clamp_min(1e-12)
            muL[k] = (mLk * w[:, None]).sum(dim=0) / w_sum
            muT[k] = (mTk * w[:, None]).sum(dim=0) / w_sum

        return F.normalize(muL, dim=1), F.normalize(muT, dim=1)


def pca2d_np(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:2].T
    Z = X @ W
    return Z.astype(np.float32)


def cos_sim_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    if b.ndim == 1:
        b = b[None, :]
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(axis=1)


@torch.no_grad()
def visualize_memory_bank(bank: UnifiedMemoryBank, out_dir: str, tag: str = "best"):
    os.makedirs(out_dir, exist_ok=True)

    mL = bank.mL.detach().cpu().float().numpy()
    mT = bank.mT.detach().cpu().float().numpy()
    y = bank.y.detach().cpu().long().numpy()

    muL_t, muT_t = bank.centers()
    muL = muL_t.detach().cpu().float().numpy()
    muT = muT_t.detach().cpu().float().numpy()

    def plot_pca_scatter(M: np.ndarray, MU: np.ndarray, name: str):
        Z = pca2d_np(M)
        Zc = pca2d_np(np.vstack([M, MU]))[-2:]

        plt.figure()
        for k in (0, 1):
            idx = (y == k)
            if idx.any():
                plt.scatter(Z[idx, 0], Z[idx, 1], s=10, alpha=0.55, label=f"class{k}")
        plt.scatter(Zc[0, 0], Zc[0, 1], marker="*", s=250, label="center0")
        plt.scatter(Zc[1, 0], Zc[1, 1], marker="*", s=250, label="center1")
        plt.title(f"{name} PCA2D ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mbank_{name}_pca2d_{tag}.png"), dpi=200)
        plt.close()

    def plot_sim_hist(M: np.ndarray, MU: np.ndarray, name: str):
        sim0 = cos_sim_np(M, MU[0])
        sim1 = cos_sim_np(M, MU[1])

        plt.figure()
        for k in (0, 1):
            idx = (y == k)
            if not idx.any():
                continue
            s_self = sim0[idx] if k == 0 else sim1[idx]
            s_other = sim1[idx] if k == 0 else sim0[idx]
            plt.hist(s_self, bins=40, alpha=0.45, density=True, label=f"class{k} -> self")
            plt.hist(s_other, bins=40, alpha=0.45, density=True, label=f"class{k} -> other")
        plt.title(f"{name} cosine sim to centers ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mbank_{name}_simhist_{tag}.png"), dpi=200)
        plt.close()

        stats = {}
        for k in (0, 1):
            idx = (y == k)
            if idx.any():
                s_self = sim0[idx] if k == 0 else sim1[idx]
                s_other = sim1[idx] if k == 0 else sim0[idx]
                stats[f"class{k}_self_mean"] = float(np.mean(s_self))
                stats[f"class{k}_self_std"] = float(np.std(s_self))
                stats[f"class{k}_other_mean"] = float(np.mean(s_other))
                stats[f"class{k}_other_std"] = float(np.std(s_other))
        with open(os.path.join(out_dir, f"mbank_{name}_simstats_{tag}.json"), "w") as f:
            json.dump(stats, f, indent=2)

    plot_pca_scatter(mL, muL, "L")
    plot_pca_scatter(mT, muT, "T")
    plot_sim_hist(mL, muL, "L")
    plot_sim_hist(mT, muT, "T")

    def cos_center(MU: np.ndarray) -> float:
        a = MU[0] / (np.linalg.norm(MU[0]) + 1e-12)
        b = MU[1] / (np.linalg.norm(MU[1]) + 1e-12)
        return float(np.dot(a, b))

    with open(os.path.join(out_dir, f"mbank_centers_{tag}.txt"), "w") as f:
        f.write(f"cos(muL0, muL1) = {cos_center(muL):.6f}\n")
        f.write(f"cos(muT0, muT1) = {cos_center(muT):.6f}\n")
        f.write(f"N = {int(y.shape[0])}, pos = {int(y.sum())}, neg = {int((y == 0).sum())}\n")


def cmcl_loss_one_view_stable(
    z: torch.Tensor,
    y: torch.Tensor,
    bank_feats: torch.Tensor,
    bank_y: torch.Tensor,
    centers: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    z = F.normalize(z.float(), dim=1)
    centers = F.normalize(centers.float(), dim=1)
    bank_feats = F.normalize(bank_feats.float(), dim=1)
    y = y.long()
    bank_y = bank_y.long()

    tau = max(1e-6, float(tau))

    pos = (z * centers[y]).sum(dim=1) / tau
    sim_all = (z @ bank_feats.t()) / tau

    neg_mask = (bank_y.unsqueeze(0) != y.unsqueeze(1))
    sim_all = sim_all.masked_fill(~neg_mask, float("-inf"))

    logits = torch.cat([pos.unsqueeze(1), sim_all], dim=1)
    denom = torch.logsumexp(logits, dim=1)
    loss = (-pos + denom).mean()
    return loss


def supcon_eq2_with_bank(
    zL: torch.Tensor,
    zT: torch.Tensor,
    y: torch.Tensor,
    bank_mL: torch.Tensor,
    bank_mT: torch.Tensor,
    bank_y: torch.Tensor,
    tau: float = 0.1,
    use_bank_neg: bool = True,
    use_batch_neg: bool = True,
    mask_bank_same_label: bool = True,
) -> torch.Tensor:
    device = zL.device
    y = y.long().to(device)
    tau = max(1e-6, float(tau))

    zL = F.normalize(zL.float(), dim=1)
    zT = F.normalize(zT.float(), dim=1)
    z = torch.cat([zL, zT], dim=0)
    y2 = torch.cat([y, y], dim=0)

    B2 = int(z.shape[0])

    sim_bb = (z @ z.t()) / tau
    self_mask = torch.eye(B2, dtype=torch.bool, device=device)
    sim_bb = sim_bb.masked_fill(self_mask, float("-inf"))

    pos_mask = (y2[:, None] == y2[None, :]) & (~self_mask)
    pos_cnt = pos_mask.sum(dim=1)
    valid = pos_cnt > 0
    if not valid.any():
        return torch.zeros((), device=device)

    logits_list = []
    if use_batch_neg:
        logits_list.append(sim_bb)

    if use_bank_neg:
        bank_mL = F.normalize(bank_mL.float(), dim=1)
        bank_mT = F.normalize(bank_mT.float(), dim=1)
        bank_feats = torch.cat([bank_mL, bank_mT], dim=0)
        bank_y2 = torch.cat([bank_y.long(), bank_y.long()], dim=0)

        sim_bk = (z @ bank_feats.t()) / tau
        if mask_bank_same_label:
            neg_mask = (bank_y2[None, :] != y2[:, None])
            sim_bk = sim_bk.masked_fill(~neg_mask, float("-inf"))
        logits_list.append(sim_bk)

    denom_logits = torch.cat(logits_list, dim=1)
    denom = torch.logsumexp(denom_logits, dim=1)

    mean_pos_sim = (sim_bb.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_cnt.clamp_min(1))
    loss = (-mean_pos_sim + denom)
    return loss[valid].mean()


def supcon_eq2_single_view_with_bank(
    z: torch.Tensor,
    y: torch.Tensor,
    bank_m: torch.Tensor,
    bank_y: torch.Tensor,
    tau: float = 0.1,
    use_bank_neg: bool = True,
    use_batch_neg: bool = True,
    mask_bank_same_label: bool = True,
) -> torch.Tensor:
    device = z.device
    y = y.long().to(device)
    tau = max(1e-6, float(tau))

    z = F.normalize(z.float(), dim=1)
    B = z.shape[0]

    sim_bb = (z @ z.t()) / tau
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim_bb = sim_bb.masked_fill(self_mask, float("-inf"))

    pos_mask = (y[:, None] == y[None, :]) & (~self_mask)
    pos_cnt = pos_mask.sum(dim=1)
    valid = pos_cnt > 0
    if not valid.any():
        return torch.zeros((), device=device)

    logits_list = []
    if use_batch_neg:
        logits_list.append(sim_bb)

    if use_bank_neg:
        bank_m = F.normalize(bank_m.float(), dim=1)
        bank_y = bank_y.long().to(device)
        sim_bk = (z @ bank_m.t()) / tau
        if mask_bank_same_label:
            neg_mask = (bank_y[None, :] != y[:, None])
            sim_bk = sim_bk.masked_fill(~neg_mask, float("-inf"))
        logits_list.append(sim_bk)

    denom_logits = torch.cat(logits_list, dim=1)
    denom = torch.logsumexp(denom_logits, dim=1)

    mean_pos_sim = (sim_bb.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_cnt.clamp_min(1))
    loss = (-mean_pos_sim + denom)
    return loss[valid].mean()


class ResNet18Multi(nn.Module):
    def __init__(self, imagenet: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if imagenet else None
        m = resnet18(weights=weights)
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        rep = F.adaptive_avg_pool2d(c5, 1).flatten(1)
        return dict(c2=c2, c3=c3, c4=c4, c5=c5, rep=rep)


def strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def load_view_pretrain_resnet18(model: ResNet18Multi, ckpt_path: str, strict: bool = False):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[WARN] Pretrain ckpt not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("model", None), dict):
            sd = ckpt["model"]
        elif isinstance(ckpt.get("state_dict", None), dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    sd = strip_prefix(sd, "module.")
    sd = strip_prefix(sd, "backbone.")

    if "conv1.weight" in sd and sd["conv1.weight"].ndim == 4:
        w = sd["conv1.weight"]
        if w.shape[1] == 1 and model.conv1.weight.shape[1] == 3:
            sd["conv1.weight"] = w.repeat(1, 3, 1, 1) / 3.0
            print("[OK] Adapted conv1 weights from 1ch -> 3ch")

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"[OK] Loaded view-pretrain backbone from: {ckpt_path}")
    print(f"     missing={len(missing)} unexpected={len(unexpected)} (strict={strict})")


class SpatialGate(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.conv(x))
        return x * g


class MSA(nn.Module):
    def __init__(self, c2=64, c3=128, c4=256, c5=512, out_ch=256):
        super().__init__()
        self.l2 = nn.Conv2d(c2, out_ch, 1)
        self.l3 = nn.Conv2d(c3, out_ch, 1)
        self.l4 = nn.Conv2d(c4, out_ch, 1)
        self.l5 = nn.Conv2d(c5, out_ch, 1)

        self.g2 = SpatialGate(out_ch)
        self.g3 = SpatialGate(out_ch)
        self.g4 = SpatialGate(out_ch)
        self.g5 = SpatialGate(out_ch)

        self.smooth2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.smooth3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.smooth4 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, c2, c3, c4, c5) -> torch.Tensor:
        p5 = self.g5(self.l5(c5))
        p4 = self.g4(self.l4(c4)) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.smooth4(p4)
        p3 = self.g3(self.l3(c3)) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth3(p3)
        p2 = self.g2(self.l2(c2)) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.smooth2(p2)
        return p2


class MoEHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, n_classes: int = 2):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.1),
                    nn.Linear(hidden, n_classes),
                )
                for _ in range(n_classes)
            ]
        )

    def forward(
        self,
        feat: torch.Tensor,
        z_cat: torch.Tensor,
        mu_cat: torch.Tensor,
        tau_gate: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_cat = F.normalize(z_cat, dim=1)
        mu_cat = F.normalize(mu_cat, dim=1)
        sims = z_cat @ mu_cat.t()
        w = F.softmax(sims / max(1e-6, float(tau_gate)), dim=1)

        exp_logits = torch.stack([e(feat) for e in self.experts], dim=1)
        logits = (exp_logits * w.unsqueeze(-1)).sum(dim=1)
        return logits, w


class CNet(nn.Module):
    def __init__(self, rep_dim: int = 128, MSA_out: int = 256, moe_hidden: int = 256):
        super().__init__()
        self.backbone = ResNet18Multi(imagenet=True)

        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, rep_dim),
        )

        self.MSA = MSA(out_ch=MSA_out)
        self.view_head = nn.Linear(MSA_out, 2)

        fuse_dim = 2 * MSA_out
        self.fuse_mlp = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, fuse_dim),
            nn.ReLU(inplace=True),
        )

        self.moe = MoEHead(in_dim=fuse_dim, hidden=moe_hidden, n_classes=2)
        self.rep_dim = rep_dim

    def forward(
        self,
        xL: torch.Tensor,
        xT: torch.Tensor,
        muL: torch.Tensor,
        muT: torch.Tensor,
        tau_gate: float,
    ) -> Dict[str, torch.Tensor]:
        fL = self.backbone(xL)
        fT = self.backbone(xT)

        zL = F.normalize(self.proj(fL["rep"]), dim=1)
        zT = F.normalize(self.proj(fT["rep"]), dim=1)

        pL = self.MSA(fL["c2"], fL["c3"], fL["c4"], fL["c5"])
        pT = self.MSA(fT["c2"], fT["c3"], fT["c4"], fT["c5"])
        vL = F.adaptive_avg_pool2d(pL, 1).flatten(1)
        vT = F.adaptive_avg_pool2d(pT, 1).flatten(1)

        logitL = self.view_head(vL)
        logitT = self.view_head(vT)

        feat = torch.cat([vL, vT], dim=1)
        feat = self.fuse_mlp(feat)

        z_cat = torch.cat([zL, zT], dim=1)
        mu_cat = torch.cat([muL, muT], dim=1)
        logitF, w = self.moe(feat, z_cat, mu_cat, tau_gate=tau_gate)

        return dict(logitF=logitF, logitL=logitL, logitT=logitT, zL=zL, zT=zT, w=w)
