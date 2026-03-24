import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import torchvision.transforms.functional as TVF
except Exception:
    TVF = None


def read_h5_array(path: str, key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f[key][:]


def read_h5_scalar(path: str, key: str) -> int:
    with h5py.File(path, "r") as f:
        return int(f[key][()])


@torch.no_grad()
def augment_maskinput_3ch_mild(
    x3: torch.Tensor,
    p_blur: float,
    blur_sigma_min: float,
    blur_sigma_max: float,
    brightness_min: float,
    brightness_max: float,
    contrast_min: float,
    contrast_max: float,
) -> torch.Tensor:
    assert x3.ndim == 3 and x3.shape[0] == 3
    img = x3[0:1]

    if (TVF is not None) and (torch.rand(1).item() < float(p_blur)):
        sigma = float(torch.empty(1).uniform_(float(blur_sigma_min), float(blur_sigma_max)).item())
        k = 3 if sigma < 0.35 else 5
        img = TVF.gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])

    bf = float(torch.empty(1).uniform_(float(brightness_min), float(brightness_max)).item())
    img = img * bf

    cf = float(torch.empty(1).uniform_(float(contrast_min), float(contrast_max)).item())
    mean = img.mean(dim=(1, 2), keepdim=True)
    img = (img - mean) * cf + mean

    return torch.cat([img, x3[1:]], dim=0)


def mask_to_plaque_vessel(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = mask.astype(np.uint8)
    mx = int(m.max()) if m.size else 0
    if mx <= 2:
        plaque = (m == 1)
        vessel = (m == 2)
    else:
        plaque = (m == 128)
        vessel = (m == 255)
    return plaque.astype(np.float32), vessel.astype(np.float32)


def make_img_mask_3ch(img: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(img.astype(np.float32))[None, ...]
    x = x - x.mean()
    x = x / (x.std() + 1e-6)

    plaque01, vessel01 = mask_to_plaque_vessel(mask)
    plaque = torch.from_numpy(plaque01)[None, ...]
    vessel = torch.from_numpy(vessel01)[None, ...]
    return torch.cat([x, plaque, vessel], dim=0)


class TwoViewH5Dataset(Dataset):
    def __init__(
        self,
        case_ids: List[str],
        images_dir: str,
        labels_dir: str,
        label_suffix: str = "_label.h5",
        train_index_map: Optional[Dict[str, int]] = None,
        aug: bool = False,
        aug_cfg: Optional[Dict[str, float]] = None,
    ):
        self.case_ids = [str(c) for c in case_ids]
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.label_suffix = label_suffix
        self.train_index_map = train_index_map
        self.aug = aug
        self.aug_cfg = aug_cfg or {}

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        cid = self.case_ids[i]
        img_path = os.path.join(self.images_dir, f"{cid}.h5")
        lab_path = os.path.join(self.labels_dir, f"{cid}{self.label_suffix}")

        long_img = read_h5_array(img_path, "long_img")
        trans_img = read_h5_array(img_path, "trans_img")
        y = read_h5_scalar(lab_path, "cls")
        long_mask = read_h5_array(lab_path, "long_mask")
        trans_mask = read_h5_array(lab_path, "trans_mask")

        xL = make_img_mask_3ch(long_img, long_mask)
        xT = make_img_mask_3ch(trans_img, trans_mask)

        if self.aug:
            xL = augment_maskinput_3ch_mild(xL, **self.aug_cfg)
            xT = augment_maskinput_3ch_mild(xT, **self.aug_cfg)

        idx = -1
        if self.train_index_map is not None and cid in self.train_index_map:
            idx = int(self.train_index_map[cid])

        return dict(xL=xL, xT=xT, y=int(y), idx=idx, cid=cid)
