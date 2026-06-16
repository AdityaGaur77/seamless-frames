"""
dataset.py
----------
PyTorch dataset for road segmentation.

Each item returns:
    img       (3, H, W)  float32, normalised
    mask      (1, H, W)  float32, 0 / 1
    occluded  (3, H, W)  float32, normalised — the *input* variant with
                                       canopy / shadow / clouds / vehicles
    meta      dict       {name, scene}

The training loop feeds the *occluded* image to the model and computes the
loss against the *clean* mask.  This is the standard "see through the
occlusion" setup described in the problem statement.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def make_aug(image_size: int = 256) -> A.Compose:
    """Augmentations applied *identically* to the clean image, the occluded
    image, and the mask.  Geometric augs use the Replay mode so the
    transformations stay in lock-step across the three tensors."""
    return A.Compose(
        [
            A.RandomCrop(width=image_size, height=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.85, 1.15), rotate=(-25, 25), p=0.5,
                     border_mode=cv2.BORDER_REFLECT),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                                 val_shift_limit=10, p=0.5),
            A.GaussNoise(p=0.2),
        ],
        additional_targets={"occluded": "image", "mask": "mask"},
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class SampleMeta:
    name: str
    scene: str


class RoadDataset(Dataset):
    """Loads triplets (clean RGB, occluded RGB, mask) produced by synth_data.py."""

    def __init__(self, root: str | Path,
                 split: str = "train",
                 image_size: int = 256,
                 augment: bool = True):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        meta_path = self.root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {self.root}")
        with open(meta_path) as f:
            self.meta = json.load(f)

        # accept either 'name' or 'id' as the per-sample identifier
        all_names: List[str] = [
            m.get("name") or m.get("id") or f"sample_{i}"
            for i, m in enumerate(self.meta)
        ]
        # 80/20 train/val split, deterministic
        all_names.sort()
        cut = int(0.8 * len(all_names))
        self.names = all_names[:cut] if split == "train" else all_names[cut:]

        self.aug = make_aug(image_size) if augment else None

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        clean_path = self.root / "images" / f"{name}.png"
        occ_path = self.root / "occluded" / f"{name}.png"
        mask_path = self.root / "masks" / f"{name}.png"

        img = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        occ = cv2.imread(str(occ_path), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        occ = cv2.cvtColor(occ, cv2.COLOR_BGR2RGB)
        msk = (msk > 127).astype(np.uint8)

        if self.aug is not None:
            out = self.aug(image=img, occluded=occ, mask=msk)
            img, occ, msk = out["image"], out["occluded"], out["mask"]

        # Normalise to [-1, 1] (helps BCE/Dice on small datasets)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        occ_t = torch.from_numpy(occ).permute(2, 0, 1).float() / 127.5 - 1.0
        msk_t = torch.from_numpy(msk).unsqueeze(0).float()

        # Build a robust lookup that supports both 'name' and 'id' keys
        meta_by_key = {}
        for m in self.meta:
            key = m.get("name") or m.get("id") or f"sample_{len(meta_by_key)}"
            meta_by_key[key] = m

        # Metadata lookup for the loaded sample
        scene = meta_by_key.get(name, {}).get("scene", "unknown")

        return {
            "image": img_t,
            "occluded": occ_t,
            "mask": msk_t,
            "name": name,
            "scene": scene,
        }


def collate(batch):
    imgs = torch.stack([b["image"] for b in batch])
    occs = torch.stack([b["occluded"] for b in batch])
    msks = torch.stack([b["mask"] for b in batch])
    names = [b["name"] for b in batch]
    scenes = [b["scene"] for b in batch]
    return {"image": imgs, "occluded": occs, "mask": msks,
            "name": names, "scene": scenes}


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ds = RoadDataset("data/synth", split="train", image_size=256)
    print(f"train size: {len(ds)}")
    s = ds[0]
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in s.items()})
