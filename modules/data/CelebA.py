import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from dataclasses import dataclass

import kagglehub
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import CelebA

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modules.utils.tensor_utils import (
    _to_long_tensor,
    _to_float_tensor,
    ArrayLike,
    DeviceLike,
    _as_tensor,
)

from modules.utils.dataset_utils import (
    _validate_dataframe,
    _validate_columns_exist,
    _infer_cat_and_cont_cols,
    to_tensor_dataset_ar,
    to_tensor_dataset_flow,
    to_tensor_dataset_predictive,
    make_loader,
    split_df,
    split_df_with_indices,
    fit_schema_px,
    transform_px,
    fit_predictor_schema,
    transform_predictor,
    TabularDataset,
    compute_pg_dirichlet,
    compute_pg_dirichlet_from_groups
)

# ============================================================
# Attribute metadata
# ============================================================

CELEBA_ATTR_NAMES: List[str] = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]

CELEBA_ATTR_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CELEBA_ATTR_NAMES)}

# ============================================================
# Config
# ============================================================

@dataclass
class CelebAConfig:
    root: str
    image_size: int = 128
    batch_size: int = 64
    num_workers: int = 4
    target_attr: str = "Smiling"
    group_attrs: Tuple[str, ...] = ("Male", "Young")
    download: bool = False
    normalize: bool = True

# ============================================================
# Helpers
# ============================================================
def validate_celeba_attrs(attrs: Sequence[str]) -> None:
    for attr in attrs:
        if attr not in CELEBA_ATTR_NAMES:
            raise ValueError(f"Invalid attribute: {attr}. Must be one of: {CELEBA_ATTR_NAMES}")

def celeba_target_transofrm(x: torch.Tensor) -> torch.Tensor:
    # Convert from {-1, 1} to {0, 1}
    return (x + 1) // 2

def buils_group_ids_from_attrs(attr_matrix_01: torch.Tensor, group_attr_indices: Sequence[int]) -> Tuple[torch.Tensor, List[str]]:
    """
    Build instersectional group IDs from selected binary attributes.
    Args:
    - attr_matrix_01: Tensor of shape (N, num_attrs) with binary values {0, 1}attr_matrix_01: [N, A] binary tensor in {0, 1}
    - group_attr_indices: List of attribute indices to use for group ID construction
    Returns:
    - group_ids: Tensor of shape (N,) with integer group IDs
    - group_id_to_attrs: List mapping group ID to the combination of attribute values
    """
    if attr_matrix_01.ndim != 2:
        raise ValueError(f"Expected attr_matrix_01 to be 2D with shape [N, A], got shape {tuple(attr_matrix_01.shape)}")
    
    k = len(group_attr_indices)
    if k == 0:
        raise ValueError("At least one group attribute index must be provided")

    selected = attr_matrix_01[:, group_attr_indices]  # [N, k]

    # binary tuple -> integer id
    weigths = (2 ** torch.arange(k, dtype=torch.long)).view(1, k)
    group_ids = (selected.long() * weigths).sum(dim=1)  # [N]

    group_names = []
    for gid in range(2 ** k):
        bits = [(gid >> j) & 1 for j in range(k)]
        name_parts = []
        for bit, attr_idx in zip(bits, group_attr_indices):
            attr_name = CELEBA_ATTR_NAMES[attr_idx]
            name_parts.append(f"{attr_name}={bit}")
        group_names.append("_".join(name_parts))
    
    return group_ids, group_names


# ============================================================
# Dataset
# ============================================================

class CelebAAttributeDataset(Dataset):
    """
    CelebA dataset wrapper for:
      - one binary prediction target y
      - one intersectional group id g
      - raw attribute vector a

    Returns a dict with:
      {
        "image": FloatTensor [3,H,W],
        "label": LongTensor [],
        "group_id": LongTensor [],
        "group_name": str,
        "attrs": LongTensor [40],
        "index": LongTensor []
      }
    """

    def __init__(
        self,
        root: str,
        split: str,
        target_attr: str,
        group_attrs: Sequence[str],
        image_size: int = 128,
        download: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'valid', 'test'")

        validate_celeba_attrs([target_attr])
        validate_celeba_attrs(group_attrs)

        self.root = root
        self.split = split
        self.target_attr = target_attr
        self.group_attrs = tuple(group_attrs)

        transform_list = [
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        if normalize:
            transform_list.append(
                transofrms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            )
        
        self.transforms = transforms.Compose(transform_list)

        self.base = CelebA(
            root=self.root,
            split=self.split,
            target_type="attr",
            transform=self.transforms,
            download=download,
        )

        #base.attr is shape [N, 40] in {-1, +1}
        attrs_pm1 = self.base.attr
        self.target_ids = CELEBA_ATTR_TO_IDX[self.target_attr]
        self.group_attr_indices = [CELEBA_ATTR_TO_IDX[attr] for attr in self.group_attrs]

        self.labels = self.attrs_01[:, self.target_idx].long()
        self.group_ids, self.group_names = buils_group_ids_from_attrs(
            self.attrs_01,
            self.group_attr_indices,
        )

        self.num_groups = len(self.group_names)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        image, _ = self.base[idx]

        label = self.labels[idx]
        group_id = self.group_ids[idx]
        attrs = self.attrs_01[idx]

        return {
            "image": image,
            "label": label,
            "group_id": group_id,
            "group_name": self.group_names[int(group_id.item())],
            "attrs": attrs,
            "index": torch.tensor(idx, dtype=torch.long),
        }

# ============================================================
# Collate
# ============================================================
def celeba_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    group_ids = torch.stack([item["group_id"] for item in batch], dim=0)
    attrs = torch.stack([item["attrs"] for item in batch], dim=0)
    indices = torch.stack([item["index"] for item in batch], dim=0)
    group_names = [item["group_name"] for item in batch]

    return {
        "image": images,
        "label": labels,
        "group_id": group_ids,
        "attrs": attrs,
        "index": indices,
        "group_name": group_names,
    }

# ============================================================
# DataLoaders
# ============================================================

def build_celeba_datasets(cfg: CelebAConfig) -> Tuple[CelebAAttributeDataset, CelebAAttributeDataset, CelebAAttributeDataset]:
    train_ds = CelebAAttributeDataset(
        root=cfg.root,
        split="train",
        target_attr=cfg.target_attr,
        group_attrs=cfg.group_attrs,
        image_size=cfg.image_size,
        download=cfg.download,
        normalize=cfg.normalize,
    )
    valid_ds = CelebAAttributeDataset(
        root=cfg.root,
        split="valid",
        target_attr=cfg.target_attr,
        group_attrs=cfg.group_attrs,
        image_size=cfg.image_size,
        download=False,
        normalize=cfg.normalize,
    )
    test_ds = CelebAAttributeDataset(
        root=cfg.root,
        split="test",
        target_attr=cfg.target_attr,
        group_attrs=cfg.group_attrs,
        image_size=cfg.image_size,
        download=False,
        normalize=cfg.normalize,
    )
    return train_ds, valid_ds, test_ds

def build_celeba_loaders(cfg: CelebAConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = build_celeba_datasets(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=celeba_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=celeba_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=celeba_collate_fn,
    )
    return train_loader, val_loader, test_loader

# ============================================================
# Quick inspection helpers
# ============================================================

@torch.no_grad()
def summarize_group_counts(dataset: CelebAAttributeDataset) -> Dict[str, int]:
    counts = torch.bincount(dataset.group_ids, minlength=dataset.num_groups)
    return {dataset.group_names[i]: int(counts[i].item()) for i in range(dataset.num_groups)}


@torch.no_grad()
def summarize_label_by_group(dataset: CelebAAttributeDataset) -> List[Dict[str, object]]:
    rows = []
    for gid in range(dataset.num_groups):
        mask = dataset.group_ids == gid
        n = int(mask.sum().item())
        if n == 0:
            positive_rate = float("nan")
        else:
            positive_rate = float(dataset.labels[mask].float().mean().item())

        rows.append(
            {
                "group_id": gid,
                "group_name": dataset.group_names[gid],
                "count": n,
                "label_positive_rate": positive_rate,
            }
        )
    return rows

# # ============================================================
# #  HOW TO USE THIS
# # ============================================================
# cfg = CelebAConfig(
#     root="/path/to/celeba",
#     image_size=128,
#     batch_size=64,
#     num_workers=4,
#     target_attr="Smiling",
#     group_attrs=("Male", "Young"),
#     download=False,
# )

# train_loader, val_loader, test_loader = build_celeba_loaders(cfg)

# # Inspect the split
# train_ds, val_ds, test_ds = build_celeba_datasets(cfg)

# print("Train groups:")
# print(summarize_group_counts(train_ds))

# print("Train label by group:")
# for row in summarize_label_by_group(train_ds):
#     print(row)

# # Check one batch
# batch = next(iter(train_loader))
# print(batch["image"].shape)     # [B, 3, H, W]
# print(batch["label"].shape)     # [B]
# print(batch["group_id"].shape)  # [B]
# print(batch["attrs"].shape)     # [B, 40]
# print(batch["group_name"][:5])  # list[str]