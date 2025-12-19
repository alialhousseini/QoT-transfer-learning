"""
Contrastive data utilities for the lightpath dataset.

Provides:
- LightpathContrastiveDataset: reuses preprocessing from lightpath_nn (scaler/encoder/imputer/flags).
- ContrastiveCollator: builds K augmented views per sample for in-batch contrastive training.
- build_contrastive_loader: convenience factory to get a DataLoader with the collator wired up.

Augmentations (lightweight, tabular-friendly):
- Numeric Gaussian noise scaled by feature std.
- Numeric feature dropout (mask to zero).
- Optional categorical masking to a reserved mask token (last id per column).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lightpath_nn import (
    CAT_COLS,
    NUMERIC_COLS,
    PreprocessConfig,
    prepare_features,
)
import pandas as pd


@dataclass
class ContrastiveAugmentConfig:
    k_views: int = 2
    num_noise_std: float = 0.01   # scale factor times per-feature std
    num_drop_prob: float = 0.05   # probability to zero a numeric feature
    cat_mask_prob: float = 0.0    # probability to mask a categorical token to the reserved id


class LightpathContrastiveDataset(Dataset):
    """Dataset wrapper that prepares features once and optionally keeps labels."""

    def __init__(
        self,
        data_path: Path,
        target_path: Path | None = None,
        sample_size: int | None = None,
        preprocess_cfg: PreprocessConfig | None = None,
    ):
        cfg = preprocess_cfg or PreprocessConfig()
        df = pd.read_csv(data_path)
        targets = None
        if target_path is not None:
            targets = pd.read_csv(target_path)["class"].astype(int)
        if sample_size is not None:
            df = df.sample(n=sample_size, random_state=42)
            if targets is not None:
                targets = targets.loc[df.index]
            df = df.reset_index(drop=True)
            if targets is not None:
                targets = targets.reset_index(drop=True)

        (
            self.cat_matrix,
            self.cat_cardinalities,
            self.num_matrix,
            self.num_feature_names,
            self.flag_cols,
        ) = prepare_features(df, cfg, NUMERIC_COLS, CAT_COLS)

        self.num_feature_std = np.std(self.num_matrix, axis=0).astype(np.float32) + 1e-6
        self.targets = None if targets is None else targets.to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.num_matrix)

    def __getitem__(self, idx: int):
        base = (
            torch.from_numpy(self.cat_matrix[idx]),
            torch.from_numpy(self.num_matrix[idx]),
        )
        if self.targets is None:
            return base
        return base + (torch.tensor(self.targets[idx]),)

    @property
    def has_labels(self) -> bool:
        return self.targets is not None


class ContrastiveCollator:
    """Creates K augmented views per sample for contrastive learning."""

    def __init__(
        self,
        cat_cardinalities: Sequence[int],
        num_feature_std: np.ndarray,
        cfg: ContrastiveAugmentConfig,
        include_labels: bool,
    ):
        self.cat_mask_ids = [c - 1 for c in cat_cardinalities]
        self.num_std = torch.from_numpy(num_feature_std.astype(np.float32))
        self.cfg = cfg
        self.include_labels = include_labels

    def _augment(self, cat: torch.Tensor, num: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cat_view = cat
        num_view = num.clone()

        if self.cfg.num_noise_std > 0:
            scale = self.cfg.num_noise_std * self.num_std.to(num.device)
            num_view = num_view + torch.randn_like(num_view) * scale
        if self.cfg.num_drop_prob > 0:
            drop_mask = torch.rand_like(num_view) < self.cfg.num_drop_prob
            num_view = num_view.masked_fill(drop_mask, 0.0)
        if self.cfg.cat_mask_prob > 0 and cat.numel() > 0:
            mask = torch.rand_like(cat.float()) < self.cfg.cat_mask_prob
            mask_ids = torch.tensor(self.cat_mask_ids, device=cat.device).unsqueeze(0)
            cat_view = torch.where(mask.bool(), mask_ids, cat)

        return cat_view, num_view

    def __call__(self, batch):
        if self.include_labels:
            cats, nums, labels = zip(*batch)
        else:
            cats, nums = zip(*batch)
            labels = None

        cat_batch = torch.stack(cats) if cats[0].numel() > 0 else torch.empty((len(cats), 0), dtype=torch.long)
        num_batch = torch.stack(nums)

        views = []
        for _ in range(self.cfg.k_views):
            v_cat, v_num = self._augment(cat_batch, num_batch)
            views.append({"cat": v_cat, "num": v_num})

        if labels is None:
            return views
        return views, torch.stack(labels)


def build_contrastive_loader(
    data_path: Path,
    target_path: Path | None = None,
    preprocess_cfg: PreprocessConfig | None = None,
    augment_cfg: ContrastiveAugmentConfig | None = None,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    sample_size: int | None = None,
) -> Tuple[DataLoader, LightpathContrastiveDataset]:
    dataset = LightpathContrastiveDataset(
        data_path=data_path,
        target_path=target_path,
        sample_size=sample_size,
        preprocess_cfg=preprocess_cfg,
    )
    collator = ContrastiveCollator(
        cat_cardinalities=dataset.cat_cardinalities,
        num_feature_std=dataset.num_feature_std,
        cfg=augment_cfg or ContrastiveAugmentConfig(),
        include_labels=dataset.has_labels,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=False,
    )
    return loader, dataset
