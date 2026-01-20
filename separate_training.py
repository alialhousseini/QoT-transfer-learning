"""
End-to-end contrastive + classification training for the QoT tabular data.

Pipeline blocks:
1) Data prep: sampler builds M+N mini-batches (random, class_balanced, m_per_class).
2) Backbone: tabular encoder -> latent z (embedding_dim).
3) Mining + contrastive loss: PML miners/losses on the batch embedding.
4) Classification head: predicts class from z (cross-entropy).

Supports:
- Separate training: contrastive (backbone+projection) then classifier.
- Joint training: optimize contrastive + classification losses together.
- Hyperparameter tuning via random/grid search over a JSON search space.

Example:
  python separate_training.py --mode joint \
    --data1 cleaned_lightpath_dataset.csv --target1 cleaned_lightpath_target.csv \
    --loss-name TripletMarginLoss --miner-name BatchHardMiner \
    --sampler m_per_class --m-per-class 64 \
    --eval data2:cleaned_lightpath_dataset2.csv:cleaned_lightpath_target_2.csv

Search space JSON example:
{
  "embedding_dim": [64, 128],
  "hidden_dims": [[256, 128], [512, 256]],
  "loss_name": ["TripletMarginLoss", "MultiSimilarityLoss"],
  "miner_name": ["BatchHardMiner", null],
  "lr": [0.001, 0.0005]
}
"""
from __future__ import annotations

import argparse
import copy
import inspect
import itertools
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)

try:
    from lightpath_nn import AVAIL_GROUPS, NUMERIC_COLS, CAT_COLS
except Exception as exc:  # pragma: no cover - fails fast if project layout changes
    raise ImportError("Missing lightpath_nn.py with column definitions.") from exc


@dataclass
class PreprocessConfig:
    num_scaler: str = "none"           # "none", "standard", "minmax"
    cat_encoder: str = "embedding"     # "embedding", "onehot"
    impute_strategy: str = "median"    # "median", "zero"
    add_availability_flags: bool = True


@dataclass
class EvalSpec:
    name: str
    data_path: Path
    target_path: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_availability_flags(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    flag_cols: List[str] = []
    for group_name, cols in AVAIL_GROUPS.items():
        flag_col = f"{group_name}_avail"
        df[flag_col] = (df[cols].ne(0).any(axis=1)).astype("int8")
        flag_cols.append(flag_col)
        for col in cols:
            df.loc[df[col] == 0, col] = np.nan
    return df, flag_cols


def _make_onehot_encoder():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class TabularPreprocessor:
    def __init__(
        self,
        cfg: PreprocessConfig,
        numeric_cols: Sequence[str],
        cat_cols: Sequence[str],
    ) -> None:
        self.cfg = cfg
        self.numeric_cols = list(numeric_cols)
        self.cat_cols = list(cat_cols)
        self.flag_cols: List[str] = []
        self.num_impute: Optional[pd.Series] = None
        self.num_scale: Optional[pd.Series] = None
        self.num_shift: Optional[pd.Series] = None
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        self.cat_cardinalities: List[int] = []
        self.onehot_encoder = None
        self.fitted = False

    def _check_columns(self, df: pd.DataFrame) -> None:
        expected = set(self.numeric_cols) | set(self.cat_cols) | set(self.flag_cols)
        missing = expected.difference(df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in dataset: {missing_list}")

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        df = df.copy()
        if self.cfg.add_availability_flags:
            df, self.flag_cols = add_availability_flags(df)
        else:
            self.flag_cols = []

        self.numeric_cols = list(self.numeric_cols) + list(self.flag_cols)
        self._check_columns(df)

        num_df = df[self.numeric_cols].copy()
        if self.cfg.impute_strategy == "median":
            self.num_impute = num_df.median()
        elif self.cfg.impute_strategy == "zero":
            self.num_impute = pd.Series(0, index=self.numeric_cols)
        else:
            raise ValueError(f"Unsupported impute_strategy: {self.cfg.impute_strategy}")

        num_df = num_df.fillna(self.num_impute)
        if self.cfg.num_scaler == "standard":
            self.num_shift = num_df.mean()
            self.num_scale = num_df.std().replace(0, 1e-6)
        elif self.cfg.num_scaler == "minmax":
            self.num_shift = num_df.min()
            self.num_scale = (num_df.max() - num_df.min()).replace(0, 1e-6)
        elif self.cfg.num_scaler == "none":
            self.num_shift = None
            self.num_scale = None
        else:
            raise ValueError(f"Unsupported num_scaler: {self.cfg.num_scaler}")

        if self.cfg.cat_encoder == "embedding":
            cat_df = df[self.cat_cols].fillna("missing").astype(str)
            self.cat_maps = {}
            self.cat_cardinalities = []
            for col in self.cat_cols:
                categories = sorted(cat_df[col].unique().tolist())
                self.cat_maps[col] = {val: idx for idx, val in enumerate(categories)}
                self.cat_cardinalities.append(len(categories) + 1)
            self.onehot_encoder = None
        elif self.cfg.cat_encoder == "onehot":
            cat_df = df[self.cat_cols].fillna("missing").astype(str)
            self.onehot_encoder = _make_onehot_encoder()
            self.onehot_encoder.fit(cat_df)
            self.cat_maps = {}
            self.cat_cardinalities = []
        else:
            raise ValueError(f"Unsupported cat_encoder: {self.cfg.cat_encoder}")

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        df = df.copy()
        if self.cfg.add_availability_flags:
            df, _ = add_availability_flags(df)

        self._check_columns(df)

        num_df = df[self.numeric_cols].copy()
        num_df = num_df.fillna(self.num_impute)
        if self.cfg.num_scaler == "standard":
            num_df = (num_df - self.num_shift) / self.num_scale
        elif self.cfg.num_scaler == "minmax":
            num_df = (num_df - self.num_shift) / self.num_scale

        num_matrix = num_df.to_numpy(dtype=np.float32)

        if self.cfg.cat_encoder == "embedding":
            cat_df = df[self.cat_cols].fillna("missing").astype(str)
            cat_matrix = np.zeros((len(df), len(self.cat_cols)), dtype=np.int64)
            for i, col in enumerate(self.cat_cols):
                mapping = self.cat_maps[col]
                unknown_idx = self.cat_cardinalities[i] - 1
                mapped = cat_df[col].map(mapping).fillna(unknown_idx).astype(np.int64)
                cat_matrix[:, i] = mapped.to_numpy()
            return cat_matrix, num_matrix

        if self.cfg.cat_encoder == "onehot":
            cat_df = df[self.cat_cols].fillna("missing").astype(str)
            onehot = self.onehot_encoder.transform(cat_df).astype(np.float32)
            if onehot.shape[1] > 0:
                num_matrix = np.concatenate([num_matrix, onehot], axis=1)
            cat_matrix = np.empty((len(df), 0), dtype=np.int64)
            return cat_matrix, num_matrix

        raise ValueError(f"Unsupported cat_encoder: {self.cfg.cat_encoder}")


class TabularDataset(Dataset):
    def __init__(self, cat_matrix: np.ndarray, num_matrix: np.ndarray, labels: np.ndarray) -> None:
        self.cat = torch.from_numpy(cat_matrix).long()
        self.num = torch.from_numpy(num_matrix).float()
        self.labels = torch.from_numpy(labels).long()
        self.label_array = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.cat[idx], self.num[idx], self.labels[idx]


def load_raw_data(
    data_path: Path,
    target_path: Path,
    target_col: str,
    sample_size: Optional[int],
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    targets = pd.read_csv(target_path)
    if target_col not in targets.columns:
        raise ValueError(f"Target column '{target_col}' not found in {target_path}")
    y = targets[target_col].astype(int)
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)
        y = y.loc[df.index]
        df = df.reset_index(drop=True)
        y = y.reset_index(drop=True)
    return df, y


def mix_with_data2(
    df1: pd.DataFrame,
    y1: pd.Series,
    df2: Optional[pd.DataFrame],
    y2: Optional[pd.Series],
    mix_percent: float,
    mix_base: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    if df2 is None or y2 is None or mix_percent <= 0:
        return df1, y1

    if mix_base == "data2":
        sample_size = int(len(df2) * mix_percent)
    elif mix_base == "data1":
        sample_size = int(len(df1) * mix_percent)
    else:
        raise ValueError("mix_base must be 'data1' or 'data2'.")

    sample_size = max(1, min(sample_size, len(df2)))
    df2_sample = df2.sample(n=sample_size, random_state=seed)
    y2_sample = y2.loc[df2_sample.index]
    df_combined = pd.concat([df1, df2_sample], ignore_index=True)
    y_combined = pd.concat([y1, y2_sample], ignore_index=True)
    return df_combined, y_combined


def split_indices(
    y: np.ndarray,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    indices = np.arange(len(y))
    holdout = val_split + test_split
    train_idx, temp_idx = train_test_split(
        indices, test_size=holdout, stratify=y, random_state=seed
    )
    if holdout == 0:
        return train_idx, np.array([], dtype=int), np.array([], dtype=int)

    if val_split > 0 and test_split > 0:
        val_ratio = val_split / holdout
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1 - val_ratio,
            stratify=y[temp_idx],
            random_state=seed,
        )
    elif val_split > 0:
        val_idx = temp_idx
        test_idx = np.array([], dtype=int)
    else:
        val_idx = np.array([], dtype=int)
        test_idx = temp_idx

    return train_idx, val_idx, test_idx


def build_dataloader(
    dataset: TabularDataset,
    batch_size: int,
    sampler_type: str,
    m_per_class: int,
    seed: int,
    num_workers: int,
    drop_last: bool,
) -> DataLoader:
    labels = dataset.label_array
    if sampler_type == "random":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last,
        )

    if sampler_type == "m_per_class":
        try:
            from pytorch_metric_learning.samplers import MPerClassSampler
        except Exception as exc:
            raise ImportError("pytorch-metric-learning is required for MPerClassSampler.") from exc
        num_classes = len(np.unique(labels))
        effective_batch = m_per_class * num_classes
        if batch_size != effective_batch:
            print(
                f"Sampler m_per_class overrides batch_size: using {effective_batch} "
                f"(m_per_class={m_per_class}, num_classes={num_classes})."
            )
        sampler = MPerClassSampler(labels, m=m_per_class, length_before_new_iter=len(labels))
        return DataLoader(
            dataset,
            batch_size=effective_batch,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
        )

    if sampler_type == "class_balanced":
        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        weights = class_weights[labels]
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
        )

    raise ValueError(f"Unknown sampler_type: {sampler_type}")


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last_dim = hidden
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class CatMLPBackbone(nn.Module):
    def __init__(
        self,
        cat_cardinalities: Sequence[int],
        numeric_dim: int,
        cat_embed_dim: int,
        hidden_dims: Sequence[int],
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, cat_embed_dim) for cardinality in cat_cardinalities]
        )
        self.num_norm = nn.LayerNorm(numeric_dim) if numeric_dim > 0 else None
        input_dim = len(cat_cardinalities) * cat_embed_dim + numeric_dim
        self.mlp = build_mlp(input_dim, hidden_dims, embedding_dim, dropout)

    def forward(self, cat_ids: torch.Tensor, num_feats: torch.Tensor) -> torch.Tensor:
        cat_tokens = [emb(cat_ids[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_flat = torch.cat(cat_tokens, dim=1) if cat_tokens else None
        if self.num_norm is not None:
            num_feats = self.num_norm(num_feats)
        if cat_flat is None:
            x = num_feats
        else:
            x = torch.cat([cat_flat, num_feats], dim=1)
        return self.mlp(x)


class DenseMLPBackbone(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        hidden_dims: Sequence[int],
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_norm = nn.LayerNorm(numeric_dim) if numeric_dim > 0 else None
        self.mlp = build_mlp(numeric_dim, hidden_dims, embedding_dim, dropout)

    def forward(self, cat_ids: torch.Tensor, num_feats: torch.Tensor) -> torch.Tensor:
        if self.num_norm is not None:
            num_feats = self.num_norm(num_feats)
        return self.mlp(num_feats)


class CatTransformerBackbone(nn.Module):
    def __init__(
        self,
        cat_cardinalities: Sequence[int],
        numeric_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        hidden_dims: Sequence[int],
        embedding_dim: int,
        dropout: float,
        flatten: bool = True,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in cat_cardinalities]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(2 * d_model, 64),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.flatten = flatten
        self.num_norm = nn.LayerNorm(numeric_dim) if numeric_dim > 0 else None
        transformer_out_dim = len(cat_cardinalities) * d_model if flatten else d_model
        input_dim = transformer_out_dim + numeric_dim
        self.mlp = build_mlp(input_dim, hidden_dims, embedding_dim, dropout)

    def forward(self, cat_ids: torch.Tensor, num_feats: torch.Tensor) -> torch.Tensor:
        cat_tokens = torch.stack(
            [emb(cat_ids[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )
        ctx = self.transformer(cat_tokens)
        if self.flatten:
            ctx = ctx.flatten(start_dim=1)
        else:
            ctx = ctx.mean(dim=1)
        if self.num_norm is not None:
            num_feats = self.num_norm(num_feats)
        x = torch.cat([ctx, num_feats], dim=1)
        return self.mlp(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dims, output_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float, num_classes: int) -> None:
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dims, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastivePipeline(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        projection: Optional[nn.Module],
        classifier: Optional[nn.Module],
        metric_from: str,
        normalize_embeddings: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = projection
        self.classifier = classifier
        self.metric_from = metric_from
        self.normalize_embeddings = normalize_embeddings

    def forward(self, cat_ids: torch.Tensor, num_feats: torch.Tensor):
        emb = self.backbone(cat_ids, num_feats)
        proj = self.projection(emb) if self.projection is not None else emb
        metric = proj if self.metric_from == "projection" else emb
        if self.normalize_embeddings:
            metric = F.normalize(metric, dim=1)
        logits = self.classifier(emb) if self.classifier is not None else None
        return emb, proj, metric, logits


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def parse_int_list(value: Optional[str], default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return [int(v) for v in value]
    value = value.strip()
    if not value:
        return list(default)
    return [int(part) for part in value.split(",") if part.strip()]


def parse_json_dict(value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    value = str(value).strip()
    if not value:
        return {}
    return json.loads(value)


def resolve_pml_component(spec, module):
    if isinstance(spec, dict):
        name = spec.get("name")
        params = spec.get("params", {})
        if name and hasattr(module, name):
            return getattr(module, name)(**params)
    if isinstance(spec, str) and hasattr(module, spec):
        return getattr(module, spec)()
    return spec


def build_pml_loss(loss_name: str, loss_params: Dict, embedding_dim: int, num_classes: int) -> nn.Module:
    try:
        from pytorch_metric_learning import losses, distances, reducers, regularizers
    except Exception as exc:
        raise ImportError("pytorch-metric-learning is required for contrastive losses.") from exc

    if not hasattr(losses, loss_name):
        raise ValueError(f"Loss '{loss_name}' not found in pytorch_metric_learning.losses")

    loss_cls = getattr(losses, loss_name)
    kwargs = dict(loss_params)
    sig = inspect.signature(loss_cls.__init__)
    if "embedding_size" in sig.parameters and "embedding_size" not in kwargs:
        kwargs["embedding_size"] = embedding_dim
    if "num_classes" in sig.parameters and "num_classes" not in kwargs:
        kwargs["num_classes"] = num_classes

    if "distance" in kwargs:
        kwargs["distance"] = resolve_pml_component(kwargs["distance"], distances)
    if "reducer" in kwargs:
        kwargs["reducer"] = resolve_pml_component(kwargs["reducer"], reducers)
    if "regularizer" in kwargs:
        kwargs["regularizer"] = resolve_pml_component(kwargs["regularizer"], regularizers)

    return loss_cls(**kwargs)


def build_pml_miner(miner_name: Optional[str], miner_params: Dict) -> Optional[nn.Module]:
    if miner_name is None or miner_name.lower() == "none":
        return None
    try:
        from pytorch_metric_learning import miners
    except Exception as exc:
        raise ImportError("pytorch-metric-learning is required for miners.") from exc

    if not hasattr(miners, miner_name):
        raise ValueError(f"Miner '{miner_name}' not found in pytorch_metric_learning.miners")

    miner_cls = getattr(miners, miner_name)
    return miner_cls(**miner_params)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception:
        metrics["pr_auc"] = float("nan")
    try:
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        else:
            metrics["roc_auc"] = float("nan")
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def evaluate_contrastive(
    model: ContrastivePipeline,
    loader: DataLoader,
    loss_fn: nn.Module,
    miner: Optional[nn.Module],
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    autocast = torch.cuda.amp.autocast
    with torch.no_grad():
        for cat_ids, num_feats, labels in loader:
            cat_ids = cat_ids.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device)
            with autocast(enabled=use_amp):
                _, _, metric, _ = model(cat_ids, num_feats)
                if miner is not None:
                    pairs = miner(metric, labels)
                    loss = loss_fn(metric, labels, pairs)
                else:
                    loss = loss_fn(metric, labels)
            total_loss += loss.item() * len(labels)
            total += len(labels)
    return total_loss / max(total, 1)


def evaluate_classifier(
    model: ContrastivePipeline,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    probs: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    autocast = torch.cuda.amp.autocast
    with torch.no_grad():
        for cat_ids, num_feats, labels in loader:
            cat_ids = cat_ids.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device)
            with autocast(enabled=use_amp):
                _, _, _, logits = model(cat_ids, num_feats)
                loss = ce_loss(logits, labels)
            total_loss += loss.item() * len(labels)
            total += len(labels)
            prob = torch.softmax(logits, dim=1)[:, 1]
            probs.append(prob.detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(labels_all) if labels_all else np.array([])
    y_prob = np.concatenate(probs) if probs else np.array([])
    metrics = compute_metrics(y_true, y_prob) if len(y_true) else {}
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def train_contrastive(
    model: ContrastivePipeline,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    loss_fn: nn.Module,
    miner: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    use_amp: bool,
    patience: int,
) -> Dict[str, float]:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state = None
    best_val = float("inf")
    patience_left = patience

    for epoch in range(epochs):
        model.train()
        if not finetune_backbone:
            model.backbone.eval()
            if model.projection is not None:
                model.projection.eval()
        total_loss = 0.0
        total = 0
        for cat_ids, num_feats, labels in train_loader:
            cat_ids = cat_ids.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, _, metric, _ = model(cat_ids, num_feats)
                if miner is not None:
                    pairs = miner(metric, labels)
                    loss = loss_fn(metric, labels, pairs)
                else:
                    loss = loss_fn(metric, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            total += len(labels)

        train_loss = total_loss / max(total, 1)
        msg = f"Epoch {epoch+1}/{epochs} | contrastive train loss={train_loss:.4f}"
        if val_loader is not None:
            val_loss = evaluate_contrastive(model, val_loader, loss_fn, miner, device, use_amp)
            msg += f" | val loss={val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_left = patience
            else:
                patience_left -= 1
        print(msg)

        if val_loader is not None and patience > 0 and patience_left <= 0:
            print("Early stopping (contrastive).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_loss": best_val}


def train_classifier(
    model: ContrastivePipeline,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    use_amp: bool,
    patience: int,
    monitor_metric: str,
    finetune_backbone: bool,
) -> Dict[str, float]:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce_loss = nn.CrossEntropyLoss()
    best_state = None
    best_score = -float("inf")
    patience_left = patience

    if not finetune_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        if model.projection is not None:
            for param in model.projection.parameters():
                param.requires_grad = False

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for cat_ids, num_feats, labels in train_loader:
            cat_ids = cat_ids.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, _, _, logits = model(cat_ids, num_feats)
                loss = ce_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            total += len(labels)

        train_loss = total_loss / max(total, 1)
        msg = f"Epoch {epoch+1}/{epochs} | classifier train loss={train_loss:.4f}"
        if val_loader is not None:
            metrics = evaluate_classifier(model, val_loader, device, use_amp)
            score = metrics.get(monitor_metric, float("nan"))
            msg += f" | val {monitor_metric}={score:.4f}"
            if not math.isnan(score) and score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                patience_left = patience
            else:
                patience_left -= 1
        print(msg)

        if val_loader is not None and patience > 0 and patience_left <= 0:
            print("Early stopping (classifier).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_score": best_score}


def parse_eval_specs(values: Optional[List[str]]) -> List[EvalSpec]:
    if not values:
        return []
    specs: List[EvalSpec] = []
    for raw in values:
        parts = raw.split(":")
        if len(parts) == 3:
            name, data_path, target_path = parts
        elif len(parts) == 2:
            data_path, target_path = parts
            name = Path(data_path).stem
        else:
            raise ValueError("Eval spec must be 'name:data_path:target_path' or 'data_path:target_path'")
        specs.append(EvalSpec(name=name, data_path=Path(data_path), target_path=Path(target_path)))
    return specs


def build_backbone(
    backbone_type: str,
    cat_cardinalities: Sequence[int],
    numeric_dim: int,
    embedding_dim: int,
    cat_embed_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    tf_d_model: int,
    tf_heads: int,
    tf_layers: int,
) -> nn.Module:
    if backbone_type == "dense_mlp":
        return DenseMLPBackbone(numeric_dim, hidden_dims, embedding_dim, dropout)
    if backbone_type == "cat_mlp":
        if not cat_cardinalities:
            raise ValueError("cat_mlp requires categorical embeddings (cat_encoder=embedding).")
        return CatMLPBackbone(cat_cardinalities, numeric_dim, cat_embed_dim, hidden_dims, embedding_dim, dropout)
    if backbone_type == "cat_tf":
        if not cat_cardinalities:
            raise ValueError("cat_tf requires categorical embeddings (cat_encoder=embedding).")
        return CatTransformerBackbone(
            cat_cardinalities=cat_cardinalities,
            numeric_dim=numeric_dim,
            d_model=tf_d_model,
            nhead=tf_heads,
            num_layers=tf_layers,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout,
            flatten=True,
        )
    raise ValueError(f"Unknown backbone_type: {backbone_type}")


def build_pipeline(
    backbone: nn.Module,
    embedding_dim: int,
    proj_dim: int,
    proj_hidden: Sequence[int],
    proj_dropout: float,
    clf_hidden: Sequence[int],
    clf_dropout: float,
    num_classes: int,
    metric_from: str,
    normalize_embeddings: bool,
) -> ContrastivePipeline:
    if proj_dim <= 0:
        projection = None
    elif proj_dim == embedding_dim and not proj_hidden:
        projection = nn.Identity()
    else:
        projection = ProjectionHead(embedding_dim, proj_hidden, proj_dim, proj_dropout)

    classifier = ClassifierHead(embedding_dim, clf_hidden, clf_dropout, num_classes)
    return ContrastivePipeline(backbone, projection, classifier, metric_from, normalize_embeddings)


def serialize_args(args: argparse.Namespace) -> Dict:
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list):
            payload[key] = value
        else:
            payload[key] = value
    return payload


def run_experiment(args: argparse.Namespace, save_outputs: bool) -> Dict[str, Dict]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    preprocess_cfg = PreprocessConfig(
        num_scaler=args.num_scaler,
        cat_encoder=args.cat_encoder,
        impute_strategy=args.impute_strategy,
        add_availability_flags=not args.no_avail_flags,
    )

    data1_path = Path(args.data1)
    target1_path = Path(args.target1)
    df1, y1 = load_raw_data(
        data1_path,
        target1_path,
        args.target_col,
        args.sample_size,
        args.seed,
    )

    df2 = y2 = None
    if args.data2 and args.target2:
        df2, y2 = load_raw_data(
            Path(args.data2),
            Path(args.target2),
            args.target_col,
            args.sample_size_data2,
            args.seed,
        )

    df_train_raw, y_train_raw = mix_with_data2(
        df1, y1, df2, y2, args.mix_percent, args.mix_base, args.seed
    )

    y_all = y_train_raw.to_numpy(dtype=np.int64)

    train_idx, val_idx, test_idx = split_indices(
        y_all, args.val_split, args.test_split, args.seed
    )

    train_df = df_train_raw.iloc[train_idx]
    val_df = df_train_raw.iloc[val_idx] if len(val_idx) else None
    test_df = df_train_raw.iloc[test_idx] if len(test_idx) else None

    preprocessor = TabularPreprocessor(preprocess_cfg, NUMERIC_COLS, CAT_COLS).fit(train_df)
    cat_train, num_train = preprocessor.transform(train_df)
    train_ds = TabularDataset(cat_train, num_train, y_all[train_idx])

    val_ds = None
    if val_df is not None:
        cat_val, num_val = preprocessor.transform(val_df)
        val_ds = TabularDataset(cat_val, num_val, y_all[val_idx])

    test_ds = None
    if test_df is not None:
        cat_test, num_test = preprocessor.transform(test_df)
        test_ds = TabularDataset(cat_test, num_test, y_all[test_idx])

    train_loader = build_dataloader(
        train_ds,
        batch_size=args.batch_size,
        sampler_type=args.sampler,
        m_per_class=args.m_per_class,
        seed=args.seed,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

    if args.cat_encoder == "onehot" and args.backbone_type != "dense_mlp":
        raise ValueError("cat_encoder=onehot requires backbone_type=dense_mlp.")

    backbone = build_backbone(
        backbone_type=args.backbone_type,
        cat_cardinalities=preprocessor.cat_cardinalities,
        numeric_dim=num_train.shape[1],
        embedding_dim=args.embedding_dim,
        cat_embed_dim=args.cat_embed_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        tf_d_model=args.tf_d_model,
        tf_heads=args.tf_heads,
        tf_layers=args.tf_layers,
    )
    model = build_pipeline(
        backbone=backbone,
        embedding_dim=args.embedding_dim,
        proj_dim=args.proj_dim,
        proj_hidden=args.proj_hidden,
        proj_dropout=args.proj_dropout,
        clf_hidden=args.clf_hidden,
        clf_dropout=args.clf_dropout,
        num_classes=args.num_classes,
        metric_from=args.metric_from,
        normalize_embeddings=args.normalize_embeddings,
    ).to(device)

    metric_dim = args.embedding_dim
    if args.metric_from == "projection":
        if args.proj_dim <= 0:
            print("metric_from=projection but proj_dim<=0; using embedding instead.")
        else:
            metric_dim = args.proj_dim

    loss_fn = build_pml_loss(
        loss_name=args.loss_name,
        loss_params=args.loss_params,
        embedding_dim=metric_dim,
        num_classes=args.num_classes,
    )
    miner = build_pml_miner(args.miner_name, args.miner_params)

    results: Dict[str, Dict] = {"train": {}, "val": {}, "test": {}, "eval": {}}

    if args.mode == "separate":
        optimizer = torch.optim.AdamW(
            list(model.backbone.parameters())
            + ([] if model.projection is None else list(model.projection.parameters())),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        results["train"]["contrastive"] = train_contrastive(
            model,
            train_loader,
            val_loader,
            loss_fn,
            miner,
            optimizer,
            device,
            args.contrastive_epochs,
            use_amp,
            args.patience,
        )
        if save_outputs:
            torch.save(model.state_dict(), args.output_dir / f"{args.run_name}_contrastive.pt")

        classifier_optimizer = torch.optim.AdamW(
            list(model.classifier.parameters())
            + (list(model.backbone.parameters()) if args.finetune_backbone else []),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        results["train"]["classifier"] = train_classifier(
            model,
            train_loader,
            val_loader,
            classifier_optimizer,
            device,
            args.classifier_epochs,
            use_amp,
            args.patience,
            args.monitor_metric,
            args.finetune_backbone,
        )
        if save_outputs:
            torch.save(model.state_dict(), args.output_dir / f"{args.run_name}_classifier.pt")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        results["train"]["joint"] = train_joint(
            model,
            train_loader,
            val_loader,
            loss_fn,
            miner,
            optimizer,
            device,
            args.epochs,
            use_amp,
            args.patience,
            args.monitor_metric,
            args.loss_alpha,
            args.loss_beta,
        )
        if save_outputs:
            torch.save(model.state_dict(), args.output_dir / f"{args.run_name}_joint.pt")

    if val_loader is not None:
        results["val"] = evaluate_classifier(model, val_loader, device, use_amp)

    if test_loader is not None:
        results["test"] = evaluate_classifier(model, test_loader, device, use_amp)

    eval_specs = parse_eval_specs(args.eval)
    for spec in eval_specs:
        eval_df, eval_y = load_raw_data(
            spec.data_path, spec.target_path, args.target_col, None, args.seed
        )
        cat_eval, num_eval = preprocessor.transform(eval_df)
        eval_ds = TabularDataset(cat_eval, num_eval, eval_y.to_numpy(dtype=np.int64))
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        results["eval"][spec.name] = evaluate_classifier(model, eval_loader, device, use_amp)

    if save_outputs:
        with open(args.output_dir / f"{args.run_name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        with open(args.output_dir / f"{args.run_name}_config.json", "w", encoding="utf-8") as f:
            json.dump(serialize_args(args), f, indent=2)

    return results


def load_search_space(path: Path) -> Dict[str, List]:
    with open(path, "r", encoding="utf-8") as f:
        space = json.load(f)
    if not isinstance(space, dict):
        raise ValueError("Search space JSON must be a dict of parameter -> list.")
    return space


def sample_from_space(space: Dict[str, List]) -> Dict[str, object]:
    return {key: random.choice(values) for key, values in space.items()}


def grid_from_space(space: Dict[str, List]) -> Iterable[Dict[str, object]]:
    keys = list(space.keys())
    for combo in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, combo))


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, object]) -> argparse.Namespace:
    new_args = copy.deepcopy(args)
    for key, value in overrides.items():
        if not hasattr(new_args, key):
            raise ValueError(f"Unknown tuning parameter: {key}")
        current = getattr(new_args, key)
        if isinstance(current, list) and isinstance(value, str):
            value = parse_int_list(value, current)
        setattr(new_args, key, value)
    return new_args


def run_tuning(args: argparse.Namespace) -> None:
    space = load_search_space(Path(args.tune_space))
    trials = args.tune_trials
    metric = args.tune_metric
    best_score = -float("inf")
    best_params = None

    if args.tune_method == "grid":
        combos = list(grid_from_space(space))
    else:
        combos = [sample_from_space(space) for _ in range(trials)]

    for idx, params in enumerate(combos, start=1):
        trial_args = apply_overrides(args, params)
        if args.tune_epochs is not None and args.tune_epochs > 0:
            trial_args.epochs = args.tune_epochs
            trial_args.contrastive_epochs = args.tune_epochs
            trial_args.classifier_epochs = args.tune_epochs
        trial_args.run_name = f"{args.run_name}_trial{idx}"
        print(f"Trial {idx}/{len(combos)} params={params}")
        results = run_experiment(trial_args, save_outputs=False)
        val_metrics = results.get("val") or results.get("test", {})
        score = val_metrics.get(metric, float("nan"))
        print(f"Trial {idx} {metric}={score}")
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_params = params

    print(f"Best {metric}={best_score} params={best_params}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QoT contrastive + classifier training")
    parser.add_argument("--mode", choices=["separate", "joint"], default="joint")
    parser.add_argument("--data1", required=True)
    parser.add_argument("--target1", required=True)
    parser.add_argument("--data2")
    parser.add_argument("--target2")
    parser.add_argument("--target-col", default="class")
    parser.add_argument("--mix-percent", type=float, default=0.0)
    parser.add_argument("--mix-base", choices=["data1", "data2"], default="data2")
    parser.add_argument("--sample-size", type=int, default=-1)
    parser.add_argument("--sample-size-data2", type=int, default=-1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--eval", action="append", help="name:data_path:target_path or data_path:target_path")

    parser.add_argument("--num-scaler", choices=["none", "standard", "minmax"], default="none")
    parser.add_argument("--cat-encoder", choices=["embedding", "onehot"], default="embedding")
    parser.add_argument("--impute-strategy", choices=["median", "zero"], default="median")
    parser.add_argument("--no-avail-flags", action="store_true")

    parser.add_argument("--sampler", choices=["random", "m_per_class", "class_balanced"], default="random")
    parser.add_argument("--m-per-class", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--backbone-type", choices=["cat_mlp", "cat_tf", "dense_mlp"], default="cat_mlp")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dims", default="512,256")
    parser.add_argument("--cat-embed-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tf-d-model", type=int, default=64)
    parser.add_argument("--tf-heads", type=int, default=4)
    parser.add_argument("--tf-layers", type=int, default=2)

    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--proj-hidden", default="256")
    parser.add_argument("--proj-dropout", type=float, default=0.1)
    parser.add_argument("--metric-from", choices=["projection", "embedding"], default="projection")
    parser.add_argument("--normalize-embeddings", action="store_true")

    parser.add_argument("--clf-hidden", default="256")
    parser.add_argument("--clf-dropout", type=float, default=0.2)
    parser.add_argument("--num-classes", type=int, default=2)

    parser.add_argument("--loss-name", default="TripletMarginLoss")
    parser.add_argument("--loss-params", default="")
    parser.add_argument("--miner-name", default="none")
    parser.add_argument("--miner-params", default="")
    parser.add_argument("--loss-alpha", type=float, default=0.5)
    parser.add_argument("--loss-beta", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--contrastive-epochs", type=int, default=20)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--monitor-metric", default="pr_auc")
    parser.add_argument("--finetune-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--run-name", default="")

    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-space", default="")
    parser.add_argument("--tune-trials", type=int, default=10)
    parser.add_argument("--tune-method", choices=["random", "grid"], default="random")
    parser.add_argument("--tune-metric", default="pr_auc")
    parser.add_argument("--tune-epochs", type=int, default=0)

    args = parser.parse_args()

    args.hidden_dims = parse_int_list(args.hidden_dims, [512, 256])
    args.proj_hidden = parse_int_list(args.proj_hidden, [256])
    args.clf_hidden = parse_int_list(args.clf_hidden, [256])
    args.loss_params = parse_json_dict(args.loss_params)
    args.miner_params = parse_json_dict(args.miner_params)
    args.sample_size = None if args.sample_size is None or args.sample_size < 0 else args.sample_size
    args.sample_size_data2 = (
        None if args.sample_size_data2 is None or args.sample_size_data2 < 0 else args.sample_size_data2
    )

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    args.run_name = run_name
    args.output_dir = Path(args.save_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.tune and not args.tune_space:
        raise ValueError("--tune-space is required when --tune is set.")

    return args


def main() -> None:
    args = parse_args()
    if args.tune:
        run_tuning(args)
        return
    results = run_experiment(args, save_outputs=True)
    print("Test metrics:")
    print(json.dumps(results.get("test", {}), indent=2))
    if results.get("eval"):
        print("Eval metrics:")
        print(json.dumps(results["eval"], indent=2))


if __name__ == "__main__":
    main()


def train_joint(
    model: ContrastivePipeline,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    loss_fn: nn.Module,
    miner: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    use_amp: bool,
    patience: int,
    monitor_metric: str,
    alpha: float,
    beta: float,
) -> Dict[str, float]:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce_loss = nn.CrossEntropyLoss()
    best_state = None
    best_score = -float("inf")
    patience_left = patience

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for cat_ids, num_feats, labels in train_loader:
            cat_ids = cat_ids.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, _, metric, logits = model(cat_ids, num_feats)
                if miner is not None:
                    pairs = miner(metric, labels)
                    cont_loss = loss_fn(metric, labels, pairs)
                else:
                    cont_loss = loss_fn(metric, labels)
                cls_loss = ce_loss(logits, labels)
                loss = alpha * cont_loss + beta * cls_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            total += len(labels)

        train_loss = total_loss / max(total, 1)
        msg = f"Epoch {epoch+1}/{epochs} | joint train loss={train_loss:.4f}"
        if val_loader is not None:
            metrics = evaluate_classifier(model, val_loader, device, use_amp)
            score = metrics.get(monitor_metric, float("nan"))
            msg += f" | val {monitor_metric}={score:.4f}"
            if not math.isnan(score) and score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                patience_left = patience
            else:
                patience_left -= 1
        print(msg)

        if val_loader is not None and patience > 0 and patience_left <= 0:
            print("Early stopping (joint).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_score": best_score}
