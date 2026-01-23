"""
End-to-end contrastive + classification training for the QoT tabular data.

Pipeline blocks:
1) Data prep: numeric-only features, sampler builds M+N mini-batches.
2) Backbone: numeric MLP encoder -> latent z (embedding_dim).
3) Mining + contrastive loss: PML miners/losses on the batch embedding.
4) Classification head: predicts class from z (cross-entropy).

Supports:
- Separate training: contrastive (backbone+projection) then classifier.
- Joint training: optimize contrastive + classification losses together.
- Optional fine-tuning: after initial training, continue supervised training on a second dataset.
- Hyperparameter tuning via random/grid search over a JSON search space.

Example:
  python separate_training.py --mode joint \
    --data1 datasets/cleaned_lightpath_dataset.csv --target1 datasets/cleaned_lightpath_target.csv \
    --loss-name TripletMarginLoss --miner-name BatchHardMiner \
    --sampler m_per_class --m-per-class 64 \
    --eval data2:datasets/cleaned_lightpath_dataset_2.csv:datasets/cleaned_lightpath_target_2.csv

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
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Import torch before numpy/pandas to avoid DLL initialization issues on Windows.
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
)
 


@dataclass
class PreprocessConfig:
    num_scaler: str = "none"           # "none", "standard", "minmax"
    impute_strategy: str = "median"    # "median", "zero"


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


class TabularPreprocessor:
    def __init__(
        self,
        cfg: PreprocessConfig,
        numeric_cols: Optional[Sequence[str]] = None,
        drop_cols: Optional[Sequence[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.numeric_cols = list(numeric_cols) if numeric_cols else []
        self.drop_cols = list(drop_cols) if drop_cols else []
        self.num_impute: Optional[pd.Series] = None
        self.num_scale: Optional[pd.Series] = None
        self.num_shift: Optional[pd.Series] = None
        self.cat_cardinalities: List[int] = []
        self.fitted = False

    def _check_columns(self, df: pd.DataFrame) -> None:
        expected = set(self.numeric_cols)
        missing = expected.difference(df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in dataset: {missing_list}")

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        df = df.copy()
        if self.drop_cols:
            df = df.drop(columns=self.drop_cols, errors="ignore")
        if not self.numeric_cols:
            self.numeric_cols = list(df.columns)
        self._check_columns(df)

        num_df = df[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
        if num_df.isna().any().any():
            nan_count = int(num_df.isna().sum().sum())
            print(f"Warning: {nan_count} missing/non-numeric values detected; imputation applied.")
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

        self.cat_cardinalities = []
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        df = df.copy()
        if self.drop_cols:
            df = df.drop(columns=self.drop_cols, errors="ignore")
        self._check_columns(df)

        num_df = df[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
        num_df = num_df.fillna(self.num_impute)
        if self.cfg.num_scaler == "standard":
            num_df = (num_df - self.num_shift) / self.num_scale
        elif self.cfg.num_scaler == "minmax":
            num_df = (num_df - self.num_shift) / self.num_scale

        num_matrix = num_df.to_numpy(dtype=np.float32)
        cat_matrix = np.empty((len(df), 0), dtype=np.int64)
        return cat_matrix, num_matrix


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


def parse_col_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    value = str(value).strip()
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_json_dict(value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    value = str(value).strip()
    if not value:
        return {}
    return json.loads(value)


def get_trainable_params(module) -> List[nn.Parameter]:
    if module is None or not hasattr(module, "parameters"):
        return []
    return [param for param in module.parameters() if param.requires_grad]


def move_module_to_device(module, device: torch.device):
    if module is None:
        return None
    if hasattr(module, "to"):
        return module.to(device)
    return module


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
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    per_prec, per_rec, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    metrics["precision_0"] = float(per_prec[0])
    metrics["recall_0"] = float(per_rec[0])
    metrics["precision_1"] = float(per_prec[1])
    metrics["recall_1"] = float(per_rec[1])
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
    if hasattr(loss_fn, "eval"):
        loss_fn.eval()
    if miner is not None and hasattr(miner, "eval"):
        miner.eval()
    total_loss = 0.0
    total = 0
    autocast = torch.amp.autocast
    with torch.no_grad():
        for cat_ids, num_feats, labels in loader:
            cat_ids = cat_ids.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device)
            with autocast(device_type=device.type, enabled=use_amp):
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
    autocast = torch.amp.autocast
    with torch.no_grad():
        for cat_ids, num_feats, labels in loader:
            cat_ids = cat_ids.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device)
            with autocast(device_type=device.type, enabled=use_amp):
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
    normalize_loss_by_batch: bool,
) -> Dict[str, float]:
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_state = None
    best_val = float("inf")
    patience_left = patience

    for epoch in range(epochs):
        model.train()
        if hasattr(loss_fn, "train"):
            loss_fn.train()
        if miner is not None and hasattr(miner, "train"):
            miner.train()
        total_loss = 0.0
        total = 0
        for cat_ids, num_feats, labels in train_loader:
            cat_ids = cat_ids.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _, _, metric, _ = model(cat_ids, num_feats)
                if miner is not None:
                    pairs = miner(metric, labels)
                    loss = loss_fn(metric, labels, pairs)
                else:
                    loss = loss_fn(metric, labels)
                if normalize_loss_by_batch:
                    loss = loss / max(batch_size, 1)
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
    scaler = torch.amp.GradScaler(enabled=use_amp)
    ce_loss = nn.CrossEntropyLoss()
    best_state = None
    best_score = -float("inf")
    patience_left = patience

    set_backbone_trainable(model, finetune_backbone)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for cat_ids, num_feats, labels in train_loader:
            cat_ids = cat_ids.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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
    normalize_loss_by_batch: bool,
) -> Dict[str, float]:
    scaler = torch.amp.GradScaler(enabled=use_amp)
    ce_loss = nn.CrossEntropyLoss()
    best_state = None
    best_score = -float("inf")
    patience_left = patience

    for epoch in range(epochs):
        model.train()
        if hasattr(loss_fn, "train"):
            loss_fn.train()
        if miner is not None and hasattr(miner, "train"):
            miner.train()
        total_loss = 0.0
        total = 0
        for cat_ids, num_feats, labels in train_loader:
            cat_ids = cat_ids.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _, _, metric, logits = model(cat_ids, num_feats)
                if miner is not None:
                    pairs = miner(metric, labels)
                    cont_loss = loss_fn(metric, labels, pairs)
                else:
                    cont_loss = loss_fn(metric, labels)
                if normalize_loss_by_batch:
                    cont_loss = cont_loss / max(batch_size, 1)
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


def parse_eval_specs(values: Optional[List[str]]) -> List[EvalSpec]:
    if values is None or values == "":
        return []
    if isinstance(values, str):
        values = [values]
    specs: List[EvalSpec] = []
    for raw in values:
        if raw is None or raw == "":
            continue
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
    numeric_dim: int,
    embedding_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
) -> nn.Module:
    if backbone_type != "dense_mlp":
        raise ValueError("Numeric-only pipeline supports backbone_type='dense_mlp'.")
    return DenseMLPBackbone(numeric_dim, hidden_dims, embedding_dim, dropout)


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


def set_backbone_trainable(model: ContrastivePipeline, trainable: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = trainable
    if model.projection is not None:
        for param in model.projection.parameters():
            param.requires_grad = trainable


def build_split_loaders(
    df: pd.DataFrame,
    y: pd.Series,
    preprocessor: TabularPreprocessor,
    args: argparse.Namespace,
    val_split: float,
    test_split: float,
    seed: int,
    fit_preprocessor: bool,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], np.ndarray]:
    y_all = y.to_numpy(dtype=np.int64)
    train_idx, val_idx, test_idx = split_indices(y_all, val_split, test_split, seed)

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx] if len(val_idx) else None
    test_df = df.iloc[test_idx] if len(test_idx) else None

    if fit_preprocessor:
        preprocessor.fit(train_df)

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
        seed=seed,
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

    return train_loader, val_loader, test_loader, num_train


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
        impute_strategy=args.impute_strategy,
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

    preprocessor = TabularPreprocessor(
        preprocess_cfg,
        numeric_cols=args.numeric_cols,
        drop_cols=args.drop_cols,
    )
    train_loader, val_loader, test_loader, num_train = build_split_loaders(
        df1,
        y1,
        preprocessor,
        args,
        args.val_split,
        args.test_split,
        args.seed,
        fit_preprocessor=True,
    )

    backbone = build_backbone(
        backbone_type=args.backbone_type,
        numeric_dim=num_train.shape[1],
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
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

    loss_fn = move_module_to_device(
        build_pml_loss(
            loss_name=args.loss_name,
            loss_params=args.loss_params,
            embedding_dim=metric_dim,
            num_classes=args.num_classes,
        ),
        device,
    )
    miner = move_module_to_device(build_pml_miner(args.miner_name, args.miner_params), device)
    loss_params = get_trainable_params(loss_fn)

    results: Dict[str, Dict] = {"train": {}, "val": {}, "test": {}, "eval": {}, "finetune": {}}

    if args.mode == "separate":
        contrastive_params = list(model.backbone.parameters())
        if model.projection is not None:
            contrastive_params += list(model.projection.parameters())
        contrastive_params += loss_params
        optimizer = torch.optim.AdamW(
            contrastive_params,
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
            args.normalize_loss_by_batch,
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
        joint_params = list(model.parameters()) + loss_params
        optimizer = torch.optim.AdamW(
            joint_params,
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
            args.normalize_loss_by_batch,
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

    if args.finetune_data or args.finetune_target:
        if not (args.finetune_data and args.finetune_target):
            raise ValueError("Both --finetune-data and --finetune-target are required for fine-tuning.")

        df2, y2 = load_raw_data(
            Path(args.finetune_data),
            Path(args.finetune_target),
            args.target_col,
            args.finetune_sample_size,
            args.seed,
        )
        ft_train_loader, ft_val_loader, ft_test_loader, _ = build_split_loaders(
            df2,
            y2,
            preprocessor,
            args,
            args.finetune_val_split,
            args.finetune_test_split,
            args.seed,
            fit_preprocessor=False,
        )

        finetune_optimizer = torch.optim.AdamW(
            list(model.classifier.parameters())
            + (list(model.backbone.parameters()) if args.finetune_backbone else []),
            lr=args.finetune_lr,
            weight_decay=args.finetune_weight_decay,
        )
        results["finetune"]["classifier"] = train_classifier(
            model,
            ft_train_loader,
            ft_val_loader,
            finetune_optimizer,
            device,
            args.finetune_epochs,
            use_amp,
            args.patience,
            args.monitor_metric,
            args.finetune_backbone,
        )
        if save_outputs:
            torch.save(model.state_dict(), args.output_dir / f"{args.run_name}_finetune.pt")

        if ft_val_loader is not None:
            results["finetune"]["val"] = evaluate_classifier(model, ft_val_loader, device, use_amp)
        if ft_test_loader is not None:
            results["finetune"]["test"] = evaluate_classifier(model, ft_test_loader, device, use_amp)

        results["finetune"]["eval"] = {}
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
            results["finetune"]["eval"][spec.name] = evaluate_classifier(
                model, eval_loader, device, use_amp
            )

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
    normalized: Dict[str, List] = {}
    for key, value in space.items():
        if isinstance(value, list):
            normalized[key] = value
        else:
            normalized[key] = [value]
    return normalized


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


def _extract_tune_controls(
    args: argparse.Namespace,
    space: Dict[str, List],
) -> Tuple[argparse.Namespace, Dict[str, List]]:
    """Extract tune control keys from space and let them override CLI args."""
    meta_keys = {
        "tune_method",
        "tune_trials",
        "tune_metric",
        "tune_epochs",
    }
    new_args = copy.deepcopy(args)
    remaining = {}

    def _first_value(value):
        if isinstance(value, list):
            if not value:
                return None
            if len(value) > 1:
                raise ValueError(
                    "Tune control keys must have a single value in the JSON (got multiple)."
                )
            return value[0]
        return value

    for key, value in space.items():
        if key in meta_keys:
            resolved = _first_value(value)
            if resolved is not None:
                if not hasattr(new_args, key):
                    raise ValueError(f"Unknown tune control parameter: {key}")
                setattr(new_args, key, resolved)
        else:
            remaining[key] = value

    return new_args, remaining


def run_tuning(args: argparse.Namespace) -> None:
    space = load_search_space(Path(args.tune_space))
    args, space = _extract_tune_controls(args, space)
    trials = args.tune_trials
    metric = args.tune_metric
    best_score = -float("inf")
    best_params = None
    failed_trials: List[Tuple[int, Dict[str, object], str]] = []

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
        try:
            results = run_experiment(trial_args, save_outputs=False)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            failed_trials.append((idx, params, err))
            print(f"Trial {idx} failed; skipping. Error: {err}")
            print(traceback.format_exc())
            continue
        val_metrics = results.get("val") or results.get("test", {})
        score = val_metrics.get(metric, float("nan"))
        print(f"Trial {idx} {metric}={score}")
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_params = params

    print(f"Best {metric}={best_score} params={best_params}")
    if failed_trials:
        print(f"Skipped {len(failed_trials)} failed trials.")
        for idx, params, err in failed_trials:
            print(f"  Trial {idx} params={params} error={err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QoT contrastive + classifier training")
    parser.add_argument("--config", default="")
    parser.add_argument("--mode", choices=["separate", "joint"], default="joint")
    parser.add_argument("--data1", default="")
    parser.add_argument("--target1", default="")
    parser.add_argument("--target-col", default="class")
    parser.add_argument("--sample-size", type=int, default=-1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--eval", action="append", help="name:data_path:target_path or data_path:target_path")

    parser.add_argument("--num-scaler", choices=["none", "standard", "minmax"], default="none")
    parser.add_argument("--impute-strategy", choices=["median", "zero"], default="median")
    parser.add_argument(
        "--numeric-cols",
        default="",
        help="Comma-separated list of numeric feature columns. Empty = use all columns.",
    )
    parser.add_argument(
        "--drop-cols",
        default="",
        help="Comma-separated list of columns to drop before training.",
    )

    parser.add_argument("--sampler", choices=["random", "m_per_class", "class_balanced"], default="random")
    parser.add_argument("--m-per-class", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--backbone-type", choices=["dense_mlp"], default="dense_mlp")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dims", default="512,256")
    parser.add_argument("--dropout", type=float, default=0.1)

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
    parser.add_argument("--normalize-loss-by-batch", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--contrastive-epochs", type=int, default=20)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--monitor-metric", default="pr_auc")
    parser.add_argument("--finetune-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--finetune-data", default="")
    parser.add_argument("--finetune-target", default="")
    parser.add_argument("--finetune-sample-size", type=int, default=-1)
    parser.add_argument("--finetune-val-split", type=float, default=None)
    parser.add_argument("--finetune-test-split", type=float, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--finetune-lr", type=float, default=None)
    parser.add_argument("--finetune-weight-decay", type=float, default=None)

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
    cli_values = vars(args).copy()

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise ValueError(f"--config path not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        if not isinstance(config_data, dict):
            raise ValueError("--config must point to a JSON object of argument overrides.")
        for key, value in config_data.items():
            if not hasattr(args, key):
                raise ValueError(f"Unknown config parameter: {key}")
            setattr(args, key, value)
        provided_dests = set()
        argv = sys.argv[1:]
        for arg in argv:
            if not arg.startswith("--"):
                continue
            opt = arg.split("=", 1)[0]
            action = parser._option_string_actions.get(opt)
            if action is not None:
                provided_dests.add(action.dest)
        for dest in provided_dests:
            setattr(args, dest, cli_values[dest])

    args.hidden_dims = parse_int_list(args.hidden_dims, [512, 256])
    args.proj_hidden = parse_int_list(args.proj_hidden, [256])
    args.clf_hidden = parse_int_list(args.clf_hidden, [256])
    args.numeric_cols = parse_col_list(args.numeric_cols)
    args.drop_cols = parse_col_list(args.drop_cols)
    if not args.numeric_cols:
        args.numeric_cols = None
    args.loss_params = parse_json_dict(args.loss_params)
    args.miner_params = parse_json_dict(args.miner_params)
    args.sample_size = None if args.sample_size is None or args.sample_size < 0 else args.sample_size
    args.finetune_sample_size = (
        None
        if args.finetune_sample_size is None or args.finetune_sample_size < 0
        else args.finetune_sample_size
    )
    args.finetune_data = args.finetune_data or None
    args.finetune_target = args.finetune_target or None
    if args.finetune_val_split is None:
        args.finetune_val_split = args.val_split
    if args.finetune_test_split is None:
        args.finetune_test_split = args.test_split
    if args.finetune_epochs is None:
        args.finetune_epochs = args.classifier_epochs
    if args.finetune_lr is None:
        args.finetune_lr = args.lr
    if args.finetune_weight_decay is None:
        args.finetune_weight_decay = args.weight_decay

    if not args.data1 or not args.target1:
        raise ValueError("--data1 and --target1 are required (via CLI or --config).")

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
    if results.get("finetune", {}).get("eval"):
        print("Fine-tune eval metrics:")
        print(json.dumps(results["finetune"]["eval"], indent=2))


if __name__ == "__main__":
    main()
