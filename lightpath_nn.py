"""
Neural network architectures for the lightpath dataset with hyperparameter-friendly
APIs (Optuna / wandb sweeps ready).

Three variants:
1) CatTransformerConcatMLP: categorical embeddings -> lightweight Transformer encoder
   -> flatten -> concat numeric (and flags) -> MLP -> binary logit.
2) TokenTransformerWithCls: categorical embeddings + numeric embeddings + [CLS] token
   -> lightweight Transformer encoder -> CLS -> MLP -> binary logit.
3) CatMLP: categorical embeddings -> concat numeric (and flags) -> MLP.
4) DenseMLP: all features treated as dense (numeric + flags + optional one-hot cats) -> MLP.

Preprocessing inside this script is now configurable from the CLI:
- num_scaler: none | standard | minmax
- cat_encoder: embedding | onehot
- impute_strategy: median | zero
- availability flags can be toggled

MLP widths are configurable per layer via a comma-separated list (e.g., --mlp-hidden 256,128).
Models accept configurable depth/width/heads via CLI args for easy tuning.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

# ----------------------------
# Column definitions (align with preprocess_dataset.py)
# ----------------------------
AVAIL_GROUPS: Dict[str, List[str]] = {
    "mod_left": ["min_mod_order_left", "max_mod_order_left"],
    "mod_right": ["min_mod_order_right", "max_mod_order_right"],
    "linerate_left": ["min_lp_linerate_left", "max_lp_linerate_left"],
    "linerate_right": ["min_lp_linerate_right", "max_lp_linerate_right"],
    "ber_left": ["min_ber_left", "max_ber_left"],
    "ber_right": ["min_ber_right", "max_ber_right"],
}

NUMERIC_COLS: List[str] = [
    "path_len",
    "avg_link_len",
    "min_link_len",
    "max_link_len",
    "sum_link_occ",
    "min_link_occ",
    "max_link_occ",
    "avg_link_occ",
    "std_link_occ",
    "max_ber",
    "min_ber",
    "avg_ber",
    "min_ber_left",
    "max_ber_left",
    "min_ber_right",
    "max_ber_right",
]

CAT_COLS: List[str] = [
    "num_links",
    "num_spans",
    "freq",
    "mod_order",
    "lp_linerate",
    "conn_linerate",
    "src_degree",
    "dst_degree",
    "min_mod_order_left",
    "max_mod_order_left",
    "min_mod_order_right",
    "max_mod_order_right",
    "min_lp_linerate_left",
    "max_lp_linerate_left",
    "min_lp_linerate_right",
    "max_lp_linerate_right",
]


@dataclass
class PreprocessConfig:
    num_scaler: str = "none"           # "none", "standard", "minmax"
    cat_encoder: str = "embedding"     # "embedding", "onehot"
    impute_strategy: str = "median"    # "median", "zero"
    add_availability_flags: bool = True


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


def factorize_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> Tuple[np.ndarray, List[int]]:
    cat_arrays = []
    cardinalities: List[int] = []
    for col in cat_cols:
        series = df[col].fillna("missing").astype(str)
        codes, uniques = pd.factorize(series, sort=True)
        cat_arrays.append(codes.astype(np.int64))
        # +1 for padding slot if ever needed
        cardinalities.append(len(uniques) + 1)
    cat_matrix = np.stack(cat_arrays, axis=1) if cat_arrays else np.empty(
        (len(df), 0), dtype=np.int64)
    return cat_matrix, cardinalities


def onehot_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    if not cat_cols:
        return np.empty((len(df), 0), dtype=np.float32), []
    cat_df = df[cat_cols].apply(lambda s: s.fillna("missing").astype(str))
    onehot_df = pd.get_dummies(cat_df, columns=cat_cols, prefix=cat_cols)
    return onehot_df.to_numpy(dtype=np.float32), list(onehot_df.columns)


def impute_numeric(df: pd.DataFrame, numeric_cols: List[str], strategy: str) -> pd.DataFrame:
    numeric_df = df[numeric_cols].copy()
    if not numeric_cols:
        return numeric_df
    if strategy == "median":
        fill_values = numeric_df.median()
    elif strategy == "zero":
        fill_values = 0
    else:
        raise ValueError(f"Unsupported impute_strategy: {strategy}")
    return numeric_df.fillna(fill_values)


def scale_numeric(num_df: pd.DataFrame, scaler: str) -> pd.DataFrame:
    if scaler == "none":
        return num_df.astype(np.float32)
    if scaler == "standard":
        mean = num_df.mean()
        std = num_df.std().replace(0, 1e-6)
        return ((num_df - mean) / std).astype(np.float32)
    if scaler == "minmax":
        min_vals = num_df.min()
        max_vals = num_df.max()
        denom = (max_vals - min_vals).replace(0, 1e-6)
        return ((num_df - min_vals) / denom).astype(np.float32)
    raise ValueError(f"Unsupported num_scaler: {scaler}")


def prepare_features(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
    numeric_cols: List[str],
    cat_cols: List[str],
) -> Tuple[np.ndarray, List[int], np.ndarray, List[str], List[str]]:
    working_df = df.copy()
    flag_cols: List[str] = []
    if cfg.add_availability_flags:
        working_df, flag_cols = add_availability_flags(working_df)

    numeric_df = impute_numeric(working_df, numeric_cols, cfg.impute_strategy)
    numeric_df = scale_numeric(numeric_df, cfg.num_scaler)
    numeric_feature_names: List[str] = list(numeric_df.columns)

    if flag_cols:
        flag_df = working_df[flag_cols].fillna(0).astype(np.float32)
        numeric_df = pd.concat([numeric_df, flag_df], axis=1)
        numeric_feature_names.extend(flag_cols)

    num_matrix = numeric_df.to_numpy(dtype=np.float32)

    if cfg.cat_encoder == "embedding":
        cat_matrix, cat_cardinalities = factorize_categoricals(
            working_df, cat_cols)
    elif cfg.cat_encoder == "onehot":
        onehot_matrix, onehot_cols = onehot_categoricals(working_df, cat_cols)
        cat_matrix = np.empty((len(working_df), 0), dtype=np.int64)
        cat_cardinalities = []
        if onehot_cols:
            num_matrix = np.concatenate([num_matrix, onehot_matrix], axis=1)
            numeric_feature_names.extend(onehot_cols)
    else:
        raise ValueError(f"Unsupported cat_encoder: {cfg.cat_encoder}")

    return cat_matrix, cat_cardinalities, num_matrix, numeric_feature_names, flag_cols


class LightpathDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        target_path: Path,
        sample_size: int | None = None,
        preprocess_cfg: PreprocessConfig | None = None,
    ):
        cfg = preprocess_cfg or PreprocessConfig()
        df = pd.read_csv(data_path)
        y_series = pd.read_csv(target_path)["class"].astype(int)
        if sample_size is not None:
            df = df.sample(n=sample_size, random_state=42)
            y_series = y_series.loc[df.index]
            df = df.reset_index(drop=True)
            y_series = y_series.reset_index(drop=True)

        (
            self.cat_matrix,
            self.cat_cardinalities,
            self.num_matrix,
            self.num_feature_names,
            self.flag_cols,
        ) = prepare_features(df, cfg, NUMERIC_COLS, CAT_COLS)

        self.targets = y_series.to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.cat_matrix[idx]),
            torch.from_numpy(self.num_matrix[idx]),
            torch.tensor(self.targets[idx]),
        )


# ----------------------------
# Model definitions
# ----------------------------
class CatTransformerConcatMLP(nn.Module):
    """Variant A: cat embeddings -> transformer -> flatten -> concat numeric -> MLP."""

    def __init__(
        self,
        cat_cardinalities: List[int],
        numeric_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        mlp_hidden: List[int] | None = None,
        flatten: bool = True,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in cat_cardinalities]
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
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.flatten = flatten
        transformer_out_dim = len(cat_cardinalities) * \
            d_model if flatten else d_model
        self.num_norm = nn.LayerNorm(numeric_dim)
        self.mlp = build_mlp(transformer_out_dim +
                             numeric_dim, mlp_hidden or [128], dropout)

    def forward(self, cat_ids: torch.Tensor, num_feats: torch.Tensor) -> torch.Tensor:
        # cat_ids: [B, C], num_feats: [B, N]
        cat_tokens = torch.stack(
            [emb(cat_ids[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )  # [B, C, d_model]
        ctx = self.transformer(cat_tokens)  # [B, C, d_model]
        if self.flatten:
            ctx = ctx.flatten(start_dim=1)
        else:
            ctx = ctx.mean(dim=1)
        num_feats = self.num_norm(num_feats)
        x = torch.cat([ctx, num_feats], dim=1)
        return self.mlp(x).squeeze(-1)


class CatMLP(nn.Module):
    """Variant C: cat embeddings -> concat numeric -> MLP (no transformer)."""

    def __init__(
        self,
        cat_cardinalities: List[int],
        numeric_dim: int,
        d_model: int = 64,
        hidden_sizes: List[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in cat_cardinalities]
        )
        self.num_norm = nn.LayerNorm(numeric_dim)
        input_dim = len(cat_cardinalities) * d_model + numeric_dim
        self.mlp = build_mlp(input_dim, hidden_sizes or [128], dropout)

    def forward(self, cat_ids: torch.Tensor, num_feats: torch.Tensor) -> torch.Tensor:
        cat_tokens = torch.stack(
            [emb(cat_ids[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )  # [B, C, d_model]
        cat_flat = cat_tokens.flatten(start_dim=1)
        num_feats = self.num_norm(num_feats)
        x = torch.cat([cat_flat, num_feats], dim=1)
        return self.mlp(x).squeeze(-1)


class DenseMLP(nn.Module):
    """Variant D: all features already dense (numeric + flags + optional one-hot)."""

    def __init__(
        self,
        numeric_dim: int,
        hidden_sizes: List[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_norm = nn.LayerNorm(numeric_dim)
        self.mlp = build_mlp(numeric_dim, hidden_sizes or [128], dropout)

    def forward(self, cat_ids: torch.Tensor, num_feats: torch.Tensor) -> torch.Tensor:
        num_feats = self.num_norm(num_feats)
        return self.mlp(num_feats).squeeze(-1)


class TokenTransformerWithCls(nn.Module):
    """Variant B: embed cat + numeric tokens + CLS -> transformer -> CLS -> MLP."""

    def __init__(
        self,
        cat_cardinalities: List[int],
        numeric_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        mlp_hidden: List[int] | None = None,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in cat_cardinalities]
        )
        # Numeric token embeddings: one linear per numeric feature
        self.num_linears = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(numeric_dim)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(2 * d_model, 64),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.mlp = build_mlp(d_model, mlp_hidden or [128], dropout)

    def forward(self, cat_ids: torch.Tensor, num_feats: torch.Tensor) -> torch.Tensor:
        cat_tokens = torch.stack(
            [emb(cat_ids[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )  # [B, C, d_model]
        num_tokens = torch.stack(
            [lin(num_feats[:, i: i + 1]) for i, lin in enumerate(self.num_linears)], dim=1
        )  # [B, N, d_model]
        B = cat_tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # [B,1,d_model]
        seq = torch.cat([cls, cat_tokens, num_tokens], dim=1)
        ctx = self.transformer(seq)
        cls_out = ctx[:, 0, :]
        return self.mlp(cls_out).squeeze(-1)


# ----------------------------
# Training utilities
# ----------------------------
def build_mlp(input_dim: int, hidden_sizes: List[int], dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = input_dim
    if not hidden_sizes:
        hidden_sizes = []
    for h in hidden_sizes:
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        last_dim = h
    layers.append(nn.Linear(last_dim, 1))
    return nn.Sequential(*layers)


@dataclass
class TrainConfig:
    model_type: str = "cat_tf"  # "cat_tf", "cls_tf", "cat_mlp", or "dense_mlp"
    batch_size: int = 256
    epochs: int = 2
    lr: float = 1e-3
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    mlp_hidden: List[int] = field(default_factory=lambda: [128])
    num_scaler: str = "none"  # "none", "standard", "minmax"
    cat_encoder: str = "embedding"  # "embedding", "onehot"
    impute_strategy: str = "median"  # "median", "zero"
    add_availability_flags: bool = True
    sample_size: int | None = 5000  # subsample for quick runs; None = full dataset
    data_path: Path = Path("cleaned_lightpath_dataset.csv")
    target_path: Path = Path("cleaned_lightpath_target.csv")


def build_model(cfg: TrainConfig, cat_cardinalities: List[int], numeric_dim: int) -> nn.Module:
    if cfg.cat_encoder == "onehot" and cfg.model_type in {"cat_tf", "cls_tf", "cat_mlp"}:
        raise ValueError(
            "cat_encoder=onehot is only supported with model_type='dense_mlp'.")
    if cfg.model_type in {"cat_tf", "cls_tf", "cat_mlp"} and not cat_cardinalities:
        raise ValueError(
            "Chosen model requires categorical embeddings but none were produced (cat_encoder='embedding' expected).")

    if cfg.model_type == "cat_tf":
        return CatTransformerConcatMLP(
            cat_cardinalities=cat_cardinalities,
            numeric_dim=numeric_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            mlp_hidden=cfg.mlp_hidden,
            flatten=True,
        )
    if cfg.model_type == "cat_mlp":
        return CatMLP(
            cat_cardinalities=cat_cardinalities,
            numeric_dim=numeric_dim,
            d_model=cfg.d_model,
            hidden_sizes=cfg.mlp_hidden,
            dropout=cfg.dropout,
        )
    if cfg.model_type == "cls_tf":
        return TokenTransformerWithCls(
            cat_cardinalities=cat_cardinalities,
            numeric_dim=numeric_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            mlp_hidden=cfg.mlp_hidden,
        )
    if cfg.model_type == "dense_mlp":
        return DenseMLP(
            numeric_dim=numeric_dim,
            hidden_sizes=cfg.mlp_hidden,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown model_type: {cfg.model_type}")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for cat_ids, num_feats, targets in loader:
        cat_ids = cat_ids.to(device)
        num_feats = num_feats.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(cat_ids, num_feats)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for cat_ids, num_feats, targets in loader:
            cat_ids = cat_ids.to(device)
            num_feats = num_feats.to(device)
            targets = targets.to(device)
            logits = model(cat_ids, num_feats)
            loss = loss_fn(logits, targets)
            total_loss += loss.item() * len(targets)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == targets.long()).sum().item()
            total += len(targets)
    return total_loss / len(loader.dataset), correct / total if total > 0 else 0.0


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Lightpath NN trainer")
    parser.add_argument(
        "--model-type",
        choices=["cat_tf", "cls_tf", "cat_mlp", "dense_mlp"],
        default="cat_tf",
        help="Choose model architecture. Use dense_mlp when --cat-encoder onehot.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--mlp-hidden",
        type=str,
        default="128",
        help="Comma-separated hidden layer sizes, e.g., '256,128' (applies to all variants).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of samples to draw for quick runs; use -1 for full dataset.",
    )
    parser.add_argument(
        "--num-scaler",
        choices=["none", "standard", "minmax"],
        default="none",
        help="Numeric scaling applied before the model (LayerNorm still applied in-model).",
    )
    parser.add_argument(
        "--cat-encoder",
        choices=["embedding", "onehot"],
        default="embedding",
        help="Embedding keeps categorical tokens for transformer/embeddings; onehot moves them into the dense feature block.",
    )
    parser.add_argument(
        "--impute-strategy",
        choices=["median", "zero"],
        default="median",
        help="Missing numeric imputation strategy (before scaling).",
    )
    parser.add_argument(
        "--no-avail-flags",
        action="store_true",
        help="Disable availability flags/zero-masking for paired columns that use 0 as sentinel.",
    )
    args = parser.parse_args()

    def parse_hidden(s: str) -> List[int]:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        return [int(p) for p in parts] if parts else [128]

    return TrainConfig(
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mlp_hidden=parse_hidden(args.mlp_hidden),
        sample_size=None if args.sample_size == -1 else args.sample_size,
        num_scaler=args.num_scaler,
        cat_encoder=args.cat_encoder,
        impute_strategy=args.impute_strategy,
        add_availability_flags=not args.no_avail_flags,
    )


def main():
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess_cfg = PreprocessConfig(
        num_scaler=cfg.num_scaler,
        cat_encoder=cfg.cat_encoder,
        impute_strategy=cfg.impute_strategy,
        add_availability_flags=cfg.add_availability_flags,
    )

    dataset = LightpathDataset(
        cfg.data_path,
        cfg.target_path,
        sample_size=cfg.sample_size,
        preprocess_cfg=preprocess_cfg,
    )
    # simple split: 80/20
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = build_model(cfg, dataset.cat_cardinalities,
                        dataset.num_matrix.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch+1}/{cfg.epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
        )

    print("Done. Model ready for hyperparameter tuning loops (Optuna/wandb sweeps).")


if __name__ == "__main__":
    main()
