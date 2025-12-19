"""
Flexible preprocessing script for the lightpath dataset.

It:
1) Builds availability/mask features for columns where 0 encodes "not available".
2) Splits columns into numeric, categorical, and binary flags.
3) Applies configurable transforms (scaler/encoder/masking) and writes a processed
   dataset ready for a neural network.

Toggle behavior by changing the CONFIG block near the top:
- NUM_SCALER:    "standard", "minmax", or "none"
- CAT_ENCODER:   "onehot", "ordinal", or "none"
- ZERO_MISSING_NUMERIC: if True, missing numeric neighbor features are zeroed
  and the availability flag is provided for masking in the model.
Output file name encodes the chosen scaler/encoder/mask mode.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

# ----------------------------
# Configurable switches
# ----------------------------
NUM_SCALER = "standard"  # "standard", "minmax", "none"
CAT_ENCODER = "onehot"   # "onehot", "ordinal", "none"
ZERO_MISSING_NUMERIC = False  # if True, numeric neighbor cols with missing get zeroed (use flag to mask)

SOURCE_DATA = Path("cleaned_lightpath_dataset.csv")
SOURCE_TARGET = Path("cleaned_lightpath_target.csv")  # expects column "class"
OUTPUT_DIR = Path("processed")

# ----------------------------
# Column definitions
# ----------------------------
# Concept groups where 0 encodes "not available"
AVAIL_GROUPS: Dict[str, List[str]] = {
    # For modulation and linerate, max is 0 only when min is 0; one flag per side.
    "mod_left": ["min_mod_order_left", "max_mod_order_left"],
    "mod_right": ["min_mod_order_right", "max_mod_order_right"],
    "linerate_left": ["min_lp_linerate_left", "max_lp_linerate_left"],
    "linerate_right": ["min_lp_linerate_right", "max_lp_linerate_right"],
    # For BER, zeros come in pairs; one flag per side.
    "ber_left": ["min_ber_left", "max_ber_left"],
    "ber_right": ["min_ber_right", "max_ber_right"],
}

# Numeric (continuous) columns to scale
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

# Discrete/categorical columns (treat with encoder)
CAT_COLS: List[str] = [
    "num_links",
    "num_spans",
    "freq",
    "mod_order",
    "lp_linerate",
    "conn_linerate",
    "src_degree",
    "dst_degree",
    # neighbor modulation/linerate (zeros -> missing)
    "min_mod_order_left",
    "max_mod_order_left",
    "min_mod_order_right",
    "max_mod_order_right",
    "min_lp_linerate_left",
    "max_lp_linerate_left",
    "min_lp_linerate_right",
    "max_lp_linerate_right",
]


def make_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "none":
        return "passthrough"
    raise ValueError(f"Unknown NUM_SCALER: {kind}")


def make_cat_encoder(kind: str):
    if kind == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
    if kind == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if kind == "none":
        return "passthrough"
    raise ValueError(f"Unknown CAT_ENCODER: {kind}")


def add_availability_and_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Add availability flags and set sentinel zeros to NaN (or 0 for optional zero-masking)."""
    df = df.copy()

    for group_name, cols in AVAIL_GROUPS.items():
        flag_col = f"{group_name}_avail"
        df[flag_col] = (df[cols].ne(0).any(axis=1)).astype("int8")
        # replace sentinel zeros with NaN
        for col in cols:
            df.loc[df[col] == 0, col] = np.nan
        # optionally zero out numeric neighbor columns (BER sides) after marking missing
        if ZERO_MISSING_NUMERIC and group_name.startswith("ber"):
            for col in cols:
                df[col] = df[col].fillna(0)

    return df


def build_preprocessor(
    numeric_cols: Iterable[str], categorical_cols: Iterable[str], flag_cols: Iterable[str]
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", make_scaler(NUM_SCALER)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_cat_encoder(CAT_ENCODER)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_cols)),
            ("cat", categorical_transformer, list(categorical_cols)),
            ("flags", "passthrough", list(flag_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def main() -> None:
    if not SOURCE_DATA.exists():
        raise FileNotFoundError(f"Missing input: {SOURCE_DATA}")
    if not SOURCE_TARGET.exists():
        raise FileNotFoundError(f"Missing target: {SOURCE_TARGET}")

    df = pd.read_csv(SOURCE_DATA)
    y = pd.read_csv(SOURCE_TARGET)["class"].astype(int).to_numpy()

    df = add_availability_and_mask(df)

    # Identify flag columns added
    flag_cols = [c for c in df.columns if c.endswith("_avail")]

    # Prepare categorical columns: cast to string, mark NaN as "missing"
    df[CAT_COLS] = df[CAT_COLS].apply(lambda s: s.astype("Int64"))
    df[CAT_COLS] = df[CAT_COLS].astype("object").fillna("missing")

    preprocessor = build_preprocessor(NUMERIC_COLS, CAT_COLS, flag_cols)
    X = preprocessor.fit_transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Recover feature names for inspection
    feature_names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        cols_list = list(cols)
        if name == "num":
            feature_names.extend(cols_list)
        elif name == "cat":
            enc = transformer.named_steps.get("encoder")
            if hasattr(enc, "get_feature_names_out"):
                feature_names.extend(enc.get_feature_names_out(cols_list))
            else:
                feature_names.extend(cols_list)
        elif name == "flags":
            feature_names.extend(cols_list)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_base = OUTPUT_DIR / f"cleaned_lightpath_dataset_{NUM_SCALER}_{CAT_ENCODER}_{'zero' if ZERO_MISSING_NUMERIC else 'nan'}"
    out_npz = out_base.with_suffix(".npz")
    np.savez_compressed(out_npz, X=X, y=y, feature_names=np.array(feature_names, dtype=object))

    print(f"Wrote {out_npz} with X shape {X.shape}, y shape {y.shape}")
    print(f"Numeric scaler: {NUM_SCALER} | Categorical encoder: {CAT_ENCODER} | Zero-masked numeric missing: {ZERO_MISSING_NUMERIC}")


if __name__ == "__main__":
    main()
