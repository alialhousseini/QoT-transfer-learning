#!/usr/bin/env python3
"""
Feature transformation analysis + recommendations for cleaned Lightpath QoT CSVs.

Inputs (your cleaned CSVs):
  - cleaned_lightpath_dataset.csv  (features only, 32 columns)
  - cleaned_lightpath_target.csv   (target, typically just 'class')

Outputs:
  - <outdir>/transform_recommendations.csv   (per-feature decision + stats)
  - <outdir>/transform_report.md            (human-readable justification)
  - <outdir>/scaling_benchmark.csv          (optional; only if sklearn installed and --evaluate)

Design goals:
  - Be data-driven: range/percentiles/skew/outlier rates/uniques.
  - Be model-aware: scaling vs encoding depends on model family.
  - Be pragmatic: recommend a default, plus an alternative.

Notes:
  - Scaling (Standard vs MinMax) does not change skewness; for skewed variables we recommend
    a monotonic transform (e.g., log) before scaling.
  - For discrete grid-like variables (e.g., freq) and low-cardinality integer-like columns,
    encoding as categorical is often better than treating them as continuous.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recommend per-feature transformations for cleaned QoT CSVs")
    p.add_argument("--data", default="cleaned_lightpath_dataset.csv", help="Feature CSV path")
    p.add_argument("--target", default="cleaned_lightpath_target.csv", help="Target CSV path (expects 'class')")
    p.add_argument("--outdir", default="transform_analysis", help="Output directory")
    p.add_argument("--seed", type=int, default=7, help="Random seed")
    p.add_argument(
        "--sample-size",
        type=int,
        default=200_000,
        help="Rows to sample for distributional stats (skew/outliers/uniques)",
    )
    p.add_argument("--chunksize", type=int, default=120_000, help="CSV chunksize for streaming")
    p.add_argument(
        "--max-unique-categorical",
        type=int,
        default=30,
        help="If integer-like and nunique <= this threshold, treat as categorical",
    )
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Benchmark StandardScaler vs MinMaxScaler pipelines (requires scikit-learn)",
    )
    p.add_argument(
        "--eval-size",
        type=int,
        default=150_000,
        help="Rows used for benchmarking if --evaluate (sampled from the dataset)",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_rows_fast(path: str) -> int:
    # Fast newline count; subtract 1 for header.
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            n += block.count(b"\n")
    return max(0, n - 1)


def select_sample_mask(n: int, k: int, seed: int) -> np.ndarray:
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    idx.sort()
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


@dataclass
class OnlineColStats:
    count: int = 0
    missing: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        v = values.astype(float, copy=False)
        # missing values handled before passing in (caller uses dropna)
        batch_count = int(v.size)
        batch_mean = float(v.mean())
        batch_m2 = float(((v - batch_mean) ** 2).sum())
        batch_min = float(v.min())
        batch_max = float(v.max())

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.min = batch_min
            self.max = batch_max
            return

        # parallel update (Chan et al.)
        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / new_count
        self.m2 = self.m2 + batch_m2 + (delta**2) * self.count * batch_count / new_count
        self.count = new_count
        self.min = min(self.min, batch_min)
        self.max = max(self.max, batch_max)

    @property
    def std(self) -> float:
        if self.count <= 1:
            return float("nan")
        return float(math.sqrt(self.m2 / (self.count - 1)))


def df_to_markdown(df: pd.DataFrame, index: bool = False, max_rows: Optional[int] = None) -> str:
    if df is None or df.empty:
        return "_(empty)_"
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    if index:
        view = view.reset_index()

    def fmt(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return ""
            return f"{v:.6g}"
        if isinstance(v, (np.floating, np.integer)):
            return str(v.item())
        s = str(v)
        s = s.replace("\n", " ").replace("|", "\\|")
        return s

    cols = [str(c) for c in view.columns.tolist()]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def is_integer_like(series: pd.Series, atol: float = 1e-8) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    v = s.values.astype(float, copy=False)
    return float(np.isclose(v, np.round(v), atol=atol, rtol=0).mean())


def outlier_rate_iqr(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return 0.0
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return float(((s < lo) | (s > hi)).mean())


def safe_skew(series: pd.Series) -> float:
    s = series.dropna()
    if s.size < 3:
        return float("nan")
    return float(s.skew())


def safe_kurt(series: pd.Series) -> float:
    s = series.dropna()
    if s.size < 4:
        return float("nan")
    return float(s.kurt())


def percentiles(series: pd.Series, qs: Sequence[float]) -> Dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    qv = s.quantile(list(qs)).to_dict()
    return {f"p{int(q*100)}": float(v) for q, v in qv.items()}


def feature_family(name: str) -> str:
    if name == "freq":
        return "grid_frequency"
    if "ber" in name:
        return "ber"
    if "mod_order" in name:
        return "mod_order"
    if "linerate" in name:
        return "line_rate"
    if "occ" in name:
        return "occupancy"
    if name in {"num_links", "num_spans"}:
        return "counts"
    if name.endswith("_degree"):
        return "degree"
    if name.endswith("_len") or name.endswith("_link_len") or name == "path_len":
        return "length"
    return "other_numeric"


def choose_monotonic_transform(name: str, fam: str, s: pd.Series) -> str:
    """
    Decide whether a monotonic pre-transform (e.g., log) is helpful BEFORE scaling.
    Returns a short token: "none", "log1p", "-log10".
    """
    sn = pd.to_numeric(s, errors="coerce").dropna()
    if sn.empty:
        return "none"

    # BER: use -log10 (common in optics; converts tiny BERs into a more spread-out "margin-like" scale).
    if fam == "ber":
        return "-log10"

    # These are discrete by nature; applying log is misleading.
    if fam in {"grid_frequency", "mod_order", "line_rate"}:
        return "none"

    minv = float(sn.min())
    if minv < 0:
        return "none"

    maxv = float(sn.max())
    # Only consider log-like transforms when the dynamic range is large enough to matter.
    if minv > 0 and (maxv / minv) < 10.0:
        return "none"

    raw_sk = safe_skew(sn)
    if not math.isfinite(raw_sk):
        return "none"

    # For positively-skewed, non-negative variables, test log1p if skew is substantial.
    if raw_sk > 1.0:
        log_sk = safe_skew(np.log1p(sn))
        if math.isfinite(log_sk) and abs(log_sk) < 0.85 * abs(raw_sk):
            return "log1p"

    return "none"


def recommend_transform(
    name: str,
    nunique: int,
    intlike: float,
    minv: float,
    maxv: float,
    skew: float,
    outlier_rate: float,
    zero_rate: float,
    max_unique_categorical: int,
    pre_transform: str,
) -> Tuple[str, str, str]:
    """
    Returns (recommended, alternative, rationale).
    recommended/alternative are short strings describing transforms.
    """
    fam = feature_family(name)

    if fam == "grid_frequency":
        rationale = (
            f"`{name}` has low-ish cardinality on a fixed grid (nunique≈{nunique}); "
            "treating it as continuous implies a linear ordering that usually does not reflect channel identity. "
            "Prefer categorical encoding."
        )
        return "one-hot encode (categorical)", "standardize (if forcing numeric)", rationale

    # BERs are typically very small, positive, and highly skewed.
    if fam == "ber":
        eps = 1e-12
        pre = f"-log10(x+{eps:g})" if pre_transform == "-log10" else f"log10(x+{eps:g})"
        rationale = (
            f"`{name}` is BER-like (min={minv:.3g}, max={maxv:.3g}, zero%={100*zero_rate:.2f}, skew≈{skew:.2f}). "
            f"BERs span orders of magnitude; applying `{pre}` spreads small values and makes this feature numerically well-scaled. "
            "Then standardize for models sensitive to feature scale (linear models, kNN, neural nets)."
        )
        return f"{pre} then standardize", f"{pre} then min-max scale", rationale

    # Modulation orders are discrete and often better expressed as bits per symbol.
    if fam == "mod_order":
        rationale = (
            f"`{name}` is modulation order-like (nunique≈{nunique}, integer-like={intlike:.3f}). "
            "These values are discrete; either one-hot encode or map to an ordinal domain feature `log2(mod_order)` "
            "when >0 (bits per symbol)."
        )
        if nunique <= max_unique_categorical and intlike > 0.98:
            return "one-hot encode (including 0 if present)", "map to bits_per_symbol=log2(x) then standardize", rationale
        return "map to bits_per_symbol=log2(x) then standardize", "one-hot encode", rationale

    # Line rates are discrete sets; numeric scaling can be OK but can introduce artificial linearity.
    if fam == "line_rate":
        rationale = (
            f"`{name}` is a (discrete) line-rate-like feature (nunique≈{nunique}, integer-like={intlike:.3f}). "
            "Because only a few rates exist, treating it as categorical avoids imposing a linear relationship between rates."
        )
        if nunique <= max_unique_categorical and intlike > 0.98:
            return "one-hot encode", "standardize numeric", rationale
        return "standardize numeric", "min-max scale", rationale

    # Occupancy often behaves like counts with right skew and potential outliers.
    if fam == "occupancy":
        if pre_transform == "log1p":
            rationale = (
                f"`{name}` looks like occupancy/count data (min={minv:.3g}, max={maxv:.3g}, skew≈{skew:.2f}, "
                f"outliers≈{100*outlier_rate:.2f}%). A `log1p` reduces right-skew; then scale."
            )
            return "log1p then standardize", "log1p then min-max scale", rationale
        rationale = (
            f"`{name}` appears reasonably well-behaved (skew≈{skew:.2f}, outliers≈{100*outlier_rate:.2f}%). "
            "Standardization is a good default."
        )
        return "standardize", "min-max scale", rationale

    # Small-count integers: scaling helps gradient/distance based models; otherwise optional.
    if fam in {"counts", "degree"}:
        if nunique <= max_unique_categorical and intlike > 0.98 and nunique <= 10:
            rationale = (
                f"`{name}` is a low-cardinality integer count (nunique≈{nunique}). "
                "You can keep it numeric (scaled) or one-hot encode if you suspect non-linear jumps."
            )
            return "standardize", "one-hot encode", rationale
        if pre_transform == "log1p":
            rationale = (
                f"`{name}` is count-like with right skew (skew≈{skew:.2f}). "
                "A `log1p` can help; then standardize."
            )
            return "log1p then standardize", "log1p then min-max scale", rationale
        rationale = (
            f"`{name}` is count-like; standardization keeps it comparable to other features for gradient-based models."
        )
        return "standardize", "min-max scale", rationale

    # Lengths and other continuous physical quantities: standardization is a robust default.
    if fam == "length":
        if pre_transform == "log1p":
            rationale = (
                f"`{name}` is non-negative and right-skewed (skew≈{skew:.2f}). "
                "A `log1p` makes the scale more linear (diminishing returns for very long paths), then standardize."
            )
            return "log1p then standardize", "standardize", rationale
        rationale = f"`{name}` is continuous length-like; standardization is a solid default across many models."
        return "standardize", "min-max scale", rationale

    # Catch-all numeric features.
    if pre_transform == "log1p":
        rationale = (
            f"`{name}` is highly right-skewed (skew≈{skew:.2f}) and non-negative. "
            "Prefer a log-like transform before scaling."
        )
        return "log1p then standardize", "log1p then min-max scale", rationale
    rationale = f"`{name}` is numeric with moderate skew/outliers; standardization is the default choice."
    return "standardize", "min-max scale", rationale


def stream_sample_and_online_stats(
    data_path: str,
    target_path: str,
    sample_size: int,
    chunksize: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, OnlineColStats], int]:
    total_rows = count_rows_fast(data_path)
    keep_frac = min(1.0, sample_size / max(1, total_rows))
    rng = np.random.default_rng(seed)

    online: Dict[str, OnlineColStats] = {}
    sample_parts: List[pd.DataFrame] = []
    y_parts: List[pd.Series] = []

    data_iter = pd.read_csv(data_path, chunksize=chunksize, low_memory=False)
    tgt_iter = pd.read_csv(target_path, chunksize=chunksize, low_memory=False)

    for chunk_i, (x_chunk, y_chunk) in enumerate(zip(data_iter, tgt_iter)):
        if "class" not in y_chunk.columns:
            raise RuntimeError(f"Target file must contain 'class' column; got {list(y_chunk.columns)}")
        # Update streaming stats (for numeric columns only)
        for col in x_chunk.columns:
            s = pd.to_numeric(x_chunk[col], errors="coerce")
            if col not in online:
                online[col] = OnlineColStats()
            online[col].missing += int(s.isna().sum())
            online[col].update(s.dropna().values)

        # Sample rows from this chunk
        if keep_frac >= 1.0:
            take = x_chunk
            take_y = y_chunk["class"]
        else:
            # independent random for each chunk, then trim later
            frac = keep_frac
            take = x_chunk.sample(frac=frac, random_state=int(rng.integers(0, 2**31 - 1)))
            take_y = y_chunk.loc[take.index, "class"]

        sample_parts.append(take)
        y_parts.append(take_y)

    sample_df = pd.concat(sample_parts, axis=0).reset_index(drop=True)
    y = pd.concat(y_parts, axis=0).reset_index(drop=True)

    if len(sample_df) > sample_size:
        idx = rng.choice(len(sample_df), size=sample_size, replace=False)
        idx.sort()
        sample_df = sample_df.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    return sample_df, y, online, total_rows


def try_run_benchmark(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: List[str],
    numeric_cols: List[str],
    ber_cols: List[str],
    seed: int,
) -> Optional[pd.DataFrame]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler
        from sklearn.linear_model import SGDClassifier
    except Exception:
        return None

    def log10_eps(arr):
        eps = 1e-12
        return np.log10(np.asarray(arr, dtype=float) + eps)

    def make_preprocess(scaler):
        # Apply log10 to BER columns then scale. Non-BER numeric columns only scale.
        ber_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("log10", FunctionTransformer(log10_eps, feature_names_out="one-to-one")),
                ("scale", scaler),
            ]
        )
        num_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", scaler)])
        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers = []
        if ber_cols:
            transformers.append(("ber", ber_pipe, ber_cols))
        if numeric_cols:
            transformers.append(("num", num_pipe, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", cat_pipe, categorical_cols))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(int), test_size=0.2, random_state=seed, stratify=y.astype(int)
    )

    configs = [
        ("standard", StandardScaler(with_mean=False)),
        ("minmax", MinMaxScaler()),
    ]
    rows = []
    for name, scaler in configs:
        preprocess = make_preprocess(scaler)
        # Fast baseline optimizer; good enough to compare scaling choices.
        model = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=40,
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=seed,
        )
        pipe = Pipeline(steps=[("prep", preprocess), ("clf", model)])
        pipe.fit(X_train, y_train)
        scores = pipe.decision_function(X_test)
        # Map scores to a pseudo-probability for thresholding; AUC uses rank so score is fine.
        proba = 1.0 / (1.0 + np.exp(-scores))
        pred = (scores >= 0).astype(int)
        rows.append(
            {
                "pipeline": name,
                "roc_auc": float(roc_auc_score(y_test, proba)),
                "accuracy": float(accuracy_score(y_test, pred)),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "n_features_out": int(pipe.named_steps["prep"].transform(X_test).shape[1]),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    if not Path(args.data).exists():
        raise SystemExit(f"Missing data file: {args.data}")
    if not Path(args.target).exists():
        raise SystemExit(f"Missing target file: {args.target}")

    sample_df, y, online, total_rows = stream_sample_and_online_stats(
        data_path=args.data,
        target_path=args.target,
        sample_size=int(args.sample_size),
        chunksize=int(args.chunksize),
        seed=int(args.seed),
    )

    # Compute sample-based stats
    records: List[Dict[str, object]] = []
    for col in sample_df.columns:
        s = pd.to_numeric(sample_df[col], errors="coerce")
        nunique = int(s.dropna().nunique())
        intlike = is_integer_like(s)
        zero_rate = float((s.fillna(0) == 0).mean())
        skew = safe_skew(s)
        kurt = safe_kurt(s)
        out_rate = outlier_rate_iqr(s)
        q = percentiles(s, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

        o = online[col]
        minv = float(o.min) if math.isfinite(o.min) else float("nan")
        maxv = float(o.max) if math.isfinite(o.max) else float("nan")

        pre = choose_monotonic_transform(col, feature_family(col), s)
        rec, alt, rationale = recommend_transform(
            name=col,
            nunique=nunique,
            intlike=intlike,
            minv=minv,
            maxv=maxv,
            skew=skew,
            outlier_rate=out_rate,
            zero_rate=zero_rate,
            max_unique_categorical=int(args.max_unique_categorical),
            pre_transform=pre,
        )

        records.append(
            {
                "feature": col,
                "family": feature_family(col),
                "nunique_sample": nunique,
                "integer_like_frac": float(intlike),
                "min_full": minv,
                "max_full": maxv,
                "mean_full": float(o.mean) if o.count else float("nan"),
                "std_full": float(o.std) if o.count else float("nan"),
                "missing_full": int(o.missing),
                "zero_rate_sample": float(zero_rate),
                "skew_sample": float(skew),
                "kurtosis_sample": float(kurt),
                "outlier_rate_iqr_sample": float(out_rate),
                "chosen_pre_transform": pre,
                **q,
                "recommended_transform": rec,
                "alternative_transform": alt,
                "rationale": rationale,
            }
        )

    rec_df = pd.DataFrame(records).sort_values(["family", "feature"])
    rec_path = outdir / "transform_recommendations.csv"
    rec_df.to_csv(rec_path, index=False)

    # Identify columns for optional benchmark
    ber_cols = [c for c in sample_df.columns if "ber" in c]
    # heuristic categoricals: grid-like freq + low-cardinality integer-like columns
    categorical_cols = ["freq"] if "freq" in sample_df.columns else []
    for col in sample_df.columns:
        if col == "freq":
            continue
        s = pd.to_numeric(sample_df[col], errors="coerce")
        nunique = int(s.dropna().nunique())
        intlike = is_integer_like(s)
        if intlike > 0.98 and nunique <= int(args.max_unique_categorical):
            categorical_cols.append(col)
    categorical_cols = sorted(set(categorical_cols))

    numeric_cols = [c for c in sample_df.columns if c not in categorical_cols and c not in ber_cols]

    # Build report
    report_lines: List[str] = []
    report_lines.append("# Feature Transformation Recommendations\n")
    report_lines.append(f"**Dataset:** `{args.data}`\n")
    report_lines.append(f"**Rows (full):** {total_rows:,}\n")
    report_lines.append(f"**Rows (sampled for distribution stats):** {len(sample_df):,}\n")
    report_lines.append("## How to read this\n")
    report_lines.append(
        "- `recommended_transform` is the default suggestion for gradient/distance-based models.\n"
        "- Tree-based models generally do not require scaling; the main exception here is treating discrete features "
        "(`freq`, modulation orders, line rates) as categorical rather than continuous.\n"
        "- For BER-like columns, scaling alone is not enough; a `log10` transform is typically the important step.\n"
    )

    report_lines.append("## Recommended transforms (per feature)\n")
    view_cols = [
        "feature",
        "family",
        "nunique_sample",
        "min_full",
        "max_full",
        "skew_sample",
        "outlier_rate_iqr_sample",
        "recommended_transform",
    ]
    report_lines.append(df_to_markdown(rec_df[view_cols], index=False, max_rows=None) + "\n")

    report_lines.append("## Detailed per-feature justification\n")
    for fam in rec_df["family"].unique().tolist():
        report_lines.append(f"### {fam}\n")
        sub = rec_df[rec_df["family"] == fam].copy()
        for _, r in sub.iterrows():
            report_lines.append(f"**{r['feature']}**\n")
            report_lines.append(
                f"- Range (full): {r['min_full']:.6g} .. {r['max_full']:.6g}\n"
                f"- Skew (sample): {r['skew_sample']:.3f}, outliers(IQR)≈{100*float(r['outlier_rate_iqr_sample']):.2f}%\n"
                f"- Unique (sample): {int(r['nunique_sample'])}, integer-like: {float(r['integer_like_frac']):.3f}\n"
                f"- Recommended: {r['recommended_transform']}\n"
                f"- Alternative: {r['alternative_transform']}\n"
                f"- Why: {r['rationale']}\n"
            )

    # Optional benchmark: sample a smaller frame for model eval
    bench_df = None
    if args.evaluate:
        eval_n = min(int(args.eval_size), len(sample_df))
        if eval_n >= 5_000:
            rng = np.random.default_rng(int(args.seed) + 1)
            idx = rng.choice(len(sample_df), size=eval_n, replace=False)
            idx.sort()
            X_eval = sample_df.iloc[idx].reset_index(drop=True)
            y_eval = y.iloc[idx].reset_index(drop=True)
            bench_df = try_run_benchmark(
                X=X_eval,
                y=y_eval,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                ber_cols=ber_cols,
                seed=int(args.seed),
            )
        if bench_df is None:
            report_lines.append("## Scaling benchmark\n")
            report_lines.append(
                "_Benchmark skipped: scikit-learn not installed (or unavailable), or insufficient sample size._\n"
            )
        else:
            bench_path = outdir / "scaling_benchmark.csv"
            bench_df.to_csv(bench_path, index=False)
            report_lines.append("## Scaling benchmark (logistic regression, sampled)\n")
            report_lines.append(df_to_markdown(bench_df, index=False, max_rows=None) + "\n")
            report_lines.append(
                "Interpretation:\n"
                "- If `standard` beats `minmax`, standardization is the better default for your chosen model family.\n"
                "- If results are close, prefer standardization (more common for linear models) unless your downstream model "
                "specifically benefits from [0,1] ranges (some neural nets / distance-based methods).\n"
            )

    report_path = outdir / "transform_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[OK] Wrote: {rec_path}")
    print(f"[OK] Wrote: {report_path}")
    if bench_df is not None:
        print(f"[OK] Wrote: {outdir / 'scaling_benchmark.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
