#!/usr/bin/env python3
"""
End-to-end EDA for the Lightpath QoT NetCDF dataset.

This script loads a NetCDF file like `lightpath_dataset.nc` containing:
  - data:   (sample, feature)
  - target: (sample, metric)
and produces a Markdown report + figures under an output directory.

By default, it runs EDA on a random sample of rows to keep runtime/memory sane.
Use `--all` if you explicitly want to materialize the full dataset as Pandas.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _import_xarray():
    import xarray as xr  # type: ignore

    return xr


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA for Lightpath QoT NetCDF datasets")
    p.add_argument(
        "--path",
        default="lightpath_dataset.nc",
        help="Path to dataset (.nc/.netcdf or .csv) (default: lightpath_dataset.nc)",
    )
    p.add_argument(
        "--target-path",
        default=None,
        help="Optional target CSV (when --path is a feature CSV). If omitted, the script tries to infer targets from --path.",
    )
    p.add_argument(
        "--join-mode",
        choices=["index", "key"],
        default="index",
        help="How to align data/target when using separate CSVs (default: index).",
    )
    p.add_argument(
        "--join-key",
        default="conn_id",
        help="Join key when --join-mode=key (default: conn_id).",
    )
    p.add_argument(
        "--csv-sep",
        default=",",
        help="CSV delimiter (default: ',').",
    )
    p.add_argument(
        "--outdir",
        default="eda_output",
        help="Output directory for report/figures (default: eda_output)",
    )
    p.add_argument(
        "--engine",
        default="netcdf4",
        help='xarray engine (default: "netcdf4")',
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=200_000,
        help="Rows to sample for Pandas-based EDA (default: 200000). Ignored with --all.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for sampling (default: 7)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run EDA on all rows (may require large RAM).",
    )
    p.add_argument(
        "--max-unique-for-discrete",
        type=int,
        default=50,
        help="Columns with <= this many unique values are treated as discrete (default: 50).",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Top-K categories to show for discrete columns (default: 20).",
    )
    return p.parse_args()


@dataclass
class DatasetInfo:
    path: str
    total_samples: int
    n_features: int
    n_metrics: int
    feature_names: List[str]
    metric_names: List[str]
    analyzed_rows: int
    analyzed_fraction: float
    pandas_data_mb: float
    pandas_target_mb: float


def safe_percent(n: int, d: int) -> float:
    return 0.0 if d == 0 else (100.0 * n / d)


def format_int(n: int) -> str:
    return f"{n:,}"


def format_float(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.{digits}f}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def select_sample_indices(total: int, sample_size: int, seed: int) -> np.ndarray:
    sample_size = max(1, min(sample_size, total))
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=sample_size, replace=False)
    idx.sort()
    return idx


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort conversion of object/string columns to numeric when possible.
    Keeps non-numeric columns unchanged.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            try:
                out[col] = pd.to_numeric(out[col])
            except (ValueError, TypeError):
                pass
    return out


def load_dataframes(
    path: str,
    engine: str,
    all_rows: bool,
    sample_size: int,
    seed: int,
    csv_sep: str = ",",
    target_path: Optional[str] = None,
    join_mode: str = "index",
    join_key: str = "conn_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, DatasetInfo]:
    suffix = Path(path).suffix.lower()
    if suffix in {".nc", ".netcdf", ".cdf"}:
        xr = _import_xarray()
        ds = xr.open_dataset(path, engine=engine)

        try:
            total_samples = int(ds.sizes.get("sample", 0))
            feature_names = [str(x) for x in ds["feature"].values.tolist()]
            metric_names = [str(x) for x in ds["metric"].values.tolist()]
        except Exception as e:  # pragma: no cover
            ds.close()
            raise RuntimeError(f"Unexpected dataset structure in {path}: {e}") from e

        if total_samples <= 0:
            ds.close()
            raise RuntimeError(f"No 'sample' dimension found or empty dataset in {path}")

        if all_rows:
            ds_sel = ds
            analyzed_rows = total_samples
        else:
            idx = select_sample_indices(total_samples, sample_size, seed)
            ds_sel = ds.isel(sample=idx)
            analyzed_rows = int(idx.size)

        try:
            data_df = ds_sel["data"].to_pandas()
            target_df = ds_sel["target"].to_pandas()
        except MemoryError:
            ds.close()
            raise
        finally:
            ds.close()

        # Make sure column labels are meaningful strings (xarray usually does this already).
        if list(data_df.columns) != feature_names:
            data_df.columns = feature_names
        if list(target_df.columns) != metric_names:
            target_df.columns = metric_names

        data_mb = float(data_df.memory_usage(deep=True).sum() / (1024**2))
        target_mb = float(target_df.memory_usage(deep=True).sum() / (1024**2))

        info = DatasetInfo(
            path=str(path),
            total_samples=total_samples,
            n_features=len(feature_names),
            n_metrics=len(metric_names),
            feature_names=feature_names,
            metric_names=metric_names,
            analyzed_rows=analyzed_rows,
            analyzed_fraction=float(analyzed_rows / total_samples),
            pandas_data_mb=data_mb,
            pandas_target_mb=target_mb,
        )
        return data_df, target_df, info

    if suffix == ".csv":
        data_df, target_df, total_samples = load_csv_dataframes(
            path=path,
            target_path=target_path,
            sep=csv_sep,
            join_mode=join_mode,
            join_key=join_key,
        )
        if not all_rows:
            idx = select_sample_indices(total_samples, sample_size, seed)
            data_df = data_df.iloc[idx].reset_index(drop=True)
            target_df = target_df.iloc[idx].reset_index(drop=True)
            analyzed_rows = int(len(idx))
        else:
            analyzed_rows = total_samples

        # Best-effort numeric conversion for CSV inputs (avoid losing numeric columns due to dtype=object).
        data_df = coerce_numeric_columns(data_df)
        target_df = coerce_numeric_columns(target_df)

        feature_names = [str(c) for c in data_df.columns.tolist()]
        metric_names = [str(c) for c in target_df.columns.tolist()]

        data_mb = float(data_df.memory_usage(deep=True).sum() / (1024**2))
        target_mb = float(target_df.memory_usage(deep=True).sum() / (1024**2))

        info = DatasetInfo(
            path=str(path),
            total_samples=total_samples,
            n_features=len(feature_names),
            n_metrics=len(metric_names),
            feature_names=feature_names,
            metric_names=metric_names,
            analyzed_rows=analyzed_rows,
            analyzed_fraction=float(analyzed_rows / total_samples),
            pandas_data_mb=data_mb,
            pandas_target_mb=target_mb,
        )
        return data_df, target_df, info

    raise RuntimeError(f"Unsupported file type for --path: {path}")


def load_csv_dataframes(
    path: str,
    target_path: Optional[str],
    sep: str,
    join_mode: str,
    join_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    CSV loader supporting:
      1) a single combined CSV containing both features and targets; or
      2) a feature CSV + a target CSV, aligned by row index (default) or by key.

    Returns (data_df, target_df, total_samples).
    """
    known_target_cols = ["class", "osnr", "snr", "ber"]
    data_df = pd.read_csv(path, sep=sep, low_memory=False)

    if target_path is None:
        present = [c for c in known_target_cols if c in data_df.columns]
        if not present:
            raise RuntimeError(
                "CSV input detected but no target columns found in --path. "
                "Provide --target-path (e.g., target.csv) or use a combined CSV containing target columns "
                f"{known_target_cols}."
            )
        target_df = data_df[present].copy()
        data_df = data_df.drop(columns=present)
        return data_df, target_df, int(len(target_df))

    target_df = pd.read_csv(target_path, sep=sep, low_memory=False)
    if join_mode == "index":
        if len(data_df) != len(target_df):
            raise RuntimeError(
                f"CSV join-mode=index requires equal row counts; got data={len(data_df)} target={len(target_df)}"
            )
        return data_df, target_df, int(len(data_df))

    # join_mode == "key"
    if join_key not in data_df.columns:
        raise RuntimeError(f"join-key '{join_key}' not found in data CSV columns")
    if join_key not in target_df.columns:
        raise RuntimeError(f"join-key '{join_key}' not found in target CSV columns")

    merged = data_df.merge(target_df, on=join_key, how="inner", suffixes=("", "_tgt"))
    present = [c for c in known_target_cols if c in merged.columns]
    if not present:
        raise RuntimeError(
            f"After key-join on '{join_key}', no target columns found. Expected one of {known_target_cols}."
        )
    merged_target = merged[present].copy()
    merged_data = merged.drop(columns=present)
    return merged_data, merged_target, int(len(merged_target))


def identify_discrete_columns(
    df: pd.DataFrame, max_unique: int, force: Optional[Sequence[str]] = None
) -> List[str]:
    force = set(force or [])
    discrete: List[str] = []
    for col in df.columns:
        if col in force:
            discrete.append(col)
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        nunq = int(s.nunique(dropna=True))
        if nunq <= max_unique:
            discrete.append(col)
            continue
        # Heuristic: treat integer-like float columns as discrete if unique isn't enormous.
        if pd.api.types.is_numeric_dtype(s):
            sample = s.sample(min(20_000, len(s)), random_state=1) if len(s) > 20_000 else s
            is_intlike = np.isclose(sample.values, np.round(sample.values), rtol=0, atol=1e-8).mean()
            if is_intlike > 0.99 and nunq <= max_unique * 5:
                discrete.append(col)
    return discrete


def markdown_table(rows: List[Tuple[str, str]], headers: Tuple[str, str] = ("Metric", "Value")) -> str:
    out = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | {v} |")
    return "\n".join(out)


def df_to_markdown(df: pd.DataFrame, index: bool = True, max_rows: Optional[int] = 30) -> str:
    """
    Minimal DataFrame -> Markdown converter without requiring the optional `tabulate` dependency.
    Intended for small summary tables in the generated report.
    """
    if df is None or df.empty:
        return "_(empty)_"

    view = df.head(max_rows).copy() if max_rows is not None else df.copy()
    if index:
        view = view.reset_index()

    def fmt(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return ""
            return format_float(v, digits=6)
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


def plot_histograms(
    df: pd.DataFrame,
    cols: Sequence[str],
    outdir: Path,
    title_prefix: str,
    bins: int = 60,
) -> List[str]:
    plt = _import_matplotlib()
    paths: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(s.values, bins=bins, color="#4C78A8", alpha=0.9)
        ax.set_title(f"{title_prefix}: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        fig.tight_layout()
        path = outdir / f"hist_{col}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_corr_heatmap(df: pd.DataFrame, outpath: Path, title: str) -> Optional[str]:
    plt = _import_matplotlib()
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr(numeric_only=True)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    return str(outpath)


def top_correlated_pairs(corr: pd.DataFrame, topn: int = 20, min_abs: float = 0.7) -> pd.DataFrame:
    c = corr.copy()
    np.fill_diagonal(c.values, np.nan)
    # Avoid collisions when index/columns share the same name (common with xarray-derived DataFrames).
    c.index.name = "a"
    c.columns.name = "b"
    pairs = (
        c.stack(future_stack=True)
        .rename("corr")
        .reset_index()
    )
    pairs = pairs.dropna(subset=["corr"])
    pairs["abs_corr"] = pairs["corr"].abs()
    pairs = pairs[pairs["abs_corr"] >= min_abs].sort_values("abs_corr", ascending=False)
    # De-duplicate symmetric pairs by sorting (a,b) keys.
    key = pairs.apply(lambda r: "||".join(sorted([str(r["a"]), str(r["b"])])), axis=1)
    pairs = pairs.loc[~key.duplicated()].head(topn)
    return pairs[["a", "b", "corr"]]


def quality_checks(
    data: pd.DataFrame, target: pd.DataFrame
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    total = len(data)

    def count(mask: pd.Series) -> Dict[str, float]:
        n = int(mask.fillna(True).sum())
        return {"count": float(n), "pct": float(safe_percent(n, total))}

    checks: Dict[str, Dict[str, float]] = {}
    meta: Dict[str, float] = {}
    if {"src_id", "dst_id"}.issubset(data.columns):
        checks["src_equals_dst"] = count(data["src_id"] == data["dst_id"])
    if {"avg_link_len", "min_link_len", "max_link_len"}.issubset(data.columns):
        checks["avg_link_len_outside_minmax"] = count(
            (data["avg_link_len"] < data["min_link_len"]) | (data["avg_link_len"] > data["max_link_len"])
        )
    if {"avg_link_len", "num_links", "path_len"}.issubset(data.columns):
        denom = data["path_len"].replace(0, np.nan).abs()
        ratio = (data["avg_link_len"] * data["num_links"]) / denom
        checks["path_len_inconsistent_num_links_avg_link_len"] = count((ratio < 0.999) | (ratio > 1.001))
    if {"num_spans", "num_links"}.issubset(data.columns):
        checks["num_spans_leq_num_links"] = count(data["num_spans"] <= data["num_links"])
    if {"avg_ber", "min_ber", "max_ber"}.issubset(data.columns):
        checks["avg_ber_outside_minmax"] = count(
            (data["avg_ber"] < data["min_ber"]) | (data["avg_ber"] > data["max_ber"])
        )
    if {"avg_link_occ", "num_links", "sum_link_occ"}.issubset(data.columns):
        denom = data["sum_link_occ"].replace(0, np.nan).abs()
        ratio = (data["avg_link_occ"] * data["num_links"]) / denom
        checks["sum_link_occ_inconsistent_avg_link_occ_num_links"] = count((ratio < 0.999) | (ratio > 1.001))
    if "freq" in data.columns:
        freq = data["freq"].dropna().astype(float)
        if not freq.empty:
            uf = np.unique(freq.values)
            uf.sort()
            if uf.size >= 2:
                diffs = np.diff(uf)
                spacing = float(np.median(diffs))
                anchor = float(uf[0])
                # Small tolerance since inferred grid already matches the dataset.
                tol = max(1e-6, spacing * 1e-3)
                k = np.round((data["freq"] - anchor) / spacing)
                nearest = anchor + k * spacing
                dist = (data["freq"] - nearest).abs()
                checks["freq_off_inferred_grid_tol"] = count(dist > tol)
                meta["freq_grid_inferred_spacing"] = spacing
                meta["freq_grid_inferred_anchor"] = anchor
                meta["freq_grid_inferred_tol"] = tol
    if "class" in target.columns:
        # Note: "bad label" means non-binary or NaN; class values in this dataset are floats.
        s = target["class"]
        checks["class_nan"] = count(s.isna())
        checks["class_non_binary"] = count(~(s.isna() | s.isin([0, 1, 0.0, 1.0])))
    return checks, meta


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    ensure_dir(figdir)

    try:
        data_df, target_df, info = load_dataframes(
            path=args.path,
            engine=args.engine,
            all_rows=bool(args.all),
            sample_size=int(args.sample_size),
            seed=int(args.seed),
            csv_sep=str(args.csv_sep),
            target_path=args.target_path,
            join_mode=str(args.join_mode),
            join_key=str(args.join_key),
        )
    except MemoryError:
        raise SystemExit(
            "MemoryError while converting NetCDF to Pandas. Re-run without `--all` or reduce `--sample-size`."
        )

    # Basic stats
    missing_data = data_df.isna().sum().sort_values(ascending=False)
    missing_target = target_df.isna().sum().sort_values(ascending=False)

    discrete_cols = identify_discrete_columns(
        data_df,
        max_unique=int(args.max_unique_for_discrete),
        force=["mod_order", "lp_linerate", "conn_linerate", "src_id", "dst_id", "src_degree", "dst_degree"],
    )

    numeric_data = data_df.select_dtypes(include=[np.number])
    numeric_target = target_df.select_dtypes(include=[np.number])

    # Descriptive statistics
    percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    data_desc = numeric_data.describe(percentiles=percentiles).T
    target_desc = numeric_target.describe(percentiles=percentiles).T

    # Discrete summaries
    topk = int(args.topk)
    discrete_counts: Dict[str, pd.DataFrame] = {}
    for col in discrete_cols:
        if col not in data_df.columns:
            continue
        vc = data_df[col].value_counts(dropna=False).head(topk)
        discrete_counts[col] = vc.rename_axis(col).reset_index(name="count")

    # Correlations (on analyzed sample/all)
    corr = numeric_data.corr(numeric_only=True)
    corr_plot = plot_corr_heatmap(numeric_data, figdir / "corr_heatmap_features.png", "Feature Correlations")
    top_pairs = top_correlated_pairs(corr, topn=20, min_abs=0.75)

    # Feature-to-target correlations (if target metrics present)
    feature_target_corr = None
    if not numeric_target.empty and not numeric_data.empty:
        joined = pd.concat([numeric_data, numeric_target], axis=1)
        jcorr = joined.corr(numeric_only=True)
        metrics = [c for c in ["ber", "snr", "osnr", "class"] if c in jcorr.columns]
        if metrics:
            feature_target_corr = jcorr.loc[numeric_data.columns, metrics].sort_index()

    # Quality checks
    checks, check_meta = quality_checks(data_df, target_df)

    # Plots
    hist_cols_data = [
        "path_len",
        "avg_link_len",
        "min_link_len",
        "max_link_len",
        "num_links",
        "num_spans",
        "freq",
        "sum_link_occ",
        "avg_link_occ",
        "std_link_occ",
        "avg_ber",
    ]
    hist_cols_target = ["osnr", "snr", "ber"]
    hist_paths = []
    hist_paths += plot_histograms(data_df, hist_cols_data, figdir, "Feature Histogram")
    hist_paths += plot_histograms(target_df, hist_cols_target, figdir, "Target Histogram")

    # Additional quick group summaries
    class_summary_md = ""
    if "class" in target_df.columns:
        cls = target_df["class"]
        vc = cls.value_counts(dropna=False)
        class_summary_md = df_to_markdown(
            vc.rename("count").to_frame().assign(pct=lambda d: d["count"] / len(cls)),
            index=True,
            max_rows=None,
        )

    # Build Markdown report
    report_lines: List[str] = []
    report_lines.append(f"# Lightpath QoT EDA Report\n")
    report_lines.append(f"**Dataset:** `{info.path}`\n")
    report_lines.append("## 1) Overview\n")
    report_lines.append(
        markdown_table(
            [
                ("Total samples", format_int(info.total_samples)),
                ("Features", format_int(info.n_features)),
                ("Target metrics", format_int(info.n_metrics)),
                ("Analyzed rows", format_int(info.analyzed_rows)),
                ("Analyzed fraction", f"{info.analyzed_fraction:.3%}"),
                ("Pandas `data` size", f"{info.pandas_data_mb:.1f} MB"),
                ("Pandas `target` size", f"{info.pandas_target_mb:.1f} MB"),
            ]
        )
        + "\n"
    )
    report_lines.append("**Features (35):**\n\n" + ", ".join(info.feature_names) + "\n")
    report_lines.append("**Targets (4):**\n\n" + ", ".join(info.metric_names) + "\n")

    report_lines.append("## 2) Missing Values (on analyzed rows)\n")
    report_lines.append("### Data\n")
    report_lines.append(df_to_markdown(missing_data.head(20).rename("missing").to_frame(), index=True, max_rows=None) + "\n")
    report_lines.append("### Target\n")
    report_lines.append(
        df_to_markdown(missing_target.head(20).rename("missing").to_frame(), index=True, max_rows=None) + "\n"
    )

    report_lines.append("## 3) Descriptive Statistics (on analyzed rows)\n")
    report_lines.append("### Key feature stats (selected columns)\n")
    selected = [c for c in hist_cols_data if c in data_desc.index]
    if selected:
        report_lines.append(df_to_markdown(data_desc.loc[selected], index=True, max_rows=None) + "\n")
    else:
        report_lines.append("_(No selected feature columns found)_\n")
    report_lines.append("### Target metric stats\n")
    report_lines.append(df_to_markdown(target_desc, index=True, max_rows=None) + "\n")

    report_lines.append("## 4) Discrete / Categorical-Like Columns\n")
    report_lines.append(
        f"Detected discrete-like columns (<= {args.max_unique_for_discrete} unique values on analyzed rows, "
        "plus a few forced IDs/rates):\n\n"
        + ", ".join(discrete_cols)
        + "\n"
    )
    for col, table in discrete_counts.items():
        report_lines.append(f"### `{col}` top-{topk}\n")
        report_lines.append(df_to_markdown(table, index=False, max_rows=None) + "\n")

    report_lines.append("## 5) Consistency / Sanity Checks (on analyzed rows)\n")
    if checks:
        check_rows = [(k, f"{format_int(int(v['count']))} ({v['pct']:.2f}%)") for k, v in checks.items()]
        report_lines.append(markdown_table(check_rows, headers=("Check", "Violations")) + "\n")
        report_lines.append(
            "Interpretation notes:\n"
            "- Some checks are strict equalities/tolerances and may flag harmless floating-point effects.\n"
            "- If `freq_off_inferred_grid_tol` is high, consider snapping `freq` to the nearest inferred grid.\n"
        )
        if any(k.startswith("freq_grid_inferred_") for k in check_meta):
            report_lines.append(
                "Inferred frequency grid:\n"
                f"- spacing: `{format_float(check_meta.get('freq_grid_inferred_spacing', float('nan')), 6)}` THz\n"
                f"- anchor(min): `{format_float(check_meta.get('freq_grid_inferred_anchor', float('nan')), 6)}` THz\n"
                f"- tol: `{format_float(check_meta.get('freq_grid_inferred_tol', float('nan')), 8)}` THz\n"
            )
    else:
        report_lines.append("_(No checks executed)_\n")

    report_lines.append("## 6) Correlations (on analyzed rows)\n")
    if corr_plot:
        report_lines.append(f"- Heatmap: `{Path(corr_plot).as_posix()}`\n")
    if not top_pairs.empty:
        report_lines.append("### Strongest feature-feature correlations (|r| ≥ 0.75)\n")
        report_lines.append(df_to_markdown(top_pairs, index=False, max_rows=None) + "\n")
    else:
        report_lines.append("_(No correlations above threshold)_\n")

    if feature_target_corr is not None:
        report_lines.append("### Feature ↔ target correlations (Pearson)\n")
        report_lines.append(df_to_markdown(feature_target_corr, index=True, max_rows=None) + "\n")

    report_lines.append("## 7) Target Class Balance (on analyzed rows)\n")
    if class_summary_md:
        report_lines.append(class_summary_md + "\n")
        report_lines.append(
            "Interpretation notes:\n"
            "- If `class` is imbalanced, prefer stratified splits and metrics like PR-AUC/F1.\n"
        )
    else:
        report_lines.append("_(No `class` column found in target)_\n")

    report_lines.append("## 8) Figures\n")
    if hist_paths:
        for p in hist_paths:
            report_lines.append(f"- `{Path(p).as_posix()}`")
        report_lines.append("")
    else:
        report_lines.append("_(No figures written)_\n")

    report_lines.append("## 9) Practical Next Steps\n")
    report_lines.append(
        "- Decide whether IDs (`conn_id`, `src_id`, `dst_id`) are allowed features; often they should be dropped.\n"
        "- Consider modeling `freq`, `mod_order`, `lp_linerate`, `conn_linerate` as categorical/discrete.\n"
        "- If sanity-check violations are common, decide whether to filter rows, fix by rule, or add features capturing the inconsistency.\n"
        "- For prediction targets (`osnr`, `snr`, `ber`), check skew and consider log-transforming `ber`.\n"
    )

    report_path = outdir / "eda_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "dataset": asdict(info),
        "discrete_columns": discrete_cols,
        "missing_data_top": missing_data.head(20).to_dict(),
        "missing_target_top": missing_target.head(20).to_dict(),
        "checks": checks,
        "check_meta": check_meta,
        "corr_heatmap": str((figdir / "corr_heatmap_features.png").as_posix()) if corr_plot else None,
    }
    (outdir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Wrote report: {report_path}")
    print(f"[OK] Wrote summary: {outdir / 'eda_summary.json'}")
    print(f"[OK] Figures in: {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
