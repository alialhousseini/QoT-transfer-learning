import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create balanced percentage subsets and their disjoint complements."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="datasets",
        help="Directory containing data1/data2 and target1/target2 CSVs.",
    )
    parser.add_argument(
        "--percs",
        default="1,2,3,4,5",
        help="Comma-separated list of percentage values to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling new percentage subsets.",
    )
    return parser.parse_args()


def _load_base(data_dir: Path, data_name: str) -> tuple[pd.DataFrame, pd.Series]:
    data_path = data_dir / f"{data_name}.csv"
    target_path = data_dir / f"target{data_name[-1]}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target: {target_path}")
    data = pd.read_csv(data_path)
    target = pd.read_csv(target_path)["class"].astype(int)
    if len(data) != len(target):
        raise ValueError(
            f"Row mismatch for {data_name}: {len(data)} data vs {len(target)} target"
        )
    return data, target


def _locate_subset_indices(
    base_data: pd.DataFrame,
    base_target: pd.Series,
    subset_data: pd.DataFrame,
    subset_target: pd.Series,
) -> np.ndarray:
    key_cols = list(base_data.columns) + ["class"]
    float_cols = base_data.select_dtypes(include=["float"]).columns

    base = base_data.copy()
    if len(float_cols) > 0:
        base[float_cols] = base[float_cols].round(6)
    base["class"] = base_target.astype(int).to_numpy()
    base["_occ"] = base.groupby(key_cols).cumcount()

    subset = subset_data.copy()
    if len(float_cols) > 0:
        subset[float_cols] = subset[float_cols].round(6)
    subset["class"] = subset_target.astype(int).to_numpy()
    subset["_occ"] = subset.groupby(key_cols).cumcount()

    merged = subset.merge(
        base.reset_index(), on=key_cols + ["_occ"], how="left"
    )
    if merged["index"].isna().any():
        missing = merged["index"].isna().sum()
        raise ValueError(
            f"Failed to match {missing} subset rows to base dataset."
        )
    return merged["index"].to_numpy()


def _sample_balanced_indices(
    target: pd.Series, sample_size: int, rng: np.random.Generator
) -> np.ndarray:
    if sample_size % 2 != 0:
        raise ValueError("Sample size must be even for balanced sampling.")
    half = sample_size // 2
    idx0 = target[target == 0].index.to_numpy()
    idx1 = target[target == 1].index.to_numpy()
    if len(idx0) < half or len(idx1) < half:
        raise ValueError(
            f"Not enough samples for balanced split: "
            f"class0={len(idx0)}, class1={len(idx1)}, needed={half}"
        )
    pick0 = rng.choice(idx0, size=half, replace=False)
    pick1 = rng.choice(idx1, size=half, replace=False)
    picks = np.concatenate([pick0, pick1])
    rng.shuffle(picks)
    return picks


def _write_dataset(path: Path, data: pd.DataFrame, target: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    target_path = path.with_name(f"{path.stem}_target.csv")
    target.to_frame("class").to_csv(target_path, index=False)


def _build_perc_name(data_name: str, perc: int) -> str:
    return f"{data_name}_{perc}perc"


def _build_minus_name(data_name: str, perc: int) -> str:
    return f"{data_name}_minus_{perc}perc"


def _resolve_existing_subset(
    data_dir: Path,
    data_name: str,
    perc: int,
) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    subset_path = data_dir / f"{_build_perc_name(data_name, perc)}.csv"
    target_path = data_dir / f"{_build_perc_name(data_name, perc)}_target.csv"
    if subset_path.exists() and target_path.exists():
        subset = pd.read_csv(subset_path)
        target = pd.read_csv(target_path)["class"].astype(int)
        if len(subset) != len(target):
            raise ValueError(
                f"Row mismatch for existing {subset_path}: "
                f"{len(subset)} vs {len(target)}"
            )
        return subset, target
    return None


def _generate_for_dataset(
    data_dir: Path, data_name: str, percs: list[int], seed: int
) -> None:
    data, target = _load_base(data_dir, data_name)
    total_rows = len(data)

    for perc in percs:
        sample_size = (total_rows * perc) // 100
        if sample_size == 0:
            raise ValueError(f"{data_name} {perc}% yields zero samples.")

        existing = _resolve_existing_subset(data_dir, data_name, perc)
        if existing is not None:
            subset_data, subset_target = existing
            subset_idx = _locate_subset_indices(
                data, target, subset_data, subset_target
            )
        else:
            rng = np.random.default_rng(seed + perc)
            subset_idx = _sample_balanced_indices(target, sample_size, rng)
            subset_data = data.iloc[subset_idx].reset_index(drop=True)
            subset_target = target.iloc[subset_idx].reset_index(drop=True)
            subset_path = data_dir / f"{_build_perc_name(data_name, perc)}.csv"
            _write_dataset(subset_path, subset_data, subset_target)

        mask = np.ones(total_rows, dtype=bool)
        mask[subset_idx] = False
        minus_data = data.loc[mask].reset_index(drop=True)
        minus_target = target.loc[mask].reset_index(drop=True)
        minus_path = data_dir / f"{_build_minus_name(data_name, perc)}.csv"
        _write_dataset(minus_path, minus_data, minus_target)


def main() -> int:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    percs = [int(p.strip()) for p in args.percs.split(",") if p.strip()]
    if not percs:
        raise ValueError("No percentages provided.")

    for data_name in ("data1", "data2"):
        _generate_for_dataset(data_dir, data_name, percs, args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
