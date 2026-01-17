import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create non-overlapping random shards from dataset2 samples."
        )
    )
    parser.add_argument(
        "--input",
        default="datasets/cleaned_lightpath_dataset_2.csv",
        help="Path to dataset2 CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/dataset2_shards",
        help="Directory to write shard CSVs.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=3,
        help="Number of shards to create.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--prefix",
        default="dataset2_shard",
        help="Filename prefix for dataset shards.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Optional target CSV aligned with dataset2.",
    )
    parser.add_argument(
        "--target-prefix",
        default="target2_shard",
        help="Filename prefix for target shards (when --target is set).",
    )
    return parser.parse_args()


def _compute_shard_sizes(total_rows: int, num_shards: int) -> list[int]:
    base = total_rows // num_shards
    sizes = [base] * num_shards
    remainder = total_rows % num_shards
    for i in range(remainder):
        sizes[i] += 1
    return sizes


def main() -> int:
    args = _parse_args()
    if args.num_shards < 2:
        raise ValueError("--num-shards must be >= 2.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)
    total_rows = len(data)
    if total_rows == 0:
        raise ValueError("Input dataset has zero rows.")

    target = None
    if args.target:
        target_path = Path(args.target)
        if not target_path.exists():
            raise FileNotFoundError(f"Target not found: {target_path}")
        target = pd.read_csv(target_path)
        if len(target) != total_rows:
            raise ValueError(
                "Target row count does not match input dataset."
            )

    sizes = _compute_shard_sizes(total_rows, args.num_shards)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(total_rows)

    start = 0
    for i, size in enumerate(sizes, start=1):
        end = start + size
        shard_idx = indices[start:end]
        shard = data.iloc[shard_idx].reset_index(drop=True)
        shard_path = output_dir / f"{args.prefix}_{i}.csv"
        shard.to_csv(shard_path, index=False)

        if target is not None:
            target_shard = target.iloc[shard_idx].reset_index(drop=True)
            target_path = output_dir / f"{args.target_prefix}_{i}.csv"
            target_shard.to_csv(target_path, index=False)

        print(f"Shard {i}: {size} rows -> {shard_path}")
        start = end

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
