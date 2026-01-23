#!/usr/bin/env python3
"""Watch for misplaced outputs and move them into runs_data2 by array id."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
import shutil

DATA2_PATH = "datasets/data2.csv"


def _parse_idx_from_config_path(path_str: str) -> int | None:
    match = re.search(r"run_(\d+)\.json", path_str)
    if match:
        return int(match.group(1))
    return None


def _parse_idx_from_run_name(run_name: str) -> int | None:
    match = re.search(r"data2_array(\d+)", run_name)
    if match:
        return int(match.group(1))
    return None


def _parse_idx_from_dir(dir_path: Path) -> int | None:
    match = re.search(r"array_(\d+)", dir_path.name)
    if match:
        return int(match.group(1))
    return None


def move_data2_outputs(root: Path, dry_run: bool = False) -> int:
    runs_dir = root / "runs"
    runs_data2 = root / "runs_data2"
    moved = 0

    search_dirs = [runs_dir] + [p for p in runs_dir.glob("array_*") if p.is_dir()]
    for dir_path in search_dirs:
        for cfg_path in dir_path.glob("*_config.json"):
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if cfg.get("data1") != DATA2_PATH:
                continue

            idx = None
            cfg_source = cfg.get("config")
            if isinstance(cfg_source, str):
                idx = _parse_idx_from_config_path(cfg_source)
            if idx is None and isinstance(cfg.get("run_name"), str):
                idx = _parse_idx_from_run_name(cfg["run_name"])
            if idx is None:
                idx = _parse_idx_from_dir(dir_path)
            if idx is None:
                continue

            prefix = cfg_path.name.replace("_config.json", "")
            dest_dir = runs_data2 / f"array_{idx}"
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)

            for file_path in dir_path.glob(prefix + "*"):
                dest_path = dest_dir / file_path.name
                if dry_run:
                    print(f"DRY RUN: would move {file_path} -> {dest_path}")
                else:
                    shutil.move(str(file_path), str(dest_path))
                    moved += 1

    return moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.once:
        moved = move_data2_outputs(root, dry_run=args.dry_run)
        print(f"Moved {moved} files.")
        return

    while True:
        moved = move_data2_outputs(root, dry_run=args.dry_run)
        if moved:
            print(f"Moved {moved} files.")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
