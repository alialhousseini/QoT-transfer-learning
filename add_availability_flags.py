"""
Create a dataset with explicit availability flags for fields where 0 means "not available".

Input : cleaned_lightpath_dataset.csv
Output: cleaned_lightpath_dataset_with_flags.csv (ignored by git via .gitignore)
"""
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path("cleaned_lightpath_dataset.csv")
OUTPUT = Path("cleaned_lightpath_dataset_with_flags.csv")

# Concept groups where 0 encodes "not available"
GROUPS = {
    # For modulation/linerate, max=0 only when min=0. One availability flag per side.
    "mod_left": ["min_mod_order_left", "max_mod_order_left"],
    "mod_right": ["min_mod_order_right", "max_mod_order_right"],
    "linerate_left": ["min_lp_linerate_left", "max_lp_linerate_left"],
    "linerate_right": ["min_lp_linerate_right", "max_lp_linerate_right"],
    # For BER, zeros come in pairs. One availability flag per side.
    "ber_left": ["min_ber_left", "max_ber_left"],
    "ber_right": ["min_ber_right", "max_ber_right"],
}


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing input file: {SOURCE}")

    df = pd.read_csv(SOURCE)

    for group_name, cols in GROUPS.items():
        # availability: any of the grouped columns is non-zero
        avail_col = f"{group_name}_avail"
        df[avail_col] = (df[cols].ne(0).any(axis=1)).astype("int8")
        # replace sentinel zeros with NaN in each column
        for col in cols:
            df.loc[df[col] == 0, col] = np.nan

    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} with shape {df.shape} (original: {SOURCE})")


if __name__ == "__main__":
    main()
