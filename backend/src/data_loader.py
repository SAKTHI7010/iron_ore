"""Loading, validating and unit-normalising the Windarling CSV.

The raw file is never modified. Values in the file are mass *fractions*
(e.g. Fe = 0.6328); for modelling / display they are converted on a copy to
percent (63.28 %).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


class DataValidationError(ValueError):
    """Raised when the dataset does not satisfy the expected schema."""


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read the raw CSV exactly as stored on disk."""
    path = Path(path) if path else C.DATA_RAW_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place Windarling_Ore_Composition.csv in backend/data/raw/."
        )
    return pd.read_csv(path)


def detect_units(df: pd.DataFrame) -> str:
    """Return 'fraction' if Fe/composition are 0-1 mass fractions, else 'percent'."""
    fe = pd.to_numeric(df[C.TARGET], errors="coerce").dropna()
    if fe.empty:
        raise DataValidationError("Fe column has no numeric values.")
    return "fraction" if fe.max() <= 1.0 else "percent"


def validate_raw(df: pd.DataFrame) -> dict:
    """Validate schema and return an inspection report (shape, dtypes, NaNs, units...)."""
    missing_cols = [c for c in C.REQUIRED_RAW_COLS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")
    numeric_cols = [C.TARGET, *C.CHEM_COLS, *C.COORD_COLS]
    for c in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise DataValidationError(f"Column '{c}' must be numeric.")
    units = detect_units(df)
    scale = 100.0 if units == "fraction" else 1.0
    comp_sum = (df[[C.TARGET, *C.GANGUE_COLS, "LOI"]] * scale).sum(axis=1)
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items()},
        "units_detected": units,
        "lithotype_counts": {str(k): int(v) for k, v in df[C.CAT_COL].value_counts().items()},
        "fe_missing": int(df[C.TARGET].isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "fe_plus_other_components_sum_pct": {
            "min": float(comp_sum.min()),
            "mean": float(comp_sum.mean()),
            "max": float(comp_sum.max()),
        },
    }


def to_percent(df: pd.DataFrame, units: str | None = None) -> pd.DataFrame:
    """Return a copy with Fe + chemistry columns expressed in percent."""
    units = units or detect_units(df)
    out = df.copy()
    if units == "fraction":
        for c in [C.TARGET, *C.CHEM_COLS]:
            if c in out.columns:
                out[c] = out[c] * 100.0
    return out


def load_dataset(path: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    """Load + validate + convert to percent. Rows with missing Fe are dropped."""
    raw = load_raw(path)
    report = validate_raw(raw)
    df = to_percent(raw, report["units_detected"])
    df = df.dropna(subset=[C.TARGET]).reset_index(drop=True)
    df[C.CAT_COL] = df[C.CAT_COL].astype(str).str.strip().str.lower()
    return df, report


def split_train_test(df: pd.DataFrame, test_size: float = C.TEST_SIZE, seed: int = C.RANDOM_STATE):
    """Random hold-out split, stratified on lithotype (very rare classes are pooled
    *for stratification only*, since e.g. basalt has a single sample)."""
    strat = df[C.CAT_COL].copy()
    counts = strat.value_counts()
    rare = counts[counts < 20].index
    strat = strat.where(~strat.isin(rare), "rare_pooled")
    if strat.value_counts().min() < 2:
        strat = None
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    return train.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    data, rep = load_dataset()
    print(rep)
    print(np.round(data.describe().T, 4))
