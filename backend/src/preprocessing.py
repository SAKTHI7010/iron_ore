"""Reusable, leakage-safe preprocessing.

``FeaturePreprocessor`` learns everything (lithotype categories, per-lithotype
medians for imputation, winsorisation caps) from the *training* data only.
Coordinates and identifiers are intentionally excluded from the model features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C

DEFAULT_CAP_COLS = ("SiO2", "Al2O3")


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Impute (median by lithotype) -> cap (1st/99th pct) -> one-hot lithotype."""

    def __init__(self, cap: bool = True, cap_cols=DEFAULT_CAP_COLS, cap_quantiles=(0.01, 0.99),
                 handle_unknown: str = "ignore"):
        self.cap = cap
        self.cap_cols = cap_cols
        self.cap_quantiles = cap_quantiles
        self.handle_unknown = handle_unknown  # 'ignore' -> all-zero one-hot, 'error' -> raise

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _clean(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[C.CAT_COL] = X[C.CAT_COL].astype(str).str.strip().str.lower()
        for c in C.CHEM_COLS:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    @property
    def feature_names_(self) -> list[str]:
        return [*C.CHEM_COLS, *[f"{C.CAT_COL}_{c}" for c in self.categories_]]

    # -- sklearn API -----------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        X = self._clean(X)
        self.categories_ = sorted(X[C.CAT_COL].unique())
        self.global_median_ = X[C.CHEM_COLS].median()
        self.median_by_lith_ = X.groupby(C.CAT_COL)[C.CHEM_COLS].median()
        lo, hi = self.cap_quantiles
        self.caps_ = {}
        if self.cap:
            for c in self.cap_cols:
                self.caps_[c] = (float(X[c].quantile(lo)), float(X[c].quantile(hi)))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._clean(X)
        unknown = sorted(set(X[C.CAT_COL]) - set(self.categories_))
        if unknown and self.handle_unknown == "error":
            raise ValueError(f"Unknown Lithotype value(s) {unknown}; expected one of {self.categories_}.")
        out = X[C.CHEM_COLS].copy()
        for c in C.CHEM_COLS:
            miss = out[c].isna()
            if miss.any():
                fill = X.loc[miss, C.CAT_COL].map(self.median_by_lith_[c])
                fill = fill.fillna(self.global_median_[c])
                out.loc[miss, c] = fill
        for c, (lo, hi) in self.caps_.items():
            out[c] = out[c].clip(lo, hi)
        for cat in self.categories_:
            out[f"{C.CAT_COL}_{cat}"] = (X[C.CAT_COL] == cat).astype(float).values
        return out.astype(float).reset_index(drop=True)


def iqr_outlier_summary(df: pd.DataFrame, cols=("Fe", "SiO2", "Al2O3")) -> dict:
    """Count IQR (1.5x) and |z|>3 outliers for the inspected columns."""
    res = {}
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = (s - s.mean()) / s.std(ddof=0)
        res[c] = {
            "iqr_low": float(lo),
            "iqr_high": float(hi),
            "n_iqr_outliers": int(((s < lo) | (s > hi)).sum()),
            "n_abs_z_gt_3": int((z.abs() > 3).sum()),
            "p01": float(s.quantile(0.01)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }
    return res
