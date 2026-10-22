import numpy as np
import pandas as pd

from src import config as C
from src.preprocessing import FeaturePreprocessor


def test_one_hot_and_columns(split):
    train, _ = split
    prep = FeaturePreprocessor().fit(train)
    F = prep.transform(train)
    assert list(F.columns) == prep.feature_names_
    assert not any(c in F.columns for c in C.COORD_COLS + [C.ID_COL])
    lith_cols = [c for c in F.columns if c.startswith("Lithotype_")]
    assert (F[lith_cols].sum(axis=1) == 1).all()


def test_imputation_uses_train_median_by_lithotype(split):
    train, _ = split
    prep = FeaturePreprocessor(cap=False).fit(train)
    row = train.iloc[[0]].copy()
    row["SiO2"] = np.nan
    out = prep.transform(row)
    expected = train[train[C.CAT_COL] == row[C.CAT_COL].iloc[0]]["SiO2"].median()
    assert out["SiO2"].iloc[0] == expected


def test_caps_fit_on_train_only(split):
    train, test = split
    prep = FeaturePreprocessor(cap=True).fit(train)
    lo, hi = prep.caps_["SiO2"]
    assert lo == train["SiO2"].quantile(0.01) and hi == train["SiO2"].quantile(0.99)
    assert prep.transform(test)["SiO2"].max() <= hi + 1e-12
