import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_loader import DataValidationError, detect_units, load_raw, to_percent, validate_raw


def test_real_file_shape_and_columns():
    raw = load_raw()
    assert raw.shape == (1600, 16)
    assert set(C.REQUIRED_RAW_COLS) <= set(raw.columns)


def test_units_detected_and_converted():
    raw = load_raw()
    assert detect_units(raw) == "fraction"
    pct = to_percent(raw)
    assert pct["Fe"].iloc[0] == pytest.approx(63.28)
    assert raw["Fe"].iloc[0] == pytest.approx(0.6328)  # raw preserved


def test_validate_report(dataset):
    _, rep = dataset
    assert rep["fe_missing"] == 0 and rep["shape"] == [1600, 16]


def test_missing_column_raises():
    raw = load_raw().drop(columns=["SiO2"])
    with pytest.raises(DataValidationError):
        validate_raw(raw)


def test_split_is_disjoint_and_sized(split):
    train, test = split
    assert len(train) + len(test) == 1600
    assert set(train["Hole_id"]).isdisjoint(set(test["Hole_id"]))
