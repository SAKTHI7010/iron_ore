import joblib
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.predict import explain_frame, predict_frame, single_prediction, validate_input


def _sample(art):
    r = art["median_by_lithotype"]["goethite"]
    return {**r, C.CAT_COL: "goethite"}


def test_artifact_reloads_and_predicts_identically(quick_artifact, split):
    path, art = quick_artifact
    _, test = split
    X = test.drop(columns=[C.TARGET])
    p1 = predict_frame(X, art)["Fe_pred_pct"].values
    art2 = joblib.load(path)
    p2 = predict_frame(X, art2)["Fe_pred_pct"].values
    np.testing.assert_allclose(p1, p2)
    assert np.isfinite(p1).all() and 30 < p1.mean() < 80


def test_single_prediction_and_interval(quick_artifact):
    _, art = quick_artifact
    res = single_prediction(_sample(art), art)
    assert not res["errors"]
    p = res["prediction"]
    assert p["Fe_pred_lower95_pct"] <= p["Fe_pred_pct"] <= p["Fe_pred_upper95_pct"]


def test_shap_attribution_is_additive(quick_artifact, split):
    _, art = quick_artifact
    _, test = split
    X = test.drop(columns=[C.TARGET]).head(25)
    phi, base = explain_frame(X, art)
    pred = predict_frame(X, art)["Fe_pred_pct"].values
    np.testing.assert_allclose(base + phi.sum(axis=1).values, pred, atol=1e-3)


def test_validation_catches_bad_input(quick_artifact):
    _, art = quick_artifact
    df = pd.DataFrame([{**_sample(art), C.CAT_COL: "unobtainium"}])
    _, errors, _ = validate_input(df, art)
    assert errors
    _, errors, _ = validate_input(pd.DataFrame({"SiO2": [1]}), art)
    assert "Missing required" in errors[0]
    df = pd.DataFrame([_sample(art)])
    _, errors, _ = validate_input(df, art, units="fraction")
    assert errors  # percent-scale values flagged as invalid fractions
