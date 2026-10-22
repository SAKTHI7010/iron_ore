"""Inference utilities: model loading, input validation, prediction and explanation."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .models import aggregate_phi

_CACHE: dict = {}


def ensure_model(path=C.MODEL_PATH) -> None:
    """Train the model on first use if no artefact exists yet."""
    if not path.exists():
        from .train_model import train_and_save

        train_and_save(path)


def load_artifact(path=C.MODEL_PATH) -> dict:
    key = str(path)
    if key not in _CACHE:
        ensure_model(path)
        _CACHE[key] = joblib.load(path)
    return _CACHE[key]


def validate_input(df: pd.DataFrame, artifact: dict, units: str = "percent") -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate a user-supplied table (single row or batch).

    Returns (clean_df_in_percent, errors, warnings). If ``errors`` is non-empty the
    returned frame must not be used for prediction.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if df is None or len(df) == 0:
        return pd.DataFrame(), ["The input contains no rows."], warnings
    missing = [c for c in C.REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        return pd.DataFrame(), [f"Missing required column(s): {missing}. Required: {C.REQUIRED_INPUT_COLS}"], warnings

    out = df.copy()
    for c in C.CHEM_COLS:
        coerced = pd.to_numeric(out[c], errors="coerce")
        bad = int((coerced.isna() & out[c].notna()).sum())
        if bad:
            errors.append(f"Column '{c}' has {bad} non-numeric value(s).")
        out[c] = coerced
    n_nan = int(out[C.CHEM_COLS].isna().any(axis=1).sum())
    if n_nan:
        warnings.append(f"{n_nan} row(s) have missing chemistry; imputed with the training median for their lithotype.")

    out[C.CAT_COL] = out[C.CAT_COL].astype(str).str.strip().str.lower()
    unknown = sorted(set(out[C.CAT_COL]) - set(artifact["categories"]))
    if unknown:
        errors.append(f"Unknown Lithotype value(s) {unknown}. Allowed: {artifact['categories']}.")

    if C.TARGET in out.columns:
        out[C.TARGET] = pd.to_numeric(out[C.TARGET], errors="coerce")

    if units == "fraction":
        if (out[C.CHEM_COLS].max().max() > 1.0):
            errors.append("Units set to 'fraction' but some values exceed 1.0 - did you mean percent?")
        else:
            for c in [*C.CHEM_COLS, *([C.TARGET] if C.TARGET in out.columns else [])]:
                out[c] = out[c] * 100.0
    if (out[C.CHEM_COLS] < 0).any().any():
        errors.append("Negative chemistry values are not physically valid.")
    if (out[C.CHEM_COLS] > 100).any().any():
        errors.append("Chemistry values above 100 % are not valid.")

    if not errors:
        for c, r in artifact["feature_ranges"].items():
            n_out = int(((out[c] < r["min"]) | (out[c] > r["max"])).sum())
            if n_out:
                warnings.append(f"{n_out} row(s) have {c} outside the training range "
                                f"[{r['min']:.4g}, {r['max']:.4g}] % - extrapolation, treat with caution.")
    return out, errors, warnings


def predict_frame(df_pct: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Hybrid prediction + 95 % interval for already-validated percent-unit rows."""
    hybrid = artifact["hybrid"]
    base, corr = hybrid.predict_components(df_pct)
    pred = base + corr
    iv = artifact["interval"]
    return pd.DataFrame({
        "Fe_pred_pct": pred,
        "Fe_pred_lower95_pct": np.clip(pred + iv["lower"], 0, 100),
        "Fe_pred_upper95_pct": np.clip(pred + iv["upper"], 0, 100),
        "Fe_baseline_pct": base,
        "Fe_ml_correction_pct": corr,
    }, index=df_pct.index)


def explain_frame(df_pct: pd.DataFrame, artifact: dict) -> tuple[pd.DataFrame, float]:
    """Per-sample additive attribution (aggregated lithotype) and the base value."""
    phi, base_value = artifact["hybrid"].explain(df_pct)
    phi = aggregate_phi(phi)
    phi.index = df_pct.index
    return phi, base_value


def single_prediction(values: dict, artifact: dict) -> dict:
    row = pd.DataFrame([values])
    clean, errors, warnings = validate_input(row, artifact, "percent")
    if errors:
        return {"errors": errors, "warnings": warnings}
    res = predict_frame(clean, artifact).iloc[0].to_dict()
    phi, base_value = explain_frame(clean, artifact)
    return {"errors": [], "warnings": warnings, "prediction": res,
            "phi": phi.iloc[0].to_dict(), "base_value": base_value, "input": clean.iloc[0].to_dict()}
