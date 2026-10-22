"""Train, tune (on training data only), evaluate on the hold-out test set and
persist the model artefact + reports.

Run from the ``backend`` folder:   python -m src.train_model
"""
from __future__ import annotations

import json
import time
from itertools import product

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data_loader import load_dataset, split_train_test
from .evaluate_model import (
    bootstrap_rmse_by_lithotype, direction_check, importance_table,
    metrics_by_lithotype, regression_metrics, save_static_figures,
)
from .models import BaselineModel, HybridFeModel, PureModel, cv_rmse, oof_predictions
from .preprocessing import iqr_outlier_summary

XGB_GRID = [dict(max_depth=d, n_estimators=n, learning_rate=0.03) for d, n in product([2, 3, 4], [200, 400])]


def _tune(make, X, y, label, log):
    scores = []
    for params in XGB_GRID:
        scores.append((cv_rmse(lambda p=params: make(p), X, y), params))
    scores.sort(key=lambda t: t[0])
    log[label] = [{"cv_rmse": float(s), "params": p} for s, p in scores]
    return scores[0][1]


def train_and_save(model_path=C.MODEL_PATH, quick: bool = False, save_reports: bool = True) -> dict:
    t0 = time.time()
    df, data_report = load_dataset()
    train, test = split_train_test(df)
    X_tr, y_tr = train.drop(columns=[C.TARGET]), train[C.TARGET].values
    X_te, y_te = test.drop(columns=[C.TARGET]), test[C.TARGET].values
    sel: dict = {}

    # ---- 1. design choices selected by CV on the TRAIN split only -----------------
    cap_cv = {str(c): cv_rmse(lambda c=c: HybridFeModel(cap=c), X_tr, y_tr) for c in (True, False)}
    cap = cap_cv["True"] <= cap_cv["False"]
    sel["outlier_capping_cv_rmse"] = cap_cv
    sel["outlier_capping_chosen"] = bool(cap)

    base_cv = {k: cv_rmse(lambda k=k: BaselineModel(k, cap), X_tr, y_tr) for k in ("mass_balance", "ridge")}
    baseline_kind = min(base_cv, key=base_cv.get)
    sel["baseline_cv_rmse"] = base_cv
    sel["baseline_chosen"] = baseline_kind

    if quick:
        hy_params, xgb_params = {}, {}
    else:
        hy_params = _tune(lambda p: HybridFeModel(baseline_kind, cap, p), X_tr, y_tr, "hybrid_xgb_grid", sel)
        xgb_params = _tune(lambda p: PureModel("xgb", cap, p), X_tr, y_tr, "pure_xgb_grid", sel)
    sel["hybrid_xgb_params"], sel["pure_xgb_params"] = hy_params, xgb_params

    # ---- 2. final fits on train ---------------------------------------------------
    baseline = BaselineModel(baseline_kind, cap).fit(X_tr, y_tr)
    ridge = PureModel("ridge", cap).fit(X_tr, y_tr)
    xgb = PureModel("xgb", cap, xgb_params).fit(X_tr, y_tr)
    hybrid = HybridFeModel(baseline_kind, cap, hy_params).fit(X_tr, y_tr)

    # ---- 3. 95 % prediction interval: empirical quantiles of cross-validated errors
    oof = oof_predictions(lambda: HybridFeModel(baseline_kind, cap, hy_params), X_tr, y_tr)
    err = y_tr - oof  # measured - predicted
    a = (1 - C.INTERVAL_LEVEL) / 2
    q_lo, q_hi = float(np.quantile(err, a)), float(np.quantile(err, 1 - a))

    # ---- 4. held-out evaluation ---------------------------------------------------
    base_t, corr_t = hybrid.predict_components(X_te)
    preds = pd.DataFrame({
        "Hole_id": test[C.ID_COL], C.CAT_COL: test[C.CAT_COL],
        "Easting": test["Easting"], "Northing": test["Northing"],
        "Fe_actual": y_te,
        "Fe_baseline": baseline.predict(X_te), "Fe_ridge": ridge.predict(X_te),
        "Fe_xgb": xgb.predict(X_te), "Fe_hybrid": base_t + corr_t, "Fe_correction": corr_t,
    })
    preds["Fe_hybrid_lower95"] = preds["Fe_hybrid"] + q_lo
    preds["Fe_hybrid_upper95"] = preds["Fe_hybrid"] + q_hi
    inside = (preds["Fe_actual"] >= preds["Fe_hybrid_lower95"]) & (preds["Fe_actual"] <= preds["Fe_hybrid_upper95"])
    preds["inside_interval"] = inside

    names = {"baseline": "Fe_baseline", "ridge": "Fe_ridge", "xgboost": "Fe_xgb", "hybrid": "Fe_hybrid"}
    comparison = {k: regression_metrics(preds["Fe_actual"], preds[v]) for k, v in names.items()}
    by_lith = {k: metrics_by_lithotype(preds, "Fe_actual", v).to_dict("records") for k, v in names.items()}
    boot = bootstrap_rmse_by_lithotype(preds, "Fe_actual", "Fe_hybrid")

    phi_h, _ = hybrid.explain(X_te)
    phi_x, _ = xgb.explain(X_te)
    F_te = hybrid.prep_.transform(X_te)
    imp_h, imp_x = importance_table(phi_h), importance_table(phi_x)

    b_int, b_coef = hybrid.baseline_.raw_coefficients(hybrid.feature_names_)
    ranges = {}
    for c in C.CHEM_COLS:
        s = X_tr[c]
        ranges[c] = {k: float(v) for k, v in
                     zip(["min", "p01", "median", "p99", "max"], s.quantile([0, .01, .5, .99, 1]))}
    ranges_by_lith = X_tr.groupby(C.CAT_COL)[C.CHEM_COLS].median().to_dict("index")

    metrics = {
        "data_report": data_report,
        "split": {"n_train": len(train), "n_test": len(test), "seed": C.RANDOM_STATE},
        "outliers_inspection_full_data": iqr_outlier_summary(df),
        "selection": sel,
        "comparison": comparison,
        "by_lithotype": by_lith,
        "rmse_bootstrap_by_lithotype": boot,
        "shap_importance": {"hybrid": imp_h, "xgboost": imp_x},
        "shap_direction_spearman_hybrid": direction_check(F_te, phi_h),
        "interval": {
            "level": C.INTERVAL_LEVEL, "offset_lower": q_lo, "offset_upper": q_hi,
            "method": "empirical quantiles of 5-fold cross-validated hybrid errors on the training split",
            "test_coverage": float(inside.mean()),
            "test_coverage_by_lithotype": {str(k): float(v) for k, v in inside.groupby(preds[C.CAT_COL]).mean().items()},
        },
        "baseline_equation": {"kind": baseline_kind, "intercept": b_int,
                              "coefficients": dict(zip(hybrid.feature_names_, map(float, b_coef)))},
        "train_seconds": round(time.time() - t0, 1),
    }

    artifact = {
        "hybrid": hybrid, "ridge": ridge, "xgboost": xgb, "baseline": baseline,
        "interval": {"lower": q_lo, "upper": q_hi, "level": C.INTERVAL_LEVEL},
        "categories": hybrid.prep_.categories_,
        "feature_ranges": ranges, "median_by_lithotype": ranges_by_lith,
        "meta": {"trained_on": "training split only", "n_train": len(train), "units": "percent",
                 "coordinates_used_in_model": False, "citation": C.DATASET_CITATION},
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    if save_reports:
        C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        preds.to_csv(C.REPORTS_DIR / "test_predictions.csv", index=False)
        pd.DataFrame(comparison).T.to_csv(C.REPORTS_DIR / "model_comparison.csv")
        save_static_figures(preds, imp_h)
    return metrics


def main():
    m = train_and_save()
    print(json.dumps({k: m[k] for k in ("split", "comparison", "interval", "selection")
                      if k != "selection"}, indent=2))
    print("selection:", {k: v for k, v in m["selection"].items() if "grid" not in k})
    print("Saved model ->", C.MODEL_PATH)


if __name__ == "__main__":
    main()
