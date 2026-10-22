"""Model classes: baseline, pure Ridge / XGBoost, and the hybrid
(mechanistic baseline + XGBoost residual correction).

Every class takes the *raw* (percent-unit) DataFrame in ``fit``/``predict`` and
owns its preprocessing, so cross-validation never leaks information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config as C
from .preprocessing import FeaturePreprocessor

DEFAULT_XGB = dict(
    n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
    random_state=C.RANDOM_STATE, n_jobs=1, verbosity=0,
)


def _xgb(params: dict | None) -> XGBRegressor:
    p = {**DEFAULT_XGB, **(params or {})}
    return XGBRegressor(**p)


# ---------------------------------------------------------------------------
# Layer 1: mechanistic / baseline proxy (linear in the preprocessed features)
# ---------------------------------------------------------------------------
class LinearBaseline:
    """``mass_balance``: Fe = a + b*Fe_oxide_proxy + c*LOI, where
    Fe_oxide_proxy = 100 - (SiO2 + Al2O3 + P + S + Mn + CL)  [% non-gangue mass].
    ``ridge``: Ridge on SiO2, Al2O3, LOI (fallback suggested in the brief).
    """

    def __init__(self, kind: str = "mass_balance"):
        if kind not in ("mass_balance", "ridge"):
            raise ValueError(kind)
        self.kind = kind

    def _design(self, F: pd.DataFrame) -> np.ndarray:
        if self.kind == "mass_balance":
            proxy = 100.0 - F[C.GANGUE_COLS].sum(axis=1)
            return np.column_stack([proxy.values, F["LOI"].values])
        return F[["SiO2", "Al2O3", "LOI"]].values

    def fit(self, F: pd.DataFrame, y):
        X = self._design(F)
        if self.kind == "mass_balance":
            self.model_ = LinearRegression().fit(X, y)
        else:
            self.model_ = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))).fit(X, y)
        return self

    def predict(self, F: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._design(F))

    def raw_coefficients(self, columns: list[str]) -> tuple[float, np.ndarray]:
        """Express the baseline as intercept + coef . x on the preprocessed columns."""
        coef = pd.Series(0.0, index=columns)
        if self.kind == "mass_balance":
            b, c = self.model_.coef_
            a = float(self.model_.intercept_)
            for g in C.GANGUE_COLS:
                coef[g] = -b
            coef["LOI"] = c
            return a + 100.0 * b, coef.values
        scaler, ridge = self.model_.steps[0][1], self.model_.steps[1][1]
        w = ridge.coef_ / scaler.scale_
        for name, wi in zip(["SiO2", "Al2O3", "LOI"], w):
            coef[name] = wi
        intercept = float(ridge.intercept_ - np.sum(ridge.coef_ * scaler.mean_ / scaler.scale_))
        return intercept, coef.values


# ---------------------------------------------------------------------------
# TreeSHAP helper (shap library, with xgboost-native fallback)
# ---------------------------------------------------------------------------
def tree_shap(model: XGBRegressor, F: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Return (phi [n x features], expected_value) for an XGBoost model."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        phi = np.asarray(explainer.shap_values(F))
        ev = float(np.ravel(explainer.expected_value)[0])
        return phi, ev
    except Exception:  # pragma: no cover - fallback when shap/xgboost versions clash
        import xgboost as xgb

        contribs = model.get_booster().predict(xgb.DMatrix(F), pred_contribs=True)
        return contribs[:, :-1], float(contribs[0, -1])


# ---------------------------------------------------------------------------
# Comparison models
# ---------------------------------------------------------------------------
class BaselineModel:
    """Mechanistic / baseline proxy only."""

    def __init__(self, kind="mass_balance", cap=True):
        self.kind, self.cap = kind, cap

    def fit(self, X: pd.DataFrame, y):
        self.prep_ = FeaturePreprocessor(cap=self.cap).fit(X)
        F = self.prep_.transform(X)
        self.baseline_ = LinearBaseline(self.kind).fit(F, np.asarray(y))
        return self

    def predict(self, X):
        return self.baseline_.predict(self.prep_.transform(X))


class PureModel:
    """Single-stage model on all approved features: ``ridge`` or ``xgb``."""

    def __init__(self, kind="xgb", cap=True, xgb_params=None):
        self.kind, self.cap, self.xgb_params = kind, cap, xgb_params

    def fit(self, X: pd.DataFrame, y):
        self.prep_ = FeaturePreprocessor(cap=self.cap).fit(X)
        F = self.prep_.transform(X)
        y = np.asarray(y)
        if self.kind == "ridge":
            self.model_ = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))).fit(F, y)
        else:
            self.model_ = _xgb(self.xgb_params).fit(F, y)
        self.feature_names_ = list(F.columns)
        return self

    def predict(self, X):
        return self.model_.predict(self.prep_.transform(X))

    def explain(self, X) -> tuple[pd.DataFrame, float]:
        if self.kind != "xgb":
            raise NotImplementedError
        F = self.prep_.transform(X)
        phi, ev = tree_shap(self.model_, F)
        return pd.DataFrame(phi, columns=self.feature_names_), ev


class HybridFeModel:
    """Proposed hybrid: baseline prediction + XGBoost-predicted residual correction.

    The residual model is trained on *out-of-fold* baseline residuals so it does
    not learn from in-sample baseline fit.
    """

    def __init__(self, baseline_kind="mass_balance", cap=True, xgb_params=None, n_splits=C.CV_FOLDS):
        self.baseline_kind, self.cap, self.xgb_params, self.n_splits = baseline_kind, cap, xgb_params, n_splits

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, dtype=float)
        self.prep_ = FeaturePreprocessor(cap=self.cap).fit(X)
        F = self.prep_.transform(X)
        oof = np.zeros(len(F))
        kf = KFold(self.n_splits, shuffle=True, random_state=C.RANDOM_STATE)
        for tr, va in kf.split(F):
            b = LinearBaseline(self.baseline_kind).fit(F.iloc[tr], y[tr])
            oof[va] = b.predict(F.iloc[va])
        self.baseline_ = LinearBaseline(self.baseline_kind).fit(F, y)
        self.resid_model_ = _xgb(self.xgb_params).fit(F, y - oof)
        self.feature_names_ = list(F.columns)
        self.train_mean_ = F.mean()
        return self

    def predict_components(self, X) -> tuple[np.ndarray, np.ndarray]:
        F = self.prep_.transform(X)
        return self.baseline_.predict(F), self.resid_model_.predict(F)

    def predict(self, X) -> np.ndarray:
        base, corr = self.predict_components(X)
        return base + corr

    def explain(self, X) -> tuple[pd.DataFrame, float]:
        """Exact additive attribution: prediction = base_value + sum(phi).

        phi = linear-baseline contribution (coef * (x - mean)) + TreeSHAP of the
        residual correction.
        """
        F = self.prep_.transform(X)
        intercept, coef = self.baseline_.raw_coefficients(self.feature_names_)
        mean = self.train_mean_.values
        lin_phi = (F.values - mean) * coef
        tree_phi, ev = tree_shap(self.resid_model_, F)
        base_value = float(intercept + coef @ mean + ev)
        return pd.DataFrame(lin_phi + tree_phi, columns=self.feature_names_), base_value


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def oof_predictions(make_model, X: pd.DataFrame, y, n_splits=C.CV_FOLDS) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    oof = np.zeros(len(X))
    kf = KFold(n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    for tr, va in kf.split(X):
        m = make_model().fit(X.iloc[tr], y[tr])
        oof[va] = m.predict(X.iloc[va])
    return oof


def cv_rmse(make_model, X, y, n_splits=C.CV_FOLDS) -> float:
    oof = oof_predictions(make_model, X, y, n_splits)
    return float(np.sqrt(np.mean((np.asarray(y) - oof) ** 2)))


def aggregate_phi(phi: pd.DataFrame) -> pd.DataFrame:
    """Collapse the one-hot Lithotype columns into a single 'Lithotype' attribution."""
    lith_cols = [c for c in phi.columns if c.startswith(f"{C.CAT_COL}_")]
    out = phi.drop(columns=lith_cols).copy()
    out[C.CAT_COL] = phi[lith_cols].sum(axis=1) if lith_cols else 0.0
    return out
