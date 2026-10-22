"""Held-out evaluation: RMSE / MAE / R2, per-lithotype metrics, bootstrap RMSE
distributions, SHAP importance and domain-direction checks, static figures."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as C
from .models import aggregate_phi


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(y_true)),
    }


def metrics_by_lithotype(df: pd.DataFrame, y_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for lith, g in df.groupby(C.CAT_COL):
        m = regression_metrics(g[y_col], g[pred_col])
        if m["n"] < 3:  # R2 meaningless on <3 points
            m["r2"] = float("nan")
        rows.append({C.CAT_COL: lith, **m})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def bootstrap_rmse_by_lithotype(df, y_col, pred_col, n_boot=300, min_n=5, seed=C.RANDOM_STATE) -> dict:
    """Bootstrap the test-set RMSE within each lithotype (>= min_n test samples)."""
    rng = np.random.default_rng(seed)
    out = {}
    for lith, g in df.groupby(C.CAT_COL):
        if len(g) < min_n:
            continue
        e = (g[pred_col].values - g[y_col].values) ** 2
        idx = rng.integers(0, len(e), size=(n_boot, len(e)))
        out[str(lith)] = [float(v) for v in np.sqrt(e[idx].mean(axis=1))]
    return out


def importance_table(phi: pd.DataFrame) -> dict:
    agg = aggregate_phi(phi)
    imp = agg.abs().mean().sort_values(ascending=False)
    return {k: float(v) for k, v in imp.items()}


def direction_check(F: pd.DataFrame, phi: pd.DataFrame) -> dict:
    """Spearman(feature value, its SHAP value): negative => higher value lowers Fe."""
    res = {}
    for c in C.CHEM_COLS:
        if F[c].nunique() > 2:
            rho = spearmanr(F[c], phi[c]).correlation
            res[c] = float(rho) if rho == rho else float("nan")
    return res


def save_static_figures(pred_df: pd.DataFrame, importance: dict, out_dir=C.FIGURES_DIR) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for lith, g in pred_df.groupby(C.CAT_COL):
        ax.scatter(g["Fe_actual"], g["Fe_hybrid"], s=14, alpha=0.7, label=lith)
    lo, hi = pred_df["Fe_actual"].min(), pred_df["Fe_actual"].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Measured Fe (%)")
    ax.set_ylabel("Predicted Fe (%) - hybrid")
    ax.legend()
    ax.set_title("Test set: predicted vs measured")
    p = out_dir / "pred_vs_actual.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(pred_df["Fe_hybrid"] - pred_df["Fe_actual"], bins=40, color="#4aa3a2")
    ax.set_xlabel("Residual = predicted - measured Fe (%)")
    ax.set_title("Hybrid residuals (test)")
    p = out_dir / "residual_hist.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = list(importance)[::-1]
    ax.barh(names, [importance[n] for n in names], color="#d98c3f")
    ax.set_xlabel("Mean |SHAP| (percentage points of Fe)")
    ax.set_title("Hybrid feature importance (test)")
    p = out_dir / "shap_importance.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(str(p))
    return paths
