"""Exploratory data analysis: summary statistics, correlations and figures.

Run from ``backend``:  python -m src.eda
Writes reports/eda_summary.json, reports/eda_summary.md and PNGs in reports/figures.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config as C
from .data_loader import load_dataset
from .preprocessing import iqr_outlier_summary

NUM_COLS = [C.TARGET, *C.CHEM_COLS]


def compute_eda(df: pd.DataFrame) -> dict:
    desc = df[NUM_COLS].describe().T
    desc["skew"] = df[NUM_COLS].skew()
    pearson = df[NUM_COLS].corr(method="pearson")[C.TARGET].drop(C.TARGET)
    spearman = df[NUM_COLS].corr(method="spearman")[C.TARGET].drop(C.TARGET)
    return {
        "describe_pct": desc.round(4).to_dict("index"),
        "fe_mean": float(df[C.TARGET].mean()),
        "fe_std": float(df[C.TARGET].std()),
        "fe_skew": float(df[C.TARGET].skew()),
        "lithotype_counts": {str(k): int(v) for k, v in df[C.CAT_COL].value_counts().items()},
        "fe_by_lithotype": df.groupby(C.CAT_COL)[C.TARGET].agg(["count", "mean", "std", "min", "max"]).round(3).to_dict("index"),
        "pearson_with_fe": {k: float(v) for k, v in pearson.items()},
        "spearman_with_fe": {k: float(v) for k, v in spearman.items()},
        "outliers": iqr_outlier_summary(df),
    }


# ----- plotly figures (used by the Streamlit app) ---------------------------------
def fig_fe_hist(df):
    f = px.histogram(df, x=C.TARGET, nbins=50, color_discrete_sequence=["#2dd4bf"], template="plotly_dark",
                     title="Distribution of measured Fe (%)")
    f.update_layout(bargap=0.03, xaxis_title="Fe (%)")
    return f


def fig_corr_heatmap(df, method="pearson"):
    corr = df[NUM_COLS].corr(method=method)
    f = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                  template="plotly_dark", title=f"{method.title()} correlation heatmap")
    f.update_layout(height=520)
    return f


def fig_fe_box(df):
    f = px.box(df, x=C.CAT_COL, y=C.TARGET, color=C.CAT_COL, points="outliers", template="plotly_dark",
               title="Measured Fe by lithotype")
    f.update_layout(showlegend=False, yaxis_title="Fe (%)")
    return f


def fig_feature_hist(df, col):
    return px.histogram(df, x=col, nbins=60, template="plotly_dark", color_discrete_sequence=["#f59e0b"],
                        title=f"Distribution of {col} (%)")


def main():
    df, rep = load_dataset()
    eda = compute_eda(df)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "eda_summary.json").write_text(json.dumps({"data_report": rep, **eda}, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.histplot(df[C.TARGET], bins=50, ax=ax, color="#2a9d8f")
    ax.set_xlabel("Fe (%)")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "eda_fe_hist.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(df[NUM_COLS].corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "eda_corr_heatmap.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x=C.CAT_COL, y=C.TARGET, ax=ax)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "eda_fe_by_lithotype.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(df["Easting"], df["Northing"], c=df[C.TARGET], s=8, cmap="turbo")
    fig.colorbar(sc, label="Fe (%)"); ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "eda_spatial_fe.png", dpi=130); plt.close(fig)

    md = [
        "# EDA summary (computed from the real data; units: percent)",
        f"- Shape: {rep['shape']}, units in file: **{rep['units_detected']}** (converted to % on a copy).",
        f"- Fe mean {eda['fe_mean']:.2f} %, std {eda['fe_std']:.2f} %, skew {eda['fe_skew']:.2f}.",
        f"- Lithotype counts: {eda['lithotype_counts']}",
        f"- Pearson(Fe, x): { {k: round(v, 3) for k, v in eda['pearson_with_fe'].items()} }",
        f"- Spearman(Fe, x): { {k: round(v, 3) for k, v in eda['spearman_with_fe'].items()} }",
    ]
    (C.REPORTS_DIR / "eda_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
