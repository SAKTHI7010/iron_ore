"""Lightweight geological Digital Twin helpers.

The Windarling data are static snapshots of one bench (no time dimension), so the
"twin" is a spatial ore-quality model: data -> model -> decision. It is NOT an
equipment-level or dynamic mechanistic twin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config as C


def grade_decision(fe_pct: float, cutoff: float) -> str:
    """Simple cut-off decision on the predicted Fe grade."""
    return "Above cut-off (ore)" if fe_pct >= cutoff else "Below cut-off (low grade)"


def nearest_context(ref: pd.DataFrame, easting: float, northing: float, k: int = 10) -> dict:
    """Measured Fe of the k nearest drilled samples around a location."""
    d = np.hypot(ref["Easting"] - easting, ref["Northing"] - northing)
    idx = d.nsmallest(k).index
    return {
        "k": k,
        "mean_measured_fe": float(ref.loc[idx, C.TARGET].mean()),
        "std_measured_fe": float(ref.loc[idx, C.TARGET].std()),
        "mean_distance_m": float(d.loc[idx].mean()),
    }


def spatial_figure(ref: pd.DataFrame, new_points: pd.DataFrame | None = None,
                   color_col: str = C.TARGET, title: str = "Measured Fe grade across the bench",
                   value_label: str = "Measured Fe", trace_name: str = "Measured (drill samples)",
                   colorscale: str = "Turbo") -> go.Figure:
    """Fe map in Easting/Northing. ``new_points`` need Easting, Northing, Fe_pred_pct."""
    fig = go.Figure(go.Scatter(
        x=ref["Easting"], y=ref["Northing"], mode="markers", name=trace_name,
        marker=dict(size=7, color=ref[color_col], colorscale=colorscale, showscale=True,
                    colorbar=dict(title=f"{value_label} (%)"), opacity=0.85),
        text=ref[C.CAT_COL], customdata=ref[color_col],
        hovertemplate="E %{x:.1f} m<br>N %{y:.1f} m<br>%{text}<br>" + value_label + " %{customdata:.2f} %<extra></extra>",
    ))
    if new_points is not None and len(new_points):
        fig.add_trace(go.Scatter(
            x=new_points["Easting"], y=new_points["Northing"], mode="markers", name="New sample (predicted)",
            marker=dict(size=15, symbol="star", color=new_points["Fe_pred_pct"], colorscale="Turbo",
                        cmin=float(ref[color_col].min()), cmax=float(ref[color_col].max()),
                        line=dict(color="white", width=2)),
            customdata=new_points["Fe_pred_pct"],
            hovertemplate="E %{x:.1f} m<br>N %{y:.1f} m<br>Predicted Fe %{customdata:.2f} %<extra></extra>",
        ))
    fig.update_layout(title=title, xaxis_title="Easting (m)", yaxis_title="Northing (m)",
                      template="plotly_dark", height=520, legend=dict(orientation="h", y=-0.15))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
