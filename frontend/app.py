"""Streamlit dashboard - Windarling iron-ore Fe-grade soft sensor.

Run from the ``frontend`` folder:   streamlit run app.py
The UI only talks to the backend through ``backend/src`` (loaded via sys.path).
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src import config as C  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402
from src.digital_twin import grade_decision, nearest_context, spatial_figure  # noqa: E402
from src.eda import (compute_eda, fig_corr_heatmap, fig_fe_box, fig_fe_hist,  # noqa: E402
                     fig_feature_hist)
from src.predict import (load_artifact, predict_frame, single_prediction,  # noqa: E402
                         validate_input)

st.set_page_config(page_title="Windarling Fe Soft Sensor", page_icon="⛏️", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, label { font-family: 'Outfit', sans-serif !important; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"],
[data-testid="stMetricLabel"] p, [data-testid="stMetricValue"], [data-testid="stSidebar"] * { color: #e2e8f0; }
.stApp header[data-testid="stHeader"] { background: transparent; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, rgba(45,212,191,.12), transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, rgba(245,158,11,.10), transparent 55%), #0b1220; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f1a2e 0%,#0b1220 100%);
                                    border-right: 1px solid rgba(148,163,184,.15); }
.hero { padding: 26px 30px; border-radius: 20px; margin-bottom: 18px;
        background: linear-gradient(120deg, rgba(45,212,191,.18), rgba(99,102,241,.16) 55%, rgba(245,158,11,.16));
        border: 1px solid rgba(148,163,184,.22); box-shadow: 0 10px 40px rgba(0,0,0,.35); }
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 700; letter-spacing: .3px;
           background: linear-gradient(90deg,#5eead4,#a5b4fc,#fbbf24); -webkit-background-clip: text;
           -webkit-text-fill-color: transparent; }
.hero p { margin: 6px 0 0 0; color: #cbd5e1; font-size: 1rem; }
.badge { display:inline-block; padding: 3px 12px; border-radius: 999px; font-size:.78rem; margin: 10px 8px 0 0;
         background: rgba(148,163,184,.14); border: 1px solid rgba(148,163,184,.25); color:#e2e8f0; }
div[data-testid="stMetric"] { background: rgba(30,41,59,.55); border: 1px solid rgba(148,163,184,.18);
        border-radius: 16px; padding: 14px 18px; backdrop-filter: blur(6px); transition: all .25s ease; }
div[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: rgba(45,212,191,.6);
        box-shadow: 0 8px 24px rgba(45,212,191,.15); }
div[data-testid="stMetricValue"] { font-weight: 700; }
.result-card { padding: 22px 26px; border-radius: 18px; text-align:center;
        background: linear-gradient(135deg, rgba(45,212,191,.22), rgba(30,41,59,.7));
        border: 1px solid rgba(45,212,191,.45); box-shadow: 0 10px 34px rgba(45,212,191,.12); }
.result-card .v { font-size: 3rem; font-weight: 700; color: #5eead4; line-height: 1.1; }
.result-card .s { color: #cbd5e1; font-size: .95rem; }
.note { padding: 12px 16px; border-left: 4px solid #f59e0b; background: rgba(245,158,11,.08);
        border-radius: 8px; color: #e2e8f0; font-size: .92rem; }
.stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 600;
        background: linear-gradient(90deg,#14b8a6,#6366f1); color: white; border: none; transition: all .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99,102,241,.35); color: white; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

DARK = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,.55)")
LITH_COLORS = {"goethite": "#f59e0b", "hematite": "#ef4444", "schist": "#38bdf8", "basalt": "#a78bfa",
               "magnetite": "#34d399"}


def hero(title: str, subtitle: str, badges: list[str]):
    b = "".join(f"<span class='badge'>{x}</span>" for x in badges)
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p>{b}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading model (trains automatically on first run)...")
def get_artifact():
    return load_artifact()


@st.cache_data(show_spinner=False)
def get_data():
    df, report = load_dataset()
    return df, report


@st.cache_data(show_spinner=False)
def get_metrics():
    p = C.REPORTS_DIR / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def get_test_predictions():
    p = C.REPORTS_DIR / "test_predictions.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def get_eda(_df_key: int):
    return compute_eda(get_data()[0])


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- charts
def waterfall_figure(phi: dict, base_value: float, values: dict, pred: float) -> go.Figure:
    items = sorted(phi.items(), key=lambda kv: abs(kv[1]), reverse=True)
    labels, deltas = [], []
    for name, v in items:
        labels.append(f"{name} = {values[name]}" if name == C.CAT_COL else f"{name} = {values[name]:.4g} %")
        deltas.append(v)
    x = ["Base value (mean model output)", *labels, "Predicted Fe"]
    y = [base_value, *deltas, pred]
    measure = ["absolute", *["relative"] * len(deltas), "total"]
    cum = base_value + np.cumsum(deltas)
    lo, hi = min(cum.min(), base_value, pred), max(cum.max(), base_value, pred)
    pad = max(0.15 * (hi - lo), 0.3)
    fig = go.Figure(go.Waterfall(
        x=x, y=y, measure=measure, orientation="v",
        text=[f"{base_value:.2f}", *[f"{d:+.2f}" for d in deltas], f"{pred:.2f}"], textposition="outside",
        increasing=dict(marker=dict(color="#34d399")), decreasing=dict(marker=dict(color="#f87171")),
        totals=dict(marker=dict(color="#6366f1")), connector=dict(line=dict(color="#64748b"))))
    fig.update_layout(title="SHAP-style contribution waterfall (percentage points of Fe)",
                      yaxis=dict(range=[lo - pad, hi + pad], title="Fe (%)"), height=470,
                      showlegend=False, xaxis=dict(tickangle=-30), **DARK)
    return fig


def detect_units_guess(df: pd.DataFrame) -> str:
    cols = [c for c in [*C.CHEM_COLS, C.TARGET] if c in df.columns]
    nums = df[cols].apply(pd.to_numeric, errors="coerce")
    return "fraction" if nums.max().max() <= 1.0 else "percent"


# --------------------------------------------------------------------------- pages
def page_overview(df, report):
    hero("Windarling Iron-Ore Fe Soft Sensor",
         "Composition-to-grade prediction: XRF-style oxide chemistry in, Fe grade out - with uncertainty and explanations.",
         ["Hybrid ML", "SHAP", "Streamlit + Plotly", "Lightweight geological Digital Twin"])
    eda = get_eda(len(df))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{report['shape'][0]:,}")
    c2.metric("Mean Fe (%)", f"{eda['fe_mean']:.2f}")
    c3.metric("Std Fe (%)", f"{eda['fe_std']:.2f}")
    c4.metric("Lithotypes", len(eda["lithotype_counts"]))

    st.markdown(
        f"<div class='note'><b>Units check:</b> the CSV stores mass <i>fractions</i> (e.g. Fe 0.6328). "
        f"They are shown here as percent (63.28 %) on a converted copy - the raw file is untouched. "
        f"This is a <b>static spatial</b> dataset (no time axis), so the tool is a composition-to-grade soft sensor, "
        f"not a dynamic one.</div>", unsafe_allow_html=True)
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Fe & chemistry distributions", "Correlations", "Lithotypes", "Data & summary"])
    with t1:
        a, b = st.columns(2)
        a.plotly_chart(fig_fe_hist(df).update_layout(**DARK), use_container_width=True)
        col = b.selectbox("Chemistry variable", C.CHEM_COLS, index=1, key="eda_col")
        b.plotly_chart(fig_feature_hist(df, col).update_layout(**DARK), use_container_width=True)
        st.caption(f"Fe skewness = {eda['fe_skew']:.2f} (strongly left-skewed: a tail of low-grade, high-silica samples).")
    with t2:
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="corr_method")
        a, b = st.columns([3, 2])
        a.plotly_chart(fig_corr_heatmap(df, method).update_layout(**DARK), use_container_width=True)
        corr = eda[f"{method}_with_fe"]
        tbl = pd.DataFrame({"Variable": list(corr), "Correlation with Fe": [round(v, 3) for v in corr.values()]})
        b.dataframe(tbl.sort_values("Correlation with Fe"), hide_index=True, use_container_width=True)
    with t3:
        a, b = st.columns([2, 3])
        counts = pd.DataFrame({"Lithotype": list(eda["lithotype_counts"]),
                               "Samples": list(eda["lithotype_counts"].values())})
        a.dataframe(counts, hide_index=True, use_container_width=True)
        a.markdown("<div class='note'>Basalt (1 sample) and schist (17) are too rare for reliable per-class "
                   "conclusions - treat their results with caution.</div>", unsafe_allow_html=True)
        b.plotly_chart(fig_fe_box(df).update_layout(**DARK), use_container_width=True)
        st.dataframe(pd.DataFrame(eda["fe_by_lithotype"]).T.reset_index().rename(columns={"index": "Lithotype"}),
                     hide_index=True, use_container_width=True)
    with t4:
        st.write("**Descriptive statistics (percent)**")
        st.dataframe(pd.DataFrame(eda["describe_pct"]).T.round(4), use_container_width=True)
        st.write("**Sample rows (percent)**")
        st.dataframe(df.head(10).round(4), use_container_width=True)
        miss = pd.Series(report["missing_by_column"])
        st.caption(f"Missing values: {int(miss.sum())} in total. Source: {C.DATASET_CITATION}")


def page_single(art, df):
    hero("Single Sample Prediction", "Enter a lab/XRF composition and get Fe grade, a 95 % interval and an explanation.",
         ["Hybrid model", "95 % prediction interval", "Additive SHAP waterfall"])
    cutoff = st.session_state["cutoff"]
    lith = st.selectbox("Lithotype", art["categories"], index=art["categories"].index("goethite")
                        if "goethite" in art["categories"] else 0, key="s_lith")
    med = art["median_by_lithotype"].get(lith) or {c: art["feature_ranges"][c]["median"] for c in C.CHEM_COLS}
    st.caption("Composition inputs in **percent (%)**. Defaults are the training median for the chosen lithotype.")
    cols = st.columns(4)
    values = {}
    for i, c in enumerate(C.CHEM_COLS):
        r = art["feature_ranges"][c]
        values[c] = cols[i % 4].number_input(f"{c} (%)", min_value=0.0, max_value=100.0, value=float(round(med[c], 4)),
                                             step=0.01, format="%.4f", key=f"in_{c}_{lith}",
                                             help=f"Training range: {r['min']:.4g} - {r['max']:.4g} %")
    with st.expander("Location on the bench (optional - used by the Digital Twin map)"):
        e_col, n_col = st.columns(2)
        e = e_col.number_input("Easting (m)", value=float(round(df["Easting"].median(), 2)), step=1.0, key="s_e")
        n = n_col.number_input("Northing (m)", value=float(round(df["Northing"].median(), 2)), step=1.0, key="s_n")

    if st.button("Predict Fe grade", key="btn_predict"):
        res = single_prediction({**values, C.CAT_COL: lith}, art)
        res["loc"] = (e, n)
        st.session_state["last_single"] = res

    res = st.session_state.get("last_single")
    if not res:
        st.info("Set the composition and press **Predict Fe grade**.")
        return
    if res["errors"]:
        for er in res["errors"]:
            st.error(er)
        return
    for w in res["warnings"]:
        st.warning(w)
    p = res["prediction"]
    a, b = st.columns([2, 3])
    with a:
        st.markdown(f"<div class='result-card'><div class='s'>Predicted Fe grade</div>"
                    f"<div class='v'>{p['Fe_pred_pct']:.2f} %</div>"
                    f"<div class='s'>95 % interval: {p['Fe_pred_lower95_pct']:.2f} - {p['Fe_pred_upper95_pct']:.2f} %</div></div>",
                    unsafe_allow_html=True)
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Mechanistic baseline", f"{p['Fe_baseline_pct']:.2f} %")
        m2.metric("ML correction", f"{p['Fe_ml_correction_pct']:+.2f} pp")
        decision = grade_decision(p["Fe_pred_pct"], cutoff)
        st.metric(f"Decision @ cut-off {cutoff:.1f} % Fe", decision)
        st.caption("Interval = empirical 95 % quantiles of cross-validated hybrid errors (constant width; "
                   "may be optimistic for rare lithotypes and out-of-range samples).")
    with b:
        st.plotly_chart(waterfall_figure(res["phi"], res["base_value"], res["input"], p["Fe_pred_pct"]),
                        use_container_width=True)

    st.markdown("##### Digital Twin update")
    ctx = nearest_context(df, *res["loc"], k=10)
    d1, d2, d3 = st.columns(3)
    d1.metric("Predicted Fe at location", f"{p['Fe_pred_pct']:.2f} %")
    d2.metric("Mean measured Fe of 10 nearest holes", f"{ctx['mean_measured_fe']:.2f} %",
              f"{p['Fe_pred_pct'] - ctx['mean_measured_fe']:+.2f} pp vs prediction", delta_color="off")
    d3.metric("Mean distance to those holes", f"{ctx['mean_distance_m']:.1f} m")
    if st.button("Pin this sample to the Digital Twin map", key="btn_pin"):
        pts = st.session_state.setdefault("twin_points", [])
        pts.append({"Easting": res["loc"][0], "Northing": res["loc"][1], "Fe_pred_pct": p["Fe_pred_pct"],
                    "Lithotype": lith})
        st.success(f"Pinned ({len(pts)} sample(s) on the map). Open the Digital Twin page.")


def page_performance(art):
    hero("Model Performance", "Held-out test-set results (20 % of samples, never used for fitting or tuning).",
         ["RMSE / MAE / R²", "Per-lithotype", "SHAP importance"])
    metrics, preds = get_metrics(), get_test_predictions()
    if metrics is None or preds is None:
        st.warning("No evaluation reports found. Run `python -m src.train_model` in the backend folder.")
        return
    names = {"baseline": "Mechanistic baseline", "ridge": "Ridge", "xgboost": "XGBoost", "hybrid": "Hybrid (baseline + XGB)"}
    comp = pd.DataFrame(metrics["comparison"]).T
    comp.index = [names[i] for i in comp.index]
    best = comp["rmse"].idxmin()
    h = metrics["comparison"]["hybrid"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hybrid RMSE (pp Fe)", f"{h['rmse']:.3f}")
    c2.metric("Hybrid MAE (pp Fe)", f"{h['mae']:.3f}")
    c3.metric("Hybrid R²", f"{h['r2']:.4f}")
    c4.metric("Interval coverage (target 95 %)", f"{metrics['interval']['test_coverage'] * 100:.1f} %")
    st.markdown(f"<div class='note'>Lowest test RMSE: <b>{best}</b>. Hybrid is not assumed to be best - see the table. "
                f"R² is very high for all models because the oxides and Fe are compositional (closure effect: the "
                f"chemistry components nearly determine Fe), so compare RMSE/MAE rather than R². "
                f"Test n = {metrics['split']['n_test']}.</div>", unsafe_allow_html=True)
    st.write("")
    st.dataframe(comp.rename(columns={"rmse": "RMSE", "mae": "MAE", "r2": "R²", "n": "n"}).round(4),
                 use_container_width=True)

    model_key = st.selectbox("Model for scatter / residual plots", list(names), index=3,
                             format_func=lambda k: names[k], key="perf_model")
    col = {"baseline": "Fe_baseline", "ridge": "Fe_ridge", "xgboost": "Fe_xgb", "hybrid": "Fe_hybrid"}[model_key]
    a, b = st.columns(2)
    fig = px.scatter(preds, x="Fe_actual", y=col, color=C.CAT_COL, color_discrete_map=LITH_COLORS,
                     labels={"Fe_actual": "Measured Fe (%)", col: "Predicted Fe (%)"},
                     title="Predicted vs measured Fe (test set)", opacity=0.8)
    lo, hi = float(preds["Fe_actual"].min()), float(preds["Fe_actual"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1", line=dict(dash="dash", color="#94a3b8")))
    a.plotly_chart(fig.update_layout(**DARK), use_container_width=True)
    resid = preds[col] - preds["Fe_actual"]
    rh = px.histogram(resid, nbins=50, title="Residuals = predicted - measured (pp Fe)",
                      color_discrete_sequence=["#6366f1"], labels={"value": "Residual (pp Fe)"})
    b.plotly_chart(rh.update_layout(showlegend=False, **DARK), use_container_width=True)

    a, b = st.columns(2)
    imp = pd.DataFrame(metrics["shap_importance"]).fillna(0).reset_index().rename(columns={"index": "Feature"})
    imp = imp.melt(id_vars="Feature", var_name="Model", value_name="Mean |SHAP| (pp Fe)")
    order = list(metrics["shap_importance"]["hybrid"])
    fi = px.bar(imp, y="Feature", x="Mean |SHAP| (pp Fe)", color="Model", barmode="group", orientation="h",
                category_orders={"Feature": order[::-1]}, title="Feature importance (mean |SHAP|, test set)",
                color_discrete_sequence=["#2dd4bf", "#f59e0b"])
    a.plotly_chart(fi.update_layout(**DARK), use_container_width=True)
    boot = metrics["rmse_bootstrap_by_lithotype"]
    bx = go.Figure()
    for lith, vals in boot.items():
        bx.add_trace(go.Box(y=vals, name=lith, marker_color=LITH_COLORS.get(lith, "#94a3b8"), boxmean=True))
    bx.update_layout(title="Hybrid RMSE by lithotype (bootstrap, test set)", yaxis_title="RMSE (pp Fe)",
                     showlegend=False, **DARK)
    b.plotly_chart(bx, use_container_width=True)
    st.caption("Boxplot shows 300 bootstrap resamples of the test set per lithotype (lithotypes with fewer than 5 "
               "test samples are listed in the table below only).")

    st.write("**Metrics by lithotype**")
    bl = pd.DataFrame(metrics["by_lithotype"][model_key]).rename(
        columns={"rmse": "RMSE", "mae": "MAE", "r2": "R²"})
    st.dataframe(bl.round(4), hide_index=True, use_container_width=True)

    with st.expander("Do the SHAP explanations agree with domain knowledge?"):
        d = metrics["shap_direction_spearman_hybrid"]
        st.write("Spearman correlation between each input and its SHAP contribution (negative = higher value lowers "
                 "predicted Fe):")
        st.dataframe(pd.DataFrame({"Feature": list(d), "Spearman(value, SHAP)": [round(v, 3) for v in d.values()]}),
                     hide_index=True, use_container_width=True)
        st.write("Higher SiO2 and Al2O3 (gangue) reduce predicted Fe, consistent with dilution. "
                 "Note that every component here lowers Fe because of closure.")
    with st.expander("Model details & selection log"):
        sel = metrics["selection"]
        st.json({k: v for k, v in sel.items() if "grid" not in k})
        st.write("Baseline equation (mass balance, percent units):")
        st.json(metrics["baseline_equation"])
        st.write("Interval coverage by lithotype:")
        st.json(metrics["interval"]["test_coverage_by_lithotype"])


def page_batch(art):
    hero("Batch Upload", "Validate a CSV, predict Fe for every row, download results and check the below-threshold share.",
         ["Schema validation", "CSV download", "Threshold analysis"])
    threshold = st.number_input("Fe threshold (%)", min_value=0.0, max_value=100.0, value=float(st.session_state["cutoff"]),
                                step=0.5, key="batch_threshold")
    required = ", ".join([*C.CHEM_COLS, C.CAT_COL])
    st.caption(f"Required columns: **{required}**. Optional: Fe (measured), Easting, Northing, Hole_id.")
    raw = pd.read_csv(C.DATA_RAW_PATH).head(25) if C.DATA_RAW_PATH.exists() else None
    if raw is not None:
        st.download_button("Download a sample input file (original fraction format)", csv_bytes(raw),
                           file_name="sample_input.csv", mime="text/csv", key="dl_sample")
    up = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
    if up is None:
        st.info("Upload a CSV to begin.")
        return
    try:
        df_in = pd.read_csv(up)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the file as CSV: {exc}")
        return
    guess = detect_units_guess(df_in) if set(C.CHEM_COLS) <= set(df_in.columns) else "percent"
    units = st.radio("Units of the chemistry (and Fe) columns", ["fraction", "percent"],
                     index=["fraction", "percent"].index(guess), horizontal=True, key=f"units_{up.name}",
                     help="'fraction' = 0-1 mass fractions as in the original Windarling file.")
    clean, errors, warnings = validate_input(df_in, art, units)
    st.write(f"Rows read: **{len(df_in):,}** | columns: {list(df_in.columns)}")
    for w in warnings:
        st.warning(w)
    if errors:
        for er in errors:
            st.error(er)
        return
    st.success("Schema validation passed.")
    out_pred = predict_frame(clean, art)
    result = df_in.copy()
    for c in out_pred.columns:
        result[c] = out_pred[c].round(4).values
    if C.TARGET in clean.columns:
        result["Fe_measured_pct"] = clean[C.TARGET].round(4).values
    c1, c2, c3 = st.columns(3)
    below = float((out_pred["Fe_pred_pct"] < threshold).mean())
    c1.metric("Samples predicted", f"{len(result):,}")
    c2.metric(f"Predicted Fe below {threshold:.1f} %", f"{below * 100:.1f} %")
    c3.metric("Mean predicted Fe (%)", f"{out_pred['Fe_pred_pct'].mean():.2f}")
    if "Fe_measured_pct" in result and result["Fe_measured_pct"].notna().any():
        m = result["Fe_measured_pct"].dropna()
        d1, d2 = st.columns(2)
        d1.metric(f"Measured Fe below {threshold:.1f} % (reference)", f"{(m < threshold).mean() * 100:.1f} %")
        err = (out_pred["Fe_pred_pct"] - clean[C.TARGET]).dropna()
        d2.metric("RMSE vs measured (pp Fe)", f"{np.sqrt((err ** 2).mean()):.3f}")
        sc = px.scatter(x=clean[C.TARGET], y=out_pred["Fe_pred_pct"], labels={"x": "Measured Fe (%)", "y": "Predicted Fe (%)"},
                        title="Predicted vs measured (uploaded data)")
        lo, hi = float(m.min()), float(m.max())
        sc.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="#94a3b8"), name="1:1"))
        st.plotly_chart(sc.update_layout(**DARK), use_container_width=True)
    hist = px.histogram(out_pred, x="Fe_pred_pct", nbins=40, title="Distribution of predicted Fe (%)",
                        color_discrete_sequence=["#2dd4bf"], labels={"Fe_pred_pct": "Predicted Fe (%)"})
    hist.add_vline(x=threshold, line_dash="dash", line_color="#f87171", annotation_text="threshold")
    st.plotly_chart(hist.update_layout(**DARK), use_container_width=True)
    st.dataframe(result.head(500), use_container_width=True)
    st.download_button("Download predictions (CSV)", csv_bytes(result), file_name="fe_predictions.csv",
                       mime="text/csv", key="dl_result")
    st.caption("Columns ending in _pct are in percent. 'Fe_pred_*' are model predictions; 'Fe_measured_pct' (if present) is the "
               "uploaded measurement.")
    if {"Easting", "Northing"} <= set(df_in.columns):
        pts = pd.DataFrame({"Easting": pd.to_numeric(df_in["Easting"], errors="coerce"),
                            "Northing": pd.to_numeric(df_in["Northing"], errors="coerce"),
                            "Fe_pred_pct": out_pred["Fe_pred_pct"].values}).dropna().head(2000)
        st.session_state["batch_points"] = pts
        st.info(f"{len(pts):,} located samples were sent to the Digital Twin map.")


def page_twin(art, df):
    hero("Lightweight Geological Digital Twin",
         "A static spatial ore-quality model of the Windarling bench: data -> model -> decision.",
         ["Prototype", "Not an equipment-level or dynamic twin", "Easting / Northing map"])
    cutoff = st.session_state["cutoff"]
    preds = get_test_predictions()
    layers = ["Measured Fe (all drill samples)"]
    if preds is not None:
        layers += ["Hybrid-predicted Fe (test samples)", "Absolute error |predicted - measured| (test samples)"]
    layer = st.radio("Map layer", layers, horizontal=True, key="twin_layer")

    pins = pd.DataFrame(st.session_state.get("twin_points", []))
    batch = st.session_state.get("batch_points")
    new_pts = pd.concat([x for x in (pins, batch) if x is not None and len(x)], ignore_index=True) \
        if (len(pins) or (batch is not None and len(batch))) else None

    if layer.startswith("Measured"):
        fig = spatial_figure(df, new_pts)
    elif layer.startswith("Hybrid"):
        ref = preds.rename(columns={"Fe_hybrid": "val"})
        fig = spatial_figure(ref, new_pts, color_col="val", title="Hybrid-predicted Fe (held-out test samples)",
                             value_label="Predicted Fe", trace_name="Predicted (test samples)")
    else:
        ref = preds.assign(val=(preds["Fe_hybrid"] - preds["Fe_actual"]).abs())
        fig = spatial_figure(ref, None, color_col="val", title="Absolute prediction error (held-out test samples)",
                             value_label="|Error|", trace_name="|Error| (test samples)", colorscale="Magma")
    st.plotly_chart(fig, use_container_width=True)

    a, b, c = st.columns(3)
    a.metric("Bench samples", f"{len(df):,}")
    share = float((df[C.TARGET] >= cutoff).mean())
    b.metric(f"Measured share >= {cutoff:.1f} % Fe", f"{share * 100:.1f} %")
    c.metric("New samples on map", 0 if new_pts is None else len(new_pts))
    if new_pts is not None:
        dec = new_pts.assign(Decision=new_pts["Fe_pred_pct"].apply(lambda v: grade_decision(v, cutoff)))
        st.write("**Estimated ore quality of new samples (predicted, not measured)**")
        st.dataframe(dec.head(200).round(3), hide_index=True, use_container_width=True)
        if st.button("Clear new samples from map", key="btn_clear"):
            st.session_state["twin_points"] = []
            st.session_state.pop("batch_points", None)
            st.rerun()
    else:
        st.info("Pin a sample from **Single Sample Prediction**, or upload located data in **Batch Upload**, "
                "to see the twin update.")
    st.caption("Static snapshots of the ore body (no time dimension). Measured values come from drilling/lab analysis; "
               "new points show *model-predicted* Fe.")


# --------------------------------------------------------------------------- main
def main():
    art = get_artifact()
    df, report = get_data()

    with st.sidebar:
        st.markdown("## ⛏️ Fe Soft Sensor")
        st.caption("Windarling iron-ore bench · Western Australia")
        page = st.radio("Navigate", ["Overview & EDA", "Single Sample Prediction", "Model Performance",
                                     "Batch Upload", "Digital Twin Map"], key="nav")
        st.divider()
        st.number_input("Fe cut-off for decisions (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.5,
                        key="cutoff")
        st.caption(f"Model trained on {art['meta']['n_train']:,} samples (composition + lithotype; "
                   f"coordinates excluded).")
        st.caption(C.DATASET_CITATION)

    if page == "Overview & EDA":
        page_overview(df, report)
    elif page == "Single Sample Prediction":
        page_single(art, df)
    elif page == "Model Performance":
        page_performance(art)
    elif page == "Batch Upload":
        page_batch(art)
    else:
        page_twin(art, df)


main()
