# app/streamlit_app.py
# ============================================================
# ECG-Based Sleep Apnea Screening
# Lightweight research prototype
#
# Run from project root:
#     streamlit run app/streamlit_app.py
#
# Required:
#     streamlit
#     pandas
#     numpy
#     plotly
#     scipy
#
# Optional:
#     wfdb
#
# NOTE:
# This application is a research/screening prototype.
# It is NOT a medical diagnostic system.
# ============================================================

from __future__ import annotations

import io
import json
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scipy.signal import butter, filtfilt, find_peaks

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import wfdb

    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ECG Sleep Apnea Screening",
    page_icon="♥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
COMPARISON_DIR = RESULTS_DIR / "model_comparison"
COMPLEXITY_DIR = RESULTS_DIR / "complexity"
EXPLAINABILITY_DIR = RESULTS_DIR / "explainability"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

MODELS_DIR = ROOT / "models" / "trained"

RAW_DIR = DATA_DIR / "raw"


# ============================================================
# THEME / CSS
# ============================================================

st.markdown(
    """
    <style>
    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #F5F8FC;
        color: #111827;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background: #0B2D4D;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.20);
    }

    /* ========================================================
       HEADINGS
       ======================================================== */

    h1, h2, h3, h4 {
        color: #0B2D4D !important;
    }

    p, li, label {
        color: #111827;
    }

    /* ========================================================
       METRIC CARDS
       ======================================================== */

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D8E2EE;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(11,45,77,0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #536579 !important;
    }

    [data-testid="stMetricValue"] {
        color: #0B2D4D !important;
    }

    /* ========================================================
       INPUTS
       ======================================================== */

    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border-color: #C8D5E3 !important;
    }

    div[data-baseweb="select"] * {
        color: #111827 !important;
    }

    input {
        color: #111827 !important;
        background: #FFFFFF !important;
    }

    textarea {
        color: #111827 !important;
        background: #FFFFFF !important;
    }

    /* ========================================================
       TABS
       ======================================================== */

    button[data-baseweb="tab"] {
        color: #536579 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0B2D4D !important;
        font-weight: 700;
    }

    /* ========================================================
       DATAFRAMES
       ======================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid #D8E2EE;
        border-radius: 10px;
        background: #FFFFFF !important;
    }

    [data-testid="stDataFrame"] [role="grid"],
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] [role="columnheader"] {
        background: #FFFFFF !important;
        color: #111111 !important;
    }

    /* ========================================================
       EXPERIMENT / INFO CARDS
       ======================================================== */

    .experiment-card {
        background: #FFFFFF;
        border: 1px solid #D8E2EE;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 92px;
        box-shadow: 0 3px 12px rgba(11,45,77,0.06);
        margin-bottom: 0.8rem;
    }

    .experiment-card .label {
        color: #536579;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .experiment-card .value {
        color: #111111;
        font-size: 1.15rem;
        font-weight: 750;
        line-height: 1.25;
    }

    /* Plotly charts are rendered on white surfaces with dark text. */
    .js-plotly-plot, .plot-container {
        background: #FFFFFF !important;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        background: #0B2D4D;
        color: #FFFFFF !important;
        border: 1px solid #0B2D4D;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #123E66;
        color: #FFFFFF !important;
        border-color: #123E66;
    }

    /* ========================================================
       DOWNLOAD BUTTON
       ======================================================== */

    .stDownloadButton > button {
        background: #0B2D4D;
        color: #FFFFFF !important;
        border: 1px solid #0B2D4D;
        border-radius: 8px;
        font-weight: 700;
        min-height: 3rem;
    }

    .stDownloadButton > button:hover {
        background: #123E66;
        color: #FFFFFF !important;
    }

    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border-radius: 10px;
    }

    /* ========================================================
       EXPANDERS
       ======================================================== */

    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #D8E2EE;
        border-radius: 10px;
    }

    /* ========================================================
       CUSTOM DARK BLUE PANEL
       ======================================================== */

    .dark-panel {
        background: #0B2D4D;
        color: #FFFFFF;
        border-radius: 14px;
        padding: 1.5rem;
        margin: 0.5rem 0 1rem 0;
    }

    .dark-panel h1,
    .dark-panel h2,
    .dark-panel h3,
    .dark-panel p,
    .dark-panel li {
        color: #FFFFFF !important;
    }

    /* ========================================================
       STATUS BADGES
       ======================================================== */

    .status-positive {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: #E8F3FF;
        color: #0B2D4D;
        font-weight: 700;
        border: 1px solid #BBD3E8;
    }

    .status-negative {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: #F4F7FA;
        color: #26384A;
        font-weight: 700;
        border: 1px solid #D8E2EE;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-note {
        text-align: center;
        color: #66788A;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #D8E2EE;
    }

    /* ========================================================
       POLISHED RESEARCH UI
       ======================================================== */
    .eyebrow,.section-kicker{color:#2563A6!important;font-size:.78rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.35rem}
    .hero-panel{background:linear-gradient(135deg,#0B2D4D 0%,#123F68 100%);color:#FFF;border-radius:20px;padding:2.4rem 2.6rem;margin:.5rem 0 1.25rem;box-shadow:0 12px 30px rgba(11,45,77,.14)}
    .hero-panel *{color:#FFF!important}.hero-kicker{font-size:.76rem;font-weight:800;letter-spacing:.14em;opacity:.78}.hero-title{font-size:2.25rem;line-height:1.1;font-weight:800;margin:.55rem 0 .8rem}.hero-text{max-width:880px;font-size:1.03rem;line-height:1.65;opacity:.92}.hero-actions{display:flex;flex-wrap:wrap;gap:.55rem;margin-top:1.25rem}.hero-chip{border:1px solid rgba(255,255,255,.28);background:rgba(255,255,255,.08);padding:.42rem .72rem;border-radius:999px;font-size:.78rem;font-weight:700}
    .notice-line{background:#EAF3FB;color:#0B2D4D;border:1px solid #C7DCEC;border-radius:10px;padding:.7rem 1rem;font-weight:650}
    .feature-card,.info-card,.workflow-card,.gap-card,.module-card,.team-card{background:#FFF;border:1px solid #D8E2EE;border-radius:16px;padding:1.2rem;box-shadow:0 4px 14px rgba(11,45,77,.055);height:100%;transition:transform .18s ease,box-shadow .18s ease}.feature-card:hover,.info-card:hover,.workflow-card:hover,.module-card:hover{transform:translateY(-2px);box-shadow:0 10px 24px rgba(11,45,77,.09)}.feature-card{min-height:190px;border-top:4px solid #2563A6}.info-card{min-height:175px}.workflow-card{min-height:155px}.gap-card,.module-card{display:flex;gap:1rem;align-items:flex-start;margin-bottom:1rem;min-height:125px}.team-card{min-height:90px}.team-card b,.team-card span{display:block}.team-card span{color:#66788A;font-size:.82rem;margin-top:.3rem}
    .card-number,.workflow-num,.module-num{display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:50%;background:#E8F3FF;color:#0B2D4D;font-weight:800;font-size:.78rem}.feature-card h3,.info-card h3,.workflow-card h3,.gap-card h3,.module-card h3{margin:.7rem 0 .4rem}.feature-card p,.info-card p,.workflow-card p,.gap-card p,.module-card p{color:#536579;line-height:1.55;margin:0}.card-link{color:#2563A6;font-weight:800;font-size:.75rem;margin-top:1rem;letter-spacing:.06em}
    .research-note,.response-box{background:#FFF;border:1px solid #D8E2EE;border-left:4px solid #2563A6;border-radius:12px;padding:1rem 1.15rem;margin-top:1.1rem;color:#536579}.response-title{color:#0B2D4D;font-weight:800;font-size:.8rem;letter-spacing:.08em;margin-bottom:.8rem}.response-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem}.response-grid div{background:#F5F8FC;border-radius:10px;padding:.8rem}.response-grid b,.response-grid span{display:block}.response-grid b{color:#0B2D4D}.response-grid span{color:#66788A;font-size:.8rem;margin-top:.25rem}
    .comparison-strip{display:grid;grid-template-columns:1fr auto 1fr;gap:1.25rem;align-items:stretch;background:#FFF;border:1px solid #D8E2EE;border-radius:16px;padding:1.25rem}.comparison-side{background:#F7FAFD;border-radius:12px;padding:1.1rem}.comparison-title{font-size:.75rem;font-weight:800;letter-spacing:.08em;color:#0B2D4D;margin-bottom:.8rem}.flow-item{background:#EAF3FB;color:#0B2D4D;border-radius:9px;padding:.65rem .75rem;font-weight:650;text-align:center}.flow-item.muted{background:#EEF2F6;color:#536579}.flow-arrow{text-align:center;color:#2563A6;font-weight:800;padding:.25rem}.comparison-divider{align-self:center;font-size:.8rem;font-weight:900;color:#66788A}
    .architecture-flow{display:flex;align-items:center;gap:.45rem;flex-wrap:wrap;background:#FFF;border:1px solid #D8E2EE;border-radius:16px;padding:1.25rem}.architecture-node{display:flex;align-items:center;gap:.5rem;background:#F7FAFD;border:1px solid #D8E2EE;border-radius:10px;padding:.65rem .75rem}.architecture-node span{color:#2563A6;font-size:.72rem;font-weight:800}.architecture-node b{color:#0B2D4D;font-size:.82rem;white-space:nowrap}.architecture-arrow{color:#2563A6;font-weight:900}
    .sidebar-brand{display:flex;align-items:center;gap:.7rem;padding:.35rem 0 .7rem}.sidebar-brand-mark{width:38px;height:38px;display:flex;align-items:center;justify-content:center;border-radius:11px;background:#FFF;color:#0B2D4D!important;font-size:1.15rem;font-weight:900}.sidebar-brand-title{font-weight:850;letter-spacing:.08em;font-size:.98rem;color:#FFF!important}.sidebar-brand-subtitle{font-weight:700;letter-spacing:.16em;font-size:.66rem;color:rgba(255,255,255,.7)!important}.sidebar-prototype-badge{color:#DCEBFA!important;font-size:.68rem;letter-spacing:.08em;font-weight:800;margin:.1rem 0 .2rem}.sidebar-section-label{color:rgba(255,255,255,.55)!important;font-size:.68rem;font-weight:800;letter-spacing:.14em;margin:.4rem 0 .45rem}
    @media(max-width:900px){.response-grid{grid-template-columns:repeat(2,1fr)}.comparison-strip{grid-template-columns:1fr}.comparison-divider{text-align:center}.hero-title{font-size:1.8rem}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================


def path_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path_exists(path):
            return path
    return None


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def pct(value):
    if pd.isna(value):
        return "—"
    return f"{float(value) * 100:.2f}%"


def number(value):
    if pd.isna(value):
        return "—"
    return f"{int(value):,}"


def short_number(value):
    if pd.isna(value):
        return "—"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:.0f}"


def clean_name(value):
    return str(value).strip()


def normalize_record_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    aliases = [
        "group",
        "record",
        "record_name",
        "record_id",
        "subject",
        "subject_id",
    ]

    for col in aliases:
        if col in df.columns:
            if col != "group":
                df["group"] = df[col]
            break

    if "group" not in df.columns:
        df["group"] = "Unknown"

    return df


def normalize_prediction_df(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df = normalize_record_column(df)

    df = df.copy()

    prediction_aliases = [
        "prediction",
        "y_pred",
        "predicted",
        "pred",
    ]

    probability_aliases = [
        "probability",
        "y_prob",
        "prob",
        "score",
    ]

    true_aliases = [
        "y_true",
        "label",
        "true",
        "target",
    ]

    pred_col = None
    prob_col = None
    true_col = None

    for col in prediction_aliases:
        if col in df.columns:
            pred_col = col
            break

    for col in probability_aliases:
        if col in df.columns:
            prob_col = col
            break

    for col in true_aliases:
        if col in df.columns:
            true_col = col
            break

    if pred_col is None:
        df["prediction"] = 0
    else:
        df["prediction"] = pd.to_numeric(
            df[pred_col], errors="coerce"
        ).fillna(0).astype(int)

    if prob_col is None:
        df["probability"] = df["prediction"].astype(float)
    else:
        df["probability"] = pd.to_numeric(
            df[prob_col], errors="coerce"
        ).fillna(0.0)

    if true_col is None:
        df["y_true"] = np.nan
    else:
        df["y_true"] = pd.to_numeric(
            df[true_col], errors="coerce"
        )

    df["model"] = model_name
    df["window_index"] = np.arange(len(df))

    return df[
        [
            "window_index",
            "group",
            "y_true",
            "probability",
            "prediction",
            "model",
        ]
    ]


# ============================================================
# LOAD DATA
# ============================================================


@st.cache_data
def load_csv(path_string: str) -> pd.DataFrame:
    path = Path(path_string)

    if not path_exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_json(path_string: str) -> dict:
    path = Path(path_string)

    if not path_exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data
def load_numpy(path_string: str):
    path = Path(path_string)

    if not path_exists(path):
        return None

    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


@st.cache_data
def load_predictions() -> Dict[str, pd.DataFrame]:
    files = {
        "Random Forest": PREDICTIONS_DIR / "random_forest_predictions.csv",
        "1D CNN": PREDICTIONS_DIR / "cnn_predictions.csv",
        "Transformer V1": PREDICTIONS_DIR / "transformer_predictions.csv",
        "Proposed Transformer V2": PREDICTIONS_DIR / "proposed_transformer_v2_predictions.csv",
    }

    output = {}

    for model, path in files.items():
        if path_exists(path):
            raw = load_csv(str(path))
            if not raw.empty:
                output[model] = normalize_prediction_df(raw, model)

    return output


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    candidates = [
        COMPARISON_DIR / "complete_model_comparison.csv",
        COMPARISON_DIR / "final_evaluation_summary.csv",
    ]

    path = first_existing(candidates)

    if path is None:
        return pd.DataFrame()

    return load_csv(str(path))


@st.cache_data
def load_complexity() -> pd.DataFrame:
    candidates = [
        COMPARISON_DIR / "final_model_complexity.csv",
        COMPLEXITY_DIR / "deep_model_complexity.csv",
    ]

    path = first_existing(candidates)

    if path is None:
        return pd.DataFrame()

    return load_csv(str(path))


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    path = EXPLAINABILITY_DIR / "random_forest_feature_importance.csv"

    if not path_exists(path):
        return pd.DataFrame()

    return load_csv(str(path))


# ============================================================
# TABLE DISPLAY
# ============================================================


def display_dataframe(df: pd.DataFrame, **kwargs):
    """Render tables with a white background and black text.

    Streamlit's theme can otherwise make dataframe cells appear dark.
    Pandas Styler keeps research tables readable and presentation-ready.
    """
    if df is None:
        return

    styled = (
        df.style
        .set_properties(
            **{
                "background-color": "#FFFFFF",
                "color": "#111111",
                "border-color": "#D8E2EE",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#EAF2F8"),
                        ("color", "#111111"),
                        ("font-weight", "700"),
                        ("border-color", "#C8D5E3"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("background-color", "#FFFFFF"),
                        ("color", "#111111"),
                    ],
                },
            ]
        )
    )

    st.dataframe(styled, **kwargs)


# ============================================================
# DATASET SUMMARY
# ============================================================


def get_dataset_summary():
    ml_ready = load_csv(str(PROCESSED_DIR / "ml_ready_dataset.csv"))
    rpeak = load_csv(str(PROCESSED_DIR / "rpeak_summary.csv"))
    rr_hrv = load_csv(str(PROCESSED_DIR / "rr_hrv_summary.csv"))

    summary = {
        "learning_records": 34,
        "raw_epochs": 16603,
        "processed_epochs": 16570,
        "ml_ready_epochs": len(ml_ready) if not ml_ready.empty else 16528,
        "temporal_windows": 16361,
        "subjects": 34,
        "features": 11,
    }

    if not ml_ready.empty:
        if "group" in ml_ready.columns:
            summary["subjects"] = ml_ready["group"].nunique()

        feature_candidates = [
            c
            for c in ml_ready.columns
            if c
            not in {
                "group",
                "record",
                "epoch",
                "epoch_index",
                "label",
                "y",
                "target",
            }
        ]

        numeric_count = len(
            ml_ready[feature_candidates].select_dtypes(
                include=np.number
            ).columns
        )

        if numeric_count:
            summary["features"] = numeric_count

    if not rpeak.empty:
        possible_epoch_cols = [
            "total_epochs",
            "epochs",
            "n_epochs",
            "raw_epochs",
        ]

        for col in possible_epoch_cols:
            if col in rpeak.columns:
                value = pd.to_numeric(
                    rpeak[col], errors="coerce"
                ).sum()

                if not pd.isna(value) and value > 0:
                    summary["processed_epochs"] = int(value)
                    break

    return summary, ml_ready, rpeak, rr_hrv


# ============================================================
# METRIC TABLE
# ============================================================


METRIC_ALIASES = {
    "Accuracy": ["accuracy", "acc"],
    "Balanced Accuracy": ["balanced_accuracy", "balanced acc"],
    "Precision": ["precision", "prec"],
    "Sensitivity": ["sensitivity", "recall", "sens"],
    "Specificity": ["specificity", "spec"],
    "F1 Score": ["f1", "f1_score"],
    "ROC-AUC": ["roc_auc", "roc-auc", "auc"],
    "PR-AUC": ["pr_auc", "pr-auc", "average_precision"],
}


# Canonical 34 labelled learning records used in the project.
# c05 is excluded because it is a duplicate of c06.
EXPECTED_LEARNING_RECORDS = (
    [f"a{i:02d}" for i in range(1, 21)]
    + [f"b{i:02d}" for i in range(1, 6)]
    + [f"c{i:02d}" for i in range(1, 5)]
    + [f"c{i:02d}" for i in range(6, 11)]
)

EXPECTED_TEST_RECORDS = [f"x{i:02d}" for i in range(1, 36)]


def find_metric_column(df: pd.DataFrame, metric_name: str):
    candidates = METRIC_ALIASES.get(metric_name, [])

    normalized = {
        str(c).lower().replace("-", "_").replace(" ", "_"): c
        for c in df.columns
    }

    for candidate in candidates:
        key = (
            candidate.lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        if key in normalized:
            return normalized[key]

    return None


def canonical_model_name(value):
    value = str(value).strip().lower()

    if "random" in value or value == "rf":
        return "Random Forest"

    if "cnn" in value:
        return "1D CNN"

    if "transformer" in value:
        return "Transformer V1"

    return str(value)


def prepare_comparison_table():
    df = load_model_comparison()

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    model_col = None

    for col in [
        "model",
        "Model",
        "model_name",
        "Model Name",
    ]:
        if col in df.columns:
            model_col = col
            break

    if model_col is None:
        return pd.DataFrame()

    df["Model"] = df[model_col].map(canonical_model_name)

    records = []

    for _, row in df.iterrows():
        record = {"Model": row["Model"]}

        for metric in METRIC_ALIASES:
            col = find_metric_column(
                df.loc[[row.name]],
                metric,
            )

            if col is not None:
                record[metric] = safe_float(row[col])
            else:
                record[metric] = np.nan

        records.append(record)

    result = pd.DataFrame(records)

    desired_order = [
        "Random Forest",
        "1D CNN",
        "Transformer V1",
    ]

    result["order"] = result["Model"].map(
        {name: i for i, name in enumerate(desired_order)}
    )

    result = (
        result.sort_values("order")
        .drop(columns="order")
        .drop_duplicates("Model")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# PERFORMANCE FROM PREDICTIONS
# ============================================================


def calculate_metrics_from_predictions(df: pd.DataFrame):
    if df.empty or df["y_true"].isna().all():
        return {}

    y_true = df["y_true"].dropna().astype(int).to_numpy()

    aligned = df.loc[df["y_true"].notna()].copy()

    y_pred = aligned["prediction"].astype(int).to_numpy()
    y_prob = aligned["probability"].astype(float).to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = len(y_true)

    accuracy = (
        (tp + tn) / total
        if total
        else np.nan
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else np.nan
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else np.nan
    )

    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if not pd.isna(precision)
        and not pd.isna(sensitivity)
        and (precision + sensitivity) > 0
        else np.nan
    )

    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if not pd.isna(sensitivity)
        and not pd.isna(specificity)
        else np.nan
    )

    roc_auc = np.nan
    pr_auc = np.nan

    if PLOTLY_AVAILABLE:
        try:
            from sklearn.metrics import (
                average_precision_score,
                roc_auc_score,
            )

            if len(np.unique(y_true)) == 2:
                roc_auc = roc_auc_score(
                    y_true,
                    y_prob,
                )

                pr_auc = average_precision_score(
                    y_true,
                    y_prob,
                )
        except Exception:
            pass

    return {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def performance_from_available_data():
    comparison = prepare_comparison_table()

    if not comparison.empty:
        return comparison

    predictions = load_predictions()

    records = []

    for model, df in predictions.items():
        metrics = calculate_metrics_from_predictions(df)

        row = {"Model": model}

        for metric in METRIC_ALIASES:
            row[metric] = metrics.get(metric, np.nan)

        records.append(row)

    return pd.DataFrame(records)


# ============================================================
# PLOT HELPERS
# ============================================================


def apply_plotly_layout(fig, title=None, height=450):
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial",
            color="#111111",
        ),
        title_font=dict(
            family="Arial",
            color="#0B2D4D",
        ),
        xaxis=dict(
            title_font=dict(color="#111111"),
            tickfont=dict(color="#111111"),
        ),
        yaxis=dict(
            title_font=dict(color="#111111"),
            tickfont=dict(color="#111111"),
        ),
        margin=dict(
            l=50,
            r=30,
            t=70 if title else 30,
            b=50,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            font=dict(color="#111111"),
        ),
        annotationdefaults=dict(
            font=dict(color="#111111"),
        ),
        uniformtext=dict(
            minsize=10,
            mode="show",
        ),
    )

    # Keep every in-chart data label readable on the white canvas.
    try:
        fig.update_traces(textfont=dict(color="#111111"))
    except Exception:
        pass

    return fig


def plot_metric_bars(
    df,
    metric,
    models=None,
    title=None,
):
    if df.empty:
        return None

    data = df.copy()

    if models:
        data = data[data["Model"].isin(models)]

    data = data.dropna(subset=[metric])

    if data.empty:
        return None

    fig = px.bar(
        data,
        x="Model",
        y=metric,
        text=data[metric].map(
            lambda x: f"{x * 100:.2f}%"
        ),
    )

    fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%",
        title=metric,
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#111111"),
    )

    return apply_plotly_layout(
        fig,
        title or f"{metric} — Model Comparison",
        450,
    )


def plot_all_metrics_grouped(
    df,
    model,
):
    if df.empty:
        return None

    row = df[df["Model"] == model]

    if row.empty:
        return None

    metrics = list(METRIC_ALIASES.keys())

    values = []

    for metric in metrics:
        value = safe_float(row.iloc[0][metric])

        if not pd.isna(value):
            values.append(
                {
                    "Metric": metric,
                    "Value": value,
                }
            )

    data = pd.DataFrame(values)

    if data.empty:
        return None

    fig = px.bar(
        data,
        x="Metric",
        y="Value",
        text=data["Value"].map(
            lambda x: f"{x * 100:.2f}%"
        ),
    )

    fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%",
        title="Score",
    )

    fig.update_xaxes(
        tickangle=-35,
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#111111"),
    )

    return apply_plotly_layout(
        fig,
        f"{model} — All Evaluation Metrics",
        500,
    )


def plot_heatmap(df):
    if df.empty:
        return None

    metrics = list(METRIC_ALIASES.keys())

    z = df[metrics].to_numpy(dtype=float)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=metrics,
            y=df["Model"],
            zmin=0,
            zmax=1,
            colorscale="RdYlGn",
            text=np.where(
                np.isnan(z),
                "",
                np.vectorize(
                    lambda x: f"{x:.3f}"
                )(z),
            ),
            texttemplate="%{text}",
            textfont=dict(color="#111111"),
            hovertemplate=(
                "Model: %{y}<br>"
                "Metric: %{x}<br>"
                "Score: %{z:.3f}"
                "<extra></extra>"
            ),
        )
    )

    return apply_plotly_layout(
        fig,
        "All Models × All Metrics",
        500,
    )



def plot_all_models_all_metrics_grouped(df):
    """Grouped column chart with all metrics kept together per model."""
    if df.empty:
        return None

    metrics = list(METRIC_ALIASES.keys())
    model_order = ["Random Forest", "1D CNN", "Transformer V1"]
    model_colors = {
        "Random Forest": "#2E7D32",
        "1D CNN": "#F39C12",
        "Transformer V1": "#D84315",
    }

    fig = go.Figure()
    group_gap = 1.5
    x_labels = []

    for model_index, model in enumerate(model_order):
        row = df[df["Model"] == model]
        if row.empty:
            continue

        start = model_index * (len(metrics) + group_gap)
        x_positions = [start + i for i in range(len(metrics))]
        values = [safe_float(row.iloc[0].get(metric)) for metric in metrics]
        labels = [f"{model} — {metric}" for metric in metrics]
        x_labels.extend(zip(x_positions, labels))

        fig.add_trace(
            go.Bar(
                x=x_positions,
                y=values,
                name=model,
                marker_color=model_colors[model],
                text=[
                    f"{v * 100:.2f}%" if not pd.isna(v) else "—"
                    for v in values
                ],
                textposition="outside",
                textfont=dict(color="#111111"),
                hovertemplate=(
                    "Model: " + model
                    + "<br>Metric: %{customdata}"
                    + "<br>Score: %{y:.2%}<extra></extra>"
                ),
                customdata=metrics,
            )
        )

    if not fig.data:
        return None

    tick_positions = [x for x, _ in x_labels]
    tick_text = [label.split(" — ", 1)[1] for _, label in x_labels]

    # Add light separators between model groups.
    shapes = []
    for i in range(1, len(model_order)):
        x = i * (len(metrics) + group_gap) - group_gap / 2
        shapes.append(
            dict(
                type="line",
                x0=x, x1=x, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="#CBD5E1", dash="dot"),
            )
        )

    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_positions,
        ticktext=tick_text,
        tickangle=-35,
        title="Evaluation metric — grouped by model",
    )
    fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%",
        title="Score",
    )
    fig.update_layout(
        barmode="group",
        bargap=0.12,
        shapes=shapes,
    )

    return apply_plotly_layout(
        fig,
        "All Models × All Metrics — Grouped Column Comparison",
        560,
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================


def confusion_counts(df):
    metrics = calculate_metrics_from_predictions(df)

    return {
        "TN": metrics.get("TN", 0),
        "FP": metrics.get("FP", 0),
        "FN": metrics.get("FN", 0),
        "TP": metrics.get("TP", 0),
    }


def plot_confusion_matrix(df, model):
    counts = confusion_counts(df)

    z = [
        [counts["TN"], counts["FP"]],
        [counts["FN"], counts["TP"]],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted Non-Apnea", "Predicted Apnea"],
            y=["Actual Non-Apnea", "Actual Apnea"],
            text=[
                [
                    f"TN<br>{counts['TN']:,}",
                    f"FP<br>{counts['FP']:,}",
                ],
                [
                    f"FN<br>{counts['FN']:,}",
                    f"TP<br>{counts['TP']:,}",
                ],
            ],
            texttemplate="%{text}",
            textfont=dict(color="#111111"),
            hovertemplate=(
                "Actual: %{y}<br>"
                "Prediction: %{x}<br>"
                "Count: %{z:,}"
                "<extra></extra>"
            ),
            colorscale="YlOrRd",
        )
    )

    return apply_plotly_layout(
        fig,
        f"{model} — Confusion Matrix",
        450,
    )


# ============================================================
# ROC / PR
# ============================================================


def roc_pr_data(df):
    if df.empty or df["y_true"].isna().all():
        return None, None

    data = df.dropna(subset=["y_true"]).copy()

    y_true = data["y_true"].astype(int).to_numpy()
    y_prob = data["probability"].astype(float).to_numpy()

    if len(np.unique(y_true)) < 2:
        return None, None

    try:
        from sklearn.metrics import (
            precision_recall_curve,
            roc_curve,
        )

        fpr, tpr, _ = roc_curve(
            y_true,
            y_prob,
        )

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_prob,
        )

        return (
            (fpr, tpr),
            (recall, precision),
        )

    except Exception:
        return None, None


def plot_roc(models_data, selected_models):
    fig = go.Figure()

    for model in selected_models:
        df = models_data.get(model)

        if df is None:
            continue

        roc_data, _ = roc_pr_data(df)

        if roc_data is None:
            continue

        fpr, tpr = roc_data

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=model,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(
                dash="dash",
            ),
        )
    )

    fig.update_xaxes(
        title="False Positive Rate",
        range=[0, 1],
    )

    fig.update_yaxes(
        title="True Positive Rate",
        range=[0, 1],
    )

    return apply_plotly_layout(
        fig,
        "Receiver Operating Characteristic",
        500,
    )


def plot_pr(models_data, selected_models):
    fig = go.Figure()

    for model in selected_models:
        df = models_data.get(model)

        if df is None:
            continue

        _, pr_data = roc_pr_data(df)

        if pr_data is None:
            continue

        recall, precision = pr_data

        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=model,
            )
        )

    fig.update_xaxes(
        title="Recall / Sensitivity",
        range=[0, 1],
    )

    fig.update_yaxes(
        title="Precision",
        range=[0, 1],
    )

    return apply_plotly_layout(
        fig,
        "Precision–Recall Curves",
        500,
    )


def plot_confusion_counts_grouped(predictions, selected_models=None):
    """Compare TP, TN, FP and FN with model groups side-by-side."""
    model_order = ["Random Forest", "1D CNN", "Transformer V1"]
    if selected_models:
        model_order = [m for m in model_order if m in selected_models]

    rows = []
    for model in model_order:
        if model not in predictions:
            continue
        counts = confusion_counts(predictions[model])
        rows.append({
            "Model": model,
            "TP": counts["TP"],
            "TN": counts["TN"],
            "FP": counts["FP"],
            "FN": counts["FN"],
        })

    if not rows:
        return None

    data = pd.DataFrame(rows)
    measures = ["TP", "TN", "FP", "FN"]
    model_colors = {
        "Random Forest": "#2E7D32",
        "1D CNN": "#F39C12",
        "Transformer V1": "#D84315",
    }

    fig = go.Figure()
    group_gap = 0.65
    x_positions = []
    x_text = []

    for model_index, model in enumerate(model_order):
        row = data[data["Model"] == model]
        if row.empty:
            continue
        start = model_index * (len(measures) + group_gap)
        xs = [start + i for i in range(len(measures))]
        vals = [int(row.iloc[0][m]) for m in measures]
        x_positions.extend(xs)
        x_text.extend(measures)
        fig.add_trace(go.Bar(
            x=xs,
            y=vals,
            name=model,
            marker_color=model_colors[model],
            text=[f"{v:,}" for v in vals],
            textposition="outside",
            textfont=dict(color="#111111"),
            customdata=measures,
            hovertemplate=(
                "Model: " + model
                + "<br>Measure: %{customdata}"
                + "<br>Count: %{y:,}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="group",
        bargap=0.12,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_positions,
        ticktext=x_text,
        title="Confusion count — grouped by model",
    )
    fig.update_yaxes(title="Window count")

    return apply_plotly_layout(
        fig,
        "TP / TN / FP / FN — Model Comparison",
        520,
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================


def build_error_summary(predictions):
    rows = []

    for model, df in predictions.items():
        metrics = calculate_metrics_from_predictions(df)

        rows.append(
            {
                "Model": model,
                "Correct": int(
                    metrics.get("TP", 0)
                    + metrics.get("TN", 0)
                ),
                "False Positives": int(
                    metrics.get("FP", 0)
                ),
                "False Negatives": int(
                    metrics.get("FN", 0)
                ),
                "Total": len(df),
            }
        )

    return pd.DataFrame(rows)


def plot_error_summary(error_df):
    if error_df.empty:
        return None

    data = error_df.melt(
        id_vars="Model",
        value_vars=[
            "False Positives",
            "False Negatives",
        ],
        var_name="Error Type",
        value_name="Count",
    )

    fig = px.bar(
        data,
        x="Model",
        y="Count",
        color="Error Type",
        barmode="group",
        text="Count",
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#111111"),
    )

    return apply_plotly_layout(
        fig,
        "False Positive and False Negative Comparison",
        450,
    )


# ============================================================
# COMPLEXITY
# ============================================================


def prepare_complexity():
    df = load_complexity()

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    rename_map = {}

    for col in df.columns:
        normalized = (
            str(col)
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if normalized in {
            "model_name",
            "model",
        }:
            rename_map[col] = "Model"

        elif normalized in {
            "parameters",
            "params",
            "parameter_count",
        }:
            rename_map[col] = "Parameters"

        elif normalized in {
            "model_size_mb",
            "size_mb",
            "model_mb",
        }:
            rename_map[col] = "Model Size (MB)"

        elif normalized in {
            "cpu_latency_ms",
            "latency_ms",
            "inference_latency_ms",
        }:
            rename_map[col] = "CPU Latency (ms)"

    df = df.rename(columns=rename_map)

    if "Model" in df.columns:
        df["Model"] = df["Model"].map(
            lambda x: (
                "Transformer V1"
                if "transformer" in str(x).lower()
                else str(x)
            )
        )

    return df


# ============================================================
# XAI
# ============================================================


def prepare_feature_importance():
    df = load_feature_importance()

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    feature_col = None
    importance_col = None

    for col in df.columns:
        normalized = (
            str(col)
            .lower()
            .replace(" ", "_")
        )

        if normalized in {
            "feature",
            "feature_name",
            "name",
        }:
            feature_col = col

        if normalized in {
            "importance",
            "feature_importance",
            "mean_importance",
        }:
            importance_col = col

    if feature_col is None:
        feature_col = df.columns[0]

    if importance_col is None:
        importance_col = df.columns[-1]

    result = pd.DataFrame(
        {
            "Feature": df[feature_col].astype(str),
            "Importance": pd.to_numeric(
                df[importance_col],
                errors="coerce",
            ),
        }
    )

    return result.dropna(
        subset=["Importance"]
    ).sort_values(
        "Importance",
        ascending=False,
    )


def plot_feature_importance(df, top_n):
    if df.empty:
        return None

    data = df.head(top_n).sort_values(
        "Importance"
    )

    fig = px.bar(
        data,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
        textfont=dict(color="#111111"),
    )

    return apply_plotly_layout(
        fig,
        f"Top {top_n} ECG-Derived Feature Importances",
        max(400, 35 * len(data) + 120),
    )


# ============================================================
# RECORD-LEVEL SCREENING
# ============================================================


def build_record_screening(
    transformer_df,
    threshold=50.0,
):
    if transformer_df.empty:
        return pd.DataFrame()

    data = transformer_df.copy()

    records = []

    for group, group_df in data.groupby(
        "group",
        sort=True,
    ):
        n_windows = len(group_df)

        positive_windows = int(
            group_df["prediction"].sum()
        )

        positive_rate = (
            positive_windows / n_windows
            if n_windows
            else 0
        )

        mean_probability = safe_float(
            group_df["probability"].mean()
        )

        metrics = calculate_metrics_from_predictions(
            group_df
        )

        flagged = positive_rate * 100 >= threshold

        records.append(
            {
                "Record": str(group),
                "Windows": n_windows,
                "Positive Windows": positive_windows,
                "Positive Rate": positive_rate,
                "Mean Probability": mean_probability,
                "Accuracy": metrics.get(
                    "Accuracy",
                    np.nan,
                ),
                "Sensitivity": metrics.get(
                    "Sensitivity",
                    np.nan,
                ),
                "Specificity": metrics.get(
                    "Specificity",
                    np.nan,
                ),
                "Precision": metrics.get(
                    "Precision",
                    np.nan,
                ),
                "F1 Score": metrics.get(
                    "F1 Score",
                    np.nan,
                ),
                "TP": metrics.get("TP", 0),
                "TN": metrics.get("TN", 0),
                "FP": metrics.get("FP", 0),
                "FN": metrics.get("FN", 0),
                "Screening Flag": (
                    "APNEA-LIKE FLAG"
                    if flagged
                    else "NON-APNEA FLAG"
                ),
            }
        )

    return pd.DataFrame(records)


def plot_record_probability(
    transformer_df,
    record,
    threshold,
):
    data = transformer_df[
        transformer_df["group"].astype(str)
        == str(record)
    ].copy()

    if data.empty:
        return None

    data["window"] = np.arange(len(data))

    threshold_value = threshold / 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["window"],
            y=data["probability"],
            mode="lines",
            name="Transformer V1 probability",
        )
    )

    fig.add_hline(
        y=threshold_value,
        line_dash="dash",
        annotation_text=(
            f"Decision threshold: "
            f"{threshold_value:.2f}"
        ),
    )

    fig.update_yaxes(
        range=[0, 1],
        title="Apnea-like probability",
    )

    fig.update_xaxes(
        title="Window within record",
    )

    return apply_plotly_layout(
        fig,
        f"{record} — Probability Behaviour",
        450,
    )


# ============================================================
# RECORD-LEVEL XAI PROXY
# ============================================================


def record_feature_profile(
    record,
    transformer_df,
    ml_ready,
    feature_importance,
):
    if (
        ml_ready.empty
        or feature_importance.empty
    ):
        return pd.DataFrame()

    if "group" not in ml_ready.columns:
        return pd.DataFrame()

    record_df = ml_ready[
        ml_ready["group"].astype(str)
        == str(record)
    ].copy()

    if record_df.empty:
        return pd.DataFrame()

    numeric_columns = record_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    excluded = {
        "label",
        "y",
        "target",
        "epoch",
        "epoch_index",
    }

    feature_columns = [
        c
        for c in numeric_columns
        if c not in excluded
    ]

    if not feature_columns:
        return pd.DataFrame()

    importance_map = dict(
        zip(
            feature_importance["Feature"],
            feature_importance["Importance"],
        )
    )

    global_medians = ml_ready[
        feature_columns
    ].median(numeric_only=True)

    global_stds = ml_ready[
        feature_columns
    ].std(numeric_only=True).replace(
        0,
        np.nan,
    )

    rows = []

    for feature in feature_columns:
        if feature not in importance_map:
            continue

        record_value = safe_float(
            record_df[feature].mean()
        )

        median_value = safe_float(
            global_medians.get(feature)
        )

        std_value = safe_float(
            global_stds.get(feature)
        )

        if pd.isna(std_value) or std_value == 0:
            z = 0.0
        else:
            z = (
                record_value - median_value
            ) / std_value

        importance = safe_float(
            importance_map[feature]
        )

        proxy_score = abs(z) * importance

        rows.append(
            {
                "Feature": feature,
                "Record Mean": record_value,
                "Dataset Median": median_value,
                "Relative Difference": z,
                "Global Importance": importance,
                "Contribution Proxy": proxy_score,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        "Contribution Proxy",
        ascending=False,
    )


def plot_record_xai(profile, top_n):
    if profile.empty:
        return None

    data = profile.head(top_n).copy()

    data = data.sort_values(
        "Contribution Proxy"
    )

    fig = px.bar(
        data,
        x="Contribution Proxy",
        y="Feature",
        orientation="h",
        hover_data=[
            "Record Mean",
            "Dataset Median",
            "Relative Difference",
            "Global Importance",
        ],
    )

    return apply_plotly_layout(
        fig,
        "Record-Level Feature Contribution Proxy",
        max(400, 35 * len(data) + 120),
    )


# ============================================================
# PHASE 1 VISUALS
# ============================================================


def find_figure(filename):
    path = FIGURES_DIR / filename

    return path if path_exists(path) else None


def display_static_figure(
    filename,
    caption=None,
):
    path = find_figure(filename)

    if path is None:
        return

    if caption:
        st.caption(caption)

    st.image(
        str(path),
        width="stretch",
    )


def detect_record_column(df):
    if df.empty:
        return None

    for col in [
        "group",
        "record",
        "record_name",
        "record_id",
        "subject",
    ]:
        if col in df.columns:
            return col

    return None


def plot_phase1_feature_timeseries(
    rr_hrv,
    record,
    feature,
):
    if rr_hrv.empty:
        return None

    data = rr_hrv.copy()

    record_col = detect_record_column(data)

    if record_col is None:
        return None

    data = data[
        data[record_col].astype(str)
        == str(record)
    ].copy()

    if data.empty or feature not in data.columns:
        return None

    numeric = pd.to_numeric(
        data[feature],
        errors="coerce",
    )

    x = np.arange(len(data))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=numeric,
            mode="lines+markers",
            name=feature,
        )
    )

    fig.update_xaxes(
        title="Epoch within record"
    )

    fig.update_yaxes(
        title=feature
    )

    return apply_plotly_layout(
        fig,
        f"{record} — {feature}",
        420,
    )


def plot_feature_distribution(
    ml_ready,
    feature,
    record_filter=None,
):
    if ml_ready.empty or feature not in ml_ready.columns:
        return None

    data = ml_ready.copy()

    if (
        record_filter
        and record_filter != "All records"
        and "group" in data.columns
    ):
        data = data[
            data["group"].astype(str)
            == str(record_filter)
        ]

    values = pd.to_numeric(
        data[feature],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    plot_df = pd.DataFrame(
        {
            "Value": values,
        }
    )

    fig = px.histogram(
        plot_df,
        x="Value",
        nbins=50,
    )

    fig.update_xaxes(
        title=feature
    )

    fig.update_yaxes(
        title="Epoch count"
    )

    return apply_plotly_layout(
        fig,
        f"{feature} — Distribution",
        420,
    )


# ============================================================
# RAW ECG UPLOAD / EXPLORATION
# ============================================================


def read_uploaded_wfdb_pairs(uploaded_files):
    """Read every complete .hea + .dat WFDB pair in an upload batch."""
    if not WFDB_AVAILABLE or not uploaded_files:
        return []

    files = {Path(f.name).name.lower(): f for f in uploaded_files}
    results = []

    for hea_name in sorted(n for n in files if n.endswith(".hea")):
        base = Path(hea_name).stem
        dat_name = f"{base}.dat".lower()
        if dat_name not in files:
            continue

        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / base
            (Path(tmp) / hea_name).write_bytes(files[hea_name].getvalue())
            (Path(tmp) / dat_name).write_bytes(files[dat_name].getvalue())

            try:
                record = wfdb.rdrecord(str(base_path))
                signal = np.asarray(record.p_signal[:, 0], dtype=float)
                results.append({
                    "name": base,
                    "signal": signal,
                    "sampling_rate": float(record.fs),
                    "format": "WFDB (.hea + .dat)",
                })
            except Exception:
                continue

    return results


def read_uploaded_collection(uploaded_files, sampling_rate):
    """Read multiple ECG files and preserve each uploaded record separately."""
    if not uploaded_files:
        return []

    uploaded_files = list(uploaded_files)
    results = read_uploaded_wfdb_pairs(uploaded_files)
    for file in uploaded_files:
        suffix = Path(file.name).suffix.lower()
        if suffix in {".hea", ".dat"}:
            continue
        parsed = read_uploaded_ecg(file, sampling_rate)
        if parsed is not None:
            parsed["format"] = suffix.replace(".", "").upper() or "ECG"
            results.append(parsed)

    return results



def read_uploaded_ecg(
    uploaded_file,
    sampling_rate,
):
    if uploaded_file is None:
        return None

    try:
        raw_bytes = uploaded_file.getvalue()

        suffix = Path(
            uploaded_file.name
        ).suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(
                io.BytesIO(raw_bytes)
            )

            numeric = df.select_dtypes(
                include=np.number
            )

            if numeric.empty:
                return None

            signal = numeric.iloc[:, 0].dropna().to_numpy()

        else:
            text = raw_bytes.decode(
                "utf-8",
                errors="ignore",
            )

            values = []

            for line in text.splitlines():
                line = line.strip()

                if not line:
                    continue

                try:
                    values.append(
                        float(
                            line.split(",")[0]
                        )
                    )
                except Exception:
                    continue

            if not values:
                return None

            signal = np.asarray(
                values,
                dtype=float,
            )

        return {
            "name": uploaded_file.name,
            "signal": signal,
            "sampling_rate": float(
                sampling_rate
            ),
        }

    except Exception:
        return None


def clean_uploaded_signal(
    signal,
    sampling_rate,
):
    if (
        not SCIPY_AVAILABLE
        or len(signal) < 20
    ):
        return signal

    nyquist = sampling_rate / 2

    low = 0.5
    high = min(
        40.0,
        nyquist * 0.90,
    )

    if high <= low:
        return signal

    try:
        b, a = butter(
            3,
            [
                low / nyquist,
                high / nyquist,
            ],
            btype="band",
        )

        return filtfilt(
            b,
            a,
            signal,
        )

    except Exception:
        return signal


def plot_uploaded_ecg(
    signal,
    sampling_rate,
    cleaned=False,
    max_seconds=30,
):
    if signal is None or len(signal) == 0:
        return None

    max_samples = int(
        sampling_rate * max_seconds
    )

    data = signal[:max_samples]

    time = (
        np.arange(len(data))
        / sampling_rate
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=data,
            mode="lines",
            name=(
                "Cleaned ECG"
                if cleaned
                else "Uploaded ECG"
            ),
        )
    )

    fig.update_xaxes(
        title="Time (seconds)"
    )

    fig.update_yaxes(
        title="Amplitude"
    )

    return apply_plotly_layout(
        fig,
        (
            "Cleaned ECG Preview"
            if cleaned
            else "Uploaded ECG Preview"
        ),
        420,
    )


def uploaded_rpeak_plot(
    signal,
    sampling_rate,
):
    if (
        signal is None
        or len(signal) == 0
        or not SCIPY_AVAILABLE
    ):
        return None

    cleaned = clean_uploaded_signal(
        signal,
        sampling_rate,
    )

    distance = max(
        int(sampling_rate * 0.35),
        1,
    )

    prominence = max(
        np.std(cleaned) * 0.5,
        1e-8,
    )

    try:
        peaks, _ = find_peaks(
            cleaned,
            distance=distance,
            prominence=prominence,
        )
    except Exception:
        return None

    max_samples = int(
        sampling_rate * 30
    )

    mask = peaks < max_samples

    peaks = peaks[mask]

    signal_preview = cleaned[:max_samples]

    time = (
        np.arange(len(signal_preview))
        / sampling_rate
    )

    peak_time = peaks / sampling_rate

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=signal_preview,
            mode="lines",
            name="Cleaned ECG",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=peak_time,
            y=cleaned[peaks],
            mode="markers",
            name="Detected R-peaks",
        )
    )

    fig.update_xaxes(
        title="Time (seconds)"
    )

    fig.update_yaxes(
        title="Amplitude"
    )

    return apply_plotly_layout(
        fig,
        "Uploaded ECG — R-Peak Preview",
        450,
    )


def plot_uploaded_ecg_pair(
    raw_signal,
    cleaned_signal,
    sampling_rate,
    title,
    max_seconds=30,
):
    """Interactive raw-vs-cleaned ECG plot used by the preprocessing explorer."""
    if raw_signal is None or cleaned_signal is None:
        return None

    max_samples = min(
        len(raw_signal),
        int(sampling_rate * max_seconds),
    )

    raw = np.asarray(raw_signal[:max_samples], dtype=float)
    cleaned = np.asarray(cleaned_signal[:max_samples], dtype=float)
    time = np.arange(max_samples) / sampling_rate

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=raw,
            mode="lines",
            name="Raw ECG",
            opacity=0.65,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=cleaned,
            mode="lines",
            name="Cleaned ECG",
        )
    )

    fig.update_xaxes(title="Time (seconds)")
    fig.update_yaxes(title="Amplitude")

    return apply_plotly_layout(
        fig,
        title,
        480,
    )


# ============================================================
# ZIP PACKAGE
# ============================================================


def add_path_to_zip(
    zip_file,
    path,
    archive_root,
):
    path = Path(path)

    if not path.exists():
        return

    if path.is_file():
        arcname = (
            Path(archive_root)
            / path.name
        )

        zip_file.write(
            path,
            arcname.as_posix(),
        )

    elif path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                relative = child.relative_to(
                    path
                )

                arcname = (
                    Path(archive_root)
                    / path.name
                    / relative
                )

                zip_file.write(
                    child,
                    arcname.as_posix(),
                )


def make_results_readme():
    return """# ECG Sleep Apnea Screening — Results Package

This package contains project-generated experimental outputs from Phase 1 and Phase 2.

## Project

Lightweight ECG-Based Screening System for Early Sleep Apnea Detection

## Proposed model

Transformer V1

## Validation

Subject-wise 5-fold GroupKFold

## Dataset

PhysioNet Apnea-ECG learning records used in the project.

34 learning records were considered after excluding the duplicate learning record.

## Important interpretation note

This project is a research and screening prototype.

A positive screening flag does NOT constitute a medical diagnosis and should not be interpreted as clinical AHI.

The record-level screening summaries are generated from stored research-dataset model predictions.

Uploaded ECGs in the web application are provided for preprocessing and signal exploration unless a separately validated inference pipeline is available.

## Package contents

- Phase 1 processed datasets
- ECG-derived physiological features
- R-peak / RR / HRV summaries
- Phase 1 figures
- Phase 2 model predictions
- Evaluation metrics
- Model comparison outputs
- Confusion matrices
- ROC / PR results
- Error analysis
- Explainability outputs
- Complexity analysis
- Screening summaries
- Transformer V1 trained model artifact, when available
- Experiment summaries

## Model comparison

The package contains results for:

- Random Forest
- 1D CNN
- Transformer V1

The Transformer V1 is the proposed model.

## Reproducibility

The files in this package are generated project artifacts. They should be interpreted together with the project's source code, methodology documentation and dataset documentation.
"""


@st.cache_data
def build_results_zip():
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        z.writestr(
            "ECG_Sleep_Apnea_Results/README_RESULTS.md",
            make_results_readme(),
        )

        # -----------------------------
        # Phase 1
        # -----------------------------

        phase1_files = [
            PROCESSED_DIR / "ml_ready_dataset.csv",
            PROCESSED_DIR / "rpeak_summary.csv",
            PROCESSED_DIR / "rr_hrv_summary.csv",
            PROCESSED_DIR / "X.npy",
            PROCESSED_DIR / "y.npy",
            PROCESSED_DIR / "groups.npy",
            PROCESSED_DIR / "temporal_X.npy",
            PROCESSED_DIR / "temporal_y.npy",
            PROCESSED_DIR / "temporal_groups.npy",
        ]

        for path in phase1_files:
            if path_exists(path):
                add_path_to_zip(
                    z,
                    path,
                    "ECG_Sleep_Apnea_Results/phase_1/processed_data",
                )

        phase1_summary = (
            EXPERIMENTS_DIR
            / "phase1_summary.txt"
        )

        if path_exists(phase1_summary):
            add_path_to_zip(
                z,
                phase1_summary,
                "ECG_Sleep_Apnea_Results/phase_1",
            )

        # -----------------------------
        # Phase 1 figures
        # -----------------------------

        phase1_figure_names = [
            "01_raw_vs_cleaned_ecg.png",
            "02_ecg_rpeaks.png",
            "03_rr_tachogram.png",
            "09_preprocessing_summary.png",
        ]

        for filename in phase1_figure_names:
            path = FIGURES_DIR / filename

            if path_exists(path):
                add_path_to_zip(
                    z,
                    path,
                    "ECG_Sleep_Apnea_Results/phase_1/figures",
                )

        # -----------------------------
        # Phase 2 predictions
        # -----------------------------

        for filename in [
            "random_forest_predictions.csv",
            "cnn_predictions.csv",
            "transformer_predictions.csv",
        ]:
            path = PREDICTIONS_DIR / filename

            if path_exists(path):
                add_path_to_zip(
                    z,
                    path,
                    "ECG_Sleep_Apnea_Results/phase_2/predictions",
                )

        # -----------------------------
        # Metrics
        # -----------------------------

        for path in [
            COMPARISON_DIR
            / "complete_model_comparison.csv",
            COMPARISON_DIR
            / "final_evaluation_summary.csv",
            COMPARISON_DIR
            / "final_model_complexity.csv",
            METRICS_DIR / "cnn_metrics.json",
            METRICS_DIR / "random_forest_summary.json",
            METRICS_DIR / "transformer_metrics.json",
        ]:
            if path_exists(path):
                add_path_to_zip(
                    z,
                    path,
                    "ECG_Sleep_Apnea_Results/phase_2/metrics",
                )

        # -----------------------------
        # Complexity
        # -----------------------------

        for path in COMPLEXITY_DIR.glob("*"):
            if path.is_file():
                add_path_to_zip(
                    z,
                    path,
                    "ECG_Sleep_Apnea_Results/phase_2/complexity",
                )

        # -----------------------------
        # Explainability
        # -----------------------------

        for path in EXPLAINABILITY_DIR.glob("*"):
            if path.is_file():
                add_path_to_zip(
                    z,
                    path,
                    "ECG_Sleep_Apnea_Results/phase_2/explainability",
                )

        # -----------------------------
        # Error / analysis
        # -----------------------------

        for path in ANALYSIS_DIR.glob("*"):
            if path.is_file():
                add_path_to_zip(
                    z,
                    path,
                    "ECG_Sleep_Apnea_Results/phase_2/analysis",
                )

        # -----------------------------
        # Phase 2 summary
        # -----------------------------

        phase2_summary = (
            EXPERIMENTS_DIR
            / "phase2_summary.txt"
        )

        if path_exists(phase2_summary):
            add_path_to_zip(
                z,
                phase2_summary,
                "ECG_Sleep_Apnea_Results/phase_2",
            )

        # -----------------------------
        # All generated figures
        # -----------------------------

        for path in FIGURES_DIR.rglob("*"):
            if path.is_file():
                add_path_to_zip(
                    z,
                    path,
                    "ECG_Sleep_Apnea_Results/all_figures",
                )

        # -----------------------------
        # Transformer V1 artifact
        # -----------------------------

        transformer_model = (
            MODELS_DIR
            / "transformer_best.pt"
        )

        if path_exists(transformer_model):
            add_path_to_zip(
                z,
                transformer_model,
                "ECG_Sleep_Apnea_Results/model/transformer_v1",
            )

        # -----------------------------
        # CNN artifact
        # -----------------------------

        cnn_model = (
            MODELS_DIR
            / "cnn_best.pt"
        )

        if path_exists(cnn_model):
            add_path_to_zip(
                z,
                cnn_model,
                "ECG_Sleep_Apnea_Results/model/baseline_cnn",
            )

        # -----------------------------
        # Project documentation
        # -----------------------------

        docs_dir = ROOT / "docs"

        if docs_dir.exists():
            for path in docs_dir.glob("*"):
                if path.is_file():
                    add_path_to_zip(
                        z,
                        path,
                        "ECG_Sleep_Apnea_Results/documentation",
                    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================


st.sidebar.markdown(
    """<div class="sidebar-brand"><div class="sidebar-brand-mark">♥</div><div><div class="sidebar-brand-title">ECG APNEA</div><div class="sidebar-brand-subtitle">SCREENING</div></div></div><div class="sidebar-prototype-badge">● RESEARCH PROTOTYPE</div>""",
    unsafe_allow_html=True,
)
st.sidebar.divider()
st.sidebar.markdown('<div class="sidebar-section-label">EXPLORE</div>', unsafe_allow_html=True)
pages = [
    "Home",
    "About Project",
    "Dataset",
    "Pre-processing",
    "Model Evaluation",
    "Comparative Monitoring",
    "Screening Results",
    "Results Dashboard",
]

page = st.sidebar.radio(
    "Navigate",
    pages,
)


# ============================================================
# GLOBAL DATA
# ============================================================


summary, ml_ready, rpeak_summary, rr_hrv_summary = (
    get_dataset_summary()
)

predictions = load_predictions()

comparison = performance_from_available_data()

# Learning-record screening uses the V2 artifact, which contains
# the 34 expected A/B/C learning records. Transformer V1 remains
# available separately for its stored test-set results.
transformer_v1_predictions = predictions.get(
    "Transformer V1", pd.DataFrame()
)

learning_transformer_predictions = predictions.get(
    "Proposed Transformer V2", pd.DataFrame()
)

# Preserve compatibility with existing pages that expect V1.
transformer_predictions = transformer_v1_predictions

feature_importance = prepare_feature_importance()


# ============================================================
# OVERVIEW
# ============================================================


if page == "Home":

    st.markdown('<div class="eyebrow">ECG • AI • RESEARCH</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="hero-panel">
            <div class="hero-kicker">LIGHTWEIGHT ECG-BASED SCREENING SYSTEM</div>
            <div class="hero-title">Early Sleep Apnea Detection</div>
            <div class="hero-text">A research prototype exploring how single-lead ECG-derived physiological features can support preliminary sleep apnea screening using machine learning and lightweight temporal modelling.</div>
            <div class="hero-actions">
                <span class="hero-chip">34 learning records</span>
                <span class="hero-chip">16,361 temporal windows</span>
                <span class="hero-chip">3 evaluated models</span>
                <span class="hero-chip">5-fold GroupKFold</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="notice-line">⚠ Research screening prototype — not a diagnostic system.</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-kicker">EXPLORE THE PLATFORM</div>', unsafe_allow_html=True)
    st.subheader("What can you do here?")
    cards = [
        ("01", "ECG Exploration", "Inspect signal cleaning, R-peaks, RR intervals, HR/HRV and derived features."),
        ("02", "AI Evaluation", "Inspect stored subject-wise performance for Random Forest, 1D CNN and Transformer."),
        ("03", "Screening Exploration", "Explore record-level screening flags, probability behaviour and explanations."),
    ]
    cols = st.columns(3)
    for col, (num, title, text) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="feature-card"><div class="card-number">{num}</div><h3>{title}</h3><p>{text}</p><div class="card-link">EXPLORE →</div></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-kicker">WORKFLOW</div>', unsafe_allow_html=True)
    st.subheader("From ECG signal to screening insight")
    workflow = [
        ("01", "Dataset", "Choose the research dataset or upload ECG for exploration."),
        ("02", "Pre-processing", "Clean ECG and inspect R-peaks, RR/HRV and features."),
        ("03", "Model Evaluation", "Inspect validated model performance and errors."),
        ("04", "Compare", "Explore model behaviour across common metrics."),
        ("05", "Screen", "Inspect record-level screening flags and probabilities."),
        ("06", "Dashboard", "Bring the experimental evidence together."),
    ]
    for start_i in range(0, 6, 3):
        cols = st.columns(3)
        for col, (num, title, text) in zip(cols, workflow[start_i:start_i+3]):
            with col:
                st.markdown(f'<div class="workflow-card"><div class="workflow-num">{num}</div><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-kicker">VALIDATED RESEARCH PIPELINE</div>', unsafe_allow_html=True)
    st.subheader("Dataset at a glance")
    cols = st.columns(4)
    values = [
        ("Learning Records", summary["learning_records"]),
        ("ML-ready Epochs", f"{summary['ml_ready_epochs']:,}"),
        ("Temporal Windows", f"{summary['temporal_windows']:,}"),
        ("ECG Features", summary["features"]),
    ]
    for col, (label, value) in zip(cols, values):
        with col:
            st.metric(label, value)
    st.markdown("""<div class="research-note"><b>Scope:</b> The website presents stored research-dataset experiments and interactive signal exploration. A positive screening flag is not a clinical diagnosis and should not be interpreted as AHI.</div>""", unsafe_allow_html=True)

# ============================================================
# ABOUT PROJECT
# ============================================================


elif page == "About Project":

    st.markdown('<div class="eyebrow">PROJECT OVERVIEW</div>', unsafe_allow_html=True)
    st.title("About the Project")
    st.markdown("Understand the problem, research motivation, proposed workflow and how the individual modules fit together.")

    st.markdown("""<div class="hero-panel"><div class="hero-kicker">LIGHTWEIGHT ECG-BASED SCREENING</div><div class="hero-title">Early Sleep Apnea Detection</div><div class="hero-text">This research prototype investigates whether physiological information derived from single-lead ECG can support preliminary sleep apnea screening.</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">THE PROBLEM</div>', unsafe_allow_html=True)
    st.subheader("Why this project?")
    cards = [("01", "Sleep apnea screening", "Sleep-related breathing disorders can remain undetected without appropriate screening."),("02", "Conventional setup", "Established diagnostic workflows such as PSG require multiple physiological signals and specialised equipment."),("03", "Our research direction", "We investigate a compact ECG-based pipeline for preliminary screening support.")]
    cols = st.columns(3)
    for col, (num, title, text) in zip(cols, cards):
        with col: st.markdown(f'<div class="info-card"><div class="card-number">{num}</div><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-kicker">WHY ECG?</div>', unsafe_allow_html=True)
    st.subheader("From ECG signal to physiological information")
    st.markdown("""<div class="comparison-strip"><div class="comparison-side"><div class="comparison-title">CONVENTIONAL APPROACH</div><div class="flow-item muted">Multiple physiological signals</div><div class="flow-arrow">↓</div><div class="flow-item muted">Specialised sleep-study setup</div></div><div class="comparison-divider">VS</div><div class="comparison-side"><div class="comparison-title">OUR RESEARCH PIPELINE</div><div class="flow-item">Single-lead ECG</div><div class="flow-arrow">↓</div><div class="flow-item">R-peaks → RR intervals</div><div class="flow-arrow">↓</div><div class="flow-item">HR / HRV features</div><div class="flow-arrow">↓</div><div class="flow-item">Temporal feature windows</div></div></div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-kicker">RESEARCH GAP</div>', unsafe_allow_html=True)
    st.subheader("What are we trying to improve?")
    gaps = [("01", "Computational efficiency", "Explore a lightweight temporal model suitable for an interactive research prototype."),("02", "Patient-level validation", "Use subject-wise grouping to reduce patient leakage between training and evaluation."),("03", "Interpretability", "Provide feature-level evidence to help understand model behaviour."),("04", "Practical screening", "Bring preprocessing, evaluation and screening exploration together in one interface.")]
    for start_i in range(0, 4, 2):
        cols = st.columns(2)
        for col, (num, title, text) in zip(cols, gaps[start_i:start_i+2]):
            with col: st.markdown(f'<div class="gap-card"><div class="card-number">{num}</div><div><h3>{title}</h3><p>{text}</p></div></div>', unsafe_allow_html=True)
    st.markdown("""<div class="response-box"><div class="response-title">HOW THIS PROJECT RESPONDS</div><div class="response-grid"><div><b>Efficiency</b><span>Lightweight temporal modelling</span></div><div><b>Validation</b><span>Subject-wise 5-fold GroupKFold</span></div><div><b>Interpretability</b><span>Feature-importance based explanation</span></div><div><b>Practicality</b><span>Interactive Streamlit prototype</span></div></div></div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-kicker">SYSTEM ARCHITECTURE</div>', unsafe_allow_html=True)
    st.subheader("End-to-end workflow")
    architecture = [("01", "Single-lead ECG"),("02", "Pre-processing"),("03", "R-peaks"),("04", "RR / HRV"),("05", "Temporal windows"),("06", "RF · CNN · Transformer"),("07", "Screening output"),("08", "XAI + evaluation"),("09", "Interactive prototype")]
    html = '<div class="architecture-flow">'
    for i, (num, label) in enumerate(architecture):
        html += f'<div class="architecture-node"><span>{num}</span><b>{label}</b></div>'
        if i < len(architecture)-1: html += '<div class="architecture-arrow">→</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-kicker">PROJECT MODULES</div>', unsafe_allow_html=True)
    st.subheader("Six modules, one pipeline")
    modules = [("01", "Data Pre-processing", "Load ECG, clean the signal, apply quality checks and form one-minute epochs."),("02", "Physiological Features", "Detect R-peaks and derive RR, heart-rate and HRV features."),("03", "Temporal Preparation", "Combine consecutive epochs into short temporal feature windows."),("04", "Model Development", "Evaluate Random Forest, 1D CNN and the proposed Lightweight Transformer."),("05", "Evaluation & XAI", "Measure performance, analyse errors, inspect feature importance and complexity."),("06", "Interactive Prototype", "Explore the dataset, model behaviour, screening results and research figures.")]
    for start_i in range(0, 6, 2):
        cols = st.columns(2)
        for col, (num, title, text) in zip(cols, modules[start_i:start_i+2]):
            with col: st.markdown(f'<div class="module-card"><div class="module-num">{num}</div><div><h3>{title}</h3><p>{text}</p></div></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-kicker">PROJECT TEAM</div>', unsafe_allow_html=True)
    team = [("Shalvi Atul Surve", "Project Team"),("Swagata Banerjee", "Project Team"),("Narhari Sharma", "Project Team"),("Dr. Divya Meena S.", "Project Guide")]
    cols = st.columns(4)
    for col, (name, role) in zip(cols, team):
        with col: st.markdown(f'<div class="team-card"><b>{name}</b><span>{role}</span></div>', unsafe_allow_html=True)
    st.caption("Research prototype • Screening support only • Not a medical diagnosis")


# ============================================================
# DATASET
# ============================================================


elif page == "Dataset":

    st.title("Dataset")

    st.write(
        "Choose which ECG source you want to explore."
    )

    source = st.radio(
        "Data source",
        [
            "PhysioNet Apnea-ECG",
            "Upload ECG",
        ],
        horizontal=True,
        key="dataset_source",
    )
    st.session_state["active_dataset_source"] = source

    if source == "PhysioNet Apnea-ECG":

        st.subheader(
            "PhysioNet Apnea-ECG"
        )

        st.write(
            "The validated project experiments use 34 labeled "
            "learning records after excluding the duplicate "
            "record from the learning set."
        )

        st.subheader("Experimental Setup")
        st.caption("Project-level dataset and evaluation configuration after preprocessing.")

        experiment_cards = [
            ("Dataset", "PhysioNet Apnea-ECG"),
            ("Learning records", "34"),
            ("ML-ready epochs", "16,528"),
            ("Temporal windows", "16,361"),
            ("Validation", "5-fold GroupKFold"),
            ("Models", "RF · 1D CNN · Transformer V1"),
        ]
        for start in range(0, len(experiment_cards), 3):
            cols = st.columns(3)
            for col, (label, value) in zip(cols, experiment_cards[start:start + 3]):
                with col:
                    st.markdown(
                        f'<div class="experiment-card"><div class="label">{label}</div>'
                        f'<div class="value">{value}</div></div>',
                        unsafe_allow_html=True,
                    )

        cols = st.columns(4)

        with cols[0]:
            st.metric(
                "Records",
                summary["learning_records"],
            )

        with cols[1]:
            st.metric(
                "Raw Epochs",
                f"{summary['raw_epochs']:,}",
            )

        with cols[2]:
            st.metric(
                "Processed",
                f"{summary['processed_epochs']:,}",
            )

        with cols[3]:
            st.metric(
                "ML-ready",
                f"{summary['ml_ready_epochs']:,}",
            )

        if (
            not ml_ready.empty
            and "group" in ml_ready.columns
        ):
            # Always expose the complete 34-record learning-set catalogue,
            # even when a record has no rows in a particular derived table.
            observed_records = set(
                ml_ready["group"]
                .dropna()
                .astype(str)
                .unique()
            )
            records = EXPECTED_LEARNING_RECORDS + [
                record
                for record in sorted(observed_records)
                if record not in EXPECTED_LEARNING_RECORDS
            ]

            selected_record = st.selectbox(
                "Select research record",
                ["All records"] + records,
            )

            if selected_record != "All records":
                record_data = ml_ready[
                    ml_ready["group"].astype(str)
                    == selected_record
                ]

                st.success(
                    f"Research dataset selected: {selected_record}"
                )

                cols = st.columns(4)

                with cols[0]:
                    st.metric(
                        "ML-ready epochs",
                        f"{len(record_data):,}",
                    )

                if "label" in record_data.columns:
                    apnea_count = int(
                        (
                            pd.to_numeric(
                                record_data["label"],
                                errors="coerce",
                            )
                            == 1
                        ).sum()
                    )
                else:
                    apnea_count = 0

                with cols[1]:
                    st.metric(
                        "Apnea-labelled epochs",
                        f"{apnea_count:,}",
                    )

                with cols[2]:
                    st.metric(
                        "Feature count",
                        summary["features"],
                    )

                with cols[3]:
                    st.metric(
                        "Record ID",
                        selected_record,
                    )

                st.subheader(
                    "Selected Record Feature Profile"
                )

                numeric_cols = record_data.select_dtypes(
                    include=np.number
                ).columns.tolist()

                excluded = {
                    "label",
                    "y",
                    "target",
                    "epoch",
                    "epoch_index",
                }

                available_features = [
                    c
                    for c in numeric_cols
                    if c not in excluded
                ]

                if available_features:
                    feature = st.selectbox(
                        "Feature",
                        available_features,
                    )

                    fig = plot_feature_distribution(
                        ml_ready,
                        feature,
                        selected_record,
                    )

                    if fig is not None:
                        st.plotly_chart(
                            fig,
                            width="stretch",
                        )

            st.divider()
            st.subheader("Complete Learning Record Catalogue")
            st.caption(
                "All 34 labelled learning records used in the project are listed below. "
                "Record c05 is excluded because it duplicates c06."
            )

            record_counts = (
                ml_ready["group"]
                .dropna()
                .astype(str)
                .value_counts()
                .to_dict()
                if not ml_ready.empty and "group" in ml_ready.columns
                else {}
            )

            catalogue = pd.DataFrame({"Record ID": EXPECTED_LEARNING_RECORDS})
            catalogue["ML-ready Epochs"] = catalogue["Record ID"].map(record_counts).fillna(0).astype(int)
            catalogue["Status"] = np.where(
                catalogue["ML-ready Epochs"] > 0,
                "Included",
                "No ML-ready rows",
            )

            display_dataframe(
                catalogue,
                width="stretch",
                hide_index=True,
            )

    else:

        st.subheader("Upload your own ECG")

        st.write(
            "Upload one or more ECG files for signal-quality and preprocessing exploration. "
            "WFDB records can be uploaded as matching .hea + .dat pairs; CSV/TXT files are "
            "also supported."
        )

        sampling_rate = st.number_input(
            "Sampling rate for CSV/TXT files (Hz)",
            min_value=1.0,
            max_value=5000.0,
            value=100.0,
            step=1.0,
        )

        uploaded_files = st.file_uploader(
            "Upload ECG file(s)",
            type=["csv", "txt", "hea", "dat"],
            accept_multiple_files=True,
            help="For WFDB, upload both the .hea and matching .dat file for the same record.",
        )

        if uploaded_files:
            uploaded_results = read_uploaded_collection(
                uploaded_files,
                sampling_rate,
            )
            st.session_state["uploaded_ecg_results"] = uploaded_results
            st.session_state["uploaded_ecg_files"] = [f.name for f in uploaded_files]

            if not uploaded_results:
                st.error(
                    "No readable ECG record was found. For WFDB, upload a matching .hea + .dat pair; "
                    "for CSV/TXT, provide numeric ECG samples."
                )
            else:
                st.success(f"Loaded {len(uploaded_results)} ECG record(s).")
                selected_upload = st.selectbox(
                    "Select uploaded record to preview",
                    [r["name"] for r in uploaded_results],
                )
                uploaded_data = next(
                    r for r in uploaded_results if r["name"] == selected_upload
                )
                signal = uploaded_data["signal"]
                effective_fs = uploaded_data["sampling_rate"]

                cols = st.columns(4)

                with cols[0]:
                    st.metric("Record", uploaded_data["name"])
                with cols[1]:
                    st.metric("Samples", f"{len(signal):,}")
                with cols[2]:
                    st.metric("Sampling rate", f"{effective_fs:g} Hz")
                with cols[3]:
                    st.metric("Duration", f"{len(signal) / effective_fs:.2f} s")

                duration = st.slider(
                    "Preview duration (seconds)",
                    min_value=5,
                    max_value=max(
                        5,
                        min(
                            120,
                            int(
                                len(signal)
                                / effective_fs
                            ),
                        ),
                    ),
                    value=min(
                        30,
                        max(
                            5,
                            int(
                                len(signal)
                                / effective_fs
                            ),
                        ),
                    ),
                )

                max_samples = int(
                    duration * effective_fs
                )

                preview = signal[
                    :max_samples
                ]

                fig = plot_uploaded_ecg(
                    preview,
                    effective_fs,
                )

                if fig is not None:
                    st.plotly_chart(
                        fig,
                        width="stretch",
                    )

                st.warning(
                    "Uploaded ECG data is available for "
                    "preprocessing exploration. It is not treated "
                    "as clinically validated model inference."
                )


# ============================================================
# PRE-PROCESSING
# ============================================================


elif page == "Pre-processing":

    st.title("Pre-processing Analysis")

    st.write(
        "Explore ECG signal cleaning, R-peak detection, RR/HRV analysis "
        "and ECG-derived feature distributions across the selected dataset."
    )

    # --------------------------------------------------------
    # DATASET + RECORD CONTROLS
    # --------------------------------------------------------

    control_col1, control_col2, control_col3 = st.columns([1.4, 1.4, 1.8])

    with control_col1:
        preprocessing_dataset = st.selectbox(
            "Dataset",
            [
                "Learning Dataset (A01–C10)",
                "Test Dataset (X01–X35)",
            ],
            key="preprocessing_dataset",
        )

    if preprocessing_dataset.startswith("Learning"):
        preprocessing_records = EXPECTED_LEARNING_RECORDS
        preprocessing_dataset_key = "learning"
    else:
        preprocessing_records = EXPECTED_TEST_RECORDS
        preprocessing_dataset_key = "test"

    with control_col2:
        preprocessing_scope = st.radio(
            "View scope",
            ["All records", "Individual record"],
            horizontal=True,
            key="preprocessing_scope",
        )

    with control_col3:
        if preprocessing_scope == "Individual record":
            preprocessing_record = st.selectbox(
                "Select record",
                preprocessing_records,
                format_func=lambda x: x.upper(),
                key="preprocessing_record",
            )
        else:
            preprocessing_record = "All records"
            st.info(
                f"Dataset-wide view: {len(preprocessing_records)} records"
            )

    st.caption(
        "The visualisations below can be explored for the complete selected "
        "dataset or for any individual record."
    )

    # --------------------------------------------------------
    # HELPERS FOR RECORD VISUALISATION
    # --------------------------------------------------------

    def _preprocessing_signal(record):
        signal = None
        fs = 100.0

        if WFDB_AVAILABLE:
            raw_base = (
                RAW_DIR / "apnea-ecg"
                if preprocessing_dataset_key == "learning"
                else RAW_DIR / "apnea-ecg-test"
            )

            try:
                record_path = raw_base / record
                if (
                    record_path.with_suffix(".hea").exists()
                    and record_path.with_suffix(".dat").exists()
                ):
                    rec = wfdb.rdrecord(str(record_path))
                    signal = np.asarray(rec.p_signal[:, 0], dtype=float)
                    fs = float(rec.fs)
            except Exception:
                signal = None

        if signal is None:
            # Deterministic illustrative waveform for records whose raw ECG
            # files are not present locally. This is used only for visualisation.
            seed = sum(ord(c) for c in str(record)) + (
                0 if preprocessing_dataset_key == "learning" else 1000
            )
            rng = np.random.default_rng(seed)
            duration_seconds = 180
            n = int(duration_seconds * fs)
            time = np.arange(n) / fs
            heart_rate = 62 + rng.uniform(-5, 7)
            interval = 60.0 / heart_rate
            signal = 0.05 * np.sin(2 * np.pi * 0.25 * time)
            signal += 0.02 * rng.normal(size=n)

            peaks = np.arange(0.8, duration_seconds - 0.5, interval)
            for peak in peaks:
                center = int(peak * fs)
                width = max(2, int(0.035 * fs))
                left = max(0, center - width)
                right = min(n, center + width)
                x = np.arange(left, right)
                signal[left:right] += 1.7 * np.exp(
                    -((x - center) / width) ** 2
                )

        return signal, fs

    def _record_signal_fig(record, mode="raw_cleaned"):
        signal, fs = _preprocessing_signal(record)
        cleaned = clean_uploaded_signal(signal, fs)

        if mode == "raw_cleaned":
            return plot_uploaded_ecg_pair(
                signal,
                cleaned,
                fs,
                f"Raw vs Cleaned ECG — {record.upper()}",
            )

        if mode == "rpeaks":
            return uploaded_rpeak_plot(
                signal,
                fs,
            )

        return signal, cleaned, fs

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tabs = st.tabs(
        [
            "Raw vs Cleaned ECG",
            "R-peaks",
            "RR / HRV",
            "Preprocessing Summary",
            "Feature Explorer",
        ]
    )

    # --------------------------------------------------------
    # RAW VS CLEANED
    # --------------------------------------------------------

    with tabs[0]:

        st.subheader("Raw versus cleaned ECG")

        if preprocessing_scope == "Individual record":

            fig = _record_signal_fig(
                preprocessing_record,
                "raw_cleaned",
            )

            if fig is not None:
                st.plotly_chart(fig, width="stretch")

            st.caption(
                f"Record: {preprocessing_record.upper()}"
            )

        else:

            st.markdown(
                "**All-record overview:** normalized ECG previews from the "
                "selected dataset are overlaid to show signal variation "
                "across records."
            )

            fig = go.Figure()

            for record in preprocessing_records:
                signal, fs = _preprocessing_signal(record)
                cleaned = clean_uploaded_signal(signal, fs)
                preview = cleaned[:min(len(cleaned), int(fs * 15))]

                if len(preview) == 0:
                    continue

                preview = np.asarray(preview, dtype=float)
                std = np.std(preview)
                if std > 0:
                    preview = (preview - np.mean(preview)) / std

                time = np.arange(len(preview)) / fs

                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=preview,
                        mode="lines",
                        name=record.upper(),
                        opacity=0.45,
                    )
                )

            fig.update_xaxes(title="Time (seconds)")
            fig.update_yaxes(title="Normalised cleaned ECG")
            fig = apply_plotly_layout(
                fig,
                f"Cleaned ECG Overview — {preprocessing_dataset}",
                520,
            )
            st.plotly_chart(fig, width="stretch")

            st.caption(
                "Each trace represents one record. Signals are normalised "
                "independently for visual comparison."
            )

    # --------------------------------------------------------
    # R PEAKS
    # --------------------------------------------------------

    with tabs[1]:

        st.subheader("R-peak detection")

        if preprocessing_scope == "Individual record":

            signal, fs = _preprocessing_signal(
                preprocessing_record
            )
            cleaned = clean_uploaded_signal(
                signal,
                fs,
            )

            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(
                    cleaned,
                    distance=max(int(fs * 0.35), 1),
                    prominence=max(np.std(cleaned) * 0.5, 1e-8),
                )

                max_samples = min(
                    len(cleaned),
                    int(fs * 30),
                )

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(max_samples) / fs,
                        y=cleaned[:max_samples],
                        mode="lines",
                        name="Cleaned ECG",
                    )
                )

                selected_peaks = peaks[peaks < max_samples]

                fig.add_trace(
                    go.Scatter(
                        x=selected_peaks / fs,
                        y=cleaned[selected_peaks],
                        mode="markers",
                        name="R-peaks",
                        marker=dict(size=7),
                    )
                )

                fig.update_xaxes(title="Time (seconds)")
                fig.update_yaxes(title="Amplitude")
                fig = apply_plotly_layout(
                    fig,
                    f"R-peak Detection — {preprocessing_record.upper()}",
                    480,
                )

                st.plotly_chart(fig, width="stretch")

                c1, c2 = st.columns(2)
                c1.metric("Detected R-peaks", f"{len(peaks):,}")
                c2.metric("Mean beats/min", f"{len(peaks) / (len(cleaned) / fs / 60):.1f}")

            if not rpeak_summary.empty:
                record_col = detect_record_column(rpeak_summary)
                if record_col:
                    record_rows = rpeak_summary[
                        rpeak_summary[record_col].astype(str)
                        == preprocessing_record
                    ]
                    if not record_rows.empty:
                        st.subheader("Stored Phase-1 record summary")
                        display_dataframe(
                            record_rows,
                            width="stretch",
                            hide_index=True,
                        )

        else:

            if not rpeak_summary.empty:
                record_col = detect_record_column(rpeak_summary)
                if record_col:
                    overview = rpeak_summary.copy()
                    overview = overview[
                        overview[record_col].astype(str).isin(
                            preprocessing_records
                        )
                    ]
                    st.subheader("R-peak summary across all records")
                    display_dataframe(
                        overview,
                        width="stretch",
                        hide_index=True,
                    )

            st.markdown(
                "**Select an individual record above** to inspect its "
                "R-peaks directly on the ECG waveform."
            )

    # --------------------------------------------------------
    # RR / HRV
    # --------------------------------------------------------

    with tabs[2]:

        st.subheader("RR intervals and HRV")

        if preprocessing_scope == "Individual record":

            signal, fs = _preprocessing_signal(
                preprocessing_record
            )
            cleaned = clean_uploaded_signal(signal, fs)

            if SCIPY_AVAILABLE:
                peaks, _ = find_peaks(
                    cleaned,
                    distance=max(int(fs * 0.35), 1),
                    prominence=max(np.std(cleaned) * 0.5, 1e-8),
                )
                rr = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])
            else:
                rr = np.array([])

            if len(rr):

                rr_fig = go.Figure()
                rr_fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(rr)),
                        y=rr,
                        mode="lines+markers",
                        name="RR interval",
                    )
                )
                rr_fig.update_xaxes(title="Beat index")
                rr_fig.update_yaxes(title="RR interval (s)")
                rr_fig = apply_plotly_layout(
                    rr_fig,
                    f"RR Tachogram — {preprocessing_record.upper()}",
                    430,
                )
                st.plotly_chart(rr_fig, width="stretch")

                rr_hrv = {
                    "Mean RR (s)": float(np.mean(rr)),
                    "Mean HR (bpm)": float(60 / np.mean(rr)),
                    "SDNN (s)": float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0,
                    "RMSSD (s)": float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 2 else 0.0,
                    "pNN50 (%)": float(np.mean(np.abs(np.diff(rr)) > 0.05) * 100) if len(rr) > 2 else 0.0,
                    "Min RR (s)": float(np.min(rr)),
                    "Max RR (s)": float(np.max(rr)),
                    "RR Range (s)": float(np.ptp(rr)),
                }

                hrv_df = pd.DataFrame(
                    list(rr_hrv.items()),
                    columns=["HRV Feature", "Value"],
                )

                display_dataframe(
                    hrv_df.round(4),
                    width="stretch",
                    hide_index=True,
                )

            else:
                st.warning("Insufficient R-peaks for RR/HRV visualisation.")

            # Stored Phase-1 HRV time series when available.
            if not rr_hrv_summary.empty:
                record_col = detect_record_column(rr_hrv_summary)
                if record_col:
                    record_df = rr_hrv_summary[
                        rr_hrv_summary[record_col].astype(str)
                        == preprocessing_record
                    ]
                    if not record_df.empty:
                        numeric_cols = record_df.select_dtypes(
                            include=np.number
                        ).columns.tolist()
                        if numeric_cols:
                            preferred = [
                                "mean_hr",
                                "sdnn",
                                "rmssd",
                                "pnn50",
                                "mean_rr",
                            ]
                            available = [
                                x for x in preferred
                                if x in numeric_cols
                            ] or numeric_cols[:8]
                            feature = st.selectbox(
                                "Stored HRV feature",
                                available,
                                key="rr_feature_updated",
                            )
                            fig = plot_phase1_feature_timeseries(
                                rr_hrv_summary,
                                preprocessing_record,
                                feature,
                            )
                            if fig is not None:
                                st.plotly_chart(fig, width="stretch")

        else:

            if not rr_hrv_summary.empty:
                record_col = detect_record_column(rr_hrv_summary)
                if record_col:
                    hrv_all = rr_hrv_summary.copy()
                    hrv_all = hrv_all[
                        hrv_all[record_col].astype(str).isin(
                            preprocessing_records
                        )
                    ]
                    st.subheader("HRV summary across all records")
                    display_dataframe(
                        hrv_all,
                        width="stretch",
                        hide_index=True,
                    )

                    numeric_cols = hrv_all.select_dtypes(
                        include=np.number
                    ).columns.tolist()

                    if numeric_cols:
                        preferred = [
                            "mean_hr",
                            "sdnn",
                            "rmssd",
                            "pnn50",
                            "mean_rr",
                        ]
                        available = [
                            x for x in preferred
                            if x in numeric_cols
                        ] or numeric_cols[:8]

                        feature = st.selectbox(
                            "HRV feature across records",
                            available,
                            key="all_rr_feature_updated",
                        )

                        plot_data = hrv_all.copy()
                        plot_data["Record"] = plot_data[record_col].astype(str).str.upper()
                        plot_data[feature] = pd.to_numeric(
                            plot_data[feature],
                            errors="coerce",
                        )
                        plot_data = plot_data.dropna(subset=[feature])

                        if not plot_data.empty:
                            fig = px.line(
                                plot_data,
                                x="Record",
                                y=feature,
                                markers=True,
                                title=f"{feature} Across All Records",
                            )
                            st.plotly_chart(
                                apply_plotly_layout(fig, None, 430),
                                width="stretch",
                            )
            else:
                st.info("Stored RR/HRV summary is not available.")

    # --------------------------------------------------------
    # PREPROCESSING SUMMARY
    # --------------------------------------------------------

    with tabs[3]:

        st.subheader("Preprocessing Summary")

        if preprocessing_scope == "Individual record":

            record_data = pd.DataFrame()

            if not ml_ready.empty and "group" in ml_ready.columns:
                record_data = ml_ready[
                    ml_ready["group"].astype(str)
                    == preprocessing_record
                ]

            st.markdown(
                f"**Selected record:** {preprocessing_record.upper()}"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "ML-ready epochs",
                f"{len(record_data):,}" if not record_data.empty else "—",
            )

            if not record_data.empty and "label" in record_data.columns:
                labels = pd.to_numeric(
                    record_data["label"],
                    errors="coerce",
                )
                apnea_count = int((labels == 1).sum())
                non_apnea_count = int((labels == 0).sum())
            else:
                apnea_count = 0
                non_apnea_count = 0

            c2.metric(
                "Apnea-labelled epochs",
                f"{apnea_count:,}",
            )
            c3.metric(
                "Non-apnea epochs",
                f"{non_apnea_count:,}",
            )
            c4.metric(
                "Features",
                summary["features"],
            )

            record_summary = pd.DataFrame([
                ["Dataset", preprocessing_dataset],
                ["Record", preprocessing_record.upper()],
                ["ML-ready epochs", len(record_data)],
                ["Apnea-labelled epochs", apnea_count],
                ["Non-apnea epochs", non_apnea_count],
                ["Feature count", summary["features"]],
                ["Temporal context", "5 consecutive 1-minute epochs"],
            ], columns=["Parameter", "Value"])

            display_dataframe(
                record_summary,
                width="stretch",
                hide_index=True,
            )

        else:

            st.markdown(
                f"**Dataset:** {preprocessing_dataset}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "Records",
                len(preprocessing_records),
            )
            c2.metric(
                "Raw epochs",
                f"{summary['raw_epochs']:,}",
            )
            c3.metric(
                "Processed",
                f"{summary['processed_epochs']:,}",
            )
            c4.metric(
                "ML-ready",
                f"{summary['ml_ready_epochs']:,}",
            )
            c5.metric(
                "Features",
                summary["features"],
            )

            display_static_figure(
                "09_preprocessing_summary.png",
                "Stored Phase-1 preprocessing summary.",
            )

            st.subheader("Processing stages")
            summary_table = pd.DataFrame([
                ["1", "Raw ECG", "Continuous single-lead ECG"],
                ["2", "Signal cleaning", "Filtered / normalised ECG"],
                ["3", "R-peak detection", "Cardiac beat locations"],
                ["4", "RR / HRV", "Cardiac interval and variability features"],
                ["5", "Quality filtering", "Removal of unusable epochs"],
                ["6", "Temporal windowing", "Five consecutive 1-minute feature steps"],
            ], columns=["Stage", "Processing", "Output"])

            display_dataframe(
                summary_table,
                width="stretch",
                hide_index=True,
            )

    # --------------------------------------------------------
    # FEATURE EXPLORER
    # --------------------------------------------------------

    with tabs[4]:

        st.subheader("Interactive Feature Explorer")

        if ml_ready.empty:
            st.warning("ML-ready dataset not available.")
        else:

            feature_data = ml_ready.copy()

            if "group" in feature_data.columns:
                feature_data["group"] = (
                    feature_data["group"].astype(str).str.lower()
                )
                feature_data = feature_data[
                    feature_data["group"].isin(preprocessing_records)
                ]

            numeric_cols = feature_data.select_dtypes(
                include=np.number
            ).columns.tolist()

            excluded = {
                "label",
                "y",
                "target",
                "epoch",
                "epoch_index",
            }

            features = [
                c for c in numeric_cols
                if c not in excluded
            ]

            if features:

                feature_col1, feature_col2 = st.columns(2)

                with feature_col1:
                    selected_feature = st.selectbox(
                        "Feature",
                        features,
                        key="feature_select_updated",
                    )

                with feature_col2:
                    feature_scope = st.radio(
                        "Feature scope",
                        ["All records", "Individual record"],
                        horizontal=True,
                        key="feature_scope_updated",
                    )

                feature_record = "All records"

                if feature_scope == "Individual record":
                    feature_record = st.selectbox(
                        "Record",
                        preprocessing_records,
                        format_func=lambda x: x.upper(),
                        key="feature_record_updated",
                    )

                fig = plot_feature_distribution(
                    feature_data,
                    selected_feature,
                    feature_record,
                )

                if fig is not None:
                    st.plotly_chart(
                        fig,
                        width="stretch",
                    )

                # Record-wise comparison
                if "group" in feature_data.columns:
                    grouped = (
                        feature_data
                        .groupby("group")[selected_feature]
                        .mean()
                        .reset_index()
                    )
                    grouped.columns = ["Record", "Mean Value"]
                    grouped["Record"] = grouped["Record"].str.upper()

                    fig = px.bar(
                        grouped,
                        x="Record",
                        y="Mean Value",
                        title=f"Mean {selected_feature} Across Records",
                    )

                    fig.update_xaxes(tickangle=-45)

                    st.plotly_chart(
                        apply_plotly_layout(fig, None, 450),
                        width="stretch",
                    )

                    display_dataframe(
                        grouped.round(4),
                        width="stretch",
                        hide_index=True,
                    )


# ============================================================
# MODEL EVALUATION
# ============================================================



elif page == "Model Evaluation":

    st.markdown('<div class="eyebrow">MODEL ANALYSIS</div>', unsafe_allow_html=True)
    st.title("Model Evaluation")

    st.write(
        "Inspect validated subject-wise performance using "
        "the stored research-dataset predictions."
    )

    if comparison.empty:

        st.warning(
            "Model comparison results are not available."
        )

    else:

        models = [
            x
            for x in [
                "Random Forest",
                "1D CNN",
                "Transformer V1",
            ]
            if x in comparison["Model"].tolist()
        ]

        selected_model = st.selectbox(
            "Model",
            models,
        )

        selected_metric = st.selectbox(
            "Metric",
            list(METRIC_ALIASES.keys()),
        )

        row = comparison[
            comparison["Model"]
            == selected_model
        ]

        if not row.empty:

            value = safe_float(
                row.iloc[0][selected_metric]
            )

            st.metric(
                selected_metric,
                pct(value),
            )

        fig = plot_metric_bars(
            comparison,
            selected_metric,
            models=[selected_model],
            title=f"{selected_model} — {selected_metric}",
        )

        if fig is not None:
            st.plotly_chart(
                fig,
                width="stretch",
            )

        st.subheader(
            "All stored metrics for selected model"
        )

        selected_row = comparison[
            comparison["Model"]
            == selected_model
        ]

        display = selected_row.copy()

        for metric in METRIC_ALIASES:
            if metric in display.columns:
                display[metric] = display[
                    metric
                ].map(
                    lambda x: (
                        f"{x * 100:.2f}%"
                        if not pd.isna(x)
                        else "—"
                    )
                )

        display_dataframe(
            display,
            width="stretch",
            hide_index=True,
        )

        if selected_model in predictions:

            st.subheader(
                "Stored Predictions"
            )

            pred_df = predictions[
                selected_model
            ]

            cols = st.columns(4)

            metrics = calculate_metrics_from_predictions(
                pred_df
            )

            with cols[0]:
                st.metric(
                    "True Positives",
                    f"{metrics.get('TP', 0):,}",
                )

            with cols[1]:
                st.metric(
                    "True Negatives",
                    f"{metrics.get('TN', 0):,}",
                )

            with cols[2]:
                st.metric(
                    "False Positives",
                    f"{metrics.get('FP', 0):,}",
                )

            with cols[3]:
                st.metric(
                    "False Negatives",
                    f"{metrics.get('FN', 0):,}",
                )


# ============================================================
# COMPARATIVE MONITORING
# ============================================================


elif page == "Comparative Monitoring":

    st.title("Comparative Monitoring")

    st.write(
        "Explore how the evaluated models behave across the "
        "same research dataset."
    )

    st.info(
        "The visualization is a retrospective playback of "
        "stored predictions, not live physiological monitoring."
    )

    st.subheader(
        "Three comparison perspectives"
    )

    st.write(
        "Use the comparison mode to switch between:"
    )

    st.markdown(
        "- **All models × selected metric**"
    )

    st.markdown(
        "- **Selected model × all metrics**"
    )

    st.markdown(
        "- **All models × all metrics**"
    )

    st.divider()

    mode = st.selectbox(
        "Comparison mode",
        [
            "All models × selected metric",
            "Selected model × all metrics",
            "All models × all metrics",
        ],
    )

    if comparison.empty:

        st.warning(
            "Model comparison data is unavailable."
        )

    else:

        # ----------------------------------------------------
        # MODE A
        # ----------------------------------------------------

        if mode == "All models × selected metric":

            metric = st.selectbox(
                "Select metric",
                list(METRIC_ALIASES.keys()),
            )

            st.subheader(
                f"Comparing all 3 models on {metric}"
            )

            fig = plot_metric_bars(
                comparison,
                metric,
                models=[
                    "Random Forest",
                    "1D CNN",
                    "Transformer V1",
                ],
            )

            if fig is not None:
                st.plotly_chart(
                    fig,
                    width="stretch",
                )

            display = comparison[
                ["Model", metric]
            ].copy()

            display[metric] = display[
                metric
            ].map(
                lambda x: (
                    f"{x * 100:.2f}%"
                    if not pd.isna(x)
                    else "—"
                )
            )

            display_dataframe(
                display,
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # MODE B
        # ----------------------------------------------------

        elif mode == "Selected model × all metrics":

            model = st.selectbox(
                "Select model",
                [
                    "Random Forest",
                    "1D CNN",
                    "Transformer V1",
                ],
            )

            st.subheader(
                f"{model} across all evaluation metrics"
            )

            fig = plot_all_metrics_grouped(
                comparison,
                model,
            )

            if fig is not None:
                st.plotly_chart(
                    fig,
                    width="stretch",
                )

            row = comparison[
                comparison["Model"]
                == model
            ].copy()

            if not row.empty:

                display = row.copy()

                for metric in METRIC_ALIASES:
                    display[metric] = display[
                        metric
                    ].map(
                        lambda x: (
                            f"{x * 100:.2f}%"
                            if not pd.isna(x)
                            else "—"
                        )
                    )

                display_dataframe(
                    display,
                    width="stretch",
                    hide_index=True,
                )

        # ----------------------------------------------------
        # MODE C
        # ----------------------------------------------------

        else:

            st.subheader(
                "All evaluated models × all metrics"
            )

            fig = plot_heatmap(
                comparison
            )

            if fig is not None:
                st.plotly_chart(
                    fig,
                    width="stretch",
                )

            st.subheader("Grouped Column Comparison")
            st.caption(
                "Each model is shown as its own colour group, with all eight evaluation metrics kept together for direct side-by-side comparison."
            )

            grouped_fig = plot_all_models_all_metrics_grouped(
                comparison
            )

            if grouped_fig is not None:
                st.plotly_chart(
                    grouped_fig,
                    width="stretch",
                )

            display = comparison.copy()

            for metric in METRIC_ALIASES:
                display[metric] = display[
                    metric
                ].map(
                    lambda x: (
                        f"{x * 100:.2f}%"
                        if not pd.isna(x)
                        else "—"
                    )
                )

            display_dataframe(
                display,
                width="stretch",
                hide_index=True,
            )

    st.divider()

    # --------------------------------------------------------
    # SCREENING SAFETY
    # --------------------------------------------------------

    st.subheader(
        "Screening-Safety Indicator"
    )

    st.write(
        "On the evaluated research dataset, compare the number "
        "of correctly detected apnea windows and missed apnea "
        "windows."
    )

    safety_rows = []

    for model in [
        "Random Forest",
        "1D CNN",
        "Transformer V1",
    ]:

        if model not in predictions:
            continue

        counts = confusion_counts(
            predictions[model]
        )

        safety_rows.append(
            {
                "Model": model,
                "True Positives": counts["TP"],
                "False Negatives": counts["FN"],
            }
        )

    safety_df = pd.DataFrame(
        safety_rows
    )

    if not safety_df.empty:

        st.subheader("TP / TN / FP / FN — All Models")
        st.caption(
            "Grouped comparison of all four confusion counts for Random Forest, 1D CNN and Transformer V1."
        )
        confusion_grouped_fig = plot_confusion_counts_grouped(predictions)
        if confusion_grouped_fig is not None:
            st.plotly_chart(confusion_grouped_fig, width="stretch")

        cols = st.columns(3)

        for col, model in zip(
            cols,
            safety_df["Model"],
        ):

            row = safety_df[
                safety_df["Model"]
                == model
            ].iloc[0]

            with col:

                st.metric(
                    f"{model} — TP",
                    f"{int(row['True Positives']):,}",
                )

                st.metric(
                    f"{model} — FN",
                    f"{int(row['False Negatives']):,}",
                )

        best_tp = safety_df.loc[
            safety_df["True Positives"].idxmax()
        ]

        lowest_fn = safety_df.loc[
            safety_df["False Negatives"].idxmin()
        ]

        if (
            best_tp["Model"]
            == "Transformer V1"
            and lowest_fn["Model"]
            == "Transformer V1"
        ):
            st.success(
                "Transformer V1 has the highest true-positive "
                "count and the lowest false-negative count among "
                "the three evaluated models on this research dataset."
            )

        st.caption(
            "This is an observation from the evaluated research "
            "dataset and is not a universal or clinical performance claim."
        )

    st.divider()

    # --------------------------------------------------------
    # RETROSPECTIVE PLAYBACK
    # --------------------------------------------------------

    st.subheader(
        "Retrospective Prediction Playback"
    )

    available_models = [
        model
        for model in [
            "Random Forest",
            "1D CNN",
            "Transformer V1",
        ]
        if model in predictions
    ]

    selected_playback_models = st.multiselect(
        "Models shown in playback",
        available_models,
        default=available_models,
    )

    if selected_playback_models:

        record_options = ["All records"]

        if not transformer_predictions.empty:

            record_options += sorted(
                transformer_predictions[
                    "group"
                ]
                .dropna()
                .astype(str)
                .unique()
            )

        playback_record = st.selectbox(
            "Record filter",
            record_options,
            key="playback_record",
        )

        playback_source = (
            transformer_predictions
        )

        if playback_record != "All records":

            playback_source = (
                transformer_predictions[
                    transformer_predictions[
                        "group"
                    ].astype(str)
                    == playback_record
                ]
            )

        if playback_source.empty:

            st.warning(
                "No playback data available."
            )

        else:

            max_index = max(
                0,
                len(playback_source) - 1,
            )

            window_range = st.slider(
                "Temporal window range",
                min_value=0,
                max_value=max_index,
                value=(
                    0,
                    min(
                        max_index,
                        500,
                    ),
                ),
            )

            threshold = st.slider(
                "Decision threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.01,
            )

            start, end = window_range

            fig = go.Figure()

            for model in selected_playback_models:

                model_df = predictions[
                    model
                ]

                if playback_record != "All records":
                    model_df = model_df[
                        model_df["group"].astype(str)
                        == playback_record
                    ]

                segment = model_df.iloc[
                    start:end + 1
                ]

                fig.add_trace(
                    go.Scatter(
                        x=np.arange(
                            start,
                            start + len(segment),
                        ),
                        y=segment[
                            "probability"
                        ],
                        mode="lines",
                        name=model,
                    )
                )

            fig.add_hline(
                y=threshold,
                line_dash="dash",
                annotation_text=(
                    f"Threshold {threshold:.2f}"
                ),
            )

            fig.update_yaxes(
                range=[0, 1],
                title="Predicted probability",
            )

            fig.update_xaxes(
                title="Temporal window",
            )

            fig = apply_plotly_layout(
                fig,
                "Stored Prediction Sequence",
                500,
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            st.caption(
                "Use the range controls and legend to inspect "
                "stored model probabilities over the selected "
                "research-dataset windows."
            )


# ============================================================
# SCREENING RESULTS
# ============================================================


elif page == "Screening Results":

    st.markdown('<div class="eyebrow">SCREENING EXPLORATION</div>', unsafe_allow_html=True)
    st.title("Screening Results")

    st.write(
        "Explore record-level screening flags, probability behaviour, "
        "model metrics and feature-level explanations."
    )

    st.info(
        "A screening flag indicates an apnea-like pattern in the evaluated "
        "ECG windows. It is not a clinical diagnosis and must not be "
        "interpreted as AHI."
    )

    active_source = st.session_state.get(
        "active_dataset_source",
        "PhysioNet Apnea-ECG",
    )

    # --------------------------------------------------------
    # SOURCE SELECTOR
    # --------------------------------------------------------

    screening_source = st.radio(
        "Dataset / screening source",
        [
            "Learning Dataset (A01–C10)",
            "Test Dataset (X01–X35)",
            "Uploaded ECG",
        ],
        horizontal=True,
        key="screening_source_updated",
    )

    threshold = st.slider(
        "Record screening threshold — percentage of windows predicted positive",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
    )

    # --------------------------------------------------------
    # LEARNING DATASET — REAL STORED PREDICTIONS
    # --------------------------------------------------------

    if screening_source.startswith("Learning"):

        screening = build_record_screening(
            learning_transformer_predictions,
            threshold,
        )

        if screening.empty:
            st.warning(
                "Stored Transformer V1 prediction data is unavailable."
            )
        else:

            screening = screening[
                screening["Record"].astype(str).isin(
                    EXPECTED_LEARNING_RECORDS
                )
            ].copy()

            positive = screening[
                screening["Screening Flag"] == "APNEA-LIKE FLAG"
            ]
            negative = screening[
                screening["Screening Flag"] == "NON-APNEA FLAG"
            ]

            st.success(
                "Using stored Proposed Transformer V2 predictions from the "
                "learning dataset. Research prototype only; not a clinical diagnosis."
            )

            cols = st.columns(4)
            cols[0].metric("Research Records", len(screening))
            cols[1].metric("Apnea-like Flag", len(positive))
            cols[2].metric("Non-apnea Flag", len(negative))
            cols[3].metric("Threshold", f"{threshold}%")

            st.subheader("Which records fall into each category?")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### APNEA-LIKE FLAG")
                st.write(
                    ", ".join(
                        positive["Record"].astype(str).str.upper().tolist()
                    ) if not positive.empty else "None"
                )

            with col2:
                st.markdown("### NON-APNEA FLAG")
                st.write(
                    ", ".join(
                        negative["Record"].astype(str).str.upper().tolist()
                    ) if not negative.empty else "None"
                )

            st.subheader("All learning-record screening results")
            display_dataframe(
                screening,
                width="stretch",
                hide_index=True,
            )

            st.divider()
            st.subheader("Inspect an individual learning record")

            selected_record = st.selectbox(
                "Select record",
                EXPECTED_LEARNING_RECORDS,
                format_func=lambda x: x.upper(),
                key="research_screening_record_updated",
            )

            selected_rows = screening[
                screening["Record"].astype(str)
                == selected_record
            ]

            if selected_rows.empty:
                st.info(
                    f"{selected_record.upper()} has no stored prediction rows."
                )
            else:
                selected = selected_rows.iloc[0]

                if selected["Screening Flag"] == "APNEA-LIKE FLAG":
                    st.success(
                        f"{selected_record.upper()} — APNEA-LIKE SCREENING FLAG"
                    )
                else:
                    st.info(
                        f"{selected_record.upper()} — NON-APNEA SCREENING FLAG"
                    )

                cols = st.columns(4)
                cols[0].metric(
                    "Windows",
                    f"{int(selected['Windows']):,}",
                )
                cols[1].metric(
                    "Positive Windows",
                    f"{int(selected['Positive Windows']):,}",
                )
                cols[2].metric(
                    "Positive Rate",
                    pct(selected["Positive Rate"]),
                )
                cols[3].metric(
                    "Mean Probability",
                    f"{selected['Mean Probability']:.3f}",
                )

                fig = plot_record_probability(
                    learning_transformer_predictions,
                    selected_record,
                    threshold,
                )

                if fig is not None:
                    st.plotly_chart(fig, width="stretch")

                st.subheader("Record-level evaluation metrics")

                metric_cols = st.columns(5)
                record_metrics = [
                    ("Accuracy", selected["Accuracy"]),
                    ("Sensitivity", selected["Sensitivity"]),
                    ("Specificity", selected["Specificity"]),
                    ("Precision", selected["Precision"]),
                    ("F1 Score", selected["F1 Score"]),
                ]

                for col, (name, value) in zip(
                    metric_cols,
                    record_metrics,
                ):
                    with col:
                        st.metric(name, pct(value))

                st.subheader("Confusion counts")
                cols = st.columns(4)
                for col, name in zip(
                    cols,
                    ["TP", "TN", "FP", "FN"],
                ):
                    with col:
                        st.metric(
                            name,
                            f"{int(selected[name]):,}",
                        )

                st.divider()
                st.subheader("Explainable Results")
                st.info(
                    "Feature-level importance is used as an interpretation "
                    "aid. It is not presented as direct Transformer attention."
                )

                if not feature_importance.empty:
                    top_n = st.slider(
                        "Features shown",
                        min_value=3,
                        max_value=min(11, len(feature_importance)),
                        value=min(8, len(feature_importance)),
                        key="screening_xai_updated",
                    )

                    fig = plot_feature_importance(
                        feature_importance,
                        top_n,
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")

                    profile = record_feature_profile(
                        selected_record,
                        learning_transformer_predictions,
                        ml_ready,
                        feature_importance,
                    )

                    if not profile.empty:
                        fig = plot_record_xai(
                            profile,
                            top_n,
                        )
                        if fig is not None:
                            st.plotly_chart(fig, width="stretch")

    # --------------------------------------------------------
    # TEST DATASET — DEMO VALUES ONLY
    # --------------------------------------------------------

    elif screening_source.startswith("Test"):

        st.markdown(
            """
            <div style="background:#FFF7E6;border:1px solid #E6B84C;
            border-radius:10px;padding:12px 16px;margin-bottom:15px;">
            <b>DEMO / ILLUSTRATIVE TEST VIEW</b><br>
            The current X01–X35 inference output is not presented as a
            validated research result because the test preprocessing was
            found to differ from the training feature pipeline. The values
            below are deterministic illustrative entries for interface
            demonstration only.
            </div>
            """,
            unsafe_allow_html=True,
        )

        rng = np.random.default_rng(20260909)
        demo_rows = []

        for record in EXPECTED_TEST_RECORDS:
            positive_rate = float(
                np.clip(
                    rng.uniform(10, 90),
                    2,
                    98,
                )
            )
            windows = int(rng.integers(390, 510))
            positive_windows = int(
                windows * positive_rate / 100
            )
            mean_probability = float(
                np.clip(
                    0.20 + positive_rate / 130 + rng.normal(0, 0.025),
                    0.05,
                    0.97,
                )
            )

            demo_rows.append({
                "Record": record,
                "Windows": windows,
                "Positive Windows": positive_windows,
                "Positive Rate": positive_rate / 100,
                "Mean Probability": mean_probability,
                "Screening Flag": (
                    "APNEA-LIKE FLAG"
                    if positive_rate >= threshold
                    else "NON-APNEA FLAG"
                ),
            })

        demo_screening = pd.DataFrame(demo_rows)

        positive = demo_screening[
            demo_screening["Screening Flag"] == "APNEA-LIKE FLAG"
        ]
        negative = demo_screening[
            demo_screening["Screening Flag"] == "NON-APNEA FLAG"
        ]

        cols = st.columns(4)
        cols[0].metric("Test Records", len(demo_screening))
        cols[1].metric("Apnea-like Flags", len(positive))
        cols[2].metric("Non-apnea Flags", len(negative))
        cols[3].metric("Threshold", f"{threshold}%")

        display_dataframe(
            demo_screening.assign(
                **{
                    "Record": demo_screening["Record"].str.upper(),
                    "Positive Rate": demo_screening["Positive Rate"].map(
                        lambda x: f"{x * 100:.2f}%"
                    ),
                    "Mean Probability": demo_screening["Mean Probability"].map(
                        lambda x: f"{x:.3f}"
                    ),
                }
            ),
            width="stretch",
            hide_index=True,
        )

        selected_record = st.selectbox(
            "Select X record",
            EXPECTED_TEST_RECORDS,
            format_func=lambda x: x.upper(),
            key="test_screening_record_updated",
        )

        selected = demo_screening[
            demo_screening["Record"] == selected_record
        ].iloc[0]

        if selected["Screening Flag"] == "APNEA-LIKE FLAG":
            st.success(
                f"{selected_record.upper()} — APNEA-LIKE SCREENING FLAG"
            )
        else:
            st.info(
                f"{selected_record.upper()} — NON-APNEA SCREENING FLAG"
            )

        cols = st.columns(4)
        cols[0].metric("Windows", f"{selected['Windows']:,}")
        cols[1].metric(
            "Positive Windows",
            f"{selected['Positive Windows']:,}",
        )
        cols[2].metric(
            "Positive Rate",
            f"{selected['Positive Rate'] * 100:.2f}%",
        )
        cols[3].metric(
            "Mean Probability",
            f"{selected['Mean Probability']:.3f}",
        )

        # Illustrative temporal probability curve
        record_seed = sum(ord(c) for c in selected_record) + 5000
        record_rng = np.random.default_rng(record_seed)
        n = min(200, int(selected["Windows"]))
        probs = np.clip(
            record_rng.normal(
                selected["Mean Probability"],
                0.10,
                n,
            ),
            0,
            1,
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(n),
                y=probs,
                mode="lines",
                name="Illustrative probability",
            )
        )
        fig.add_hline(
            y=threshold / 100,
            line_dash="dash",
            annotation_text=f"Threshold = {threshold / 100:.2f}",
        )
        fig.update_xaxes(title="Temporal window")
        fig.update_yaxes(
            title="Apnea-like probability",
            range=[0, 1],
        )
        st.plotly_chart(
            apply_plotly_layout(
                fig,
                f"Illustrative Screening Probability — {selected_record.upper()}",
                450,
            ),
            width="stretch",
        )

        st.subheader("Illustrative evaluation summary")

        # Fixed deterministic demo values — clearly labelled.
        demo_metrics = {
            "Accuracy": 0.74,
            "Sensitivity": 0.71,
            "Specificity": 0.77,
            "Precision": 0.68,
            "F1 Score": 0.695,
        }

        metric_cols = st.columns(5)
        for col, (name, value) in zip(
            metric_cols,
            demo_metrics.items(),
        ):
            with col:
                st.metric(name, f"{value * 100:.1f}%")

        st.subheader("Illustrative confusion counts")
        demo_counts = {
            "TP": 31,
            "TN": 38,
            "FP": 11,
            "FN": 13,
        }
        cols = st.columns(4)
        for col, name in zip(cols, demo_counts):
            with col:
                st.metric(name, f"{demo_counts[name]}")

        st.subheader("Illustrative feature-level explanation")

        demo_xai = pd.DataFrame({
            "Feature": [
                "cvrr",
                "edr_amp_std",
                "sdnn",
                "rmssd",
                "pnn50",
                "mean_rr",
            ],
            "Contribution Proxy": [
                0.0366,
                0.0353,
                0.0224,
                0.0207,
                0.0202,
                0.0105,
            ],
        })

        fig = px.bar(
            demo_xai.sort_values("Contribution Proxy"),
            x="Contribution Proxy",
            y="Feature",
            orientation="h",
            title="Illustrative Feature Contribution Proxy",
            text="Contribution Proxy",
        )
        fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
        )
        st.plotly_chart(
            apply_plotly_layout(fig, None, 420),
            width="stretch",
        )

    # --------------------------------------------------------
    # UPLOADED ECG — DEMO VALUES + ACTUAL SIGNAL EXPLORATION
    # --------------------------------------------------------

    else:

        st.markdown(
            """
            <div style="background:#FFF7E6;border:1px solid #E6B84C;
            border-radius:10px;padding:12px 16px;margin-bottom:15px;">
            <b>UPLOADED ECG — DEMONSTRATION MODE</b><br>
            Uploaded ECG signals can be explored and preprocessed. The
            screening metrics and model outputs below are illustrative
            interface values because arbitrary external ECG recordings are
            not part of the validated project inference workflow.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_names = st.session_state.get(
            "uploaded_ecg_files",
            [],
        )

        if not uploaded_names:
            st.info(
                "No ECG files are currently loaded. Go to Dataset → Upload ECG "
                "to upload .hea + .dat, CSV or TXT files."
            )

            # Still provide a demo record so the review UI can be explored.
            uploaded_record_names = [
                "Uploaded_ECG_Demo_01",
                "Uploaded_ECG_Demo_02",
                "Uploaded_ECG_Demo_03",
            ]
            st.caption(
                "Demo records are shown only to demonstrate the screening interface."
            )
        else:
            uploaded_record_names = sorted({
                Path(name).stem
                for name in uploaded_names
                if Path(name).suffix.lower()
                in {".hea", ".csv", ".txt"}
            })

            if not uploaded_record_names:
                uploaded_record_names = sorted({
                    Path(name).stem
                    for name in uploaded_names
                })

        uploaded_demo_rows = []
        upload_rng = np.random.default_rng(9092026)

        for name in uploaded_record_names:
            positive_rate = float(
                np.clip(
                    upload_rng.uniform(20, 85),
                    5,
                    95,
                )
            )
            windows = int(upload_rng.integers(300, 500))
            uploaded_demo_rows.append({
                "Record": name,
                "Windows": windows,
                "Positive Windows": int(
                    windows * positive_rate / 100
                ),
                "Positive Rate": positive_rate / 100,
                "Mean Probability": float(
                    np.clip(
                        0.22 + positive_rate / 140,
                        0.05,
                        0.95,
                    )
                ),
                "Screening Flag": (
                    "APNEA-LIKE FLAG"
                    if positive_rate >= threshold
                    else "NON-APNEA FLAG"
                ),
            })

        uploaded_demo = pd.DataFrame(
            uploaded_demo_rows
        )

        positive = uploaded_demo[
            uploaded_demo["Screening Flag"] == "APNEA-LIKE FLAG"
        ]
        negative = uploaded_demo[
            uploaded_demo["Screening Flag"] == "NON-APNEA FLAG"
        ]

        cols = st.columns(4)
        cols[0].metric("Uploaded Records", len(uploaded_demo))
        cols[1].metric("Apnea-like Flags", len(positive))
        cols[2].metric("Non-apnea Flags", len(negative))
        cols[3].metric("Threshold", f"{threshold}%")

        display_dataframe(
            uploaded_demo.assign(
                **{
                    "Positive Rate": uploaded_demo["Positive Rate"].map(
                        lambda x: f"{x * 100:.2f}%"
                    ),
                    "Mean Probability": uploaded_demo["Mean Probability"].map(
                        lambda x: f"{x:.3f}"
                    ),
                }
            ),
            width="stretch",
            hide_index=True,
        )

        selected_upload = st.selectbox(
            "Select uploaded record",
            uploaded_record_names,
            key="uploaded_screening_record_updated",
        )

        selected = uploaded_demo[
            uploaded_demo["Record"] == selected_upload
        ].iloc[0]

        if selected["Screening Flag"] == "APNEA-LIKE FLAG":
            st.success(
                f"{selected_upload} — APNEA-LIKE SCREENING FLAG"
            )
        else:
            st.info(
                f"{selected_upload} — NON-APNEA SCREENING FLAG"
            )

        cols = st.columns(4)
        cols[0].metric("Windows", f"{selected['Windows']:,}")
        cols[1].metric(
            "Positive Windows",
            f"{selected['Positive Windows']:,}",
        )
        cols[2].metric(
            "Positive Rate",
            f"{selected['Positive Rate'] * 100:.2f}%",
        )
        cols[3].metric(
            "Mean Probability",
            f"{selected['Mean Probability']:.3f}",
        )

        # ----------------------------------------------------
        # Actual uploaded signal, if available
        # ----------------------------------------------------

        actual_uploaded = None

        for item in st.session_state.get(
            "uploaded_ecg_results",
            [],
        ):
            if str(item.get("name")) == str(selected_upload):
                actual_uploaded = item
                break

        if actual_uploaded is not None:

            signal = np.asarray(
                actual_uploaded["signal"],
                dtype=float,
            )
            fs = float(
                actual_uploaded["sampling_rate"]
            )

            st.subheader("Uploaded ECG signal exploration")

            preview_seconds = min(
                30,
                max(
                    5,
                    int(len(signal) / fs),
                ),
            )

            st.plotly_chart(
                plot_uploaded_ecg(
                    signal,
                    fs,
                    cleaned=False,
                    max_seconds=preview_seconds,
                ),
                width="stretch",
            )

            cleaned = clean_uploaded_signal(
                signal,
                fs,
            )

            st.plotly_chart(
                plot_uploaded_ecg(
                    cleaned,
                    fs,
                    cleaned=True,
                    max_seconds=preview_seconds,
                ),
                width="stretch",
            )

            peak_fig = uploaded_rpeak_plot(
                signal,
                fs,
            )
            if peak_fig is not None:
                st.plotly_chart(
                    peak_fig,
                    width="stretch",
                )

        # ----------------------------------------------------
        # Illustrative probability curve
        # ----------------------------------------------------

        upload_seed = sum(ord(c) for c in selected_upload)
        upload_rng2 = np.random.default_rng(upload_seed)
        n = min(200, int(selected["Windows"]))
        probs = np.clip(
            upload_rng2.normal(
                selected["Mean Probability"],
                0.10,
                n,
            ),
            0,
            1,
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(n),
                y=probs,
                mode="lines",
                name="Illustrative probability",
            )
        )
        fig.add_hline(
            y=threshold / 100,
            line_dash="dash",
            annotation_text=f"Threshold = {threshold / 100:.2f}",
        )
        fig.update_xaxes(title="Temporal window")
        fig.update_yaxes(
            title="Apnea-like probability",
            range=[0, 1],
        )

        st.plotly_chart(
            apply_plotly_layout(
                fig,
                f"Illustrative Screening Probability — {selected_upload}",
                450,
            ),
            width="stretch",
        )

        st.subheader("Illustrative evaluation summary")

        demo_metrics = {
            "Accuracy": 0.76,
            "Sensitivity": 0.72,
            "Specificity": 0.79,
            "Precision": 0.69,
            "F1 Score": 0.705,
        }

        metric_cols = st.columns(5)
        for col, (name, value) in zip(
            metric_cols,
            demo_metrics.items(),
        ):
            with col:
                st.metric(name, f"{value * 100:.1f}%")

        st.subheader("Illustrative confusion counts")

        demo_counts = {
            "TP": 34,
            "TN": 41,
            "FP": 11,
            "FN": 13,
        }

        cols = st.columns(4)
        for col, name in zip(cols, demo_counts):
            with col:
                st.metric(
                    name,
                    f"{demo_counts[name]}",
                )

        st.subheader("Illustrative feature-level explanation")

        demo_xai = pd.DataFrame({
            "Feature": [
                "cvrr",
                "edr_amp_std",
                "sdnn",
                "rmssd",
                "pnn50",
                "mean_rr",
            ],
            "Contribution Proxy": [
                0.0366,
                0.0353,
                0.0224,
                0.0207,
                0.0202,
                0.0105,
            ],
        })

        fig = px.bar(
            demo_xai.sort_values("Contribution Proxy"),
            x="Contribution Proxy",
            y="Feature",
            orientation="h",
            title="Illustrative Feature Contribution Proxy",
            text="Contribution Proxy",
        )
        fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
        )
        st.plotly_chart(
            apply_plotly_layout(fig, None, 420),
            width="stretch",
        )

        st.caption(
            "Uploaded-record screening values are illustrative UI/demo values. "
            "They are not model-validated predictions."
        )


# ============================================================
# RESULTS DASHBOARD
# ============================================================



elif page == "Results Dashboard":

    st.title("Results Dashboard")

    st.write(
        "Use the controls below to decide exactly which "
        "Phase-1 or Phase-2 result you want to inspect."
    )

    # --------------------------------------------------------
    # DASHBOARD CONTROLS
    # --------------------------------------------------------

    st.subheader(
        "Dashboard Controls"
    )

    section_options = [
        "Pipeline Overview",
        "Model Performance",
        "Screening Safety",
        "Confusion Matrix",
        "ROC / PR",
        "Error Analysis",
        "Explainability",
        "Computational Efficiency",
        "Record-Level Screening",
        "Generated Figures",
        "Download Results",
    ]

    selected_sections = st.multiselect(
        "Sections to display",
        section_options,
        default=section_options,
    )

    selected_models = st.multiselect(
        "Models",
        [
            "Random Forest",
            "1D CNN",
            "Transformer V1",
        ],
        default=[
            "Random Forest",
            "1D CNN",
            "Transformer V1",
        ],
    )

    selected_metrics = st.multiselect(
        "Metrics",
        list(METRIC_ALIASES.keys()),
        default=[
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC",
        ],
    )

    st.divider()

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    if "Pipeline Overview" in selected_sections:

        st.subheader(
            "Pipeline Overview"
        )

        cols = st.columns(5)

        values = [
            (
                "Learning Records",
                summary["learning_records"],
            ),
            (
                "Processed Epochs",
                summary["processed_epochs"],
            ),
            (
                "ML-ready Epochs",
                summary["ml_ready_epochs"],
            ),
            (
                "Temporal Windows",
                summary["temporal_windows"],
            ),
            (
                "Features",
                summary["features"],
            ),
        ]

        for col, (label, value) in zip(
            cols,
            values,
        ):
            with col:
                st.metric(
                    label,
                    f"{value:,}",
                )

        st.caption(
            "The dashboard reports the stored Phase-1 and "
            "Phase-2 project artifacts."
        )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    if "Model Performance" in selected_sections:

        st.divider()

        st.subheader(
            "Model Performance"
        )

        if not selected_metrics:
            st.info(
                "Select at least one metric above."
            )

        elif comparison.empty or "Model" not in comparison.columns:
            st.warning(
                "Model comparison data is unavailable or malformed. "
                "This section will appear when valid evaluation artifacts "
                "are available."
            )

        else:
            available_models = [
                model for model in selected_models
                if model in comparison["Model"].astype(str).tolist()
            ]
            available_metrics = [
                metric for metric in selected_metrics
                if metric in comparison.columns
            ]

            if not available_models:
                st.warning(
                    "No selected models are available in the comparison data."
                )
            elif not available_metrics:
                st.warning(
                    "None of the selected metrics are available in the comparison data."
                )
            else:
                metric = st.selectbox(
                    "Performance metric",
                    available_metrics,
                    key="dashboard_metric",
                )

                fig = plot_metric_bars(
                    comparison,
                    metric,
                    available_models,
                    title=f"{metric} - Selected Models",
                )

                if fig is not None:
                    st.plotly_chart(
                        fig,
                        width="stretch",
                    )

                display = comparison[
                    comparison["Model"].astype(str).isin(
                        available_models
                    )
                ][["Model"] + available_metrics].copy()

                for col in available_metrics:
                    display[col] = display[col].map(
                        lambda value: (
                            f"{value * 100:.2f}%"
                            if pd.notna(value)
                            else "?"
                        )
                    )

                display_dataframe(
                    display,
                    width="stretch",
                    hide_index=True,
                )

    # --------------------------------------------------------
    # SCREENING SAFETY
    # --------------------------------------------------------

    if "Screening Safety" in selected_sections:

        st.divider()

        st.subheader(
            "Screening Safety Comparison"
        )

        st.write(
            "Comparison of true-positive and false-negative "
            "counts on the evaluated research dataset."
        )

        safety_rows = []

        for model in selected_models:

            if model not in predictions:
                continue

            counts = confusion_counts(
                predictions[model]
            )

            safety_rows.append(
                {
                    "Model": model,
                    "True Positives": counts["TP"],
                    "False Negatives": counts["FN"],
                }
            )

        safety_df = pd.DataFrame(
            safety_rows
        )

        if not safety_df.empty:

            st.subheader("TP / TN / FP / FN — All Models")
            st.caption(
                "Grouped comparison of all four confusion counts for the selected models."
            )
            confusion_grouped_fig = plot_confusion_counts_grouped(
                predictions,
                selected_models,
            )
            if confusion_grouped_fig is not None:
                st.plotly_chart(confusion_grouped_fig, width="stretch")

            cols = st.columns(
                len(safety_df)
            )

            for col, (_, row) in zip(
                cols,
                safety_df.iterrows(),
            ):
                with col:

                    st.metric(
                        f"{row['Model']} — TP",
                        f"{int(row['True Positives']):,}",
                    )

                    st.metric(
                        f"{row['Model']} — FN",
                        f"{int(row['False Negatives']):,}",
                    )

            safety_plot = safety_df.melt(
                id_vars="Model",
                value_vars=[
                    "True Positives",
                    "False Negatives",
                ],
                var_name="Measure",
                value_name="Count",
            )

            fig = px.bar(
                safety_plot,
                x="Model",
                y="Count",
                color="Measure",
                barmode="group",
                text="Count",
            )

            fig.update_traces(
                textposition="outside"
            )

            fig = apply_plotly_layout(
                fig,
                "True Positives vs False Negatives",
                450,
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            if (
                "Transformer V1"
                in safety_df["Model"].tolist()
            ):

                transformer_row = safety_df[
                    safety_df["Model"]
                    == "Transformer V1"
                ].iloc[0]

                best_tp = (
                    transformer_row[
                        "True Positives"
                    ]
                    == safety_df[
                        "True Positives"
                    ].max()
                )

                lowest_fn = (
                    transformer_row[
                        "False Negatives"
                    ]
                    == safety_df[
                        "False Negatives"
                    ].min()
                )

                if best_tp and lowest_fn:
                    st.success(
                        "Transformer V1 shows the highest true-positive "
                        "count and lowest false-negative count among "
                        "the evaluated models on this research dataset."
                    )

            st.caption(
                "This comparison is specific to the stored research "
                "dataset and should not be interpreted as clinical validation."
            )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    if "Confusion Matrix" in selected_sections:

        st.divider()

        st.subheader(
            "Confusion Matrix"
        )

        confusion_model = st.selectbox(
            "Model",
            [
                model
                for model in selected_models
                if model in predictions
            ],
            key="dashboard_confusion_model",
        )

        if confusion_model:

            fig = plot_confusion_matrix(
                predictions[
                    confusion_model
                ],
                confusion_model,
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    # --------------------------------------------------------
    # ROC / PR
    # --------------------------------------------------------

    if "ROC / PR" in selected_sections:

        st.divider()

        st.subheader(
            "ROC and Precision–Recall Analysis"
        )

        if selected_models:

            tab_roc, tab_pr = st.tabs(
                [
                    "ROC",
                    "Precision–Recall",
                ]
            )

            with tab_roc:

                fig = plot_roc(
                    predictions,
                    selected_models,
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

            with tab_pr:

                fig = plot_pr(
                    predictions,
                    selected_models,
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

    # --------------------------------------------------------
    # ERROR ANALYSIS
    # --------------------------------------------------------

    if "Error Analysis" in selected_sections:

        st.divider()

        st.subheader(
            "Error Analysis"
        )

        error_df = build_error_summary(
            predictions
        )

        if not error_df.empty:

            error_df = error_df[
                error_df["Model"].isin(
                    selected_models
                )
            ]

            fig = plot_error_summary(
                error_df
            )

            if fig is not None:
                st.plotly_chart(
                    fig,
                    width="stretch",
                )

            display_dataframe(
                error_df,
                width="stretch",
                hide_index=True,
            )

    # --------------------------------------------------------
    # XAI
    # --------------------------------------------------------

    if "Explainability" in selected_sections:

        st.divider()

        st.subheader(
            "Explainability"
        )

        st.write(
            "The current XAI evidence is based on feature-level "
            "importance analysis using ECG-derived physiological "
            "features."
        )

        st.info(
            "This is not presented as direct Transformer attention."
        )

        if not feature_importance.empty:

            xai_n = st.slider(
                "Features shown",
                min_value=3,
                max_value=min(
                    11,
                    len(feature_importance),
                ),
                value=min(
                    8,
                    len(feature_importance),
                ),
                key="dashboard_xai_n",
            )

            fig = plot_feature_importance(
                feature_importance,
                xai_n,
            )

            if fig is not None:
                st.plotly_chart(
                    fig,
                    width="stretch",
                )

    # --------------------------------------------------------
    # COMPLEXITY
    # --------------------------------------------------------

    if "Computational Efficiency" in selected_sections:

        st.divider()

        st.subheader(
            "Computational Efficiency"
        )

        complexity = prepare_complexity()

        if not complexity.empty:

            if "Model" in complexity.columns:

                complexity = complexity[
                    complexity["Model"].isin(
                        selected_models
                    )
                    | complexity["Model"].str.contains(
                        "Transformer",
                        case=False,
                        na=False,
                    )
                ]

            display_dataframe(
                complexity,
                width="stretch",
                hide_index=True,
            )

            numeric_columns = [
                c
                for c in [
                    "Parameters",
                    "Model Size (MB)",
                    "CPU Latency (ms)",
                ]
                if c in complexity.columns
            ]

            if numeric_columns:

                complexity_metric = st.selectbox(
                    "Complexity metric",
                    numeric_columns,
                    key="complexity_metric",
                )

                chart_data = complexity[
                    [
                        "Model",
                        complexity_metric,
                    ]
                ].dropna()

                if not chart_data.empty:

                    fig = px.bar(
                        chart_data,
                        x="Model",
                        y=complexity_metric,
                        text=complexity_metric,
                    )

                    fig.update_traces(
                        textposition="outside"
                    )

                    fig = apply_plotly_layout(
                        fig,
                        f"{complexity_metric} Comparison",
                        430,
                    )

                    st.plotly_chart(
                        fig,
                        width="stretch",
                    )

    # --------------------------------------------------------
    # RECORD SCREENING
    # --------------------------------------------------------

    if "Record-Level Screening" in selected_sections:

        st.divider()

        st.subheader(
            "Record-Level Screening"
        )

        dashboard_threshold = st.slider(
            "Screening threshold (% positive windows)",
            min_value=1,
            max_value=100,
            value=50,
            key="dashboard_threshold",
        )

        screening = build_record_screening(
            transformer_predictions,
            dashboard_threshold,
        )

        if not screening.empty:

            cols = st.columns(3)

            positive = screening[
                screening["Screening Flag"]
                == "APNEA-LIKE FLAG"
            ]

            negative = screening[
                screening["Screening Flag"]
                == "NON-APNEA FLAG"
            ]

            with cols[0]:
                st.metric(
                    "Records",
                    len(screening),
                )

            with cols[1]:
                st.metric(
                    "Apnea-like flags",
                    len(positive),
                )

            with cols[2]:
                st.metric(
                    "Non-apnea flags",
                    len(negative),
                )

            fig = px.bar(
                pd.DataFrame(
                    {
                        "Category": [
                            "Apnea-like flag",
                            "Non-apnea flag",
                        ],
                        "Records": [
                            len(positive),
                            len(negative),
                        ],
                    }
                ),
                x="Category",
                y="Records",
                text="Records",
            )

            fig.update_traces(
                textposition="outside"
            )

            fig = apply_plotly_layout(
                fig,
                "Record-Level Screening Overview",
                400,
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    # --------------------------------------------------------
    # GENERATED FIGURES
    # --------------------------------------------------------

    if "Generated Figures" in selected_sections:

        st.divider()

        st.subheader(
            "Generated Research Figures"
        )

        figure_files = sorted(
            [
                p
                for p in FIGURES_DIR.rglob("*")
                if p.is_file()
                and p.suffix.lower()
                in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                }
            ]
        )

        if not figure_files:

            st.info(
                "No generated figures were found."
            )

        else:

            selected_figure_name = st.selectbox(
                "Select generated figure",
                [
                    p.relative_to(
                        FIGURES_DIR
                    ).as_posix()
                    for p in figure_files
                ],
            )

            selected_figure = (
                FIGURES_DIR
                / selected_figure_name
            )

            st.image(
                str(selected_figure),
                width="stretch",
            )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    if "Download Results" in selected_sections:

        st.divider()

        st.subheader(
            "Research Results Package"
        )

        st.write(
            "Download the complete project-generated results "
            "package, including Phase-1 processed data, Phase-1 "
            "figures, Phase-2 predictions, evaluation metrics, "
            "comparative analysis, explainability outputs, "
            "complexity analysis, screening summaries and "
            "available trained model artifacts."
        )

        zip_bytes = build_results_zip()

        st.download_button(
            label="Download Complete Results Package (.zip)",
            data=zip_bytes,
            file_name=(
                "ECG_Sleep_Apnea_Complete_Results.zip"
            ),
            mime="application/zip",
            width="stretch",
        )

        st.caption(
            "The package contains project-generated artifacts "
            "available in the repository. The original PhysioNet "
            "raw dataset is not bundled."
        )


# ============================================================
# FOOTER
# ============================================================


st.markdown(
    """
    <div class="footer-note">
        ECG-Based Sleep Apnea Screening • Research prototype •
        Screening support only • Not a medical diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)