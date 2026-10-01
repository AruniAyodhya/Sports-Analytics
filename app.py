"""European Soccer Match Outcome Predictor | Decision Support System.

Interactive Streamlit Dashboard for Supervised Multi-Class Match Outcome Prediction.
Group: 2026-AI-09 | SLIIT Machine Learning Module IT3091
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from typing import Dict, Any

# Ensure project root is on sys.path across all execution environments
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from src.inference import SoccerInferenceEngine

# =============================================================================
# THEME CONFIGURATION — Inline HTML & Plotly Chart Color Palettes
# =============================================================================
_THEME_PALETTE = {
    "dark": {
        "heading": "#f8fafc",
        "subtext": "#94a3b8",
        "body": "#f1f5f9",
        "muted": "#64748b",
        "market_bg": "rgba(255, 255, 255, 0.03)",
        "market_value": "#f1f5f9",
        "market_vig": "#94a3b8",
        "chart_title": "#e2e8f0",
        "chart_axis": "#94a3b8",
        "chart_ytick": "#f1f5f9",
        "chart_grid": "rgba(255, 255, 255, 0.07)",
        "chart_bar_line": "rgba(255, 255, 255, 0.15)",
        "hover_bg": "#1e293b",
        "hover_border": "#334155",
        "hover_text": "#f8fafc",
        "model_rf": "#10b981",
        "model_lr": "#38bdf8",
        "model_dt": "#f59e0b",
        "model_xgb": "#ec4899",
        "baseline_line": "#f43f5e",
    },
    "light": {
        "heading": "#0f172a",
        "subtext": "#475569",
        "body": "#1e293b",
        "muted": "#64748b",
        "market_bg": "rgba(15, 23, 42, 0.03)",
        "market_value": "#1e293b",
        "market_vig": "#475569",
        "chart_title": "#1e293b",
        "chart_axis": "#475569",
        "chart_ytick": "#1e293b",
        "chart_grid": "rgba(0, 0, 0, 0.07)",
        "chart_bar_line": "rgba(0, 0, 0, 0.1)",
        "hover_bg": "#ffffff",
        "hover_border": "#cbd5e1",
        "hover_text": "#0f172a",
        "model_rf": "#059669",
        "model_lr": "#0284c7",
        "model_dt": "#d97706",
        "model_xgb": "#be185d",
        "baseline_line": "#e11d48",
    },
}

# =============================================================================
# PAGE SETUP & CSS LOADING
# =============================================================================
st.set_page_config(
    page_title="Soccer Outcome Predictor | Decision Support",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_custom_css() -> None:
    """Load external CSS stylesheet from assets/style.css."""
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom stylesheet assets/style.css not found. Falling back to default styles.")


load_custom_css()


# =============================================================================
# CACHED INFERENCE ENGINE
# =============================================================================
@st.cache_resource(show_spinner="Initializing Model Engine & Baseline Vectors...")
def get_engine(model_key: str = "random_forest") -> SoccerInferenceEngine:
    """Initialize and cache the SoccerInferenceEngine singleton."""
    return SoccerInferenceEngine(model_type=model_key)


# =============================================================================
# REFINED SIDEBAR NAVIGATION & SPECS
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2 class="sidebar-brand-title">⚽ SoccerOutcome AI</h2>
            <p class="sidebar-brand-sub">European League Match Decision Support</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### ⚙️ Configuration")
    model_options = {
        "Random Forest (Tuned) [Active]": "random_forest",
        "Logistic Regression (L1 Tuned) [Active]": "logistic_regression",
        "Decision Tree (Tuned RFE) [Active]": "decision_tree",
        "XGBoost Classifier (Tuned) [Active]": "xgboost",
    }
    selected_model_label = st.selectbox(
        "Active Model Pipeline",
        list(model_options.keys()),
        index=0,
        help="Select the trained classification pipeline artifact.",
    )
    active_model_key = model_options[selected_model_label]

    model_meta = {
        "random_forest": {"name": "Tuned RF", "selected": "34 Features", "selector": "MDI SFM"},
        "logistic_regression": {"name": "L1 Logistic", "selected": "18 Features", "selector": "L1 SFM"},
        "decision_tree": {"name": "Tuned DT", "selected": "20 Features", "selector": "RFE Selected"},
        "xgboost": {"name": "Tuned XGB", "selected": "50 Features", "selector": "Gain SFM"},
    }.get(active_model_key, {"name": "Tuned RF", "selected": "34 Features", "selector": "SFM Selected"})

    st.markdown("##### 🧭 Navigation")
    nav_mode = st.radio(
        "Analysis Mode",
        [
            "Mode 1: Pre-Match Simulator",
            "Mode 2: Historical Evaluator",
            "Mode 3: Model Comparison",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("##### 🔬 Pipeline Architecture")
    st.markdown(
        f"""
        <div class="arch-grid">
            <div class="arch-tile">
                <div class="arch-tile-label">Target Setup</div>
                <div class="arch-tile-value">3 Classes</div>
            </div>
            <div class="arch-tile">
                <div class="arch-tile-label">Stage 1 Drop</div>
                <div class="arch-tile-value">18 Features</div>
            </div>
            <div class="arch-tile">
                <div class="arch-tile-label">Pipeline Input</div>
                <div class="arch-tile-value">68 Features</div>
            </div>
            <div class="arch-tile">
                <div class="arch-tile-label">{model_meta['selector']}</div>
                <div class="arch-tile-value">{model_meta['selected']}</div>
            </div>
            <div class="arch-tile">
                <div class="arch-tile-label">Base Model</div>
                <div class="arch-tile-value">{model_meta['name']}</div>
            </div>
            <div class="arch-tile">
                <div class="arch-tile-label">Leakage Protocol</div>
                <div class="arch-tile-value">Pre-Match Only</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.caption("Group **2026-AI-09** • SLIIT ML Module IT3091")


# ---------------------------------------------------------------------------
# Resolve active theme, state callback, and inject light-mode CSS flag
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    if "theme_radio" in st.session_state and "Light" in str(st.session_state.theme_radio):
        st.session_state.theme = "light"
    else:
        st.session_state.theme = "dark"

active_theme = st.session_state.get("theme", "dark")
tp = _THEME_PALETTE.get(active_theme, _THEME_PALETTE["dark"])
is_dark = active_theme == "dark"

if active_theme == "light":
    st.markdown(
        '<div id="light-theme-flag" aria-hidden="true" '
        'style="display:none;position:absolute;pointer-events:none;"></div>',
        unsafe_allow_html=True,
    )


def toggle_theme() -> None:
    """Toggle between Modern Stadium Dark and Clean Turf Light modes."""
    st.session_state.theme = "light" if st.session_state.get("theme", "dark") == "dark" else "dark"


# Initialize engine with graceful error handling
engine = None
model_error = None
try:
    engine = get_engine(active_model_key)
except FileNotFoundError as fnf_err:
    model_error = str(fnf_err)
except Exception as exc:
    model_error = f"Engine initialization error: {exc}"


# =============================================================================
# HEADER SECTION & MATCHDAY TACTICAL COMMAND CENTER
# =============================================================================
with st.container(key="header_card"):
    head_left, head_right = st.columns([0.74, 0.26], vertical_alignment="center")
    with head_left:
        st.markdown(
            """
            <div class="command-center-header">
                <div class="matchday-badge">
                    <span class="badge-beacon"></span>
                    <span class="badge-tag">OFFICIAL ACCREDITATION</span>
                    <span class="badge-sep">•</span>
                    <span class="badge-org">SLIIT IT3091</span>
                    <span class="badge-sep">•</span>
                    <span class="badge-group">GROUP 2026-AI-09</span>
                </div>
                <h1 class="tactical-title">European Soccer Match Outcome Predictor</h1>
                <p class="tactical-subtitle">
                    Institutional Decision Support System • High-Dimensional Stochastic Outcome Forecasting Powered by Rolling Form Differentials & Market Consensus Intelligence
                </p>
                <div class="tactical-telemetry">
                    <span class="telemetry-chip chip-active"><span class="telemetry-dot"></span>TACTICAL RADAR ACTIVE</span>
                    <span class="telemetry-chip">3-CLASS PROBABILISTIC MATRIX</span>
                    <span class="telemetry-chip">PRE-MATCH PARITY PROTOCOL</span>
                    <span class="telemetry-chip">ZERO IN-MATCH LEAKAGE</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with head_right:
        floodlight_status = "STADIUM NIGHT • ACTIVE" if is_dark else "DAYLIGHT TURF • ACTIVE"
        floodlight_pill_cls = "pill-night" if is_dark else "pill-day"
        btn_label = "☀️ DAYLIGHT TURF" if is_dark else "🌙 STADIUM NIGHT"
        btn_help = (
            "Switch to Clean Turf Daylight Mode"
            if is_dark
            else "Switch to Stadium Night Floodlight Mode"
        )

        st.markdown(
            f"""
            <div class="floodlight-console">
                <div class="console-header">
                    <span class="console-icon">🏟️</span>
                    <span class="console-title">FLOODLIGHT SYSTEM</span>
                    <span class="console-beacon {'beacon-on' if is_dark else 'beacon-turf'}"></span>
                </div>
                <div class="console-status-row">
                    <span class="console-state-badge {floodlight_pill_cls}">{floodlight_status}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            btn_label,
            key="theme_toggle_btn",
            help=btn_help,
            on_click=toggle_theme,
            use_container_width=True,
        )

if model_error:
    st.error(f"⚠️ **Artifact Notice**: {model_error}")
    st.info(
        "💡 Please select **'Random Forest (Tuned) [Active]'** in the sidebar to run live predictions."
    )
    st.stop()


# =============================================================================
# PLOTLY PROBABILITY DISTRIBUTION CHART
# =============================================================================
def plot_probabilities(
    probabilities: dict[str, float],
    title: str = "Class Probabilities",
    theme: str = "dark",
):
    """Render a clean, minimalist Plotly horizontal bar chart for the 3 classes."""
    tc = _THEME_PALETTE.get(theme, _THEME_PALETTE["dark"])
    classes = ["Away Win", "Draw", "Home Win"]
    probs = [probabilities.get(c, 0.0) * 100 for c in classes]
    colors = ["#f43f5e", "#f59e0b", "#10b981"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=probs,
            y=classes,
            orientation="h",
            text=[f"<b>{p:.1f}%</b>" for p in probs],
            textposition="auto",
            textfont=dict(family="Plus Jakarta Sans", size=13, color="#ffffff"),
            marker=dict(
                color=colors,
                line=dict(color=tc["chart_bar_line"], width=1),
                cornerradius=6,
            ),
            hoverinfo="y+x",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color=tc["chart_title"], family="Plus Jakarta Sans"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(
                text="Model Probability (%)",
                font=dict(size=11, color=tc["chart_axis"], family="Plus Jakarta Sans"),
            ),
            range=[0, 100],
            gridcolor=tc["chart_grid"],
            tickfont=dict(size=11, color=tc["chart_axis"]),
            zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=13, color=tc["chart_ytick"], family="Plus Jakarta Sans"),
            showgrid=False,
        ),
        margin=dict(l=10, r=20, t=35, b=25),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        height=190,
    )
    return fig


# =============================================================================
# MULTI-MODEL BENCHMARK DATA & LINE CHART RENDERER (MODE 3)
# =============================================================================
@st.cache_data
def load_model_comparison_data() -> dict[str, Any]:
    """Load benchmark and lifecycle metrics across all 4 models directly from research artifacts."""
    base_path = Path(__file__).resolve().parent
    lifecycle_stages = ["Baseline Validation", "Tuned Validation", "Final Test"]
    split_stages = ["Training Split", "Validation Split", "Holdout Testing"]
    model_order = ["Random Forest", "Logistic Regression", "Decision Tree", "XGBoost"]

    # Exact calibrated metrics from notebooks/Model Comparison.ipynb and final_comparison/
    lifecycle_data = {
        "Accuracy": {
            "Random Forest": [0.4636, 0.4641, 0.4501],
            "Logistic Regression": [0.4665, 0.4707, 0.4517],
            "Decision Tree": [0.4246, 0.4678, 0.4483],
            "XGBoost": [0.4688, 0.4646, 0.4532],
        },
        "Macro F1": {
            "Random Forest": [0.4543, 0.4538, 0.4405],
            "Logistic Regression": [0.4561, 0.4599, 0.4394],
            "Decision Tree": [0.4259, 0.4581, 0.4345],
            "XGBoost": [0.4385, 0.4440, 0.4331],
        },
        "ROC-AUC": {
            "Random Forest": [0.6635, 0.6632, 0.6503],
            "Logistic Regression": [0.6641, 0.6647, 0.6525],
            "Decision Tree": [0.6522, 0.6559, 0.6362],
            "XGBoost": [0.6579, 0.6605, 0.6474],
        },
        "Macro Precision": {
            "Random Forest": [0.4637, 0.4628, 0.4506],
            "Logistic Regression": [0.4707, 0.4732, 0.4505],
            "Decision Tree": [0.4710, 0.4710, 0.4450],
            "XGBoost": [0.4375, 0.4453, 0.4342],
        },
        "Macro Recall": {
            "Random Forest": [0.4572, 0.4558, 0.4411],
            "Logistic Regression": [0.4561, 0.4603, 0.4382],
            "Decision Tree": [0.4394, 0.4580, 0.4324],
            "XGBoost": [0.4444, 0.4460, 0.4342],
        },
        "Weighted F1": {
            "Random Forest": [0.4738, 0.4746, 0.4606],
            "Logistic Regression": [0.4784, 0.4820, 0.4615],
            "Decision Tree": [0.4373, 0.4792, 0.4571],
            "XGBoost": [0.4670, 0.4694, 0.4568],
        },
    }

    # Training, Tuned Validation, and Final Holdout Testing splits
    split_data = {
        "Accuracy": {
            "Random Forest": [0.5440, 0.4641, 0.4501],
            "Logistic Regression": [0.5207, 0.4707, 0.4517],
            "Decision Tree": [0.5193, 0.4678, 0.4483],
            "XGBoost": [0.5299, 0.4646, 0.4532],
        },
        "Macro F1": {
            "Random Forest": [0.5728, 0.4538, 0.4405],
            "Logistic Regression": [0.4850, 0.4599, 0.4394],
            "Decision Tree": [0.4416, 0.4581, 0.4345],
            "XGBoost": [0.4812, 0.4440, 0.4331],
        },
        "ROC-AUC": {
            "Random Forest": [0.7679, 0.6632, 0.6503],
            "Logistic Regression": [0.6720, 0.6647, 0.6525],
            "Decision Tree": [0.6510, 0.6559, 0.6362],
            "XGBoost": [0.6895, 0.6605, 0.6474],
        },
    }

    # Dynamically overlay directly from CSV files in final_comparison/
    p_comp = base_path / "final_comparison"
    try:
        p_b = p_comp / "comparison_baseline_validation.csv"
        p_v = p_comp / "comparison_validation.csv"
        p_t = p_comp / "comparison_test.csv"
        if p_b.exists() and p_v.exists() and p_t.exists():
            df_b = pd.read_csv(p_b, index_col=0)
            df_v = pd.read_csv(p_v, index_col=0)
            df_t = pd.read_csv(p_t, index_col=0)
            for m in ["Accuracy", "Macro F1", "ROC-AUC", "Macro Precision", "Macro Recall", "Weighted F1"]:
                if m in df_b.columns:
                    for model in model_order:
                        if model in df_b.index:
                            vb = round(float(df_b.loc[model, m]), 4)
                            vv = round(float(df_v.loc[model, m]), 4)
                            vt = round(float(df_t.loc[model, m]), 4)
                            lifecycle_data[m][model] = [vb, vv, vt]
                            if m in split_data:
                                split_data[m][model][1] = vv
                                split_data[m][model][2] = vt
    except Exception:
        pass

    # Build comprehensive 6-metric lifecycle dataframe
    all_metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "ROC-AUC"]
    matrix_rows = []
    for model in model_order:
        for m in all_metrics:
            vals = lifecycle_data.get(m, {}).get(model, [0.0, 0.0, 0.0])
            b_val, v_val, t_val = vals[0], vals[1], vals[2]
            uplift = v_val - b_val
            drop = t_val - v_val
            matrix_rows.append({
                "Model Family": model,
                "Metric": m,
                "Baseline Val": f"{b_val:.4f}",
                "Tuned Val": f"{v_val:.4f}",
                "Final Test": f"{t_val:.4f}",
                "Tuning Uplift": f"{uplift:+.4f}",
                "Generalization Drop": f"{drop:+.4f}",
            })
    lifecycle_matrix_df = pd.DataFrame(matrix_rows)

    return {
        "model_order": model_order,
        "lifecycle_stages": lifecycle_stages,
        "split_stages": split_stages,
        "lifecycle_data": lifecycle_data,
        "split_data": split_data,
        "lifecycle_matrix_df": lifecycle_matrix_df,
        "majority_acc": 0.4421,
        "majority_f1": 0.2047,
    }


def plot_comparison_metric(
    metric_name: str,
    stages: list[str],
    model_data: dict[str, list[float]],
    theme: str = "dark",
    ref_line: float | None = None,
    ref_label: str = "Majority Baseline",
    height: int = 420,
    show_labels: bool = False,
    y_title: str | None = None,
) -> go.Figure:
    """Render an un-cramped, interactive multi-model progression line chart with theme-aware styling."""
    tc = _THEME_PALETTE.get(theme, _THEME_PALETTE["dark"])
    is_dark = theme == "dark"

    # High-contrast, brand-aligned colors and distinct marker shapes per model family
    model_styles = {
        "Random Forest": {
            "color": tc.get("model_rf", "#10b981" if is_dark else "#059669"),
            "symbol": "circle",
            "offset": "top left",
        },
        "Logistic Regression": {
            "color": tc.get("model_lr", "#38bdf8" if is_dark else "#0284c7"),
            "symbol": "square",
            "offset": "top right",
        },
        "Decision Tree": {
            "color": tc.get("model_dt", "#f59e0b" if is_dark else "#d97706"),
            "symbol": "diamond",
            "offset": "bottom left",
        },
        "XGBoost": {
            "color": tc.get("model_xgb", "#ec4899" if is_dark else "#be185d"),
            "symbol": "triangle-up",
            "offset": "bottom right",
        },
    }

    fig = go.Figure()

    # Determine dynamic, unclipped y-axis bounds with generous headroom
    all_vals: list[float] = [v for vals in model_data.values() for v in vals if v is not None]
    if ref_line is not None:
        all_vals.append(ref_line)
    
    val_min = min(all_vals)
    val_max = max(all_vals)
    span = max(0.015, val_max - val_min)
    y_min = max(0.0, val_min - span * 0.20)
    y_max = min(1.0, val_max + span * 0.22)

    marker_border = "#090d16" if is_dark else "#ffffff"

    for m_name, vals in model_data.items():
        style = model_styles.get(m_name, {"color": "#94a3b8", "symbol": "circle", "offset": "top center"})
        trace_mode = "lines+markers+text" if show_labels else "lines+markers"

        fig.add_trace(
            go.Scatter(
                x=stages,
                y=vals,
                mode=trace_mode,
                name=m_name,
                line=dict(color=style["color"], width=3),
                marker=dict(
                    size=9,
                    symbol=style["symbol"],
                    color=style["color"],
                    line=dict(color=marker_border, width=1.5),
                ),
                text=[f"{v:.4f}" for v in vals] if show_labels else None,
                textposition=style.get("offset", "top center"),
                textfont=dict(size=11, color=tc["chart_ytick"], family="Plus Jakarta Sans"),
                hovertemplate=f"<b>{m_name}</b>: %{{y:.4f}}<extra></extra>",
            )
        )

    # Clean horizontal reference line (e.g., Majority Class Baseline)
    if ref_line is not None:
        ref_color = tc.get("baseline_line", "#f43f5e" if is_dark else "#e11d48") if "Macro F1" in metric_name else tc["subtext"]
        fig.add_hline(
            y=ref_line,
            line_dash="dot",
            line_color=ref_color,
            line_width=1.75,
            annotation_text=f"<b>{ref_label} ({ref_line:.4f})</b>",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color=ref_color, family="Plus Jakarta Sans"),
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{metric_name}</b> <span style='font-size:12px; font-weight:normal; color:{tc['subtext']};'>— Multi-Model Progression Trajectory</span>",
            font=dict(size=15, color=tc["chart_title"], family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98,
        ),
        xaxis=dict(
            title=dict(
                text="Progression Stage",
                font=dict(size=12, color=tc["chart_axis"], family="Plus Jakarta Sans"),
                standoff=12,
            ),
            gridcolor=tc["chart_grid"],
            linecolor=tc["chart_grid"],
            tickfont=dict(size=12, color=tc["chart_axis"], family="Plus Jakarta Sans"),
            zeroline=False,
            showline=True,
        ),
        yaxis=dict(
            title=dict(
                text=y_title or metric_name,
                font=dict(size=12, color=tc["chart_axis"], family="Plus Jakarta Sans"),
                standoff=16,
            ),
            range=[y_min, y_max],
            tickformat=".4f",
            gridcolor=tc["chart_grid"],
            linecolor=tc["chart_grid"],
            tickfont=dict(size=11, color=tc["chart_ytick"], family="Plus Jakarta Sans"),
            zeroline=False,
            showline=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color=tc["chart_axis"], family="Plus Jakarta Sans"),
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=tc.get("hover_bg", "#1e293b" if is_dark else "#ffffff"),
            bordercolor=tc.get("hover_border", "#334155" if is_dark else "#cbd5e1"),
            font=dict(
                family="Plus Jakarta Sans",
                size=12,
                color=tc.get("hover_text", "#f8fafc" if is_dark else "#0f172a"),
            ),
        ),
        margin=dict(l=75, r=35, t=80, b=60),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        height=height,
    )
    return fig


def plot_generalization_gap_chart(
    model_order: list[str],
    f1_drops: list[float],
    acc_drops: list[float],
    theme: str = "dark",
    height: int = 300,
) -> go.Figure:
    """Render an interactive horizontal bar chart comparing validation-to-test performance drops."""
    tc = _THEME_PALETTE.get(theme, _THEME_PALETTE["dark"])
    is_dark = theme == "dark"

    fig = go.Figure()

    # Macro F1 Drop trace
    fig.add_trace(
        go.Bar(
            name="Macro F1 Drop (Δ points)",
            y=model_order,
            x=[abs(d) for d in f1_drops],
            orientation="h",
            marker=dict(
                color="#f43f5e" if is_dark else "#e11d48",
                line=dict(color=tc["chart_bar_line"], width=1),
                cornerradius=4,
            ),
            text=[f"{d:+.4f}" for d in f1_drops],
            textposition="auto",
            textfont=dict(family="Plus Jakarta Sans", size=11, color="#ffffff"),
            hovertemplate="<b>%{y}</b><br>Macro F1 Drop: <b>%{text}</b><extra></extra>",
        )
    )

    # Accuracy Drop trace
    fig.add_trace(
        go.Bar(
            name="Accuracy Drop (Δ % / 100)",
            y=model_order,
            x=[abs(d) for d in acc_drops],
            orientation="h",
            marker=dict(
                color="#38bdf8" if is_dark else "#0284c7",
                line=dict(color=tc["chart_bar_line"], width=1),
                cornerradius=4,
            ),
            text=[f"{d:+.4f}" for d in acc_drops],
            textposition="auto",
            textfont=dict(family="Plus Jakarta Sans", size=11, color="#ffffff"),
            hovertemplate="<b>%{y}</b><br>Accuracy Drop: <b>%{text}</b><extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Validation-to-Test Performance Decay Comparison</b>",
            font=dict(size=14, color=tc["chart_title"], family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98,
        ),
        xaxis=dict(
            title=dict(
                text="Absolute Drop Magnitude (Lower is More Stable)",
                font=dict(size=11, color=tc["chart_axis"], family="Plus Jakarta Sans"),
            ),
            gridcolor=tc["chart_grid"],
            tickfont=dict(size=11, color=tc["chart_axis"], family="Plus Jakarta Sans"),
            zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12, color=tc["chart_ytick"], family="Plus Jakarta Sans"),
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color=tc["chart_axis"], family="Plus Jakarta Sans"),
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        hoverlabel=dict(
            bgcolor=tc.get("hover_bg", "#1e293b" if is_dark else "#ffffff"),
            bordercolor=tc.get("hover_border", "#334155" if is_dark else "#cbd5e1"),
            font=dict(family="Plus Jakarta Sans", size=12, color=tc.get("hover_text", "#f8fafc" if is_dark else "#0f172a")),
        ),
        margin=dict(l=130, r=30, t=70, b=50),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        height=height,
    )
    return fig


# =============================================================================
# MODE 1: PRE-MATCH PERFORMANCE SIMULATOR (LIVE REACTIVE DASHBOARD)
# =============================================================================
if nav_mode == "Mode 1: Pre-Match Simulator":
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
            <div>
                <h4 style="margin: 0; font-weight: 700; color: {tp['heading']};">🎯 Live Pre-Match Simulator</h4>
                <p style="color: {tp['subtext']}; font-size: 0.88rem; margin: 2px 0 0 0;">
                    Dynamic fixture scenario modeling • Instantly re-evaluates probabilities on widget interaction.
                </p>
            </div>
            <div class="live-pulse-badge">
                <span class="pulse-dot"></span> Live Telemetry Active
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.25, 1], gap="medium")

    # Column 1: Match & League Context
    with col1:
        st.markdown(
            """
            <div class="card-title">
                <span class="card-title-icon">🏟️</span> Match & League Context
            </div>
            """,
            unsafe_allow_html=True,
        )
        league_choices = list(SoccerInferenceEngine.CLEAN_LEAGUE_MAP.keys())
        selected_league = st.selectbox(
            "League Competition",
            league_choices,
            index=0,
            help="Select one of 10 domestic European league competitions.",
        )
        stage_val = st.slider(
            "Match Stage / Matchweek",
            min_value=1,
            max_value=38,
            value=19,
            step=1,
            help="Week of the domestic season (1 to 38).",
        )
        st.caption("ℹ️ League encoded via one-hot representation; stage is standardized.")

    # Column 2: Relative Performance Metrics (Secondary Analytical Lens)
    with col2:
        st.markdown(
            """
            <div class="card-title">
                <span class="card-title-icon">📈</span> Relative Form & Strength Differentials
            </div>
            """,
            unsafe_allow_html=True,
        )
        ppm_diff = st.slider(
            "Points/Match Difference (Home - Away)",
            min_value=-3.0,
            max_value=3.0,
            value=0.5,
            step=0.1,
            help="Full-season Points-Per-Match differential (+ favours Home, - favours Away).",
        )
        recent_pts_diff = st.slider(
            "Recent 5-Match Points Difference",
            min_value=-3.0,
            max_value=3.0,
            value=0.8,
            step=0.1,
            help="Rolling short-term points trajectory across last 5 competitive matches.",
        )
        goals_diff = st.slider(
            "Goal Difference/Match (Home - Away)",
            min_value=-3.0,
            max_value=3.0,
            value=0.4,
            step=0.1,
            help="Average goal differential per game between the two teams.",
        )
        win_rate_diff = st.slider(
            "Recent Win Rate Difference",
            min_value=-1.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Difference in recent winning percentage (+0.2 = Home has 20% higher recent win rate).",
        )

    # Column 3: Market Consensus Odds
    with col3:
        st.markdown(
            """
            <div class="card-title">
                <span class="card-title-icon">💰</span> Market Consensus Odds
            </div>
            """,
            unsafe_allow_html=True,
        )
        b365_h = st.number_input(
            "Bet365 Home Win (B365H)",
            min_value=1.05,
            max_value=30.0,
            value=2.10,
            step=0.05,
            help="Market decimal payout for Home Win.",
        )
        b365_d = st.number_input(
            "Bet365 Draw (B365D)",
            min_value=1.20,
            max_value=20.0,
            value=3.40,
            step=0.05,
            help="Market decimal payout for Draw.",
        )
        b365_a = st.number_input(
            "Bet365 Away Win (B365A)",
            min_value=1.05,
            max_value=30.0,
            value=3.60,
            step=0.05,
            help="Market decimal payout for Away Win.",
        )

        # Normalized market implied probabilities & bookmaker overround
        inv_h = 1.0 / b365_h if b365_h > 0 else 0.0
        inv_d = 1.0 / b365_d if b365_d > 0 else 0.0
        inv_a = 1.0 / b365_a if b365_a > 0 else 0.0
        total_inv = inv_h + inv_d + inv_a
        overround_pct = (total_inv - 1.0) * 100 if total_inv > 0 else 0.0

        imp_h = (inv_h / total_inv) * 100 if total_inv > 0 else 33.33
        imp_d = (inv_d / total_inv) * 100 if total_inv > 0 else 33.33
        imp_a = (inv_a / total_inv) * 100 if total_inv > 0 else 33.34

        st.markdown(
            f"""
            <div style="background: {tp['market_bg']}; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 10px 12px; margin-top: 8px;">
                <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                    Normalized Market Implied Probabilities
                </div>
                <div style="font-size: 0.86rem; font-weight: 600; color: {tp['market_value']}; display: flex; justify-content: space-between;">
                    <span>H: <b style="color: #34d399;">{imp_h:.1f}%</b></span>
                    <span>D: <b style="color: #fbbf24;">{imp_d:.1f}%</b></span>
                    <span>A: <b style="color: #fb7185;">{imp_a:.1f}%</b></span>
                </div>
                <div style="font-size: 0.72rem; color: {tp['market_vig']}; margin-top: 5px;">
                    Bookmaker Vig / Margin: <b>+{overround_pct:.1f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Automatic, Real-Time Inference Execution (Zero Button Latency)
    sim_dict = {
        "league": selected_league,
        "stage": stage_val,
        "B365H": b365_h,
        "B365D": b365_d,
        "B365A": b365_a,
        "points_per_match_difference": ppm_diff,
        "recent_points_difference": recent_pts_diff,
        "goals_for_difference": goals_diff,
        "win_rate_difference": win_rate_diff,
    }

    sim_res = engine.predict_simulation(sim_dict)
    pred_label = sim_res["predicted_label"]
    confidence = sim_res["confidence"]
    probs = sim_res["probabilities"]

    card_variant = {
        "Home Win": "outcome-home-min",
        "Draw": "outcome-draw-min",
        "Away Win": "outcome-away-min",
    }.get(pred_label, "outcome-home-min")

    icon = {"Home Win": "🏠", "Draw": "🤝", "Away Win": "✈️"}.get(pred_label, "⚽")

    st.markdown("<hr/>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns([1, 1.45], gap="large")

    with res_col1:
        st.markdown(
            f"""
            <div class="outcome-card-minimal {card_variant}">
                <span class="outcome-tag">Live Predicted Outcome</span>
                <div class="outcome-main">{icon} {pred_label}</div>
                <div class="outcome-confidence">
                    Model Confidence: <span class="outcome-conf-val">{confidence:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        st.plotly_chart(
            plot_probabilities(probs, title="Real-Time Probability Distribution", theme=active_theme),
            use_container_width=True,
        )


# =============================================================================
# MODE 2: HISTORICAL TEST MATCH EVALUATOR
# =============================================================================
elif nav_mode == "Mode 2: Historical Evaluator":
    st.markdown("#### 🔍 Historical Holdout Match Audit")
    st.markdown(
        f"<p style='color:{tp['subtext']}; font-size:0.9rem; margin-top:-8px; margin-bottom:18px;'>"
        "Audit model predictions against the unseen holdout test set (3,923 fixtures) to inspect ground-truth accuracy."
        "</p>",
        unsafe_allow_html=True,
    )

    test_X, test_y = engine.get_test_data()
    max_idx = len(test_X) - 1

    if "fixture_input" not in st.session_state:
        st.session_state["fixture_input"] = 0

    def pick_random_fixture():
        st.session_state["fixture_input"] = random.randint(0, max_idx)

    tool_col1, tool_col2 = st.columns([2.5, 1], gap="medium")

    with tool_col1:
        match_idx = st.number_input(
            f"Holdout Fixture Index (0 to {max_idx:,})",
            min_value=0,
            max_value=max_idx,
            step=1,
            key="fixture_input",
            help="Select a fixture index from the holdout test dataset (0 to 3,922).",
        )

    with tool_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.button(
            "🎲 Random Fixture",
            on_click=pick_random_fixture,
            type="secondary",
            use_container_width=True,
            help="Randomly select an unseen fixture from the holdout test set and immediately audit it.",
        )

    # Perform inference on fixture immediately
    test_res = engine.predict_test_match(int(match_idx))
    meta = test_res["metadata"]
    odds = meta.get("odds", {})

    # Fixture context bar
    st.markdown(
        f"""
        <div class="fixture-bar">
            <div class="fixture-pill">
                <span>Fixture ID:</span> <strong>#{match_idx}</strong>
            </div>
            <div class="fixture-pill">
                <span>League:</span> <strong>{meta.get('league')}</strong>
            </div>
            <div class="fixture-pill">
                <span>Stage:</span> <strong>Matchweek {meta.get('stage')}</strong>
            </div>
            <div class="fixture-pill">
                <span>Bet365 Odds:</span> 
                <strong>{odds.get('Home Win (B365H)', '-')} / {odds.get('Draw (B365D)', '-')} / {odds.get('Away Win (B365A)', '-')}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3 Minimalist KPI Tiles
    pred_label = test_res["predicted_label"]
    actual_label = test_res["actual_label"]
    is_correct = test_res["is_correct"]
    confidence = test_res["confidence"]

    m1, m2, m3 = st.columns(3, gap="medium")

    with m1:
        color = "#10b981" if pred_label == "Home Win" else ("#f59e0b" if pred_label == "Draw" else "#f43f5e")
        st.markdown(
            f"""
            <div class="kpi-tile" style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">
                <div class="kpi-label">Predicted Outcome</div>
                <div class="kpi-value" style="color: {color};">{pred_label}</div>
                <div class="kpi-sub">Confidence: <b>{confidence:.1f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="kpi-tile" style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">
                <div class="kpi-label">Actual Ground Truth</div>
                <div class="kpi-value">{actual_label}</div>
                <div class="kpi-sub">Official Final Result</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        status_html = (
            '<span class="status-pill-ok">Match Correct ✅</span>'
            if is_correct
            else '<span class="status-pill-wrong">Misclassified ❌</span>'
        )
        status_sub = "Accurate Prediction" if is_correct else "Outcome Divergence"
        st.markdown(
            f"""
            <div class="kpi-tile" style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">
                <div class="kpi-label">Verification Status</div>
                <div class="kpi-value kpi-status-container">{status_html}</div>
                <div class="kpi-sub">{status_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.plotly_chart(
        plot_probabilities(test_res["probabilities"], title=f"Fixture #{match_idx} Model Probability Distribution", theme=active_theme),
        use_container_width=True,
    )


# =============================================================================
# MODE 3: MULTI-MODEL BENCHMARK & COMPARISON DASHBOARD
# =============================================================================
elif nav_mode == "Mode 3: Model Comparison":
    comp_data = load_model_comparison_data()
    majority_acc = comp_data["majority_acc"]
    majority_f1 = comp_data["majority_f1"]

    st.markdown("#### 🏆 Multi-Model Benchmark & Lifecycle Evaluation")
    st.markdown(
        f"<p style='color:{tp['subtext']}; font-size:0.9rem; margin-top:-8px; margin-bottom:18px;'>"
        "Comprehensive comparative technical review across all four model families (Random Forest, Logistic Regression, Decision Tree, and XGBoost) "
        "tracking classification performance trajectories, generalization retention on unseen holdout test data (3,923 matches), and overfitting resistance directly from "
        "<code>notebooks/Model Comparison.ipynb</code>."
        "</p>",
        unsafe_allow_html=True,
    )

    # 4 Executive KPI Tiles
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown(
            f"""
            <div class="kpi-tile" style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">
                <div class="kpi-label">Selected Primary Model</div>
                <div class="kpi-value" style="color: {tp['model_lr']};">Logistic Regression</div>
                <div class="kpi-sub">Val Macro F1: <b>0.4599</b> (Optimal Benchmark)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-tile" style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">
                <div class="kpi-label">Top Holdout Macro F1</div>
                <div class="kpi-value" style="color: {tp['model_rf']};">Random Forest</div>
                <div class="kpi-sub">Test F1: <b>0.4405</b> (Decay: -1.33%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-tile" style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">
                <div class="kpi-label">Top Holdout Accuracy</div>
                <div class="kpi-value" style="color: {tp['model_xgb']};">XGBoost</div>
                <div class="kpi-sub">Test Acc: <b>45.32%</b> (Decay: -1.14%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-tile" style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">
                <div class="kpi-label">Heuristic Majority Baseline</div>
                <div class="kpi-value" style="color: {tp['model_dt']};">Home Win (44.21%)</div>
                <div class="kpi-sub">Macro F1: <b>0.2047</b> (Beaten by +115%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Control Toolbar: Progression Dimension Selector & Data Point Label Toggle
    ctrl_col1, ctrl_col2 = st.columns([0.68, 0.32], vertical_alignment="center")
    with ctrl_col1:
        prog_mode = st.radio(
            "Evaluation Progression Dimension",
            [
                "📊 Lifecycle Tuning Stages (Baseline Val → Tuned Val → Final Test)",
                "🎯 Dataset Split Generalization (Training Split → Validation Split → Holdout Testing)",
            ],
            index=0,
            horizontal=True,
            help="Toggle between lifecycle hyperparameter tuning stages and raw dataset train/val/test splits for overfitting inspection.",
        )
    with ctrl_col2:
        show_data_labels = st.checkbox(
            "🏷️ Overlay Direct Data Point Values",
            value=False,
            help="Overlay numerical values directly on chart data points using smart collision offsets. When unchecked, hover over any point for the high-precision inspector.",
        )

    is_lifecycle = "Lifecycle" in prog_mode
    active_stages = comp_data["lifecycle_stages"] if is_lifecycle else comp_data["split_stages"]
    active_dataset = comp_data["lifecycle_data"] if is_lifecycle else comp_data["split_data"]

    # Interactive Visual Tabs: Clean, Spacious, and Un-cramped
    tab_acc, tab_f1, tab_auc, tab_stack, tab_gap, tab_matrix = st.tabs([
        "🎯 Accuracy Benchmark",
        "⚖️ Macro F1-Score (Primary)",
        "📈 ROC-AUC (Separation)",
        "🔬 Multi-Metric Stack",
        "📋 Generalization & Overfitting",
        "📊 Complete Lifecycle Matrix",
    ])

    with tab_acc:
        st.plotly_chart(
            plot_comparison_metric(
                "Accuracy",
                active_stages,
                active_dataset["Accuracy"],
                theme=active_theme,
                ref_line=majority_acc,
                ref_label="Majority Baseline",
                height=440,
                show_labels=show_data_labels,
                y_title="Classification Accuracy",
            ),
            use_container_width=True,
        )
        st.markdown(
            f"""
            <div style="background: {tp['market_bg']}; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 14px 18px; margin-top: 8px;">
                <div style="font-size: 0.88rem; color: {tp['heading']}; font-weight: 600; margin-bottom: 6px;">
                    🎯 Accuracy Progression & Generalization Analysis
                </div>
                <div style="font-size: 0.84rem; color: {tp['subtext']}; line-height: 1.6;">
                    • <b>Baseline Outperformance</b>: All 4 models comfortably surpass the 44.21% naive majority-class baseline on unseen holdout test data (3,923 matches).<br/>
                    • <b>Hyperparameter Tuning Uplift</b>: Decision Tree achieved the highest accuracy gain during hyperparameter tuning (+4.32%), successfully recovering from default depth instability.<br/>
                    • <b>Generalization Stability</b>: Performance drop from validation to test is tightly bounded between <b>-1.14% (XGBoost)</b> and <b>-1.95% (Decision Tree)</b>, confirming zero pathological overfitting.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_f1:
        st.plotly_chart(
            plot_comparison_metric(
                "Macro F1",
                active_stages,
                active_dataset["Macro F1"],
                theme=active_theme,
                ref_line=majority_f1,
                ref_label="Majority Baseline",
                height=440,
                show_labels=show_data_labels,
                y_title="Macro F1-Score (All 3 Classes)",
            ),
            use_container_width=True,
        )
        st.markdown(
            f"""
            <div style="background: {tp['market_bg']}; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 14px 18px; margin-top: 8px;">
                <div style="font-size: 0.88rem; color: {tp['heading']}; font-weight: 600; margin-bottom: 6px;">
                    ⚖️ Macro F1-Score & Minority Class Balance Analysis (Primary Evaluation Metric)
                </div>
                <div style="font-size: 0.84rem; color: {tp['subtext']}; line-height: 1.6;">
                    • <b>Massive Uplift Over Heuristics (+110% to +115%)</b>: While naive majority guessing achieves a dismal 0.2047 Macro F1 due to ignoring Away Wins and Draws, all four trained models achieve <b>0.4331 to 0.4405</b> on unseen holdout test data.<br/>
                    • <b>Primary Model Selection Decision</b>: <b>Logistic Regression</b> achieved the peak validation Macro F1 (<b>0.4599</b>), earning its selection as the primary classifier in the research paper.<br/>
                    • <b>Ensemble Robustness</b>: <b>Random Forest</b> achieved the highest final test Macro F1 (<b>0.4405</b>) with minimal validation-to-test drop (-0.0133), proving exceptional stochastic balance across all three match outcomes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_auc:
        st.plotly_chart(
            plot_comparison_metric(
                "ROC-AUC",
                active_stages,
                active_dataset["ROC-AUC"],
                theme=active_theme,
                ref_line=None,
                height=440,
                show_labels=show_data_labels,
                y_title="One-vs-Rest Multiclass ROC-AUC",
            ),
            use_container_width=True,
        )
        st.markdown(
            f"""
            <div style="background: {tp['market_bg']}; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 14px 18px; margin-top: 8px;">
                <div style="font-size: 0.88rem; color: {tp['heading']}; font-weight: 600; margin-bottom: 6px;">
                    📈 Multiclass Discriminative Separation (ROC-AUC) Analysis
                </div>
                <div style="font-size: 0.84rem; color: {tp['subtext']}; line-height: 1.6;">
                    • <b>Continuous Rank Quality</b>: One-vs-Rest (OvR) multiclass ROC-AUC evaluates predicted continuous probabilities across all three match outcomes.<br/>
                    • <b>Strong Calibration</b>: All models sustain robust discrimination between <b>0.6362 and 0.6525</b> on unseen holdout test data.<br/>
                    • <b>Linear Regularization Excellence</b>: <b>Logistic Regression</b> leads ROC-AUC across both validation (<b>0.6647</b>) and testing (<b>0.6525</b>), benefiting from smooth continuous log-odds probability surfaces without greedy tree step discontinuities.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_stack:
        st.markdown("##### 🔬 Multi-Metric Progression Stack (Full-Width Un-cramped View)")
        st.markdown(
            f"<p style='color:{tp['subtext']}; font-size:0.85rem; margin-top:-6px; margin-bottom:14px;'>"
            "Vertically stacked, full-width performance curves allowing uncluttered, side-by-side technical evaluation across all three core metrics."
            "</p>",
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            plot_comparison_metric(
                "Accuracy",
                active_stages,
                active_dataset["Accuracy"],
                theme=active_theme,
                ref_line=majority_acc,
                ref_label="Majority Baseline",
                height=360,
                show_labels=show_data_labels,
                y_title="Accuracy",
            ),
            use_container_width=True,
        )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        st.plotly_chart(
            plot_comparison_metric(
                "Macro F1",
                active_stages,
                active_dataset["Macro F1"],
                theme=active_theme,
                ref_line=majority_f1,
                ref_label="Majority Baseline",
                height=360,
                show_labels=show_data_labels,
                y_title="Macro F1",
            ),
            use_container_width=True,
        )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        st.plotly_chart(
            plot_comparison_metric(
                "ROC-AUC",
                active_stages,
                active_dataset["ROC-AUC"],
                theme=active_theme,
                ref_line=None,
                height=360,
                show_labels=show_data_labels,
                y_title="ROC-AUC",
            ),
            use_container_width=True,
        )

    with tab_gap:
        st.markdown("##### 📋 Generalization Degradation & Overfitting Audit")
        st.markdown(
            f"<p style='color:{tp['subtext']}; font-size:0.85rem; margin-top:-6px; margin-bottom:14px;'>"
            "Quantitative validation-to-test performance decay analysis confirming that no model exhibits pathological memorization or feature leakage."
            "</p>",
            unsafe_allow_html=True,
        )

        val_accs = [comp_data["lifecycle_data"]["Accuracy"][m][1] for m in comp_data["model_order"]]
        test_accs = [comp_data["lifecycle_data"]["Accuracy"][m][2] for m in comp_data["model_order"]]
        val_f1s = [comp_data["lifecycle_data"]["Macro F1"][m][1] for m in comp_data["model_order"]]
        test_f1s = [comp_data["lifecycle_data"]["Macro F1"][m][2] for m in comp_data["model_order"]]
        val_aucs = [comp_data["lifecycle_data"]["ROC-AUC"][m][1] for m in comp_data["model_order"]]
        test_aucs = [comp_data["lifecycle_data"]["ROC-AUC"][m][2] for m in comp_data["model_order"]]

        f1_drops_list = []
        acc_drops_list = []
        gap_rows = []
        for idx, m in enumerate(comp_data["model_order"]):
            acc_drop = (test_accs[idx] - val_accs[idx]) * 100
            f1_drop = test_f1s[idx] - val_f1s[idx]
            auc_drop = test_aucs[idx] - val_aucs[idx]
            retention = (test_f1s[idx] / val_f1s[idx]) * 100
            risk = "Very Low (Stable)" if abs(f1_drop) <= 0.015 else "Low (Expected Decay)"

            f1_drops_list.append(f1_drop)
            acc_drops_list.append((test_accs[idx] - val_accs[idx]))

            gap_rows.append({
                "Model Family": m,
                "Val Macro F1": f"{val_f1s[idx]:.4f}",
                "Test Macro F1": f"{test_f1s[idx]:.4f}",
                "F1 Drop": f"{f1_drop:+.4f}",
                "F1 Retention": f"{retention:.1f}%",
                "Val Accuracy": f"{val_accs[idx]*100:.2f}%",
                "Test Accuracy": f"{test_accs[idx]*100:.2f}%",
                "Acc Drop": f"{acc_drop:+.2f}%",
                "Test ROC-AUC": f"{test_aucs[idx]:.4f}",
                "Overfitting Assessment": risk,
            })

        df_gap = pd.DataFrame(gap_rows)
        st.dataframe(df_gap, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Interactive horizontal bar chart comparing drops
        st.plotly_chart(
            plot_generalization_gap_chart(
                comp_data["model_order"],
                f1_drops_list,
                acc_drops_list,
                theme=active_theme,
                height=280,
            ),
            use_container_width=True,
        )

        st.markdown(
            f"""
            <div class="fixture-bar" style="margin-top: 12px; display: flex; flex-wrap: wrap; gap: 14px;">
                <div class="fixture-pill">
                    <span>Model Selection Criterion:</span> <strong>Validation Macro F1 (Logistic Regression: 0.4599)</strong>
                </div>
                <div class="fixture-pill">
                    <span>Lowest F1 Generalization Drop:</span> <strong>XGBoost (-0.0109) &amp; Random Forest (-0.0133)</strong>
                </div>
                <div class="fixture-pill">
                    <span>Generalization Retention:</span> <strong>All Models Retain &gt;94.8% of Validation Performance</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_matrix:
        st.markdown("##### 📊 Complete 6-Metric Lifecycle Evaluation Matrix")
        st.markdown(
            f"<p style='color:{tp['subtext']}; font-size:0.85rem; margin-top:-6px; margin-bottom:14px;'>"
            "Full lifecycle performance matrix displaying all six primary classification metrics across Baseline Validation, Tuned Validation, and Final Holdout Testing "
            "directly extracted from <code>notebooks/Model Comparison.ipynb</code> and <code>final_comparison/</code> artifacts."
            "</p>",
            unsafe_allow_html=True,
        )

        matrix_df = comp_data.get("lifecycle_matrix_df", pd.DataFrame())
        matrix_filter = st.selectbox(
            "Filter Model Family",
            ["All Models (Unified Matrix)", "Random Forest", "Logistic Regression", "Decision Tree", "XGBoost"],
            index=0,
            help="Filter matrix rows to inspect a specific classifier lifecycle.",
        )

        if matrix_filter != "All Models (Unified Matrix)":
            filtered_df = matrix_df[matrix_df["Model Family"] == matrix_filter]
        else:
            filtered_df = matrix_df

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            <div style="background: {tp['market_bg']}; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 14px 18px; margin-top: 10px;">
                <div style="font-size: 0.88rem; color: {tp['heading']}; font-weight: 600; margin-bottom: 6px;">
                    📖 Lifecycle Evaluation Methodology & Definitions
                </div>
                <div style="font-size: 0.82rem; color: {tp['subtext']}; line-height: 1.6;">
                    • <b>Baseline Validation</b>: Out-of-the-box model architecture with default hyperparameters evaluated on the validation fold.<br/>
                    • <b>Tuned Validation</b>: Performance after hyperparameter tuning and model-specific feature selection (SFM / RFE).<br/>
                    • <b>Final Test</b>: Out-of-sample holdout evaluation on 3,923 strictly unseen historical matches.<br/>
                    • <b>Tuning Uplift</b>: Metric gain achieved through hyperparameter optimization (Tuned Val − Baseline Val).<br/>
                    • <b>Generalization Drop</b>: Performance variance when deploying from validation to final holdout test (Final Test − Tuned Val).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# METHODOLOGY & ANTI-LEAKAGE EXPANDER
# =============================================================================
st.markdown("<hr/>", unsafe_allow_html=True)
with st.expander("📚 Research Methodology & Anti-Leakage Protocol (SLIIT Group 2026-AI-09)"):
    st.markdown(
        """
        #### 1. Problem Formulation & Multi-Class Target
        - **Objective**: Supervised 3-class prediction of match outcomes:
            - **Class 0**: `Away Win`
            - **Class 1**: `Draw`
            - **Class 2**: `Home Win`
        - **Primary Analytical Lens**: Pre-match outcome probability estimation for tactical planning and decision support.
        - **Secondary Analytical Lens**: Rolling team form differentials, historical Elo ratings, and venue-specific strength indicators.

        #### 2. Anti-Leakage Compliance
        - In-match XML live events (goals, cards, corners, fouls, possession timestamps) were strictly excluded.
        - The models ingest only signals available **before kickoff**: historical head-to-head records, 3/5/10-match rolling points, goal differentials, rest days, and venue splits.
        - Cold-start burn-in protocol: Matches where either team has fewer than 5 prior competitive matches were removed from model training.

        #### 3. Feature Selection & Multi-Model Architecture
        - **Feature-Engineered Space**: 93 model-ready features from `Final Preprocessing.ipynb` (unscaled for tree classifiers, standardized for Logistic Regression).
        - **Stage 1 (Collinearity Reduction)**: 18 multicollinear features with correlation $|r| > 0.90$ removed across all pipelines.
        - **Pipeline Ingestion**: 68 informative predictors fed into model-specific feature selection.
        - **Supported Classifiers**:
            - **Random Forest (Tuned)**: MDI `SelectFromModel` selecting 34 predictors with entropy splitting ($d=8$).
            - **Logistic Regression (L1 Tuned)**: L1 Sparsity selection with Saga solver ($C=0.01$).
            - **Decision Tree (Tuned RFE)**: Recursive Feature Elimination selecting the top 20 predictors ($d=4$).
            - **XGBoost Classifier (Tuned)**: Gain-based selection selecting 50 predictors with regularized gradient boosting.
        - **Calibrated Optimal Draw Thresholding**: Unified grid calibration ($t=0.360$) applied to mitigate empirical draw under-prediction.
        """
    )
