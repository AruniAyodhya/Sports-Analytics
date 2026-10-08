"""European Soccer Match Outcome Predictor | Decision Support System.

Interactive Streamlit Dashboard for Supervised Multi-Class Match Outcome Prediction.
Group: 2026-AI-09 | SLIIT Machine Learning Module IT3091
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path across all execution environments
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from src.inference import SoccerInferenceEngine
from src.theme import load_custom_css, get_theme_palette, render_theme_toggle, toggle_theme
from views import render_simulator, render_evaluator, render_comparison


# =============================================================================
# PAGE SETUP & CSS LOADING
# =============================================================================
st.set_page_config(
    page_title="SoccerOutcome AI",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load consolidated design system from assets/style.css
load_custom_css()


# =============================================================================
# CACHED INFERENCE ENGINE
# =============================================================================
@st.cache_resource(show_spinner="Initializing Model Engine & Baseline Vectors...")
def get_engine(model_key: str = "random_forest") -> SoccerInferenceEngine:
    """Initialize and cache the SoccerInferenceEngine singleton."""
    return SoccerInferenceEngine(model_type=model_key)


# =============================================================================
# SIDEBAR NAVIGATION & PIPELINE SPECS
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2 class="sidebar-brand-title">⚽︎ SoccerOutcome AI</h2>
            <p class="sidebar-brand-sub">European League Match Decision Support</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Configuration")
    model_options = {
        "Logistic Regression (L1 Tuned)": "logistic_regression",
        "Random Forest (Tuned)": "random_forest",
        "XGBoost Classifier (Tuned)": "xgboost",
        "Decision Tree (Tuned RFE)": "decision_tree",
        
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

    nav_mode = st.radio(
        "",
        [
            "Pre-Match Simulator",
            "Historical Evaluator",
            "Model Comparison",
        ],
        index=0,
        label_visibility="collapsed",
        key="sidebar_nav",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("##### Pipeline Architecture")
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
# Theme state callback & light-mode CSS flag
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    if "theme_radio" in st.session_state and "Light" in str(st.session_state.theme_radio):
        st.session_state.theme = "light"
    else:
        st.session_state.theme = "dark"

active_theme = st.session_state.get("theme", "dark")
tp = get_theme_palette(active_theme)
is_dark = active_theme == "dark"

# Fixed theme flag container: renders identically in both modes to prevent any DOM height/layout shift
theme_flag_html = (
    f'<div id="theme-flag" class="{active_theme}-mode" aria-hidden="true" '
    f'style="display:none;position:fixed;top:0;left:0;width:0;height:0;margin:0;padding:0;overflow:hidden;pointer-events:none;"></div>'
)
if active_theme == "light":
    theme_flag_html += (
        '<div id="light-theme-flag" aria-hidden="true" '
        'style="display:none;position:fixed;top:0;left:0;width:0;height:0;margin:0;padding:0;overflow:hidden;pointer-events:none;"></div>'
    )
st.markdown(theme_flag_html, unsafe_allow_html=True)


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
# HEADER SECTION & TACTICAL COMMAND CENTER
# =============================================================================
with st.container(key="header_card"):
    head_left, head_right = st.columns([0.78, 0.22], vertical_alignment="top")
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
        render_theme_toggle()

if model_error:
    st.error(f"⚠️ **Artifact Notice**: {model_error}")
    st.info(
        "💡 Please select **'Random Forest (Tuned) [Active]'** in the sidebar to run live predictions."
    )
    st.stop()


# =============================================================================
# PAGE ROUTING
# =============================================================================
if nav_mode == "Pre-Match Simulator":
    render_simulator(engine, active_theme)
elif nav_mode == "Historical Evaluator":
    render_evaluator(engine, active_theme)
elif nav_mode == "Model Comparison":
    render_comparison(active_theme)
