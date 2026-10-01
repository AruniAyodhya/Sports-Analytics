"""European Soccer Match Outcome Predictor | Decision Support System.

Interactive Streamlit Dashboard for Supervised Multi-Class Match Outcome Prediction.
Group: 2026-AI-09 | SLIIT Machine Learning Module IT3091
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st

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
        "chart_grid": "rgba(255, 255, 255, 0.06)",
        "chart_bar_line": "rgba(255, 255, 255, 0.15)",
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
        "chart_grid": "rgba(0, 0, 0, 0.06)",
        "chart_bar_line": "rgba(0, 0, 0, 0.1)",
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
