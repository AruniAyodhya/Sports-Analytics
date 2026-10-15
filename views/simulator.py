"""Mode 1: Pre-Match Performance Simulator View.

Interactive live scenario forecasting dashboard for simulated soccer fixtures.
Group: 2026-AI-09 | SLIIT Machine Learning Module IT3091
"""

from __future__ import annotations

from typing import Dict, Any
import streamlit as st
from src.inference import SoccerInferenceEngine
from src.features import get_simulator_slider_specs
from .charts import plot_probabilities


def render_simulator(engine: SoccerInferenceEngine, active_theme: str = "dark") -> None:
    """Render Mode 1: Pre-Match Performance Simulator."""
    st.markdown(
        """
        <div class="simulator-header-row">
            <div>
                <h4 class="simulator-title">Live Pre-Match Simulator</h4>
                <p class="simulator-subtitle">
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
                Match & League Context
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
        st.caption("🕮 League encoded via one-hot representation; stage is standardized.")

    # Column 2: Relative Performance Metrics (Secondary Analytical Lens)
    with col2:
        st.markdown(
            """
            <div class="card-title">
                Relative Form & Strength Differentials
            </div>
            """,
            unsafe_allow_html=True,
        )
        core_sliders = {}
        for feature_key, spec in get_simulator_slider_specs().items():
            core_sliders[feature_key] = st.slider(
                spec["label"],
                min_value=spec["min"],
                max_value=spec["max"],
                value=spec["default"],
                step=spec["step"],
                help=spec["help"],
            )

    # Column 3: Market Consensus Odds
    with col3:
        st.markdown(
            """
            <div class="card-title">
                Market Consensus Odds
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
            <div class="market-implied-box">
                <div class="market-implied-header">
                    Normalized Market Implied Probabilities
                </div>
                <div class="market-implied-row">
                    <span>H: <b class="market-prob-home">{imp_h:.1f}%</b></span>
                    <span>D: <b class="market-prob-draw">{imp_d:.1f}%</b></span>
                    <span>A: <b class="market-prob-away">{imp_a:.1f}%</b></span>
                </div>
                <div class="market-implied-vig">
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
        **core_sliders,
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

    icon = {"Home Win": "🏠︎", "Draw": "🤝", "Away Win": "✈︎"}.get(pred_label, "⚽")

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
