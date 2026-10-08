"""Mode 2: Historical Holdout Match Evaluator View.

Audit predictions against the unseen holdout test set (3,923 fixtures).
Group: 2026-AI-09 | SLIIT Machine Learning Module IT3091
"""

from __future__ import annotations

import random
from typing import Dict, Any
import streamlit as st
from src.inference import SoccerInferenceEngine
from .charts import plot_probabilities


def render_evaluator(engine: SoccerInferenceEngine, active_theme: str = "dark") -> None:
    """Render Mode 2: Historical Test Match Evaluator."""
    st.markdown("#### Historical Holdout Match Audit")
    st.markdown(
        """
        <p class="page-subtitle">
            Audit model predictions against the unseen holdout test set (3,923 fixtures) to inspect ground-truth accuracy.
        </p>
        """,
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
        st.markdown("<div class='spacer-lg'></div>", unsafe_allow_html=True)
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
        val_cls = "kpi-val-home" if pred_label == "Home Win" else ("kpi-val-draw" if pred_label == "Draw" else "kpi-val-away")
        outcome_color = "#10b981" if pred_label == "Home Win" else ("#f59e0b" if pred_label == "Draw" else "#f43f5e")
        st.markdown(
            f"""
            <div class="kpi-tile kpi-tile-stretched">
                <div class="kpi-label">Predicted Outcome</div>
                <div class="kpi-value {val_cls}" style="color: {outcome_color};">{pred_label}</div>
                <div class="kpi-sub">Confidence: <b>{confidence:.1f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="kpi-tile kpi-tile-stretched">
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
            <div class="kpi-tile kpi-tile-stretched">
                <div class="kpi-label">Verification Status</div>
                <div class="kpi-value kpi-status-container">{status_html}</div>
                <div class="kpi-sub">{status_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)
    st.plotly_chart(
        plot_probabilities(test_res["probabilities"], title=f"Fixture #{match_idx} Model Probability Distribution", theme=active_theme),
        use_container_width=True,
    )
