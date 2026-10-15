"""Centralized Feature Registry & Simulator Feature Definitions.

Single Source of Truth for interactive pre-match simulation differentials,
connecting the inference engine and the Streamlit presentation layer.
Group: 2026-AI-09 | SLIIT ML Module IT3091
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# 1. CORE SIMULATOR DIFFERENTIAL SPECIFICATIONS (SINGLE SOURCE OF TRUTH)
# -----------------------------------------------------------------------------
SIMULATOR_CORE_FEATURES: Dict[str, Dict[str, Any]] = {
    "points_per_match_difference": {
        "label": "Points/Match Difference (Home - Away)",
        "min": -3.0,
        "max": 3.0,
        "default": 0.5,
        "step": 0.1,
        "help": "Full-season Points-Per-Match differential (+ favours Home, - favours Away).",
        "aliases": ["ppm_diff", "points_per_match_diff"],
    },
    "recent_points_difference": {
        "label": "Recent 5-Match Points Difference",
        "min": -3.0,
        "max": 3.0,
        "default": 0.8,
        "step": 0.1,
        "help": "Rolling short-term points trajectory across last 5 competitive matches.",
        "aliases": ["recent_pts_diff", "points_last5_difference"],
    },
    "goals_for_difference": {
        "label": "Goal Difference/Match (Home - Away)",
        "min": -3.0,
        "max": 3.0,
        "default": 0.4,
        "step": 0.1,
        "help": "Average goal differential per game between the two teams.",
        "aliases": [
            "goal_difference_per_match",
            "goals_diff",
            "goal_difference_per_match_difference",
        ],
    },
    "win_rate_difference": {
        "label": "Recent Win Rate Difference",
        "min": -1.0,
        "max": 1.0,
        "default": 0.2,
        "step": 0.05,
        "help": "Difference in recent winning percentage (+0.2 = Home has 20% higher recent win rate).",
        "aliases": [
            "recent_win_rate_difference",
            "win_rate_diff",
            "win_rate_last5_difference",
        ],
    },
}


# -----------------------------------------------------------------------------
# 2. HELPER UTILITIES FOR INFERENCE & UI
# -----------------------------------------------------------------------------
def get_simulator_default(feature_key: str, fallback: float = 0.0) -> float:
    """Retrieve the registered default value for a core simulator feature."""
    if feature_key in SIMULATOR_CORE_FEATURES:
        return float(SIMULATOR_CORE_FEATURES[feature_key].get("default", fallback))
    for spec in SIMULATOR_CORE_FEATURES.values():
        if feature_key in spec.get("aliases", []):
            return float(spec.get("default", fallback))
    return fallback


def extract_simulator_inputs(input_dict: Dict[str, Any]) -> Dict[str, float]:
    """Dynamically extract core simulator differentials from an input dictionary.

    Checks primary feature keys and registered aliases, automatically falling back
    to the centralized default values if a parameter is omitted.
    """
    resolved: Dict[str, float] = {}

    for key, spec in SIMULATOR_CORE_FEATURES.items():
        val = None
        # Check primary key
        if key in input_dict:
            val = input_dict[key]
        else:
            # Check aliases
            for alias in spec.get("aliases", []):
                if alias in input_dict:
                    val = input_dict[alias]
                    break

        # Fallback to configured default
        if val is None:
            val = spec.get("default", 0.0)

        resolved[key] = float(val)

    return resolved


def get_simulator_slider_specs() -> Dict[str, Dict[str, Any]]:
    """Retrieve slider UI specifications for dynamic Streamlit widget rendering."""
    return SIMULATOR_CORE_FEATURES
