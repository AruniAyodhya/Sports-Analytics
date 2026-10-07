from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import streamlit as st

THEME_PALETTE: Dict[str, Dict[str, str]] = {
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


def get_theme_palette(theme: str = "dark") -> Dict[str, str]:
    """Retrieve theme-specific color tokens for Plotly visualizations."""
    return THEME_PALETTE.get(theme, THEME_PALETTE["dark"])


def load_custom_css() -> None:
    """Load and inject all modular CSS stylesheets from the assets/ directory."""
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    modular_files = ["base.css", "sidebar.css", "cards.css", "components.css"]

    css_chunks = []
    for fname in modular_files:
        fpath = assets_dir / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                css_chunks.append(f.read())

    if css_chunks:
        combined_css = "\n\n".join(css_chunks)
        st.markdown(f"<style>{combined_css}</style>", unsafe_allow_html=True)
    else:
        # Fallback to master style.css if modular files are absent
        fallback_path = assets_dir / "style.css"
        if fallback_path.exists():
            with open(fallback_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            st.warning("Custom stylesheets not found in assets/. Falling back to default styles.")

