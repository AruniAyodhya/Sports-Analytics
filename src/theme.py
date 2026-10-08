from __future__ import annotations

import base64
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


@st.cache_data(show_spinner=False)
def load_toggle_icons() -> Dict[str, str]:
    """Dynamically load and base64-encode PNG toggle icons from assets/toggle."""
    root_dir = Path(__file__).resolve().parent.parent
    toggle_dir = root_dir / "assets" / "toggle"
    if not toggle_dir.exists():
        toggle_dir = Path("assets/toggle").resolve()

    icons: Dict[str, str] = {}
    for name in ["day-mode-black", "day-mode-white", "night-mode-black", "night-mode-white"]:
        p = toggle_dir / f"{name}.png"
        if p.exists():
            icons[name] = f"data:image/png;base64,{base64.b64encode(p.read_bytes()).decode('utf-8')}"
    return icons


def toggle_theme() -> None:
    """Toggle between Stadium Night (Dark) and Daylight Turf (Light) modes."""
    current = st.session_state.get("theme", "dark")
    st.session_state.theme = "light" if current == "dark" else "dark"


def render_theme_toggle() -> None:
    """Render a fully functional, interactive pill-shaped theme toggle switch using assets/toggle PNGs."""
    current_theme = st.session_state.get("theme", "dark")
    is_dark = current_theme == "dark"
    tooltip = (
        "Switch to Daylight Turf (Light Theme)"
        if is_dark
        else "Switch to Stadium Night (Dark Theme)"
    )

    icons = load_toggle_icons()
    # Dynamic selection of icons based on active/inactive state and theme mode:
    # In Stadium Night (Dark Mode):
    #   - Inactive Sun: day-mode-white.png (rendered on the dark track)
    #   - Active Moon: night-mode-white.png (rendered inside the highlighted slate thumb)
    # In Daylight Turf (Light Mode):
    #   - Active Sun: day-mode-black.png (rendered inside the highlighted white thumb)
    #   - Inactive Moon: night-mode-black.png (rendered on the light track)
    sun_inactive = icons.get("day-mode-white", "")
    moon_active = icons.get("night-mode-white", "")
    sun_active = icons.get("day-mode-black", "")
    moon_inactive = icons.get("night-mode-black", "")

    if is_dark:
        dynamic_css = f"""
        <style>
        .st-key-theme_toggle_btn {{
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .st-key-theme_toggle_btn .stButton {{
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            width: auto !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .st-key-theme_toggle_btn button,
        .st-key-theme_toggle_btn button[kind="secondary"],
        .st-key-theme_toggle_btn button[data-testid="stBaseButton-secondary"] {{
            position: relative !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 88px !important;
            min-width: 88px !important;
            max-width: 88px !important;
            height: 38px !important;
            min-height: 38px !important;
            max-height: 38px !important;
            border-radius: 9999px !important;
            background-color: #090e18 !important;
            border: 1.5px solid rgba(255, 255, 255, 0.14) !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.6), 0 2px 8px rgba(0, 0, 0, 0.25) !important;
            padding: 0 !important;
            margin: 0 !important;
            cursor: pointer !important;
            overflow: hidden !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
            user-select: none !important;
        }}
        .st-key-theme_toggle_btn button:hover {{
            border-color: rgba(16, 185, 129, 0.5) !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.6), 0 0 16px rgba(16, 185, 129, 0.25) !important;
            transform: translateY(-1px) scale(1.02) !important;
        }}
        .st-key-theme_toggle_btn button:active {{
            transform: translateY(0) scale(0.98) !important;
        }}
        .st-key-theme_toggle_btn button:focus,
        .st-key-theme_toggle_btn button:focus-visible {{
            outline: none !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.6), 0 0 0 2px rgba(16, 185, 129, 0.45) !important;
        }}
        /* Inactive Sun Icon on Left Track */
        .st-key-theme_toggle_btn button::after {{
            content: '' !important;
            position: absolute !important;
            left: 11px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            width: 20px !important;
            height: 20px !important;
            background-image: url('{sun_inactive}') !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            opacity: 0.6 !important;
            pointer-events: none !important;
            z-index: 1 !important;
            transition: opacity 0.2s ease !important;
        }}
        .st-key-theme_toggle_btn button:hover::after {{
            opacity: 0.85 !important;
        }}
        /* Active Moon Indicator Thumb on Right */
        .st-key-theme_toggle_btn button::before {{
            content: '' !important;
            position: absolute !important;
            right: 3px !important;
            top: 3px !important;
            bottom: 3px !important;
            width: 38px !important;
            border-radius: 9999px !important;
            background-color: #1e293b !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            background-image: url('{moon_active}') !important;
            background-size: 19px 19px !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            pointer-events: none !important;
            z-index: 2 !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }}
        /* Hide all text inside the button */
        .st-key-theme_toggle_btn button [data-testid="stMarkdownContainer"],
        .st-key-theme_toggle_btn button p,
        .st-key-theme_toggle_btn button span,
        .st-key-theme_toggle_btn button div,
        .st-key-theme_toggle_btn button svg {{
            display: none !important;
        }}
        </style>
        """
    else:
        dynamic_css = f"""
        <style>
        .st-key-theme_toggle_btn {{
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .st-key-theme_toggle_btn .stButton {{
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            width: auto !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .st-key-theme_toggle_btn button,
        .st-key-theme_toggle_btn button[kind="secondary"],
        .st-key-theme_toggle_btn button[data-testid="stBaseButton-secondary"] {{
            position: relative !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 88px !important;
            min-width: 88px !important;
            max-width: 88px !important;
            height: 38px !important;
            min-height: 38px !important;
            max-height: 38px !important;
            border-radius: 9999px !important;
            background-color: #e2e8f0 !important;
            border: 1.5px solid rgba(15, 23, 42, 0.14) !important;
            box-shadow: inset 0 2px 4px rgba(15, 23, 42, 0.08), 0 2px 8px rgba(15, 23, 42, 0.05) !important;
            padding: 0 !important;
            margin: 0 !important;
            cursor: pointer !important;
            overflow: hidden !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
            user-select: none !important;
        }}
        .st-key-theme_toggle_btn button:hover {{
            border-color: rgba(5, 150, 105, 0.45) !important;
            box-shadow: inset 0 2px 4px rgba(15, 23, 42, 0.08), 0 0 16px rgba(16, 185, 129, 0.2) !important;
            transform: translateY(-1px) scale(1.02) !important;
        }}
        .st-key-theme_toggle_btn button:active {{
            transform: translateY(0) scale(0.98) !important;
        }}
        .st-key-theme_toggle_btn button:focus,
        .st-key-theme_toggle_btn button:focus-visible {{
            outline: none !important;
            box-shadow: inset 0 2px 4px rgba(15, 23, 42, 0.08), 0 0 0 2px rgba(16, 185, 129, 0.4) !important;
        }}
        /* Inactive Moon Icon on Right Track */
        .st-key-theme_toggle_btn button::after {{
            content: '' !important;
            position: absolute !important;
            right: 11px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            width: 20px !important;
            height: 20px !important;
            background-image: url('{moon_inactive}') !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            opacity: 0.5 !important;
            pointer-events: none !important;
            z-index: 1 !important;
            transition: opacity 0.2s ease !important;
        }}
        .st-key-theme_toggle_btn button:hover::after {{
            opacity: 0.8 !important;
        }}
        /* Active Sun Indicator Thumb on Left */
        .st-key-theme_toggle_btn button::before {{
            content: '' !important;
            position: absolute !important;
            left: 3px !important;
            top: 3px !important;
            bottom: 3px !important;
            width: 38px !important;
            border-radius: 9999px !important;
            background-color: #ffffff !important;
            border: 1px solid rgba(15, 23, 42, 0.08) !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.14), 0 1px 2px rgba(15, 23, 42, 0.08) !important;
            background-image: url('{sun_active}') !important;
            background-size: 19px 19px !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            pointer-events: none !important;
            z-index: 2 !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }}
        /* Hide all text inside the button */
        .st-key-theme_toggle_btn button [data-testid="stMarkdownContainer"],
        .st-key-theme_toggle_btn button p,
        .st-key-theme_toggle_btn button span,
        .st-key-theme_toggle_btn button div,
        .st-key-theme_toggle_btn button svg {{
            display: none !important;
        }}
        </style>
        """

    st.markdown(dynamic_css, unsafe_allow_html=True)
    st.button(
        "Toggle Theme",
        key="theme_toggle_btn",
        help=tooltip,
        on_click=toggle_theme,
    )

