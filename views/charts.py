from __future__ import annotations

from typing import Dict, List, Optional
import plotly.graph_objects as go
from src.theme import get_theme_palette


def plot_probabilities(
    probabilities: Dict[str, float],
    title: str = "Class Probabilities",
    theme: str = "dark",
) -> go.Figure:
    """Render a clean, minimalist Plotly horizontal bar chart for the 3 classes."""
    tc = get_theme_palette(theme)
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


def plot_comparison_metric(
    metric_name: str,
    stages: List[str],
    model_data: Dict[str, List[float]],
    theme: str = "dark",
    ref_line: Optional[float] = None,
    ref_label: str = "Majority Baseline",
    height: int = 420,
    show_labels: bool = False,
    y_title: Optional[str] = None,
) -> go.Figure:
    """Render an un-cramped, interactive multi-model progression line chart with theme-aware styling."""
    tc = get_theme_palette(theme)
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
    all_vals: List[float] = [v for vals in model_data.values() for v in vals if v is not None]
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
    model_order: List[str],
    f1_drops: List[float],
    acc_drops: List[float],
    theme: str = "dark",
    height: int = 300,
) -> go.Figure:
    """Render an interactive horizontal bar chart comparing validation-to-test performance drops."""
    tc = get_theme_palette(theme)
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
