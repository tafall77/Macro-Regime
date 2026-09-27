"""Plotly figure builders shared by the dashboard and the HTML report.

Palette: categorical slots validated for colour-vision deficiency (see the
dataviz reference palette). Actual = blue, Scenario = orange; regimes use a
semantic red / yellow / green trio, which needs the secondary encoding every
regime chart here carries (legend + direct labels + the state table).
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.graph_objects as go

SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BORDER = "rgba(11,11,11,0.10)"
ACTUAL = "#2a78d6"
ACTUAL_TRACK = "#cde2fb"
SCENARIO = "#eb6834"
SCENARIO_TRACK = "#fbdccd"
WARNING = "#fab219"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
REGIME_COLORS = {"bearish": "#e34948", "transition": "#eda100", "bullish": "#008300"}
_FALLBACK_STATE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"]


def state_color(state: int, label: str | None) -> str:
    key = (label or "").lower()
    return REGIME_COLORS.get(key, _FALLBACK_STATE_COLORS[state % len(_FALLBACK_STATE_COLORS)])


def gauge_figure(title: str, prob: float, color: str, track: str, reference: float | None = None) -> go.Figure:
    indicator: dict = dict(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": INK, "family": FONT}},
        title={"text": title, "font": {"size": 14, "color": INK2, "family": FONT}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "ticksuffix": "%",
                     "tickcolor": MUTED, "tickfont": {"color": MUTED, "size": 11}},
            "bar": {"color": color, "thickness": 0.55},
            "bgcolor": track,
            "borderwidth": 0,
        },
    )
    if reference is not None:
        indicator["mode"] = "gauge+number+delta"
        indicator["delta"] = {"reference": round(reference * 100, 1), "suffix": " pts", "valueformat": "+.1f",
                              "font": {"size": 16, "color": INK2},
                              "increasing": {"color": INK2}, "decreasing": {"color": INK2}}
    fig = go.Figure(go.Indicator(**indicator))
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, color=INK))
    return fig


def distribution_figure(names: Sequence[str], actual: Sequence[float], scenario: Sequence[float]) -> go.Figure:
    a = [v * 100 for v in actual]
    s = [v * 100 for v in scenario]
    fig = go.Figure()
    fig.add_bar(name="Actual", x=list(names), y=a, marker_color=ACTUAL, marker_line_width=0,
                text=[f"{v:.0f}%" for v in a], textposition="outside", textfont={"color": INK2},
                hovertemplate="Actual · %{x}: %{y:.1f}%<extra></extra>")
    fig.add_bar(name="Scenario", x=list(names), y=s, marker_color=SCENARIO, marker_line_width=0,
                text=[f"{v:.0f}%" for v in s], textposition="outside", textfont={"color": INK2},
                hovertemplate="Scenario · %{x}: %{y:.1f}%<extra></extra>")
    fig.update_layout(
        barmode="group", bargap=0.35, bargroupgap=0.08, height=260,
        margin=dict(l=40, r=20, t=10, b=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK2, size=12),
        legend=dict(orientation="h", y=1.12, x=0, font={"color": INK2}),
        yaxis=dict(range=[0, 110], ticksuffix="%", gridcolor=GRID, zerolinecolor=GRID, tickfont={"color": MUTED}),
        xaxis=dict(showgrid=False, tickfont={"color": INK}),
    )
    return fig


def history_figure(posteriors: pd.DataFrame, labels: dict[int, str] | None = None,
                   training_end: pd.Timestamp | None = None, title: str | None = None) -> go.Figure:
    """Stacked area of filtered state probabilities over time (100% stack)."""
    labels = labels or {}
    fig = go.Figure()
    for k, col in enumerate(posteriors.columns):
        name = labels.get(k, f"State {k}")
        name = name.capitalize() if isinstance(name, str) else str(name)
        fig.add_scatter(
            x=posteriors.index, y=posteriors[col] * 100, name=name, mode="lines", stackgroup="one",
            line=dict(width=0.5, color=state_color(k, labels.get(k))), fillcolor=state_color(k, labels.get(k)),
            hovertemplate=f"{name}: %{{y:.0f}}%<br>%{{x|%b %Y}}<extra></extra>",
        )
    if training_end is not None and posteriors.index[-1] > training_end:
        fig.add_vline(x=training_end, line_width=1, line_dash="dot", line_color=INK2)
        fig.add_annotation(x=training_end, y=100, yref="y", text="last refit", showarrow=False,
                           font={"size": 11, "color": INK2}, xanchor="right", yanchor="bottom")
    fig.update_layout(
        height=280, margin=dict(l=40, r=20, t=30 if title else 10, b=40), title=title,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        font=dict(family=FONT, color=INK2, size=12), legend=dict(orientation="h", y=1.12, x=0, traceorder="normal"),
        yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor=GRID, tickfont={"color": MUTED}),
        xaxis=dict(showgrid=False, tickfont={"color": MUTED}),
    )
    return fig


def indicators_figure(panel: pd.DataFrame, labels_by_key: dict[str, str]) -> go.Figure:
    """Small multiples of the headline panel: one line per indicator, shared x, own y."""
    from plotly.subplots import make_subplots

    keys = list(panel.columns)
    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[labels_by_key.get(k, k) for k in keys])
    for i, k in enumerate(keys, start=1):
        fig.add_scatter(x=panel.index, y=panel[k], mode="lines", line=dict(width=2, color=ACTUAL), name=labels_by_key.get(k, k),
                        showlegend=False, hovertemplate="%{y:.2f}<br>%{x|%b %Y}<extra></extra>", row=i, col=1)
        fig.update_yaxes(gridcolor=GRID, tickfont={"color": MUTED, "size": 10}, row=i, col=1)
        fig.update_xaxes(showgrid=False, tickfont={"color": MUTED, "size": 10}, row=i, col=1)
    fig.update_layout(height=120 * len(keys) + 40, margin=dict(l=40, r=20, t=30, b=30), hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=INK2, size=12))
    for ann in fig.layout.annotations:
        ann.font.update(size=12, color=INK2)
        ann.x = 0
        ann.xanchor = "left"
    return fig
