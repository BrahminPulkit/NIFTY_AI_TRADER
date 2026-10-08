"""Reusable Plotly charts aligned with the QuantEdge design system."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from apps.research_ui.components.theme import chart_palette, preferences


def _sequence() -> list[str]:
    colors = chart_palette()
    return [
        colors["accent"], colors["probability"], colors["bull"],
        colors["neutral"], colors["vwap"], colors["bear"], colors["ema50"],
    ]


def style(fig: go.Figure, height: int = 390) -> go.Figure:
    colors, pref = chart_palette(), preferences()
    density = {"Compact": 330, "Balanced": height, "Detailed": max(height, 450)}
    chart_height = density.get(pref["chart_density"], height)
    fig.update_layout(
        template="plotly_dark",
        height=chart_height,
        margin=dict(l=24, r=20, t=54, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16,24,39,.35)",
        font=dict(family="Inter, SF Pro Display, sans-serif",
                  color="#AAB7C8", size=11),
        title=dict(font=dict(family="Manrope, Inter, sans-serif",
                             color="#F8FAFC", size=14), x=.02),
        colorway=_sequence(),
        hoverlabel=dict(
            bgcolor="#1C2940", bordercolor="rgba(255,255,255,.12)",
            font=dict(color="#F8FAFC", family="Inter")),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, x=.02,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)", color="#6B7280",
            activecolor=colors["accent"]),
        transition=dict(duration=180 if pref["animations"] else 0),
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,.055)", linecolor="rgba(255,255,255,.08)",
        zerolinecolor="rgba(255,255,255,.08)", showspikes=True,
        spikemode="across", spikesnap="cursor", spikecolor=colors["accent"])
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,.055)", linecolor="rgba(255,255,255,.08)",
        zerolinecolor="rgba(255,255,255,.08)")
    return fig


def line(data: pd.DataFrame, x: str, y: str, color: str | None, title: str):
    fig = px.line(
        data, x=x, y=y, color=color, title=title,
        color_discrete_sequence=_sequence())
    fig.update_traces(line_width=2)
    return style(fig)


def bars(data: pd.DataFrame, x: str, y: str, color: str | None, title: str):
    fig = px.bar(
        data, x=x, y=y, color=color, title=title, barmode="group",
        color_discrete_sequence=_sequence())
    fig.update_traces(marker_line_width=0)
    return style(fig)


def gauge(value: float, title: str, maximum: float = 1.0):
    colors = chart_palette()
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"valueformat": ".1%"} if maximum == 1 else {},
        title={"text": title},
        gauge={
            "axis": {"range": [0, maximum], "tickcolor": "#6B7280"},
            "bar": {"color": colors["probability"], "thickness": .2},
            "bgcolor": "#1C2940",
            "bordercolor": "rgba(255,255,255,.08)",
            "steps": [
                {"range": [0, maximum * .4], "color": "#141F33"},
                {"range": [maximum * .4, maximum * .7], "color": "#1C2940"},
                {"range": [maximum * .7, maximum], "color": "#243552"},
            ],
        },
    ))
    return style(fig, 280)


def confusion_matrix(tn: int, fp: int, fn: int, tp: int, title: str):
    colors = chart_palette()
    fig = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted no", "Predicted yes"],
        y=["Actual no", "Actual yes"],
        text=[[tn, fp], [fn, tp]],
        texttemplate="%{text}",
        colorscale=[[0, "#141F33"], [1, colors["accent"]]],
        colorbar=dict(outlinewidth=0),
    ))
    fig.update_layout(title=title)
    return style(fig)
