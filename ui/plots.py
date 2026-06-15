"""Helpers Plotly purs pour le dashboard Streamlit."""

from __future__ import annotations

from typing import List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.presentation import compact_feature_title, pretty_feature_name


# Style par defaut (duplique volontairement pour garder des fonctions pures et reusables)
C_LINE_WIDTH = 2
COLOR_ANOM = "#38BDF8"
ANOM_MARKER_SIZE = 6.5
COLOR_LAME = "#fb923c"
LAME_MARKER_SIZE = 7.5
COLOR_INSTABILITY = "#a855f7"
COLOR_EP = "#ef4444"
EP_MARKER_SIZE = 9.0


def build_multi_panel_figure(df: pd.DataFrame, plot_cols: List[str], title: str = "") -> go.Figure:
    """Graphique multi-panel avec superposition des alertes comportementales."""
    n_panels = min(3, len(plot_cols))
    if n_panels == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n_panels,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[compact_feature_title(c) for c in plot_cols[:n_panels]],
    )

    colors = ["#9bd3ff", "#a7f3d0", "#fcd34d"]

    for i, col in enumerate(plot_cols[:n_panels], start=1):
        if col not in df.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=df["T"],
                y=df[col],
                mode="lines",
                name=pretty_feature_name(col),
                line=dict(color=colors[i - 1], width=C_LINE_WIDTH),
                showlegend=False,
            ),
            row=i,
            col=1,
        )

        if "if_anomaly_point" in df.columns:
            anom = df[df["if_anomaly_point"] == 1]
            if len(anom) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=anom["T"],
                        y=anom[col],
                        mode="markers",
                        name="Anomalie IF",
                        marker=dict(color=COLOR_ANOM, size=ANOM_MARKER_SIZE, opacity=1.0),
                        showlegend=(i == 1),
                    ),
                    row=i,
                    col=1,
                )

        if "behavioral_warning_episode" in df.columns:
            hypo = df[df["behavioral_warning_episode"] == 1]
            if len(hypo) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=hypo["T"],
                        y=hypo[col],
                        mode="markers",
                        name="Épisode HYPO",
                        marker=dict(color=COLOR_LAME, size=LAME_MARKER_SIZE, opacity=1.0),
                        showlegend=(i == 1),
                    ),
                    row=i,
                    col=1,
                )
        elif "pred_lameness_episode" in df.columns:
            legacy = df[df["pred_lameness_episode"] == 1]
            if len(legacy) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=legacy["T"],
                        y=legacy[col],
                        mode="markers",
                        name="Alerte comparative",
                        marker=dict(color=COLOR_LAME, size=LAME_MARKER_SIZE, opacity=1.0),
                        showlegend=(i == 1),
                    ),
                    row=i,
                    col=1,
                )

        if "instability_warning_episode" in df.columns:
            instability = df[df["instability_warning_episode"] == 1]
            if len(instability) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=instability["T"],
                        y=instability[col],
                        mode="markers",
                        name="Épisode INSTABILITÉ",
                        marker=dict(color=COLOR_INSTABILITY, size=LAME_MARKER_SIZE, opacity=0.9),
                        showlegend=(i == 1),
                    ),
                    row=i,
                    col=1,
                )

        notif_col = "hybrid_warning_notification" if "hybrid_warning_notification" in df else "notif_lameness"
        if notif_col in df.columns:
            notif = df[df[notif_col] == 1]
            if len(notif) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=notif["T"],
                        y=notif[col],
                        mode="markers",
                        name="Notification",
                        marker=dict(
                            color=COLOR_EP,
                            size=EP_MARKER_SIZE,
                            symbol="star",
                            line=dict(width=1.2, color="white"),
                        ),
                        showlegend=(i == 1),
                    ),
                    row=i,
                    col=1,
                )

    fig.update_layout(
        height=200 * n_panels + 100,
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=18)),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=13),
        ),
        margin=dict(l=10, r=10, t=62, b=36),
    )

    fig.update_annotations(font=dict(size=13))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    return fig


def build_small_fig(df: pd.DataFrame, col: str) -> go.Figure:
    """Mini graphique pour affichage compact."""
    fig = go.Figure()

    if col not in df.columns:
        return fig

    fig.add_trace(
        go.Scatter(
            x=df["T"],
            y=df[col],
            mode="lines",
            line=dict(color="#9bd3ff", width=1.5),
            showlegend=False,
        )
    )

    episode_col = "hybrid_warning_episode" if "hybrid_warning_episode" in df else "pred_lameness_episode"
    if episode_col in df.columns:
        lame = df[df[episode_col] == 1]
        if len(lame) > 0:
            fig.add_trace(
                go.Scatter(
                    x=lame["T"],
                    y=lame[col],
                    mode="markers",
                    marker=dict(color=COLOR_LAME, size=5),
                    showlegend=False,
                )
            )

    fig.update_layout(
        height=180,
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig
