"""Helpers Plotly purs pour le dashboard Streamlit."""

from __future__ import annotations

from typing import List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.presentation import compact_feature_title, pretty_feature_name


# Style par defaut (duplique volontairement pour garder des fonctions pures et reusables)
C_LINE_WIDTH = 2
# Les quatre marqueurs portent le sens, la courbe n'est que le contexte : ils
# reprennent donc la semantique du manuscrit, ou le rouge signale une alerte et
# l'ambre une surveillance. Le cyan precedent se confondait avec la courbe bleu
# pale, et l'orange de l'episode HYPO avec le rouge de la notification.
COLOR_ANOM = "#2196A6"
ANOM_MARKER_SIZE = 6.5
COLOR_LAME = "#AF4646"
LAME_MARKER_SIZE = 7.5
COLOR_INSTABILITY = "#D9A441"
COLOR_EP = "#6A3D9A"
# 14 et non 9 : a diametre egal une etoile porte bien moins de matiere qu'un
# disque, si bien que la notification, marqueur le plus important de la vue,
# paraissait plus discrete que les episodes traces a 7,5. A 14 elle passe
# devant eux, et de 0,80 a 1,24 mm une fois la capture imprimee au chapitre 5.
EP_MARKER_SIZE = 14.0

# Vignettes « Vaches a verifier en priorite ». Le bleu tres pale employe
# jusqu'ici se lisait mal une fois la vignette reduite. Ces deux teintes sont
# celles du memoire (ifblue et rawred de main.tex), ce qui accorde la capture
# d'ecran du chapitre 5 aux figures generees du chapitre 6.
# Variantes essayees : bleu marine + terre cuite #BE7832, ardoise #4A5568 +
# rouge franc #C0392B. Changer les deux constantes suffit a basculer.
MINI_COLOR_LINE = "#375A7F"
MINI_COLOR_ALERT = "#AF4646"


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

    # Une seule teinte neutre pour les trois panneaux. Le bleu, vert et jaune
    # pales employes jusqu'ici variaient sans rien signifier -- chaque panneau
    # porte deja son titre -- tout en volant du contraste aux marqueurs, qui
    # eux distinguent quatre etats du systeme.
    colors = ["#5B6670", "#5B6670", "#5B6670"]

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


def build_small_fig(
    df: pd.DataFrame,
    col: str,
    *,
    y_max: float | None = None,
    show_legend: bool = False,
) -> go.Figure:
    """Vignette de reperage pour la vue troupeau.

    Les axes portent des graduations et la vignette accepte un maximum commun.
    Sans eux, la vignette ne disait ni sur quelle periode ni a quelle amplitude
    elle tracait, et deux vaches placees cote a cote n'etaient pas comparables
    puisque chacune se calait sur son propre maximum. ``show_legend`` ne sert que
    sur la premiere vignette de la grille : une legende par vache repeterait six
    fois la meme cle.
    """
    fig = go.Figure()

    if col not in df.columns:
        return fig

    fig.add_trace(
        go.Scatter(
            x=df["T"],
            y=df[col],
            mode="lines",
            line=dict(color=MINI_COLOR_LINE, width=1.5),
            name="Signal",
            showlegend=show_legend,
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
                    marker=dict(color=MINI_COLOR_ALERT, size=5),
                    name="Épisode d'alerte",
                    showlegend=show_legend,
                )
            )

    # Hauteur portee de 180 a 240 px : a 180, la vignette etait cinq fois plus
    # large que haute et tout le signal restait ecrase contre l'axe, seuls les
    # pics ressortant.
    fig.update_layout(
        height=240,
        margin=dict(l=44, r=6, t=6, b=30),
        xaxis=dict(showticklabels=True, showgrid=False, tickformat="%d %b",
                   nticks=6, tickfont=dict(size=10)),
        yaxis=dict(showticklabels=True, showgrid=True, gridcolor="rgba(0,0,0,0.08)",
                   nticks=4, tickfont=dict(size=10),
                   range=[0, y_max] if y_max else None),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Legende posee a l'interieur du trace : au-dessus, elle ajoutait sa
        # hauteur a la seule vignette qui la porte et desalignait les deux
        # colonnes de la grille.
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01,
                    font=dict(size=11), bgcolor="rgba(255,255,255,0.75)",
                    borderwidth=0),
    )

    return fig
