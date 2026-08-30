"""Panneaux Streamlit liés au troupeau (classement, comparaison journalière, export)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.io import COW
from core.pipeline import run_pipeline_herd
from ui.presentation import plain_feature_name
from ui.plots import build_small_fig

PipelineKwargs = Dict[str, Any]
PlotFn = Callable[..., Any]
KeyFn = Callable[..., str]
CacheKeyFn = Callable[[], str]


def _run_herd_pipeline(df: pd.DataFrame, *, pipeline_kwargs: PipelineKwargs, max_cows: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exécute le pipeline troupeau avec le même contrat que l'app."""
    return run_pipeline_herd(df, max_cows=max_cows, **pipeline_kwargs)


def _ensure_full_herd_cache(
    *,
    df: pd.DataFrame,
    pipeline_kwargs: PipelineKwargs,
    build_cache_key_full: CacheKeyFn,
    spinner_text: str,
) -> None:
    """Calcule et met en cache les sorties troupeau complètes pour les onglets 3/4."""
    ck_full = build_cache_key_full()
    if st.session_state.get("herd_cache_key_full") == ck_full:
        return

    with st.spinner(spinner_text):
        summary_df_full, out_df_full = _run_herd_pipeline(df, pipeline_kwargs=pipeline_kwargs, max_cows=None)
        st.session_state["summary_df_full"] = summary_df_full
        st.session_state["out_df_full"] = out_df_full
        st.session_state["herd_cache_key_full"] = ck_full


def render_tab_herd(
    *,
    df: pd.DataFrame,
    interval: str,
    plot_pref: List[str],
    file_hash: str,
    st_plotly: PlotFn,
    mk_key: KeyFn,
    build_cache_key_full: CacheKeyFn,
    pipeline_kwargs: PipelineKwargs,
) -> None:
    """Rend l'onglet de classement troupeau."""
    st.subheader("Classement du troupeau par priorité de vérification")

    max_cows = st.number_input("Nombre max. de vaches à analyser (0 = tous)", value=0, step=1, key="tab2_max")
    max_cows = None if int(max_cows) <= 0 else int(max_cows)

    cache_key = (
        f"{file_hash}_{pipeline_kwargs['interval']}_{pipeline_kwargs['window_baseline']}_{pipeline_kwargs['contamination']}_{pipeline_kwargs['baseline_ratio']}_"
        f"{pipeline_kwargs['random_state']}_{pipeline_kwargs['persist_hours']}_{pipeline_kwargs['alert_min']}_{pipeline_kwargs['mix_mode']}_{pipeline_kwargs['mix_rate_thr']}_"
        f"{pipeline_kwargs['z_low_thr']}_{pipeline_kwargs['z_high_thr']}_{pipeline_kwargs['cooldown_hours']}_{pipeline_kwargs['mi_z_high_thr']}_{pipeline_kwargs['coverage_min_pct']}_"
        f"{max_cows}"
    )

    if st.session_state.get("herd_cache_key") != cache_key:
        with st.spinner("Analyse du troupeau en cours..."):
            summary_df, out_df = _run_herd_pipeline(df, pipeline_kwargs=pipeline_kwargs, max_cows=max_cows)
            st.session_state["summary_df"] = summary_df
            st.session_state["out_df"] = out_df
            st.session_state["herd_computed"] = True
            st.session_state["herd_cache_key"] = cache_key
            if max_cows is None:
                st.session_state["summary_df_full"] = summary_df
                st.session_state["out_df_full"] = out_df
                st.session_state["herd_cache_key_full"] = build_cache_key_full()

    if not st.session_state.get("herd_computed", False):
        return

    summary_df = st.session_state["summary_df"]
    out_df = st.session_state["out_df"]

    st.caption(f"Analyse terminée : {len(summary_df)} vaches")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Vaches analysées", len(summary_df))
    k2.metric("Notifications fusionnées", int(summary_df["hybrid_warning_notifs"].sum()))
    k3.metric("Vaches à vérifier", int((summary_df["hybrid_warning_notifs"] > 0).sum()))
    k4.metric("Moy. anomalies IF (comparateur)", f"{summary_df['if_anomaly_points'].mean():.1f}")

    display_cols = [
        COW,
        "n_bins",
        "hybrid_warning_notifs",
        "behavioral_warning_notifs",
        "instability_warning_notifs",
        "if_anomaly_points",
        "coverage_mean",
        "coverage_min",
    ]
    display_cols = [col for col in display_cols if col in summary_df.columns]
    # En-tetes abreges. Les trois colonnes de notifications repetaient le mot
    # « Notifications », deja porte par la metrique placee au-dessus, et les
    # vignettes du bas de l'onglet nomment deja ces memes decomptes « Fusion »,
    # « HYPO » et « INSTABILITE ». Les libelles longs imposaient au tableau une
    # largeur que la page du memoire reduit ensuite a un corps illisible.
    display_names = {
        COW: "Vache",
        "n_bins": "Intervalles",
        "hybrid_warning_notifs": "Fusion",
        "behavioral_warning_notifs": "HYPO",
        "instability_warning_notifs": "INSTABILITÉ",
        "if_anomaly_points": "Anomalies IF (comp.)",
        "coverage_mean": "Couv. moy. (%)",
        "coverage_min": "Couv. min. (%)",
    }
    summary_display = summary_df[display_cols].rename(columns=display_names)
    # Rang explicite a partir de 1 : le tableau est deja trie par priorite de
    # verification, mais seul l'index technique de pandas le donnait a lire, en
    # commencant a zero et sans en-tete.
    summary_display.insert(0, "Rang", range(1, len(summary_display) + 1))
    # Hauteur calee sur le nombre de lignes plutot que figee a 400 px : le
    # tableau s'affiche en entier au lieu de defiler dans son propre cadre, ce
    # qui le rend capturable d'un seul tenant. Lignes resserrees a 22 px au lieu
    # des 35 par defaut : a 35, les vingt-huit vaches depassaient la hauteur
    # d'ecran et aucune capture ne pouvait les prendre d'un seul coup.
    # L'en-tete de la grille garde une hauteur fixe, independante de row_height :
    # le compter comme une ligne de plus tronquait la vingt-huitieme vache.
    row_height = 22
    header_height = 36
    st.dataframe(
        summary_display,
        width="stretch",
        height=header_height + len(summary_display) * row_height + 6,
        hide_index=True,
        row_height=row_height,
    )

    st.markdown("### Vaches à vérifier en priorité")
    # Six vignettes sur deux colonnes plutot que neuf sur trois : a largeur de
    # capture egale, chaque vignette dispose de la moitie en plus, et les
    # decomptes qui l'accompagnent restent lisibles une fois la figure reduite
    # a la largeur d'une page du memoire.
    top_cows = summary_df.head(6)[COW].tolist()

    # Variable tracee et maximum commun aux six vignettes. Chacune se calait
    # auparavant sur son propre maximum : deux vaches cote a cote paraissaient
    # d'amplitude comparable alors qu'un facteur trois les separait. Le titre
    # nomme la variable, qu'aucune vignette ne portait.
    colonne_tracee = next((pc for pc in plot_pref if pc in out_df.columns), None)
    y_commun = None
    if colonne_tracee is not None:
        retenues = out_df[out_df[COW].astype(str).isin({str(c) for c in top_cows})]
        maxi = pd.to_numeric(retenues[colonne_tracee], errors="coerce").max()
        if pd.notna(maxi) and float(maxi) > 0:
            y_commun = float(maxi) * 1.05
        # Au corps du texte plutot qu'en legende, comme les intitules de vignettes :
        # st.caption rend en gris clair a environ 87 pour cent de cette taille, ce
        # qui devient illisible une fois la vue reduite a la largeur d'une page.
        # Le nom de la variable est celui du manuscrit : « Motion Index », et non
        # le nom de colonne brut ni la forme longue a trois segments.
        st.markdown(
            f"**{plain_feature_name(colonne_tracee)}**, "
            "échelle verticale commune aux six vignettes."
        )

    cols_grid = st.columns(2)
    slot = 0
    for cid in top_cows:
        g = out_df[out_df[COW].astype(str) == str(cid)].copy()
        if len(g) == 0:
            continue

        plot_col = None
        for pc in plot_pref:
            if pc in g.columns:
                plot_col = pc
                break

        if plot_col:
            with cols_grid[slot % 2]:
                r = summary_df[summary_df[COW] == cid].iloc[0]
                # Nom et decomptes sur une seule ligne, au corps du texte plutot
                # qu'en legende : st.caption rend a environ 87 pour cent de cette
                # taille, ce qui devenait illisible une fois la vue reduite a la
                # largeur d'une page du memoire.
                st.markdown(
                    f"**Vache {cid}** | Fusion : {int(r['hybrid_warning_notifs'])} | "
                    f"HYPO : {int(r['behavioral_warning_notifs'])} | "
                    f"INSTABILITÉ : {int(r['instability_warning_notifs'])}"
                )
                fig_mini = build_small_fig(
                    g, plot_col, y_max=y_commun, show_legend=(slot == 0)
                )
                st_plotly(fig_mini, "tab2", "mini", cid, file_hash, width="stretch")
            slot += 1

    st.download_button(
        "Télécharger le résumé troupeau",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"herd_summary_{interval}_{file_hash}.csv",
        mime="text/csv",
        key=mk_key("dl_herd_sum", file_hash),
    )


def render_tab_daily_comparison(
    *,
    df: pd.DataFrame,
    file_hash: str,
    st_plotly: PlotFn,
    build_cache_key_full: CacheKeyFn,
    pipeline_kwargs: PipelineKwargs,
) -> None:
    """Rend l'onglet de comparaison journalière inter-vaches."""
    st.subheader("Comparaison inter-vaches (vue journalière)")
    _ensure_full_herd_cache(
        df=df,
        pipeline_kwargs=pipeline_kwargs,
        build_cache_key_full=build_cache_key_full,
        spinner_text="Calcul des données troupeau (complet)...",
    )

    out_df = st.session_state["out_df_full"]

    if "Day" not in out_df.columns:
        out_df["Day"] = pd.to_datetime(out_df["T"]).dt.floor("D")

    episode_col = "hybrid_warning_episode"
    notif_col = "hybrid_warning_notification"
    daily = (
        out_df.groupby([COW, "Day"])
        .agg(
            day_alert=(episode_col, "max"),
            alert_rate=(episode_col, "mean"),
            n_anomalies=("if_anomaly_point", "sum"),
            n_bins=(episode_col, "size"),
            day_notifs=(notif_col, "sum"),
        )
        .reset_index()
    )

    # Sévérité graduée : combine proportion de bins en alerte + densité d'anomalies
    daily["severity_raw"] = (
        0.6 * daily["alert_rate"]
        + 0.4 * (daily["n_anomalies"] / daily["n_bins"]).clip(0, 1)
    )
    daily.loc[daily["alert_rate"] == 0, "severity_raw"] = 0

    # Normaliser : le jour le plus sévère = 1.0 (rouge foncé)
    sev_max = daily["severity_raw"].max()
    daily["severity"] = daily["severity_raw"] / sev_max if sev_max > 0 else daily["severity_raw"]

    # Pas de fillna : un couple (vache, jour) sans mesure doit rester vide. Le
    # combler par un zero l'afficherait comme une journee observee sans alerte,
    # alors qu'aucune donnee n'existe. Le fond gris du trace les distingue.
    pivot = daily.pivot(index=COW, columns="Day", values="severity")
    severity = pivot.sum(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[severity.index]

    # Colonnes reindexees sur le calendrier complet, et axe des dates reel
    # plutot que categoriel : les journees sans aucune mesure forment une bande
    # vide au lieu d'etre escamotees, et l'axe coincide avec celui de la courbe
    # placee dessous. Deux dates identiques s'y lisent donc a la meme abscisse.
    jours = pd.date_range(daily["Day"].min(), daily["Day"].max(), freq="D")
    pivot = pivot.reindex(columns=jours)

    herd_daily = (
        daily.groupby("Day")
        .agg(cows=("day_alert", "size"), cows_alert=("day_alert", "sum"), total_notifs=("day_notifs", "sum"))
        .reset_index()
    )

    # Deux effectifs plutot qu'une proportion : l'effectif observe varie d'un
    # jour a l'autre, si bien qu'une part de 100 % peut recouvrir une seule
    # vache aussi bien que tout le troupeau. Les tracer ensemble rend le
    # denominateur visible sur la figure imprimee, ou aucune infobulle ne
    # s'affiche.
    #
    # La reindexation sur le calendrier complet laisse vides les journees sans
    # aucune mesure : la courbe s'y interrompt au lieu d'etre franchie par un
    # segment droit, qui donnerait a lire une evolution la ou rien n'est mesure.
    herd_plot = herd_daily.set_index("Day").reindex(jours).rename_axis("Day").reset_index()

    # Un seul graphique a deux rangees plutot que deux graphiques empiles : les
    # abscisses sont alors partagees, si bien qu'une meme date tombe exactement
    # au meme endroit dans la carte et dans la courbe. Les dates ne sont ecrites
    # qu'une fois, sous la rangee du bas.
    fig_compare = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.70, 0.30], vertical_spacing=0.09,
        subplot_titles=(
            "Carte thermique des alertes journalières",
            "Vaches en alerte et vaches observées, par jour",
        ),
    )

    fig_compare.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=[f"Vache {c}" for c in pivot.index],
            colorscale="YlOrRd",
            zmin=0, zmax=1,
            hovertemplate="Jour=%{x}<br>%{y}<br>Sévérité=%{z:.0%}<extra></extra>",
            colorbar=dict(title="Sévérité", len=0.70, y=1.0, yanchor="top"),
        ),
        row=1, col=1,
    )
    fig_compare.add_trace(
        go.Scatter(
            x=herd_plot["Day"], y=herd_plot["cows_alert"],
            name="Vaches en alerte", mode="lines+markers",
            line=dict(color="#1f77b4"),
            hovertemplate="Jour=%{x|%d %b}<br>Vaches en alerte=%{y}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig_compare.add_trace(
        go.Scatter(
            x=herd_plot["Day"], y=herd_plot["cows"],
            name="Vaches observées", mode="lines+markers",
            line=dict(color="#9e9e9e", dash="dot"), marker=dict(size=4),
            hovertemplate="Jour=%{x|%d %b}<br>Vaches observées=%{y}<extra></extra>",
        ),
        row=2, col=1,
    )

    # Le fond gris ne couvre que la carte : il signale les couples sans mesure,
    # notion qui n'a pas de sens sous la courbe.
    fig_compare.add_shape(
        type="rect", xref="x domain", yref="y domain",
        x0=0, x1=1, y0=0, y1=1,
        fillcolor="#e0e0e0", line_width=0, layer="below",
        row=1, col=1,
    )

    # Sous-titres ramenes a gauche, legende posee a droite sur la meme ligne que
    # celui de la rangee du bas : les deux ne se genent pas et le titre cesse
    # d'etre colle a la legende.
    #
    # Les sous-titres sont en outre remontes de quelques pixels : places par
    # defaut au ras du trace, ils touchaient la premiere rangee de vaches. La
    # legende suit le meme decalage pour rester sur la ligne du sous-titre du
    # bas, ce qui suppose de le convertir en fraction de la hauteur.
    hauteur_px = 820
    decalage_px = 14
    for annotation in fig_compare.layout.annotations:
        annotation.update(x=0.0, xanchor="left", yshift=decalage_px)
    haut_rangee_basse = fig_compare.layout.yaxis2.domain[1] + decalage_px / hauteur_px

    fig_compare.update_yaxes(title_text="Nombre de vaches", row=2, col=1)
    fig_compare.update_xaxes(title_text="Jour", row=2, col=1)
    fig_compare.update_layout(
        height=hauteur_px,
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=haut_rangee_basse,
            xanchor="right", x=1.0,
        ),
    )
    st_plotly(fig_compare, "tab3", "compare_daily", file_hash, width="stretch")


def render_tab_export(
    *,
    df: pd.DataFrame,
    interval: str,
    file_hash: str,
    mk_key: KeyFn,
    build_cache_key_full: CacheKeyFn,
    pipeline_kwargs: PipelineKwargs,
) -> None:
    """Rend l'onglet export des résultats CSV."""
    st.subheader("Export des résultats")
    _ensure_full_herd_cache(
        df=df,
        pipeline_kwargs=pipeline_kwargs,
        build_cache_key_full=build_cache_key_full,
        spinner_text="Calcul des données troupeau pour l'export...",
    )

    summary_df = st.session_state["summary_df_full"]
    out_df = st.session_state["out_df_full"]

    st.markdown("### Fichiers disponibles")

    st.download_button(
        "Résumé troupeau (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"summary_{interval}_{file_hash}.csv",
        mime="text/csv",
        key=mk_key("export_sum", file_hash),
    )

    st.download_button(
        "Données complètes troupeau (CSV)",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"full_herd_{interval}_{file_hash}.csv",
        mime="text/csv",
        key=mk_key("export_full", file_hash),
    )

    episodes = out_df[out_df.get("hybrid_warning_start", 0) == 1].copy()
    if len(episodes) > 0:
        st.download_button(
            "Alertes comportementales uniquement (CSV)",
            data=episodes.to_csv(index=False).encode("utf-8"),
            file_name=f"episodes_{interval}_{file_hash}.csv",
            mime="text/csv",
            key=mk_key("export_ep", file_hash),
        )

        st.markdown(f"**{len(episodes)}** alertes comportementales détectées")
        st.dataframe(episodes.head(50), width="stretch", height=300)
    else:
        st.info("Aucune alerte comportementale détectée.")


__all__ = [
    "render_tab_herd",
    "render_tab_daily_comparison",
    "render_tab_export",
]
