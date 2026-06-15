"""Panneaux Streamlit liés au troupeau (classement, comparaison journalière, export)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.io import COW
from core.pipeline import run_pipeline_herd
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

    max_cows = st.number_input("Nombre max. d'animaux a analyser (0 = tous)", value=0, step=1, key="tab2_max")
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

    st.caption(f"Analyse terminée : {len(summary_df)} animaux")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Animaux analysés", len(summary_df))
    k2.metric("Notifications fusionnées", int(summary_df["hybrid_warning_notifs"].sum()))
    k3.metric("Animaux à vérifier", int((summary_df["hybrid_warning_notifs"] > 0).sum()))
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
    display_names = {
        COW: "Animal",
        "n_bins": "Intervalles",
        "hybrid_warning_notifs": "Notifications fusionnées",
        "behavioral_warning_notifs": "Notifications HYPO",
        "instability_warning_notifs": "Notifications INSTABILITÉ",
        "if_anomaly_points": "Anomalies IF (comparateur)",
        "coverage_mean": "Couverture moyenne (%)",
        "coverage_min": "Couverture minimale (%)",
    }
    summary_display = summary_df[display_cols].rename(columns=display_names)
    st.dataframe(summary_display, width="stretch", height=400, hide_index=True)

    st.markdown("### Animaux à vérifier en priorité")
    top_cows = summary_df.head(9)[COW].tolist()
    cols_grid = st.columns(3)
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
            with cols_grid[slot % 3]:
                r = summary_df[summary_df[COW] == cid].iloc[0]
                st.markdown(f"**Animal {cid}**")
                st.caption(
                    f"Fusion : {int(r['hybrid_warning_notifs'])} | "
                    f"HYPO : {int(r['behavioral_warning_notifs'])} | "
                    f"INSTABILITÉ : {int(r['instability_warning_notifs'])}"
                )
                fig_mini = build_small_fig(g, plot_col)
                st_plotly(fig_mini, "tab2", "mini", cid, file_hash, width="stretch")
            slot += 1

    st.download_button(
        "Telecharger le resume troupeau",
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
    """Rend l'onglet de comparaison journalière inter-animaux."""
    st.subheader("Comparaison inter-animaux (vue journalière)")
    _ensure_full_herd_cache(
        df=df,
        pipeline_kwargs=pipeline_kwargs,
        build_cache_key_full=build_cache_key_full,
        spinner_text="Calcul des donnees troupeau (complet)...",
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

    pivot = daily.pivot(index=COW, columns="Day", values="severity").fillna(0)
    severity = pivot.sum(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[severity.index]
    x_days = pd.to_datetime(pivot.columns).strftime("%b %d").tolist()

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=x_days,
            y=[f"Animal {c}" for c in pivot.index],
            colorscale="YlOrRd",
            zmin=0, zmax=1,
            hovertemplate="Jour=%{x}<br>%{y}<br>Sévérité=%{z:.0%}<extra></extra>",
            colorbar=dict(title="Sévérité"),
        )
    )
    fig_heat.update_layout(height=600, title="Carte thermique des alertes journalières", margin=dict(l=10, r=10, t=60, b=10))
    st_plotly(fig_heat, "tab3", "heatmap", file_hash, width="stretch")

    herd_daily = (
        daily.groupby("Day")
        .agg(cows=("day_alert", "size"), cows_alert=("day_alert", "sum"), total_notifs=("day_notifs", "sum"))
        .reset_index()
    )
    herd_daily["pct_alert"] = (herd_daily["cows_alert"] / herd_daily["cows"]) * 100.0

    fig_line = px.line(
        herd_daily.sort_values("Day"),
        x="Day",
        y="pct_alert",
        markers=True,
        title="Pourcentage d'animaux en alerte par jour",
    )
    fig_line.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    st_plotly(fig_line, "tab3", "line", file_hash, width="stretch")


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
    st.subheader("Export des resultats")
    _ensure_full_herd_cache(
        df=df,
        pipeline_kwargs=pipeline_kwargs,
        build_cache_key_full=build_cache_key_full,
        spinner_text="Calcul des donnees troupeau pour l'export...",
    )

    summary_df = st.session_state["summary_df_full"]
    out_df = st.session_state["out_df_full"]

    st.markdown("### Fichiers disponibles")

    st.download_button(
        "Resume troupeau (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"summary_{interval}_{file_hash}.csv",
        mime="text/csv",
        key=mk_key("export_sum", file_hash),
    )

    st.download_button(
        "Donnees completes troupeau (CSV)",
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
