"""Panneau Streamlit d'analyse individuelle."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List

import pandas as pd
import streamlit as st

from ui.clinical_ui import build_episode_why_table, load_clinical_kb, render_clinical_guidance
from ui.plots import build_multi_panel_figure
from ui.presentation import build_filterable_table, filter_detection_table

PipelineKwargs = Dict[str, Any]
PlotFn = Callable[..., Any]
KeyFn = Callable[..., str]


def render_tab_individual(
    *,
    df: pd.DataFrame,
    cows: List[str],
    interval: str,
    plot_mode: str,
    plot_pref: List[str],
    file_hash: str,
    compute_cow_cached: Callable[..., pd.DataFrame],
    st_plotly: PlotFn,
    mk_key: KeyFn,
    pipeline_kwargs: PipelineKwargs,
) -> None:
    """Rend l'onglet d'analyse individuelle."""
    st.subheader("Analyse détaillée par vache")

    cow_sel = st.selectbox("Sélectionner une vache", cows, index=0, key="tab1_cow")

    with st.spinner(f"Analyse de la vache {cow_sel}..."):
        it = compute_cow_cached(df, cow_sel, **pipeline_kwargs)

    c1, c2, c3 = st.columns(3)
    c1.metric("Intervalles", len(it))
    c2.metric("Anomalies IF (comparateur)", int(it.get("if_anomaly_point", 0).sum()))
    c3.metric("Notifications fusionnées", int(it.get("hybrid_warning_notification", 0).sum()))
    c4, c5, c6 = st.columns(3)
    c4.metric("Candidats HYPO", int(it.get("behavioral_warning_candidate", 0).sum()))
    c5.metric("Candidats INSTABILITÉ", int(it.get("instability_warning_candidate", 0).sum()))
    c6.metric("Intervalles fusionnés", int(it.get("hybrid_warning_episode", 0).sum()))

    st.markdown("---")
    col_d1, col_d2 = st.columns(2)

    min_date = it["T"].min().date() if len(it) > 0 else datetime.now().date()
    max_date = it["T"].max().date() if len(it) > 0 else datetime.now().date()

    with col_d1:
        d1 = st.date_input("Date debut", value=min_date, min_value=min_date, max_value=max_date, key="tab1_d1")
    with col_d2:
        d2 = st.date_input("Date fin", value=max_date, min_value=min_date, max_value=max_date, key="tab1_d2")

    it_filt = it[(it["T"].dt.date >= d1) & (it["T"].dt.date <= d2)].copy()

    if len(it_filt) == 0:
        st.warning("Aucune donnee dans cette periode.")
        return

    st.caption(f"Periode : {d1} à {d2} ({len(it_filt)} intervalles)")

    st.markdown("### Graphiques multi-variables")
    plot_cols = [c for c in plot_pref if c in it_filt.columns]
    if len(plot_cols) == 0:
        if "rrz" in plot_mode.lower() or "diagnostic" in plot_mode.lower():
            plot_cols = [c for c in it_filt.columns if c.endswith("_rrz")][:3]
        else:
            raw_pool = [
                c for c in it_filt.columns
                if (c.endswith("_sum") or c.endswith("_mean"))
                and not c.endswith(("_rz", "_rrz"))
                and not c.startswith(("if_", "pred_", "notif_", "expected_raw_per_bin"))
            ]
            plot_cols = raw_pool[:3]

    if len(plot_cols) > 0:
        fig_multi = build_multi_panel_figure(it_filt, plot_cols, title=f"Vache {cow_sel} ({d1} à {d2})")
        st_plotly(fig_multi, "tab1", cow_sel, d1, d2, file_hash, width="stretch")

    kb = load_clinical_kb()
    st.markdown("### Épisodes détectés (pourquoi ?)")
    ep_why = build_episode_why_table(it_filt, kb=kb)
    if len(ep_why) > 0:
        ep_why_display = ep_why.drop(columns=["consigne_clinique"], errors="ignore")
        st.dataframe(ep_why_display, width="stretch", height=250, hide_index=True)
        if "alert_level" in ep_why.columns:
            render_clinical_guidance(ep_why["alert_level"].astype(str).tolist(), kb)
        else:
            render_clinical_guidance(["normal"], kb)
    else:
        st.info("Aucun épisode détecté sur cette période.")
        render_clinical_guidance(["normal"], kb)

    st.markdown("### Table filtrable")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        only_episode = st.checkbox(
            "Épisodes fusionnés",
            key=mk_key("flt_ep", cow_sel, d1, d2),
        )
    with f2:
        only_notification = st.checkbox(
            "Notifications fusionnées",
            key=mk_key("flt_notif", cow_sel, d1, d2),
        )
    with f3:
        only_hypo_candidate = st.checkbox(
            "Candidats HYPO",
            key=mk_key("flt_hypo", cow_sel, d1, d2),
        )
    with f4:
        only_if = st.checkbox(
            "Anomalies IF (comparateur)",
            key=mk_key("flt_if", cow_sel, d1, d2),
        )

    table_df = build_filterable_table(it_filt)
    table_df = filter_detection_table(
        table_df,
        only_episode=only_episode,
        only_notification=only_notification,
        only_hypo_candidate=only_hypo_candidate,
        only_if=only_if,
    )

    st.caption(f"{len(table_df)} ligne(s) après filtres")
    st.dataframe(table_df, width="stretch", height=430, hide_index=True)

    st.download_button(
        "Télécharger les données affichées",
        data=table_df.to_csv(index=False).encode("utf-8"),
        file_name=f"vache_{cow_sel}_{interval}_{d1}_{d2}_{file_hash[:8]}.csv",
        mime="text/csv",
        key=mk_key("dl_cow", cow_sel, file_hash),
    )


__all__ = ["render_tab_individual"]
