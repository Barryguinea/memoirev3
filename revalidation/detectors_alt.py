"""Détecteurs alternatifs orientés BAISSE d'activité (vraie signature de boiterie).

Deux pistes, testées sans modifier ``core/`` :

  - Option 2 : règle d'hypoactivité à 15 min — épisode si, sur une fenêtre de
    persistance, l'activité (Motion Index) est durablement BASSE (rrz <= -2) et
    le couchage durablement HAUT (rrz >= +2), SANS exiger la densité d'anomalies
    IF (``anom_rate_k >= 0.24``) qui bloque les baisses.

  - Option 3 : détection sur AGRÉGATS JOURNALIERS — z-score robuste par vache sur
    pas/jour, MI/jour, couchage/jour ; épisode si l'activité chute (z <= -2) et le
    couchage monte (z >= +2) sur >= N jours consécutifs. C'est l'échelle utilisée
    dans la littérature (Alsaaod, Chapinal).

But : voir si ces logiques détectent une boiterie réaliste (que le pipeline
actuel manque) tout en restant spécifiques (peu de faux positifs).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core import config as C
from core.features import build_interval_features, interval_to_minutes, robust_z, rolling_robust_z
from core.io import COW, LYING, MI, STEPS, TIME, TRANSITIONS, available_base_cols


# ===========================================================================
# Option 2 : hypoactivité à 15 minutes
# ===========================================================================
def detect_hypoactivity_15min(
    df_cow_raw: pd.DataFrame,
    *,
    interval: str = "15T",
    window_baseline: int = 24,
    persist_hours: int = 7,
    low_act_rate_thr: float = 0.30,
    rest_rate_thr: float = 0.30,
    z_thr: float = 2.0,
    coverage_min_pct: float = 25.0,
) -> pd.DataFrame:
    """Marque les bins en épisode d'hypoactivité (activité basse + repos haut soutenus)."""
    it = build_interval_features(
        df_cow_raw, time_col=TIME, interval=interval,
        cols=available_base_cols(df_cow_raw), window_baseline=window_baseline,
    )
    mins = interval_to_minutes(interval)
    k = max(1, int(round(persist_hours * 60 / mins)))

    mi_col = "Motion Index_sum_log_rrz" if "Motion Index_sum_log_rrz" in it else "Motion Index_sum_rrz"
    lying_col = "Lying Time_sum_rrz"
    mi_z = pd.to_numeric(it.get(mi_col, pd.Series(0.0, index=it.index)), errors="coerce").fillna(0.0)
    lying_z = pd.to_numeric(it.get(lying_col, pd.Series(0.0, index=it.index)), errors="coerce").fillna(0.0)

    low_act = (mi_z <= -float(z_thr)).astype(int)        # activité basse
    high_rest = (lying_z >= float(z_thr)).astype(int)    # couchage haut

    # gating couverture
    if "coverage_pct" in it:
        ok = pd.to_numeric(it["coverage_pct"], errors="coerce").fillna(0.0) >= float(coverage_min_pct)
        low_act *= ok.astype(int); high_rest *= ok.astype(int)

    low_rate = low_act.rolling(k, min_periods=1).mean()
    rest_rate = high_rest.rolling(k, min_periods=1).mean()
    it["hypo_episode"] = ((low_rate >= low_act_rate_thr) & (rest_rate >= rest_rate_thr)).astype(int)
    return it


# ===========================================================================
# Option 3 : agrégats journaliers
# ===========================================================================
def daily_aggregate(df_cow_raw: pd.DataFrame) -> pd.DataFrame:
    """Agrège par jour (somme) : pas, MI, minutes couchées, transitions."""
    x = df_cow_raw.copy()
    x[TIME] = pd.to_datetime(x[TIME])
    x = x.set_index(TIME).sort_index()
    cols = {c: "sum" for c in [STEPS, MI, LYING, TRANSITIONS] if c in x.columns}
    daily = x.resample("1D").agg(cols)
    daily = daily[daily.sum(axis=1) > 0]  # jours sans donnée écartés
    return daily.reset_index()


def detect_daily_hypoactivity(
    df_cow_raw: pd.DataFrame,
    *,
    z_thr: float = 2.0,
    min_consecutive_days: int = 2,
) -> pd.DataFrame:
    """Épisode journalier : activité basse (z<=-2) ET couchage haut (z>=+2) sur >=N jours."""
    daily = daily_aggregate(df_cow_raw)
    if len(daily) < 5:
        daily["daily_episode"] = 0
        return daily
    steps_z = robust_z(daily[STEPS].to_numpy()) if STEPS in daily else np.zeros(len(daily))
    mi_z = robust_z(daily[MI].to_numpy()) if MI in daily else np.zeros(len(daily))
    lying_z = robust_z(daily[LYING].to_numpy()) if LYING in daily else np.zeros(len(daily))
    daily["steps_z"] = steps_z; daily["mi_z"] = mi_z; daily["lying_z"] = lying_z

    low_activity = ((steps_z <= -z_thr) | (mi_z <= -z_thr))
    high_rest = (lying_z >= z_thr)
    flag = (low_activity & high_rest).astype(int)

    # persistance : >= N jours consécutifs
    episode = np.zeros(len(daily), dtype=int)
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        if run >= int(min_consecutive_days):
            episode[i - run + 1 : i + 1] = 1
    daily["daily_episode"] = episode
    return daily
