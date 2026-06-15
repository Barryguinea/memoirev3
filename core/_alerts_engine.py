"""Implémentation interne de la logique d'alertes métier (episodes + notifications)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from core._alerts_helpers import _family_presence, _mi_spike_series, _pick_rrz_candidates
from core.config import (
    CONFIDENCE_W_ANOMRATE,
    CONFIDENCE_W_FAMILY,
    CONFIDENCE_W_IF_SCORE,
    CONFIDENCE_W_MI_SPIKE,
)
from core.features import interval_to_minutes


def apply_alert_logic(
    interval_df: pd.DataFrame,
    *,
    time_col: str = "T",
    interval: str = "1H",
    persist_hours: int = 7,
    alert_min: int = 2,
    mix_mode: str = "MIX",  # "IF-ONLY" ou "MIX"
    mix_rate_thr: float = 0.24,
    z_low_thr: float = -2.0,
    z_high_thr: float = +2.0,
    require_two_signals: bool = False,
    cooldown_hours: int = 12,
    pct_crit_thr: float = 10.0,
    mi_z_high_thr: float = +2.2,
    episode_requires_coherence: bool = True,
    episode_min_families: int = 1,
    episode_allow_mi_alone: bool = True,
    boiterie_min_families: int = 2,
    require_mi_for_boiterie: bool = False,
    coverage_min_pct: float = 25.0,
    # préférences candidates rrz (sans (s) pour éviter l’incohérence)
    pref_rrz_cols: Optional[List[str]] = None,
    mi_name: str = "Motion Index",
) -> pd.DataFrame:
    df = interval_df.sort_values(time_col).copy()

    mix_mode = str(mix_mode).strip().upper()
    if mix_mode not in ("IF-ONLY", "MIX"):
        mix_mode = "MIX"
    mix_on = mix_mode != "IF-ONLY"

    mins = interval_to_minutes(interval)
    window_k = max(1, int(round((float(persist_hours) * 60) / mins)))
    df["K"] = window_k

    # --- compat IF anomaly point ---
    if "if_anomaly_point" in df.columns:
        if_anom = pd.to_numeric(df["if_anomaly_point"], errors="coerce").fillna(0).astype(int)
    elif "if_anomaly" in df.columns:
        if_anom = pd.to_numeric(df["if_anomaly"], errors="coerce").fillna(0).astype(int)
    elif "if_pred" in df.columns:
        if_anom = (
            pd.to_numeric(df["if_pred"], errors="coerce").fillna(1).astype(int) == -1
        ).astype(int)
    else:
        if_anom = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    df["if_anomaly_point"] = if_anom

    # --- PHASE 1 : early coverage gating sur signaux bruts ---
    # Calculer le masque une seule fois, reutilise dans les phases 1b et 1c.
    _low_cov = pd.Series(False, index=df.index)
    if "coverage_pct" in df.columns:
        _low_cov = (df["coverage_pct"] < float(coverage_min_pct)).fillna(True)

    # Phase 1a : zero IF anomaly sur faible couverture AVANT rolling
    if _low_cov.any():
        df.loc[_low_cov, "if_anomaly_point"] = 0

    # IF persistance
    df["if_anom_k"] = (
        df["if_anomaly_point"].rolling(window_k, min_periods=1).sum().fillna(0).astype(int)
    )
    df["anom_rate_k"] = df["if_anom_k"] / float(window_k)

    # candidates familles
    if pref_rrz_cols is None:
        pref_rrz_cols = [
            "Motion Index_sum_log_rrz",
            "Motion Index_sum_rrz",
            "Steps_sum_rrz",
            "Lying Time_sum_rrz",
            "Standing Time_sum_rrz",
            "Transitions_sum_rrz",
        ]

    candidates = _pick_rrz_candidates(df, pref_rrz_cols)
    fam_rate_thr = 0.10 if not mix_on else 0.35

    fam = _family_presence(
        df,
        candidates[:10],
        z_low=float(z_low_thr),
        z_high=float(z_high_thr),
        window_k=window_k,
        rate_thr=fam_rate_thr,
    )
    for c in fam.columns:
        df[c] = fam[c].values

    # Phase 1b : zero familles/hits sur faible couverture AVANT rolling n_families_k
    if _low_cov.any():
        for _fc in [
            "fam_activity",
            "fam_rest",
            "fam_transitions",
            "fam_other",
            "hit_activity",
            "hit_rest",
            "hit_transitions",
            "hit_other",
        ]:
            if _fc in df.columns:
                df.loc[_low_cov, _fc] = 0

    df["n_families"] = (
        df[["fam_activity", "fam_rest", "fam_transitions", "fam_other"]].sum(axis=1).astype(int)
    )
    df["n_families_k"] = (
        df["n_families"].rolling(window_k, min_periods=1).max().fillna(0).astype(int)
    )

    # MI spike
    mi_sig, mi_used_col = _mi_spike_series(df, mi_name=mi_name, mi_z_high_thr=mi_z_high_thr)
    df["mi_spike"] = mi_sig
    df["mi_feat_used"] = mi_used_col if mi_used_col else ""

    # Phase 1c : zero MI spike sur faible couverture AVANT rolling mi_spike_k
    if _low_cov.any():
        df.loc[_low_cov, "mi_spike"] = 0

    df["mi_spike_k"] = df["mi_spike"].rolling(window_k, min_periods=1).sum().fillna(0).astype(int)

    # COHERENCE EPISODE
    fam_min_for_episode = 2 if bool(require_two_signals) else int(episode_min_families)
    if mix_on:
        fam_min_for_episode = max(2, fam_min_for_episode)
        episode_allow_mi_alone = False

    df["has_core_family"] = (((df["fam_activity"] == 1) | (df["fam_rest"] == 1)).astype(int))

    if episode_requires_coherence:
        if episode_allow_mi_alone:
            df["flag_coherent_episode"] = (
                ((df["n_families"] >= fam_min_for_episode) & (df["has_core_family"] == 1))
                | (df["mi_spike"] == 1)
            ).astype(int)
        else:
            df["flag_coherent_episode"] = (
                (df["n_families"] >= fam_min_for_episode) & (df["has_core_family"] == 1)
            ).astype(int)
    else:
        df["flag_coherent_episode"] = 1

    # COHERENCE BOITERIE
    df["coherence_boiterie"] = (df["n_families"] >= int(boiterie_min_families)).astype(int)
    df["coherence_boiterie"] = (
        (df["coherence_boiterie"] == 1) | ((df["fam_activity"] == 1) & (df["mi_spike"] == 1))
    ).astype(int)

    if bool(require_mi_for_boiterie):
        df["coherence_boiterie"] = (
            (df["coherence_boiterie"] & (df["mi_spike"] == 1)).astype(int)
        )

    df["n_signals"] = (df["n_families"] + df["mi_spike"].astype(int)).astype(int)

    # EPISODE ANORMAL
    if not mix_on:
        df["episode_anormal"] = (
            (df["anom_rate_k"] >= float(mix_rate_thr)) & (df["if_anom_k"] >= int(alert_min))
        ).astype(int)
    else:
        df["episode_anormal"] = (
            (df["anom_rate_k"] >= float(mix_rate_thr))
            & (df["if_anom_k"] >= int(alert_min))
            & (df["flag_coherent_episode"] == 1)
        ).astype(int)

    df["episode_anormal_start"] = (
        (df["episode_anormal"] == 1) & (df["episode_anormal"].shift(1).fillna(0) == 0)
    ).astype(int)

    # BOITERIE PROBABLE
    df["boiterie_probable"] = (
        (df["episode_anormal"] == 1) & (df["coherence_boiterie"] == 1)
    ).astype(int)

    df["boiterie_start"] = (
        (df["boiterie_probable"] == 1) & (df["boiterie_probable"].shift(1).fillna(0) == 0)
    ).astype(int)

    # CRITIQUE (IF score) — inchangé
    if "if_score" in df.columns and len(df) > 10 and df["if_score"].notna().sum() > 5:
        q05 = float(np.nanquantile(df["if_score"], 0.05))
    else:
        q05 = np.nan
    df["if_score_q05"] = q05

    if "if_score" in df.columns and df["if_score"].notna().sum() > 0:
        df["if_score_pct"] = (df["if_score"].rank(pct=True, ascending=True) * 100.0).astype(float)
    else:
        df["if_score_pct"] = np.nan

    # ------- sorties normalisées (IMPORTANT) -------
    df["pred_problem_episode"] = df["episode_anormal"].astype(int)
    df["pred_problem_start"] = df["episode_anormal_start"].astype(int)

    df["pred_lameness_episode"] = df["boiterie_probable"].astype(int)
    df["pred_lameness_start"] = df["boiterie_start"].astype(int)

    # ------- coverage gating AVANT cooldown -------
    # Les intervalles à faible couverture ne doivent pas générer de signaux
    if "coverage_pct" in df.columns:
        low = (df["coverage_pct"] < float(coverage_min_pct)).fillna(True)
        if low.any():
            cols_zero = [
                "if_anomaly_point",
                "if_anom_k",
                "anom_rate_k",
                "mi_spike",
                "mi_spike_k",
                "flag_coherent_episode",
                "episode_anormal",
                "episode_anormal_start",
                "boiterie_probable",
                "boiterie_start",
                "coherence_boiterie",
                "n_signals",
                "n_families",
                "n_families_k",
                "fam_activity",
                "fam_rest",
                "fam_transitions",
                "fam_other",
                "hit_activity",
                "hit_rest",
                "hit_transitions",
                "hit_other",
                "pred_problem_episode",
                "pred_problem_start",
                "pred_lameness_episode",
                "pred_lameness_start",
            ]
            for c in cols_zero:
                if c in df.columns:
                    df.loc[low, c] = 0

    # ------- notification boiterie (cooldown) -------
    cooldown_intervals = max(1, int(round((float(cooldown_hours) * 60) / mins)))
    df["in_cooldown"] = 0
    df["notif_lameness"] = 0
    last_fire_idx = -(10**9)

    bs_idx = df.columns.get_loc("pred_lameness_start")
    cool_idx = df.columns.get_loc("in_cooldown")
    notif_idx = df.columns.get_loc("notif_lameness")

    for i in range(len(df)):
        if i - last_fire_idx <= cooldown_intervals:
            df.iat[i, cool_idx] = 1
        if int(df.iat[i, bs_idx]) == 1:
            if i - last_fire_idx > cooldown_intervals:
                df.iat[i, notif_idx] = 1
                last_fire_idx = i

    # alert_level
    df["alert_level"] = "normal"
    df.loc[df["if_anomaly_point"] == 1, "alert_level"] = "suspect"
    df.loc[df["pred_lameness_episode"] == 1, "alert_level"] = "probable"

    if np.isfinite(q05) and "if_score" in df.columns and "if_score_pct" in df.columns:
        df.loc[
            (df["pred_lameness_episode"] == 1)
            & (df["if_score"] <= q05)
            & (df["if_score_pct"] <= float(pct_crit_thr)),
            "alert_level",
        ] = "critique"

    df["is_critique"] = (df["alert_level"] == "critique").astype(int)

    # Appliquer aussi le gating sur les colonnes post-cooldown
    if "coverage_pct" in df.columns:
        low = (df["coverage_pct"] < float(coverage_min_pct)).fillna(True)
        if low.any():
            for c in ["in_cooldown", "notif_lameness", "is_critique"]:
                if c in df.columns:
                    df.loc[low, c] = 0
            df.loc[low, "alert_level"] = "normal"

    # lame_confidence
    # LIMITATION : les poids sont empiriques (voir core/config.py), pas calibres
    # contre des donnees labelisees. Le score est utile pour le classement
    # relatif mais ne doit pas etre interprete comme une probabilite.
    df["lame_confidence"] = 0.0
    fam_score = (df.get("n_families_k", 0) / 3.0).clip(0, 1)
    anom_score = df.get("anom_rate_k", 0).clip(0, 1)
    mi_score = (df.get("mi_spike_k", 0) / df.get("K", 1).replace(0, np.nan)).fillna(0).clip(0, 1)

    if "if_score_pct" in df.columns:
        if_score_norm = (1 - df["if_score_pct"] / 100.0).clip(0, 1).fillna(0)
    else:
        if_score_norm = 0.0

    df["lame_confidence"] = (
        CONFIDENCE_W_FAMILY * fam_score
        + CONFIDENCE_W_ANOMRATE * anom_score
        + CONFIDENCE_W_MI_SPIKE * mi_score
        + CONFIDENCE_W_IF_SCORE * if_score_norm
    ).round(1)

    # Zero lame_confidence sur faible couverture
    if _low_cov.any():
        df.loc[_low_cov, "lame_confidence"] = 0.0

    return df

