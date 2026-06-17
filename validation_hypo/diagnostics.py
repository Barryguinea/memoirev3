"""Diagnostic : le pipeline détecte-t-il une vraie signature de boiterie (baisse d'activité) ?

But : comprendre POURQUOI la détection échoue/réussit, en inspectant les signaux
internes du pipeline sur une fenêtre d'injection réaliste (activité réduite,
couchage augmenté), placée dans la zone post-baseline. Aucun chiffre du mémoire n'est
touché ; ceci sert seulement à choisir un modèle d'injection honnête et juste.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from core import config as C
from core.features import build_interval_features
from core.io import (
    COW, LYING, MI, STANDING, STEPS, TIME, TR_DOWN, TR_UP, TRANSITIONS, available_base_cols, load_csv,
)
from core.pipeline import run_pipeline_one_cow
from validation_hypo.campaign import _heldout_start_time, final_params


def inject_sustained_lameness(
    df_cow_raw: pd.DataFrame,
    *,
    heldout_start: pd.Timestamp,
    duration_bins: int,
    steps_factor: float,
    mi_factor: float,
    lying_target_min: float,
    standing_factor: float,
    trans_factor: float,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Injecte une baisse d'activité soutenue après la baseline.

    steps/mi/standing/trans sont multipliés (réduction) ; lying est poussé vers un
    plancher minimal (la vache couchée plus longtemps).
    """
    df = df_cow_raw.sort_values(TIME).copy().reset_index(drop=True)
    times = pd.to_datetime(df[TIME])
    start_pos = int(np.searchsorted(times.to_numpy(), np.datetime64(heldout_start), side="left")) + 4
    start_pos = min(start_pos, len(df) - duration_bins - 2)
    g = np.arange(start_pos, start_pos + duration_bins)

    for col, fac in [(STEPS, steps_factor), (MI, mi_factor), (STANDING, standing_factor),
                     (TRANSITIONS, trans_factor), (TR_UP, trans_factor), (TR_DOWN, trans_factor)]:
        if col in df.columns:
            df.loc[g, col] = np.maximum(0.0, pd.to_numeric(df.loc[g, col], errors="coerce").fillna(0.0).to_numpy() * fac)
    if LYING in df.columns:
        cur = pd.to_numeric(df.loc[g, LYING], errors="coerce").fillna(0.0).to_numpy()
        df.loc[g, LYING] = np.maximum(cur, float(lying_target_min))  # couchée plus longtemps
    return df, pd.Timestamp(df.loc[g[0], TIME]), pd.Timestamp(df.loc[g[-1], TIME])


def diagnose(
    cow: str = "8147",
    *,
    raw_csv: str = "data/brut.csv",
    duration_bins: int = 192,           # ~2 jours
    steps_factor: float = 0.0,
    mi_factor: float = 0.05,
    lying_target_min: float = 14.0,
    standing_factor: float = 0.1,
    trans_factor: float = 0.1,
    params: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Injecte une signature et renvoie les signaux internes du pipeline dans la fenêtre."""
    params = params or final_params()
    df_all = load_csv(raw_csv); df_all[COW] = df_all[COW].astype(str)
    raw_cow = df_all[df_all[COW] == str(cow)]

    heldout = _heldout_start_time(
        raw_cow, interval=str(params["interval"]), window_baseline=int(params["window_baseline"]),
        baseline_ratio=float(params["baseline_ratio"]), coverage_min_pct=float(params["coverage_min_pct"]),
    )
    df_inj, t0, t1 = inject_sustained_lameness(
        raw_cow, heldout_start=heldout, duration_bins=duration_bins,
        steps_factor=steps_factor, mi_factor=mi_factor, lying_target_min=lying_target_min,
        standing_factor=standing_factor, trans_factor=trans_factor,
    )
    it = run_pipeline_one_cow(df_inj, cow, **params)
    it[TIME] = pd.to_datetime(it[TIME])
    win = it[(it[TIME] >= t0) & (it[TIME] <= t1)]

    def col_stat(name, fn=np.nanmin):
        return round(float(fn(pd.to_numeric(win.get(name, pd.Series(dtype=float)), errors="coerce"))), 3) if name in win else None

    return {
        "cow": cow, "duration_bins": duration_bins, "n_win_bins": int(len(win)),
        "factors": {"steps": steps_factor, "mi": mi_factor, "lying_min": lying_target_min},
        # --- z-scores robustes roulants dans la fenêtre (négatifs = baisse détectée) ---
        "Steps_sum_rrz_min": col_stat("Steps_sum_rrz"),
        "MI_sum_log_rrz_min": col_stat("Motion Index_sum_log_rrz"),
        "Lying_sum_rrz_max": col_stat("Lying Time_sum_rrz", np.nanmax),
        # --- signaux de règles ---
        "if_anom_k_max": col_stat("if_anom_k", np.nanmax),
        "anom_rate_k_max": col_stat("anom_rate_k", np.nanmax),
        "n_families_max": col_stat("n_families", np.nanmax),
        "fam_activity_any": col_stat("fam_activity", np.nanmax),
        "fam_rest_any": col_stat("fam_rest", np.nanmax),
        "mi_spike_any": col_stat("mi_spike", np.nanmax),
        # --- détection finale ---
        "episode_detected": int(pd.to_numeric(win.get("pred_problem_episode", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if "pred_problem_episode" in win else 0,
        "boiterie_detected": int(pd.to_numeric(win.get("pred_lameness_episode", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if "pred_lameness_episode" in win else 0,
        "notif": int(pd.to_numeric(win.get("notif_lameness", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "notif_lameness" in win else 0,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(diagnose(), ensure_ascii=False, indent=2))
