"""Injections sur donnees traitees/predictions pour validations (processed)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from core.features import interval_to_minutes
from core.io import COW, TIME
from scripts.validation_common import to_numeric_cols as _to_numeric_cols
from scripts.validation_injection_raw import InjectedEvent, choose_start_index
from scripts.validation_selection import _pick_window_for_cow, pick_cows_for_injection

def pick_rrz_col(df: pd.DataFrame, keys: Iterable[str]) -> str:
    """Retourne la première colonne finissant par ``'_rrz'`` dont le nom contient une des *keys*."""
    for c in df.columns:
        sc = str(c)
        if sc.endswith("_rrz") and any(k in sc for k in keys):
            return sc
    return ""


def inject_two_events_processed(
    pred_base: pd.DataFrame,
    summary_base: pd.DataFrame,
    *,
    interval: str,
    persist_hours: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Injecte un événement détectable + un caché dans des données de prédiction traitées."""
    rng = np.random.default_rng(seed)
    df = pred_base.sort_values([COW, TIME]).copy().reset_index(drop=True)
    _to_numeric_cols(df, ["if_anomaly_point"])

    minutes = interval_to_minutes(interval)
    k = max(1, int(round((float(persist_hours) * 60) / max(1, minutes))))
    detectable_len = max(k + 8, 36)
    hidden_len = max(6, int(round(0.35 * k)))

    c_activity = pick_rrz_col(df, ["Steps", "Motion Index"])
    c_rest = pick_rrz_col(df, ["Lying Time", "Standing Time"])
    c_trans = pick_rrz_col(df, ["Transitions"])
    rrz_cols = [c for c in [c_activity, c_rest, c_trans] if c]
    _to_numeric_cols(df, rrz_cols)

    cow_detect, cow_hidden = pick_cows_for_injection(summary_base)
    events: List[InjectedEvent] = []

    idx_a = np.flatnonzero(df[COW].astype(str).values == str(cow_detect))
    if len(idx_a) >= detectable_len + 10:
        s_a = choose_start_index(len(idx_a), detectable_len, 0.65, rng)
        g_a = idx_a[s_a : s_a + detectable_len]
        df.loc[g_a, "if_anomaly_point"] = 1.0
        for c in rrz_cols:
            df.loc[g_a, c] = np.where((np.arange(len(g_a)) % 2) == 0, 4.0, -4.0)
        events.append(
            InjectedEvent(
                event_id="inj_detectable_1",
                cow=str(cow_detect),
                start=pd.Timestamp(df.loc[g_a[0], TIME]),
                end=pd.Timestamp(df.loc[g_a[-1], TIME]),
                duration_bins=int(len(g_a)),
                profile="detectable_processed_visible_pattern",
                expected_detected=1,
                design_reason="Injected on anomaly/coherence drivers in processed data to create visible lameness pattern.",
            )
        )

    idx_b = np.flatnonzero(df[COW].astype(str).values == str(cow_hidden))
    if len(idx_b) >= hidden_len + 10:
        s_b = choose_start_index(len(idx_b), hidden_len, 0.20, rng)
        g_b = idx_b[s_b : s_b + hidden_len]
        df.loc[g_b, "if_anomaly_point"] = 0.0
        if c_activity:
            df.loc[g_b, c_activity] = 0.6
        events.append(
            InjectedEvent(
                event_id="inj_hidden_1",
                cow=str(cow_hidden),
                start=pd.Timestamp(df.loc[g_b[0], TIME]),
                end=pd.Timestamp(df.loc[g_b[-1], TIME]),
                duration_bins=int(len(g_b)),
                profile="non_detectable_processed_short_single_signal",
                expected_detected=0,
                design_reason="Short and weak one-family perturbation, expected under persistence/coherence thresholds.",
            )
        )

    out_events = pd.DataFrame([asdict(e) for e in events])
    return df, out_events


def inject_manual_plan_processed(
    pred_base: pd.DataFrame,
    *,
    plan: List[Dict[str, object]],
    interval: str,
    persist_hours: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # NOTE: seed est accepte pour compatibilite d'API avec inject_manual_plan_raw(),
    # mais les injections processed sont deterministes par construction
    # (placement geometrique via _pick_window_for_cow, valeurs fixes).
    del seed  # rend explicite que le parametre n'est pas utilise
    df = pred_base.sort_values([COW, TIME]).copy().reset_index(drop=True)
    _to_numeric_cols(df, ["if_anomaly_point"])

    minutes = interval_to_minutes(interval)
    k = max(1, int(round((float(persist_hours) * 60) / max(1, minutes))))
    detectable_len = max(k + 8, 36)
    hidden_len = max(6, int(round(0.35 * k)))

    c_activity = pick_rrz_col(df, ["Steps", "Motion Index"])
    c_rest = pick_rrz_col(df, ["Lying Time", "Standing Time"])
    c_trans = pick_rrz_col(df, ["Transitions"])
    rrz_cols = [c for c in [c_activity, c_rest, c_trans] if c]
    _to_numeric_cols(df, rrz_cols)

    events: List[InjectedEvent] = []
    by_cow: Dict[str, np.ndarray] = {}
    occupied: Dict[str, np.ndarray] = {}

    for p in plan:
        cow = str(p["cow"])
        prof = str(p["profile"])
        exp = int(p["expected_detected"])
        ratio = float(p.get("preferred_ratio", 0.65))
        if cow not in by_cow:
            idx = np.flatnonzero(df[COW].astype(str).values == cow)
            by_cow[cow] = idx
            occupied[cow] = np.zeros(len(idx), dtype=bool)
        idx_cow = by_cow[cow]
        if len(idx_cow) == 0:
            continue
        dur = detectable_len if exp == 1 else hidden_len
        if exp == 1 and "duration_scale" in p:
            try:
                scale = max(1.0, float(p.get("duration_scale", 1.0)))
            except Exception:
                scale = 1.0
            dur = max(dur, int(round(dur * scale)))
        placed = _pick_window_for_cow(n=len(idx_cow), duration=dur, preferred_ratio=ratio, occupied=occupied[cow], margin=8)
        if placed is None:
            continue
        s, e = placed
        g = idx_cow[s : e + 1]

        if exp == 1:
            df.loc[g, "if_anomaly_point"] = 1.0
            for c in rrz_cols:
                df.loc[g, c] = np.where((np.arange(len(g)) % 2) == 0, 4.0, -4.0)
            reason = "Injected on anomaly/coherence drivers in processed data to create visible lameness pattern."
        else:
            df.loc[g, "if_anomaly_point"] = 0.0
            if c_activity:
                df.loc[g, c_activity] = 0.6
            reason = "Short and weak one-family perturbation, expected under persistence/coherence thresholds."

        events.append(
            InjectedEvent(
                event_id=str(p["event_id"]),
                cow=cow,
                start=pd.Timestamp(df.loc[g[0], TIME]),
                end=pd.Timestamp(df.loc[g[-1], TIME]),
                duration_bins=int(len(g)),
                profile=prof,
                expected_detected=exp,
                design_reason=reason,
            )
        )
    return df, pd.DataFrame([asdict(e) for e in events])


