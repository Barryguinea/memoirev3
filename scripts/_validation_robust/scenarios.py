"""Scenarios d'injection utilises par la validation robuste."""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from core.features import interval_to_minutes
from core.io import COW, LYING, MI, STANDING, STEPS, TIME, TRANSITIONS, TR_DOWN, TR_UP
from scripts.validation_common import to_numeric_cols as _to_numeric_cols
from scripts.validation_injection_raw import InjectedEvent, apply_event_on_raw
from scripts.validation_selection import (
    _augment_summary_with_injectability,
    _pick_window_for_cow,
    _resolve_target_cow,
    pick_cows_for_injection,
)

def _inject_scenario_raw(
    df_base: pd.DataFrame,
    summary_base: pd.DataFrame,
    *,
    interval: str,
    persist_hours: int,
    seed: int,
    scenario: str,
    n_events: int = 1,
    detect_cow: Optional[str] = None,
    hidden_cow: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    df = df_base.sort_values([COW, TIME]).copy().reset_index(drop=True)
    _to_numeric_cols(df, [STEPS, MI, LYING, STANDING, TRANSITIONS, TR_UP, TR_DOWN])

    minutes = interval_to_minutes(interval)
    k = max(1, int(round((float(persist_hours) * 60) / max(1, minutes))))

    if scenario == "detectable_strong":
        prof = "detectable_strong"
        dur = max(int(round(2.0 * k)), 52)
        ratios = [0.58, 0.70, 0.82]
        expected = 1
        reason = "Strong and persistent multi-signal episode; should be detected."
    elif scenario == "detectable_borderline":
        prof = "detectable_borderline"
        dur = max(int(round(1.45 * k)), 40)
        ratios = [0.54, 0.68, 0.82]
        expected = 1
        reason = "Borderline but still coherent episode; intended to be partially detectable."
    elif scenario == "non_detectable_short":
        prof = "non_detectable_short_single_signal"
        dur = max(6, int(round(0.30 * k)))
        ratios = [0.20]
        expected = 0
        reason = "Short and weak one-family perturbation; expected non-detection."
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    summary_auto = _augment_summary_with_injectability(summary_base, df_base)
    if detect_cow is None or hidden_cow is None:
        cow_detect, cow_hidden = pick_cows_for_injection(summary_auto)
    else:
        cow_detect = _resolve_target_cow(summary_auto, str(detect_cow))
        cow_hidden = _resolve_target_cow(summary_auto, str(hidden_cow))
        if cow_hidden == cow_detect:
            s = summary_auto.copy()
            s[COW] = s[COW].astype(str)
            alt = [c for c in s[COW].tolist() if c != cow_detect]
            if alt:
                cow_hidden = str(alt[0])
    target_cow = cow_hidden if expected == 0 else cow_detect

    idx = np.flatnonzero(df[COW].astype(str).values == str(target_cow))
    events: List[InjectedEvent] = []
    if len(idx) >= dur + 10:
        n_target = max(1, int(n_events)) if expected == 1 else 1
        occupied = np.zeros(len(idx), dtype=bool)
        for i in range(n_target):
            ratio = float(ratios[i % len(ratios)])
            placed = _pick_window_for_cow(n=len(idx), duration=dur, preferred_ratio=ratio, occupied=occupied, margin=8)
            if placed is None:
                continue
            s, e = placed
            g = idx[s : e + 1]
            apply_event_on_raw(df, g, prof, rng)
            events.append(
                InjectedEvent(
                    event_id=f"{scenario}_{seed}_{i+1}",
                    cow=str(target_cow),
                    start=pd.Timestamp(df.loc[g[0], TIME]),
                    end=pd.Timestamp(df.loc[g[-1], TIME]),
                    duration_bins=int(len(g)),
                    profile=prof,
                    expected_detected=int(expected),
                    design_reason=reason,
                )
            )
    return df, pd.DataFrame([asdict(e) for e in events])

