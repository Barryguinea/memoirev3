"""Orchestrateurs d'injection raw (auto et plan manuel)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.features import interval_to_minutes
from core.io import COW, LYING, MI, STANDING, STEPS, TIME, TRANSITIONS, TR_DOWN, TR_UP
from scripts.validation_common import to_numeric_cols as _to_numeric_cols
from scripts.validation_selection import _augment_summary_with_injectability, _pick_window_for_cow, pick_cows_for_injection

from scripts._validation_injection_raw.profiles import InjectedEvent, apply_event_on_raw, choose_start_index


def inject_two_events_raw(
    df_base: pd.DataFrame,
    summary_base: pd.DataFrame,
    *,
    interval: str,
    persist_hours: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Injecte un événement détectable + un caché dans des données capteurs brutes (sélection auto des vaches)."""
    rng = np.random.default_rng(seed)
    df = df_base.sort_values([COW, TIME]).copy().reset_index(drop=True)
    _to_numeric_cols(df, [STEPS, MI, LYING, STANDING, TRANSITIONS, TR_UP, TR_DOWN])
    summary_auto = _augment_summary_with_injectability(summary_base, df_base)

    minutes = interval_to_minutes(interval)
    k = max(1, int(round((float(persist_hours) * 60) / max(1, minutes))))
    # Garder des fenêtres détectables assez longues pour la persistance, sans
    # les rendre trop larges, afin de préserver une IoU interprétable.
    detectable_len = max(int(round(1.5 * k)), 36)
    hidden_len = max(6, int(round(0.35 * k)))

    cow_detect, cow_hidden = pick_cows_for_injection(summary_auto)
    events: List[InjectedEvent] = []

    idx_a = np.flatnonzero(df[COW].astype(str).values == str(cow_detect))
    if len(idx_a) >= detectable_len + 10:
        s_a = choose_start_index(len(idx_a), detectable_len, 0.65, rng)
        g_a = idx_a[s_a : s_a + detectable_len]
        apply_event_on_raw(df, g_a, "detectable_multisignal_persistent", rng)
        events.append(
            InjectedEvent(
                event_id="inj_detectable_1",
                cow=str(cow_detect),
                start=pd.Timestamp(df.loc[g_a[0], TIME]),
                end=pd.Timestamp(df.loc[g_a[-1], TIME]),
                duration_bins=int(len(g_a)),
                profile="detectable_multisignal_persistent",
                expected_detected=1,
                design_reason="Long coherent multi-signal pattern, close to naturally detected episodes.",
            )
        )

    idx_b = np.flatnonzero(df[COW].astype(str).values == str(cow_hidden))
    if len(idx_b) >= hidden_len + 10:
        s_b = choose_start_index(len(idx_b), hidden_len, 0.20, rng)
        g_b = idx_b[s_b : s_b + hidden_len]
        apply_event_on_raw(df, g_b, "non_detectable_short_single_signal", rng)
        events.append(
            InjectedEvent(
                event_id="inj_hidden_1",
                cow=str(cow_hidden),
                start=pd.Timestamp(df.loc[g_b[0], TIME]),
                end=pd.Timestamp(df.loc[g_b[-1], TIME]),
                duration_bins=int(len(g_b)),
                profile="non_detectable_short_single_signal",
                expected_detected=0,
                design_reason="Short + weak + mostly one-family signal, expected under persistence/coherence thresholds.",
            )
        )

    out_events = pd.DataFrame([asdict(e) for e in events])
    return df, out_events


def inject_manual_plan_raw(
    df_base: pd.DataFrame,
    *,
    plan: List[Dict[str, object]],
    interval: str,
    persist_hours: int,
    seed: int,
    baseline_pred: Optional[pd.DataFrame] = None,
    cooldown_hours: Optional[int] = None,
    baseline_ratio: Optional[float] = None,
    manual_injection_variant: str = "legacy",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Injecte des événements dans les données brutes selon un plan de placement explicite."""
    rng = np.random.default_rng(seed)
    df = df_base.sort_values([COW, TIME]).copy().reset_index(drop=True)
    _to_numeric_cols(df, [STEPS, MI, LYING, STANDING, TRANSITIONS, TR_UP, TR_DOWN])
    manual_injection_variant = str(manual_injection_variant).strip().lower()

    minutes = interval_to_minutes(interval)
    k = max(1, int(round((float(persist_hours) * 60) / max(1, minutes))))
    # Garder des fenêtres détectables assez longues pour la persistance, sans
    # les rendre trop larges, afin de préserver une IoU interprétable.
    detectable_len = max(int(round(1.5 * k)), 36)
    hidden_len = max(6, int(round(0.35 * k)))

    events: List[InjectedEvent] = []
    by_cow: Dict[str, np.ndarray] = {}
    occupied: Dict[str, np.ndarray] = {}
    detect_cows = {str(p["cow"]) for p in plan if int(p.get("expected_detected", 0)) == 1}
    cooldown_bins = None
    if cooldown_hours is not None:
        cooldown_bins = max(0, int(round((float(cooldown_hours) * 60) / max(1, minutes))))
    baseline_pred_sorted = None
    if baseline_pred is not None and manual_injection_variant == "v2_1":
        baseline_pred_sorted = baseline_pred.copy()
        if TIME in baseline_pred_sorted.columns:
            baseline_pred_sorted[TIME] = pd.to_datetime(baseline_pred_sorted[TIME], errors="coerce")
        if COW in baseline_pred_sorted.columns:
            baseline_pred_sorted[COW] = baseline_pred_sorted[COW].astype(str)

    for p in plan:
        cow = str(p["cow"])
        prof = str(p["profile"])
        exp = int(p["expected_detected"])
        ratio = float(p.get("preferred_ratio", 0.65))
        if cow not in by_cow:
            idx = np.flatnonzero(df[COW].astype(str).values == cow)
            by_cow[cow] = idx
            occupied[cow] = np.zeros(len(idx), dtype=bool)
            # v2.1 manual placement: avoid baseline notification cooldown windows
            # on the detectable cow so each injected detectable event can still
            # generate its own notification under stricter defaults.
            if (
                manual_injection_variant == "v2_1"
                and cow in detect_cows
                and baseline_pred_sorted is not None
                and "notif_lameness" in baseline_pred_sorted.columns
                and cooldown_bins is not None
                and cooldown_bins > 0
                and len(idx) > 0
            ):
                base_c = baseline_pred_sorted[baseline_pred_sorted[COW] == cow].copy()
                if len(base_c):
                    notif = pd.to_numeric(
                        base_c.get("notif_lameness", pd.Series(dtype=float)),
                        errors="coerce",
                    ).fillna(0).astype(int)
                    notif_times = pd.to_datetime(
                        base_c.loc[notif == 1, TIME],
                        errors="coerce",
                    ).dropna().tolist()
                    if notif_times:
                        t_cow = pd.to_datetime(df.loc[idx, TIME], errors="coerce")
                        for nt in notif_times:
                            # Block an interval around existing baseline notifications.
                            dt = (t_cow - pd.Timestamp(nt)).abs()
                            occupied[cow] |= (
                                dt <= pd.Timedelta(minutes=minutes * cooldown_bins)
                            ).to_numpy()
            if (
                manual_injection_variant == "v2_1"
                and cow in detect_cows
                and baseline_ratio is not None
                and len(idx) > 0
            ):
                # v2.1 manual placement: avoid injecting inside the IF fit segment
                # (baseline_ratio) to reduce global model drift that can remove
                # pre-existing notifications on the target cow.
                try:
                    br = float(baseline_ratio)
                except Exception:
                    br = np.nan
                if np.isfinite(br) and 0.0 < br < 1.0:
                    fit_n = int(np.floor(len(idx) * br))
                    if fit_n > 0:
                        occupied[cow][: min(fit_n, len(idx))] = True
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
        placed = _pick_window_for_cow(
            n=len(idx_cow),
            duration=dur,
            preferred_ratio=ratio,
            occupied=occupied[cow],
            margin=8,
        )
        if placed is None:
            continue
        s, e = placed
        g = idx_cow[s : e + 1]
        apply_event_on_raw(df, g, prof, rng)
        reason = (
            "Long coherent multi-signal pattern, close to naturally detected episodes."
            if exp == 1
            else "Short + weak + mostly one-family signal, expected under persistence/coherence thresholds."
        )
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

