"""Planification de sélection des vaches et de placement des injections manuelles."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.io import COW


def pick_cows_for_injection(summary_df: pd.DataFrame) -> Tuple[str, str]:
    """Choisit (detectable, hidden) en priorisant séries longues, injectables, peu alertées."""
    s = summary_df.copy()
    s[COW] = s[COW].astype(str)
    if "injectability_score" not in s.columns:
        s["injectability_score"] = 0.5
    if "injectability_ok" not in s.columns:
        s["injectability_ok"] = 1
    if "series_length_score" not in s.columns:
        s["series_length_score"] = 0.5
    if "long_series_ok" not in s.columns:
        s["long_series_ok"] = 1
    s["injectability_score"] = pd.to_numeric(s["injectability_score"], errors="coerce").fillna(0.0)
    s["injectability_ok"] = pd.to_numeric(s["injectability_ok"], errors="coerce").fillna(0).astype(int)
    s["series_length_score"] = pd.to_numeric(s["series_length_score"], errors="coerce").fillna(0.0)
    s["long_series_ok"] = pd.to_numeric(s["long_series_ok"], errors="coerce").fillna(0).astype(int)

    s = s.sort_values(
        [
            "lameness_starts",
            "lameness_notifs",
            "long_series_ok",
            "injectability_ok",
            "series_length_score",
            "injectability_score",
            "if_anomaly_points",
        ],
        ascending=[True, True, False, False, False, False, False],
    ).reset_index(drop=True)

    if len(s) == 0:
        raise ValueError("No cows available for injection.")

    long_good = s[(s["long_series_ok"] == 1) & (s["injectability_ok"] == 1)].copy()
    long_only = s[s["long_series_ok"] == 1].copy()
    zeros = s[
        pd.to_numeric(s.get("lameness_starts", pd.Series(dtype=float)), errors="coerce")
        .fillna(0)
        .astype(int)
        == 0
    ].copy()
    zeros_good = zeros[zeros["injectability_ok"] == 1].copy()
    detect_priority = [long_good, long_only, zeros_good, zeros, s]
    hidden_priority = [long_good, long_only, zeros_good, zeros, s]

    for pool in detect_priority:
        if len(pool) == 0:
            continue
        cow_detect = str(pool.iloc[0][COW])
        for hpool in hidden_priority:
            if len(hpool) == 0:
                continue
            others = hpool[hpool[COW].astype(str) != cow_detect]
            if len(others):
                return cow_detect, str(others.iloc[0][COW])
        return cow_detect, cow_detect

    cow_detect = str(s.iloc[0][COW])
    cow_hidden = str(s.iloc[1][COW]) if len(s) > 1 else cow_detect
    return cow_detect, cow_hidden


def _pick_window_for_cow(
    *,
    n: int,
    duration: int,
    preferred_ratio: float,
    occupied: np.ndarray,
    margin: int = 8,
) -> Optional[Tuple[int, int]]:
    if duration <= 0 or n <= duration:
        return None
    max_start = n - duration
    preferred = int(np.clip(round(n * preferred_ratio), 0, max_start))
    starts = sorted(range(max_start + 1), key=lambda s: abs(s - preferred))
    for s in starts:
        e = s + duration - 1
        l = max(0, s - margin)
        r = min(n - 1, e + margin)
        if occupied[l : r + 1].any():
            continue
        occupied[l : r + 1] = True
        return s, e
    return None


def _resolve_target_cow(summary_df: pd.DataFrame, requested: str) -> str:
    s = summary_df.copy()
    s[COW] = s[COW].astype(str)
    if str(requested) in set(s[COW].tolist()):
        return str(requested)
    if "injectability_score" not in s.columns:
        s["injectability_score"] = 0.5
    if "injectability_ok" not in s.columns:
        s["injectability_ok"] = 1
    if "series_length_score" not in s.columns:
        s["series_length_score"] = 0.5
    if "long_series_ok" not in s.columns:
        s["long_series_ok"] = 1
    x = s.sort_values(
        [
            "lameness_starts",
            "lameness_notifs",
            "long_series_ok",
            "injectability_ok",
            "series_length_score",
            "injectability_score",
            "if_anomaly_points",
        ],
        ascending=[True, True, False, False, False, False, False],
    ).reset_index(drop=True)
    return str(x.iloc[0][COW])


def _manual_injection_plan(
    summary_df: pd.DataFrame,
    *,
    detect_cow: str,
    hidden_cow: str,
    n_detectable: int,
    variant: str = "legacy",
) -> List[Dict[str, object]]:
    variant_norm = str(variant).strip().lower()
    if variant_norm not in {"legacy", "v2_1"}:
        raise ValueError("manual injection variant must be one of: 'legacy', 'v2_1'.")
    cow_d = _resolve_target_cow(summary_df, str(detect_cow))
    cow_h = _resolve_target_cow(summary_df, str(hidden_cow))
    if cow_h == cow_d:
        s = summary_df.copy()
        s[COW] = s[COW].astype(str)
        alt = [c for c in s[COW].tolist() if c != cow_d]
        if alt:
            cow_h = alt[0]

    ratios = [0.55, 0.66, 0.78, 0.84]
    plan: List[Dict[str, object]] = []
    for i in range(max(1, int(n_detectable))):
        ratio = float(ratios[i % len(ratios)])
        if variant_norm == "v2_1" and i == 0:
            ratio = 0.60
        row = {
            "event_id": f"inj_detectable_{i+1}",
            "cow": cow_d,
            "profile": "detectable_processed_visible_pattern",
            "expected_detected": 1,
            "preferred_ratio": ratio,
        }
        if variant_norm == "v2_1" and i == 0:
            row["duration_scale"] = 1.10
        if variant_norm == "v2_1" and i == 1:
            row["duration_scale"] = 1.20
        plan.append(row)
    plan.append(
        {
            "event_id": "inj_hidden_1",
            "cow": cow_h,
            "profile": "non_detectable_processed_short_single_signal",
            "expected_detected": 0,
            "preferred_ratio": 0.20,
        }
    )
    return plan

