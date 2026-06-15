"""Helpers d'évaluation de validation (règles, appariement, diagnostics)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.alerts import apply_alert_logic
from core.features import interval_to_minutes
from core.io import COW, TIME
from scripts.validation_common import iou, overlap_seconds, segments_from_binary


def rerun_alert_logic_processed(df: pd.DataFrame, params: Dict[str, object]) -> pd.DataFrame:
    """Réapplique les règles métier sur des données traitées modifiées (par vache)."""
    outs = []
    for cow, gx in df.groupby(COW, sort=False):
        g = gx.copy().sort_values(TIME)
        out = apply_alert_logic(
            g,
            time_col=TIME,
            interval=str(params["interval"]),
            persist_hours=int(params["persist_hours"]),
            alert_min=int(params["alert_min"]),
            mix_mode=str(params["mix_mode"]),
            mix_rate_thr=float(params["mix_rate_thr"]),
            z_low_thr=float(params["z_low_thr"]),
            z_high_thr=float(params["z_high_thr"]),
            cooldown_hours=int(params["cooldown_hours"]),
            mi_z_high_thr=float(params["mi_z_high_thr"]),
            coverage_min_pct=float(params["coverage_min_pct"]),
        )
        out[COW] = str(cow)
        outs.append(out)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


def event_windows_mask(pred_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.Series:
    """Masque booléen des lignes de prédiction incluses dans des fenêtres injectées."""
    m = pd.Series(False, index=pred_df.index)
    if events_df is None or len(events_df) == 0:
        return m
    for r in events_df.itertuples(index=False):
        cow_mask = pred_df[COW].astype(str) == str(r.cow)
        time_mask = (pred_df[TIME] >= pd.Timestamp(r.start)) & (pred_df[TIME] <= pd.Timestamp(r.end))
        m = m | (cow_mask & time_mask)
    return m


def evaluate_injected_events(
    pred_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    interval: str,
    persist_hours: int,
    alert_min: int,
    mix_rate_thr: float,
) -> pd.DataFrame:
    """Évalue la détection de chaque événement injecté (IoU + diagnostics)."""
    if events_df is None or len(events_df) == 0:
        return pd.DataFrame()

    pred = pred_df.copy()
    pred[TIME] = pd.to_datetime(pred[TIME], errors="coerce")
    pred = pred.dropna(subset=[TIME]).copy()
    pred[COW] = pred[COW].astype(str)

    minutes = interval_to_minutes(interval)
    k = max(1, int(round((float(persist_hours) * 60) / max(1, minutes))))
    bin_seconds = int(minutes * 60)

    rows: List[Dict[str, object]] = []
    for r in events_df.itertuples(index=False):
        cow = str(r.cow)
        s = pd.Timestamp(r.start)
        e = pd.Timestamp(r.end)
        gx = pred[pred[COW].astype(str) == cow].copy().sort_values(TIME)
        segs = segments_from_binary(gx, "pred_lameness_episode")
        best_iou = max([iou((s, e), seg, bin_seconds) for seg in segs], default=0.0)
        matched_any = int(any(overlap_seconds((s, e), seg, bin_seconds) > 0 for seg in segs))
        matched_iou20 = int(best_iou >= 0.20)

        win = gx[(gx[TIME] >= s) & (gx[TIME] <= e)].copy()
        if_anom_k_max = (
            float(pd.to_numeric(win.get("if_anom_k", pd.Series(dtype=float)), errors="coerce").max())
            if len(win)
            else np.nan
        )
        anom_rate_k_max = (
            float(pd.to_numeric(win.get("anom_rate_k", pd.Series(dtype=float)), errors="coerce").max())
            if len(win)
            else np.nan
        )
        coherence_mean = (
            float(pd.to_numeric(win.get("coherence_boiterie", pd.Series(dtype=float)), errors="coerce").mean())
            if len(win)
            else np.nan
        )
        cooldown_ratio = (
            float(pd.to_numeric(win.get("in_cooldown", pd.Series(dtype=float)), errors="coerce").fillna(0).mean())
            if len(win)
            else np.nan
        )

        why = []
        if matched_any:
            why.append("Overlap with at least one predicted behavioral-warning episode.")
        else:
            if int(r.duration_bins) < int(k):
                why.append(f"Duration too short for persistence window: {int(r.duration_bins)} < K={int(k)}.")
            if np.isfinite(anom_rate_k_max) and anom_rate_k_max < float(mix_rate_thr):
                why.append(f"anom_rate_k peak {anom_rate_k_max:.3f} below threshold {float(mix_rate_thr):.3f}.")
            if np.isfinite(if_anom_k_max) and if_anom_k_max < float(alert_min):
                why.append(f"if_anom_k peak {if_anom_k_max:.1f} below alert_min {int(alert_min)}.")
            if np.isfinite(coherence_mean) and coherence_mean < 0.5:
                why.append("Low coherence_boiterie on injected window.")
            if np.isfinite(cooldown_ratio) and cooldown_ratio >= 0.8:
                why.append("Most of the window is under cooldown.")
            if not why:
                why.append("Signal remained below decision rules or was too fragmented.")

        rows.append(
            {
                "event_id": r.event_id,
                "cow": cow,
                "start": s,
                "end": e,
                "duration_bins": int(r.duration_bins),
                "profile": r.profile,
                "expected_detected": int(r.expected_detected),
                "detected_any_overlap": int(matched_any),
                "detected_iou20": int(matched_iou20),
                "best_iou": float(best_iou),
                "if_anom_k_max": if_anom_k_max,
                "anom_rate_k_max": anom_rate_k_max,
                "coherence_boiterie_mean": coherence_mean,
                "cooldown_ratio": cooldown_ratio,
                "why": " ".join(why),
            }
        )
    return pd.DataFrame(rows)
