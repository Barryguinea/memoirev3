"""Résumé d'épisodes et enrichissement de sélection pour validations."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.io import COW, LYING, MI, STANDING, STEPS, TIME, TRANSITIONS
from scripts.validation_common import segments_from_binary


def summarize_from_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les comptes de détection par vache à partir des prédictions pipeline."""
    rows = []
    for cow, gx in pred_df.groupby(COW, sort=False):
        rows.append(
            {
                COW: str(cow),
                "n_bins": int(len(gx)),
                "if_anomaly_points": int(pd.to_numeric(gx.get("if_anomaly_point", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
                "problem_points": int(pd.to_numeric(gx.get("pred_problem_episode", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
                "lameness_points": int(pd.to_numeric(gx.get("pred_lameness_episode", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
                "problem_starts": int(pd.to_numeric(gx.get("pred_problem_start", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
                "lameness_starts": int(pd.to_numeric(gx.get("pred_lameness_start", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
                "lameness_notifs": int(pd.to_numeric(gx.get("notif_lameness", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
                "critique_points": int(pd.to_numeric(gx.get("is_critique", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
                "coverage_mean": float(pd.to_numeric(gx.get("coverage_pct", pd.Series(dtype=float)), errors="coerce").mean()),
                "coverage_min": float(pd.to_numeric(gx.get("coverage_pct", pd.Series(dtype=float)), errors="coerce").min()),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["lameness_notifs", "lameness_starts", "if_anomaly_points"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return out


def extract_detected_episodes(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Liste les épisodes de boiterie contigus avec quelques statistiques de fenêtre."""
    rows: List[Dict[str, object]] = []
    for cow, gx in pred_df.groupby(COW, sort=False):
        segs = segments_from_binary(gx, "pred_lameness_episode")
        for i, (s, e) in enumerate(segs, start=1):
            win = gx[(gx[TIME] >= s) & (gx[TIME] <= e)].copy()
            rows.append(
                {
                    "cow": str(cow),
                    "episode_idx": int(i),
                    "start": s,
                    "end": e,
                    "duration_bins": int(len(win)),
                    "notif_in_episode": int(win["notif_lameness"].sum()) if "notif_lameness" in win.columns else 0,
                    "mean_if_anom_k": float(pd.to_numeric(win.get("if_anom_k", pd.Series(dtype=float)), errors="coerce").mean()) if len(win) else np.nan,
                    "mean_anom_rate_k": float(pd.to_numeric(win.get("anom_rate_k", pd.Series(dtype=float)), errors="coerce").mean()) if len(win) else np.nan,
                    "mean_coherence_boiterie": float(pd.to_numeric(win.get("coherence_boiterie", pd.Series(dtype=float)), errors="coerce").mean()) if len(win) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _augment_summary_with_injectability(summary_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features de sélection automatique (injectabilité + longueur de série) par vache."""
    s = summary_df.copy()
    s[COW] = s[COW].astype(str)
    if raw_df is None or len(raw_df) == 0:
        s["injectability_score"] = 0.5
        s["injectability_ok"] = 1
        s["series_bins"] = 0
        s["series_length_score"] = 0.5
        s["long_series_ok"] = 1
        return s

    r = raw_df.copy()
    r[COW] = r[COW].astype(str)
    metrics = pd.DataFrame({COW: sorted(r[COW].unique().tolist())})
    bins = r.groupby(COW, as_index=False).size().rename(columns={"size": "series_bins"})
    metrics = metrics.merge(bins, on=COW, how="left")

    if TIME in r.columns:
        tt = pd.to_datetime(r[TIME], errors="coerce")
        tdf = pd.DataFrame({COW: r[COW], "_t": tt}).dropna(subset=["_t"])
        if len(tdf):
            span = tdf.groupby(COW, as_index=False)["_t"].agg(["min", "max"]).reset_index()
            span["series_span_days"] = (
                (span["max"] - span["min"]).dt.total_seconds() / 86400.0
            ).clip(lower=0.0)
            span = span[[COW, "series_span_days"]]
            metrics = metrics.merge(span, on=COW, how="left")
        else:
            metrics["series_span_days"] = 0.0
    else:
        metrics["series_span_days"] = 0.0

    cols = [STEPS, MI, LYING, STANDING, TRANSITIONS]
    for col in cols:
        if col in r.columns:
            v = pd.to_numeric(r[col], errors="coerce")
            g = pd.DataFrame({COW: r[COW], "_nonzero": (v.fillna(0) > 0).astype(int), "_std_src": v})
            nz = g.groupby(COW, as_index=False)["_nonzero"].mean().rename(columns={"_nonzero": f"{col}_nonzero"})
            st = g.groupby(COW, as_index=False)["_std_src"].std().rename(columns={"_std_src": f"{col}_std"})
            st[f"{col}_std"] = pd.to_numeric(st[f"{col}_std"], errors="coerce").fillna(0.0)
            metrics = metrics.merge(nz, on=COW, how="left").merge(st, on=COW, how="left")
        else:
            metrics[f"{col}_nonzero"] = 0.0
            metrics[f"{col}_std"] = 0.0

    core = [STEPS, MI, LYING, STANDING]
    core_nz_cols = [f"{c}_nonzero" for c in core]
    core_std_cols = [f"{c}_std" for c in core]

    for c in core_nz_cols + core_std_cols + [f"{TRANSITIONS}_nonzero", f"{TRANSITIONS}_std", "series_bins", "series_span_days"]:
        if c in metrics.columns:
            metrics[c] = pd.to_numeric(metrics[c], errors="coerce").fillna(0.0)

    metrics["core_nonzero_mean"] = metrics[core_nz_cols].mean(axis=1)
    metrics["core_std_positive_mean"] = (metrics[core_std_cols] > 0).astype(int).mean(axis=1)
    metrics["transitions_nonzero"] = metrics.get(
        f"{TRANSITIONS}_nonzero",
        pd.Series(0.0, index=metrics.index),
    ).astype(float)
    metrics["injectability_score"] = (
        0.60 * metrics["core_nonzero_mean"]
        + 0.25 * metrics["core_std_positive_mean"]
        + 0.15 * metrics["transitions_nonzero"].clip(0, 1)
    ).clip(0, 1)
    metrics["injectability_ok"] = (
        (metrics.get(f"{STEPS}_nonzero", 0.0) >= 0.20)
        & (metrics.get(f"{MI}_nonzero", 0.0) >= 0.20)
        & (metrics.get(f"{LYING}_nonzero", 0.0) >= 0.20)
        & (metrics.get(f"{STANDING}_nonzero", 0.0) >= 0.20)
    ).astype(int)

    max_bins = float(max(1.0, metrics["series_bins"].max()))
    max_span = float(max(1.0, metrics["series_span_days"].max()))
    metrics["series_length_score"] = (
        0.80 * (metrics["series_bins"] / max_bins).clip(0, 1)
        + 0.20 * (metrics["series_span_days"] / max_span).clip(0, 1)
    ).clip(0, 1)
    bins_thr = int(max(96, np.quantile(metrics["series_bins"], 0.60)))
    metrics["long_series_ok"] = (metrics["series_bins"] >= float(bins_thr)).astype(int)

    keep = metrics[
        [COW, "injectability_score", "injectability_ok", "series_bins", "series_length_score", "long_series_ok"]
    ]
    out = s.merge(keep, on=COW, how="left")
    out["injectability_score"] = pd.to_numeric(out["injectability_score"], errors="coerce").fillna(0.0)
    out["injectability_ok"] = pd.to_numeric(out["injectability_ok"], errors="coerce").fillna(0).astype(int)
    out["series_bins"] = pd.to_numeric(out["series_bins"], errors="coerce").fillna(0).astype(int)
    out["series_length_score"] = pd.to_numeric(out["series_length_score"], errors="coerce").fillna(0.0)
    out["long_series_ok"] = pd.to_numeric(out["long_series_ok"], errors="coerce").fillna(0).astype(int)
    return out

