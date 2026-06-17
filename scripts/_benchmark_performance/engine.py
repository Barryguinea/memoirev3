"""Moteur de benchmark de performance du pipeline final (timing, agrégation)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.alerts import apply_alert_logic
from core.config import (
    DEFAULT_ALERT_MIN,
    DEFAULT_BASELINE_RATIO,
    DEFAULT_CONTAMINATION,
    DEFAULT_COOLDOWN_HOURS,
    DEFAULT_COVERAGE_MIN_PCT,
    DEFAULT_INTERVAL,
    DEFAULT_MIX_MODE,
    DEFAULT_MIX_RATE_THR,
    DEFAULT_MI_Z_HIGH_THR,
    DEFAULT_PERSIST_HOURS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SENSOR_WARMUP_BINS,
    DEFAULT_WINDOW_BASELINE,
    DEFAULT_Z_HIGH_THR,
    DEFAULT_Z_LOW_THR,
)
from core.features import build_interval_features
from core.hybrid_warning import apply_hybrid_warning
from core.io import COW, TIME, available_base_cols
from core.model_if import run_if_core
from core.pipeline import summarize_one_cow


@dataclass
class PipelineParams:
    """Conteneur avec valeurs par défaut pour tous les paramètres du pipeline."""

    interval: str = DEFAULT_INTERVAL
    window_baseline: int = DEFAULT_WINDOW_BASELINE
    contamination: float = DEFAULT_CONTAMINATION
    baseline_ratio: Optional[float] = DEFAULT_BASELINE_RATIO
    random_state: int = DEFAULT_RANDOM_STATE
    persist_hours: int = DEFAULT_PERSIST_HOURS
    alert_min: int = DEFAULT_ALERT_MIN
    mix_mode: str = DEFAULT_MIX_MODE
    mix_rate_thr: float = DEFAULT_MIX_RATE_THR
    z_low_thr: float = DEFAULT_Z_LOW_THR
    z_high_thr: float = DEFAULT_Z_HIGH_THR
    cooldown_hours: int = DEFAULT_COOLDOWN_HOURS
    mi_z_high_thr: float = DEFAULT_MI_Z_HIGH_THR
    coverage_min_pct: float = DEFAULT_COVERAGE_MIN_PCT


def _now_stamp() -> str:
    """Retourne la date/heure courante sous forme d'horodatage compact."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_fractions(s: str) -> List[float]:
    """Analyse une chaîne de fractions (0, 1] séparées par des virgules et retourne des valeurs uniques triées."""
    vals = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if not (0 < v <= 1):
            raise ValueError(f"fraction invalide: {v} (attendu dans ]0,1])")
        vals.append(v)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("Aucune fraction valide fournie.")
    return vals


def _subset_by_cows(df: pd.DataFrame, n_cows: int) -> pd.DataFrame:
    """Retourne un sous-ensemble du DataFrame avec uniquement les `n_cows` premières vaches (triées)."""
    cows = sorted(df[COW].astype(str).unique().tolist())
    chosen = set(cows[: int(n_cows)])
    return df[df[COW].astype(str).isin(chosen)].copy()


def _run_one_cow_timed(df_all: pd.DataFrame, cow_id: str, p: PipelineParams) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Exécute le pipeline complet pour une vache et retourne les temps par étape."""
    times: Dict[str, float] = {}
    t0 = perf_counter()
    df_cow = df_all[df_all[COW] == str(cow_id)].copy()
    base_cols = available_base_cols(df_cow)
    times["filter_seconds"] = perf_counter() - t0

    t1 = perf_counter()
    it = build_interval_features(
        df_cow=df_cow,
        time_col=TIME,
        interval=p.interval,
        cols=base_cols,
        window_baseline=int(p.window_baseline),
        mi_name="Motion Index",
    )
    times["features_seconds"] = perf_counter() - t1

    t2 = perf_counter()
    it = run_if_core(
        it,
        time_col="T",
        contamination=float(p.contamination),
        random_state=int(p.random_state),
        baseline_ratio=p.baseline_ratio,
        coverage_min_pct=float(p.coverage_min_pct),
        sensor_warmup_bins=DEFAULT_SENSOR_WARMUP_BINS,
    )
    times["if_seconds"] = perf_counter() - t2

    t3 = perf_counter()
    it = apply_alert_logic(
        it,
        time_col="T",
        interval=p.interval,
        persist_hours=int(p.persist_hours),
        alert_min=int(p.alert_min),
        mix_mode=p.mix_mode,
        mix_rate_thr=float(p.mix_rate_thr),
        z_low_thr=float(p.z_low_thr),
        z_high_thr=float(p.z_high_thr),
        cooldown_hours=int(p.cooldown_hours),
        mi_z_high_thr=float(p.mi_z_high_thr),
        coverage_min_pct=float(p.coverage_min_pct),
    )
    times["alerts_seconds"] = perf_counter() - t3

    t4 = perf_counter()
    it = apply_hybrid_warning(it, interval=p.interval)
    times["hybrid_seconds"] = perf_counter() - t4

    it[COW] = str(cow_id)
    times["cow_total_seconds"] = sum(times.values())
    times["n_rows_cow_raw"] = float(len(df_cow))
    times["n_rows_cow_interval"] = float(len(it))
    return it, times


def run_benchmark_herd(
    df_all: pd.DataFrame,
    p: PipelineParams,
    *,
    max_cows: Optional[int] = None,
    collect_output: bool = False,
) -> Dict[str, float]:
    """Mesure les performances du pipeline sur tout le troupeau et retourne des métriques de temps agrégées."""
    cows = sorted(df_all[COW].astype(str).unique().tolist())
    if max_cows is not None:
        cows = cows[: int(max_cows)]

    timers = {
        "filter_seconds": 0.0,
        "features_seconds": 0.0,
        "if_seconds": 0.0,
        "alerts_seconds": 0.0,
        "hybrid_seconds": 0.0,
        "concat_seconds": 0.0,
        "summary_seconds": 0.0,
    }
    n_rows_interval_total = 0
    summaries = []
    outs: List[pd.DataFrame] = []

    t_total = perf_counter()
    for cow_id in cows:
        it, t = _run_one_cow_timed(df_all, cow_id, p)
        for k in [
            "filter_seconds",
            "features_seconds",
            "if_seconds",
            "alerts_seconds",
            "hybrid_seconds",
        ]:
            timers[k] += float(t[k])
        n_rows_interval_total += int(t["n_rows_cow_interval"])
        s = summarize_one_cow(it)
        s[COW] = str(cow_id)
        summaries.append(s)
        if collect_output:
            outs.append(it)

    t_summary = perf_counter()
    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["lameness_notifs", "lameness_starts", "if_anomaly_points"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    timers["summary_seconds"] = perf_counter() - t_summary

    t_concat = perf_counter()
    if collect_output and outs:
        _ = pd.concat(outs, ignore_index=True)
    timers["concat_seconds"] = perf_counter() - t_concat

    total_seconds = perf_counter() - t_total
    n_rows_raw = int(len(df_all[df_all[COW].astype(str).isin(cows)]))
    n_cows = int(len(cows))

    out: Dict[str, float] = {
        "n_cows": float(n_cows),
        "n_rows_raw": float(n_rows_raw),
        "n_rows_interval_total": float(n_rows_interval_total),
        "total_seconds": float(total_seconds),
        "rows_per_sec": float(n_rows_raw / total_seconds) if total_seconds > 0 else np.nan,
        "interval_rows_per_sec": float(n_rows_interval_total / total_seconds) if total_seconds > 0 else np.nan,
        "cows_per_sec": float(n_cows / total_seconds) if total_seconds > 0 else np.nan,
        "collect_output": 1.0 if collect_output else 0.0,
    }
    out.update({k: float(v) for k, v in timers.items()})
    compute_core = sum(
        timers[key]
        for key in [
            "filter_seconds",
            "features_seconds",
            "if_seconds",
            "alerts_seconds",
            "hybrid_seconds",
        ]
    )
    out["instrumented_core_seconds"] = float(compute_core)
    for stage in [
        "features_seconds",
        "if_seconds",
        "alerts_seconds",
        "hybrid_seconds",
    ]:
        out[f"{stage.replace('_seconds', '')}_share_of_core"] = float(timers[stage] / compute_core) if compute_core > 0 else np.nan
    return out


def _aggregate_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Agrège des répétitions de benchmark en statistiques mean/median/std/min/max."""
    if runs_df.empty:
        return pd.DataFrame()

    group_cols = ["fraction", "n_cows", "n_rows_raw"]
    metric_cols = [
        "total_seconds",
        "rows_per_sec",
        "interval_rows_per_sec",
        "cows_per_sec",
        "filter_seconds",
        "features_seconds",
        "if_seconds",
        "alerts_seconds",
        "hybrid_seconds",
        "summary_seconds",
        "concat_seconds",
        "instrumented_core_seconds",
        "features_share_of_core",
        "if_share_of_core",
        "alerts_share_of_core",
        "hybrid_share_of_core",
    ]

    rows = []
    for keys, g in runs_df.groupby(group_cols, dropna=False):
        item = dict(zip(group_cols, keys))
        item["repeats"] = int(len(g))
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            item[f"{col}_mean"] = float(vals.mean())
            item[f"{col}_median"] = float(vals.median())
            item[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
            item[f"{col}_min"] = float(vals.min())
            item[f"{col}_max"] = float(vals.max())
        item["seconds_per_1000_rows_median"] = float(item["total_seconds_median"] / max(item["n_rows_raw"], 1) * 1000.0)
        rows.append(item)

    summary = pd.DataFrame(rows).sort_values(["fraction", "n_cows"]).reset_index(drop=True)
    full_frac = float(summary["fraction"].max())
    full = summary[summary["fraction"] == full_frac]
    if not full.empty:
        ref = full.iloc[0]
        summary["rows_ratio_vs_full"] = summary["n_rows_raw"] / float(ref["n_rows_raw"])
        summary["time_ratio_vs_full"] = summary["total_seconds_median"] / float(ref["total_seconds_median"])
    return summary


__all__ = [
    "PipelineParams",
    "run_benchmark_herd",
    "_now_stamp",
    "_parse_fractions",
    "_subset_by_cows",
    "_aggregate_runs",
]
