"""Helpers reproduisant exactement la sélection d'entraînement de production."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import config as C


def production_split_indices(
    interval_df: pd.DataFrame,
    *,
    baseline_ratio: float = C.DEFAULT_BASELINE_RATIO,
    coverage_min_pct: float = C.DEFAULT_COVERAGE_MIN_PCT,
    sensor_warmup_bins: int = C.DEFAULT_SENSOR_WARMUP_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne ``(train, future, candidats)`` comme ``core.model_if.run_if_core``."""
    df = interval_df.reset_index(drop=True)
    if "coverage_pct" in df.columns:
        valid_cov = (
            pd.to_numeric(df["coverage_pct"], errors="coerce").fillna(0.0)
            >= float(coverage_min_pct)
        )
    else:
        valid_cov = pd.Series(np.ones(len(df), dtype=bool), index=df.index)

    warmup_k = max(1, int(sensor_warmup_bins))
    active_mask = valid_cov.copy()
    if warmup_k > 1:
        streak = valid_cov.astype(int).rolling(warmup_k, min_periods=warmup_k).sum()
        active_candidates = streak[streak >= warmup_k]
        if len(active_candidates) > 0:
            first_active_idx = int(active_candidates.index[0])
            active_mask = pd.Series(df.index >= first_active_idx, index=df.index)

    candidate_idx = np.flatnonzero((valid_cov & active_mask).to_numpy())
    if candidate_idx.size == 0:
        candidate_idx = np.flatnonzero(valid_cov.to_numpy())
    if candidate_idx.size == 0:
        candidate_idx = np.arange(len(df))

    br = max(0.05, min(0.95, float(baseline_ratio)))
    n_train = min(max(30, int(round(len(candidate_idx) * br))), len(candidate_idx))
    train_idx = candidate_idx[:n_train]
    future_idx = candidate_idx[n_train:]
    return train_idx, future_idx, candidate_idx


def add_production_split_columns(
    interval_df: pd.DataFrame,
    *,
    baseline_ratio: float = C.DEFAULT_BASELINE_RATIO,
    coverage_min_pct: float = C.DEFAULT_COVERAGE_MIN_PCT,
    sensor_warmup_bins: int = C.DEFAULT_SENSOR_WARMUP_BINS,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Ajoute les colonnes de split de production et retourne les indices train."""
    df = interval_df.reset_index(drop=True).copy()
    train_idx, future_idx, candidate_idx = production_split_indices(
        df,
        baseline_ratio=baseline_ratio,
        coverage_min_pct=coverage_min_pct,
        sensor_warmup_bins=sensor_warmup_bins,
    )
    df["dataset_split"] = "excluded"
    df.loc[future_idx, "dataset_split"] = "futur"
    df.loc[train_idx, "dataset_split"] = "baseline"
    df["if_train_candidate"] = 0
    df.loc[candidate_idx, "if_train_candidate"] = 1
    df["if_train_point"] = 0
    df.loc[train_idx, "if_train_point"] = 1
    return df, train_idx
