"""Alerte comportementale precoce fondee sur des changements persistants.

Ce module ne produit pas un diagnostic de boiterie. Il compare chaque vache a
sa propre baseline et signale une degradation comportementale multivariee qui
doit ensuite etre verifiee sur le terrain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.features import interval_to_minutes


@dataclass(frozen=True)
class EarlyWarningConfig:
    """Parametres fixes du detecteur temporel non supervise."""

    aggregation_hours: float = 12.0
    persistence_hours: float = 6.0
    cooldown_hours: float = 24.0
    family_min_change: float = 0.10
    posture_min_change: float = 0.05
    score_threshold: float = 0.12
    cusum_drift: float = 0.070
    cusum_threshold: float = 1.20
    min_families: int = 3
    coverage_min_pct: float = 25.0
    active_families: tuple[str, ...] = ("steps", "motion", "transitions", "posture")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").astype(float)
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    floor = max(1e-6, float(denom[denom > 0].median()) * 0.05) if (denom > 0).any() else 1.0
    return (num / denom.clip(lower=floor)).replace([np.inf, -np.inf], np.nan).clip(0.0, 3.0)


def _baseline_expected_by_slot(
    values: pd.Series,
    slots: pd.Series,
    baseline_mask: pd.Series,
) -> pd.Series:
    """Medianes de baseline par heure de la journee, avec repli global."""
    base = pd.DataFrame({"value": values, "slot": slots}).loc[baseline_mask & values.notna()]
    global_median = float(base["value"].median()) if len(base) else float(values.median())
    if not np.isfinite(global_median):
        global_median = 0.0
    by_slot = base.groupby("slot")["value"].median() if len(base) else pd.Series(dtype=float)
    expected = slots.map(by_slot).astype(float).fillna(global_median)
    return expected


def _rolling_total(df: pd.DataFrame, column: str, window_bins: int) -> pd.Series:
    if column not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    values = pd.to_numeric(df[column], errors="coerce").astype(float)
    return values.rolling(window_bins, min_periods=max(4, window_bins // 2)).sum()


def _one_sided_cusum(evidence: pd.Series, drift: float) -> pd.Series:
    values = pd.to_numeric(evidence, errors="coerce").fillna(0.0).to_numpy(float)
    out = np.zeros(len(values), dtype=float)
    state = 0.0
    for i, value in enumerate(values):
        state = max(0.0, state + float(value) - float(drift))
        out[i] = state
    return pd.Series(out, index=evidence.index)


def apply_behavioral_early_warning(
    interval_df: pd.DataFrame,
    *,
    interval: str,
    config: EarlyWarningConfig | None = None,
) -> pd.DataFrame:
    """Ajoute les sorties d'alerte comportementale a un tableau d'intervalles.

    La baseline est celle deja materialisee par ``run_if_core`` dans
    ``dataset_split``. Les ratios sont calcules sur des totaux glissants et
    compares aux medianes individuelles du meme creneau horaire.
    """
    cfg = config or EarlyWarningConfig()
    df = interval_df.sort_values("T").copy().reset_index(drop=True)
    minutes = interval_to_minutes(interval)
    window_bins = max(4, int(round(cfg.aggregation_hours * 60 / minutes)))
    persist_bins = max(2, int(round(cfg.persistence_hours * 60 / minutes)))
    cooldown_bins = max(1, int(round(cfg.cooldown_hours * 60 / minutes)))

    if "dataset_split" in df:
        baseline_mask = df["dataset_split"].astype(str).eq("baseline")
        future_mask = df["dataset_split"].astype(str).eq("futur")
    else:
        split = max(30, int(round(0.60 * len(df))))
        baseline_mask = pd.Series(df.index < split, index=df.index)
        future_mask = ~baseline_mask

    times = pd.to_datetime(df["T"], errors="coerce")
    slots = times.dt.hour * 60 + times.dt.minute
    valid_cov = (
        pd.to_numeric(df.get("coverage_pct", 100.0), errors="coerce").fillna(0.0)
        >= float(cfg.coverage_min_pct)
    )

    signal_columns = {
        "steps": "Steps_sum",
        "motion": "Motion Index_sum",
        "transitions": "Transitions_sum",
        "lying": "Lying Time_sum",
        "standing": "Standing Time_sum",
    }
    ratios: dict[str, pd.Series] = {}
    day_bins = max(4, int(round(24 * 60 / minutes)))
    for family, column in signal_columns.items():
        total = _rolling_total(df, column, window_bins)
        expected = _baseline_expected_by_slot(total, slots, baseline_mask)
        baseline_ratio = _safe_ratio(total, expected)
        local_ratio = _safe_ratio(total, total.shift(day_bins))
        if family == "lying":
            ratios[family] = pd.concat([baseline_ratio, local_ratio], axis=1).max(axis=1)
        else:
            ratios[family] = pd.concat([baseline_ratio, local_ratio], axis=1).min(axis=1)
        df[f"warning_ratio_{family}"] = ratios[family]

    deficits = {
        "steps": (1.0 - ratios["steps"]).clip(0.0, 1.0),
        "motion": (1.0 - ratios["motion"]).clip(0.0, 1.0),
        "transitions": (1.0 - ratios["transitions"]).clip(0.0, 1.0),
    }
    lying_excess = (ratios["lying"] - 1.0).clip(0.0, 1.0)
    standing_deficit = (1.0 - ratios["standing"]).clip(0.0, 1.0)
    posture = pd.concat([lying_excess, standing_deficit], axis=1).mean(axis=1, skipna=True)

    for family, values in deficits.items():
        df[f"warning_change_{family}"] = values
    df["warning_change_posture"] = posture

    # Familles actives : par defaut les quatre, ce qui laisse la production
    # inchangee ; une ablation peut restreindre le detecteur a un seul canal
    # (par exemple les pas seuls, a la maniere d'un podometre).
    base_weights = {"steps": 0.30, "motion": 0.30, "transitions": 0.20, "posture": 0.20}
    family_terms = {
        "steps": deficits["steps"].fillna(0.0),
        "motion": deficits["motion"].fillna(0.0),
        "transitions": deficits["transitions"].fillna(0.0),
        "posture": posture.fillna(0.0),
    }
    active = [name for name in base_weights if name in cfg.active_families]
    if not active:
        active = list(base_weights)
    total_weight = sum(base_weights[name] for name in active)
    # Renormalisation a 1.0 : le sous-ensemble actif garde la meme echelle de
    # score (et donc le meme seuil) que le detecteur complet.
    score = sum((base_weights[name] / total_weight) * family_terms[name] for name in active)

    activity_families = [name for name in ("steps", "motion", "transitions") if name in active]
    if not activity_families:
        activity_families = ["steps", "motion", "transitions"]
    activity_ratio = pd.concat(
        [ratios[name] for name in activity_families], axis=1
    ).mean(axis=1, skipna=True)
    lag_bins = max(2, int(round(6 * 60 / minutes)))
    progressive_drop = (activity_ratio.shift(lag_bins) - activity_ratio).clip(0.0, 1.0).fillna(0.0)
    hit_thresholds = {
        "steps": cfg.family_min_change,
        "motion": cfg.family_min_change,
        "transitions": cfg.family_min_change,
        "posture": cfg.posture_min_change,
    }
    hit_values = {
        "steps": deficits["steps"],
        "motion": deficits["motion"],
        "transitions": deficits["transitions"],
        "posture": posture,
    }
    family_hits = pd.DataFrame(
        {name: hit_values[name] >= hit_thresholds[name] for name in active}
    )
    n_families = family_hits.sum(axis=1).astype(int)
    core_names = [name for name in ("steps", "motion") if name in active]
    if core_names:
        core_activity = family_hits[core_names].any(axis=1)
    else:
        core_activity = pd.Series(True, index=df.index)
    point_candidate = (
        (score >= cfg.score_threshold)
        & (n_families >= cfg.min_families)
        & core_activity
        & valid_cov
        & future_mask
    )
    evidence = (score + 0.35 * progressive_drop).where(point_candidate, 0.0)
    cusum = _one_sided_cusum(evidence, cfg.cusum_drift)
    persistent = (
        point_candidate.astype(float).rolling(persist_bins, min_periods=persist_bins).mean()
        >= 0.45
    )
    episode = (persistent & (cusum >= cfg.cusum_threshold)).astype(int)
    start = ((episode == 1) & (episode.shift(1, fill_value=0) == 0)).astype(int)

    notification = np.zeros(len(df), dtype=int)
    last_fire = -10**9
    for i, is_start in enumerate(start.to_numpy(int)):
        if is_start and i - last_fire > cooldown_bins:
            notification[i] = 1
            last_fire = i

    df["behavioral_warning_score"] = score.clip(0.0, 1.0)
    df["behavioral_warning_cusum"] = cusum
    df["behavioral_warning_families"] = n_families
    df["behavioral_warning_candidate"] = point_candidate.astype(int)
    df["behavioral_warning_episode"] = episode
    df["behavioral_warning_start"] = start
    df["behavioral_warning_notification"] = notification
    df["behavioral_warning_scope"] = "alerte comportementale a verifier, non diagnostique"
    return df


__all__ = ["EarlyWarningConfig", "apply_behavioral_early_warning"]
