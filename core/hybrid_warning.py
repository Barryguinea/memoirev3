"""Détecteur comportemental bidirectionnel exploratoire de MemoireV3.

La branche HYPO réutilise le détecteur temporel de MemoireV2. La branche
INSTABILITE ne cherche pas une hausse générale d'activité: elle recherche un
mouvement disproportionné aux pas, une densité de transitions élevée et une
fragmentation posturale. Cette distinction réduit la confusion avec l'exercice
ou l'œstrus, sans prétendre identifier la cause clinique d'une alerte.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.early_warning import (
    EarlyWarningConfig,
    _baseline_expected_by_slot,
    _one_sided_cusum,
    _rolling_total,
    _safe_ratio,
    apply_behavioral_early_warning,
)
from core.features import interval_to_minutes


@dataclass(frozen=True)
class InstabilityWarningConfig:
    """Paramètres techniques de la branche d'instabilité comportementale."""

    aggregation_hours: float = 3.0
    persistence_hours: float = 2.0
    cooldown_hours: float = 12.0
    restless_motion_min_change: float = 0.20
    transition_density_min_change: float = 0.20
    fragmentation_min_change: float = 0.20
    posture_volatility_min_change: float = 0.20
    score_threshold: float = 0.18
    cusum_drift: float = 0.06
    cusum_threshold: float = 0.70
    min_families: int = 2
    coordinated_activity_min_change: float = 0.15
    coordinated_activity_tolerance: float = 0.20
    coverage_min_pct: float = 25.0


@dataclass(frozen=True)
class HybridFusionConfig:
    """Règles de fusion des deux branches.

    ``HIERARCHICAL`` conserve une instabilité isolée comme surveillance et
    n'émet une notification fusionnée que lors d'une hypoactivité, d'une
    convergence simultanée ou d'une séquence instabilité puis hypoactivité.
    """

    mode: str = "HIERARCHICAL"
    cooldown_hours: float = 12.0
    sequence_min_hours: float = 0.0
    sequence_max_hours: float = 72.0


FUSION_MODES = {
    "HYPO_ONLY",
    "INSTABILITY_ONLY",
    "OR",
    "HIERARCHICAL",
    "SEQUENTIAL",
}


def _positive_ratio_against_references(
    values: pd.Series,
    *,
    slots: pd.Series,
    baseline_mask: pd.Series,
    day_bins: int,
) -> pd.Series:
    expected = _baseline_expected_by_slot(values, slots, baseline_mask)
    baseline_ratio = _safe_ratio(values, expected)
    local_ratio = _safe_ratio(values, values.shift(day_bins))
    return pd.concat([baseline_ratio, local_ratio], axis=1).min(axis=1)


def _rolling_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    window_bins: int,
) -> pd.Series:
    minimum = max(4, window_bins // 2)
    num = pd.to_numeric(numerator, errors="coerce").rolling(
        window_bins, min_periods=minimum
    ).sum()
    den = pd.to_numeric(denominator, errors="coerce").rolling(
        window_bins, min_periods=minimum
    ).sum()
    floor = max(1e-6, float(den[den > 0].median()) * 0.05) if (den > 0).any() else 1.0
    return (num / den.clip(lower=floor)).replace([np.inf, -np.inf], np.nan)


def _posture_volatility(df: pd.DataFrame, window_bins: int) -> pd.Series:
    missing = pd.Series(np.nan, index=df.index, dtype=float)
    lying = pd.to_numeric(df.get("Lying Time_sum", missing), errors="coerce")
    standing = pd.to_numeric(df.get("Standing Time_sum", missing), errors="coerce")
    total = lying + standing
    fraction = (lying / total.where(total > 0)).clip(0.0, 1.0)
    minimum = max(4, window_bins // 2)
    return fraction.diff().abs().rolling(window_bins, min_periods=minimum).mean()


def _recent_prior_event(
    starts: pd.Series,
    *,
    min_bins: int,
    max_bins: int,
) -> pd.Series:
    values = pd.to_numeric(starts, errors="coerce").fillna(0).astype(int).to_numpy()
    out = np.zeros(len(values), dtype=bool)
    previous: list[int] = []
    for i, value in enumerate(values):
        previous = [index for index in previous if i - index <= max_bins]
        out[i] = any(min_bins <= i - index <= max_bins for index in previous)
        if value:
            previous.append(i)
    return pd.Series(out, index=starts.index)


def _cooldown_notifications(starts: pd.Series, cooldown_bins: int) -> np.ndarray:
    notification = np.zeros(len(starts), dtype=int)
    last_fire = -10**9
    for i, is_start in enumerate(starts.to_numpy(int)):
        if is_start and i - last_fire > cooldown_bins:
            notification[i] = 1
            last_fire = i
    return notification


def apply_instability_warning(
    interval_df: pd.DataFrame,
    *,
    interval: str,
    config: InstabilityWarningConfig | None = None,
) -> pd.DataFrame:
    """Ajoute une surveillance d'instabilité, distincte de l'hyperactivité.

    Une hausse coordonnée des pas et du Motion Index est filtrée comme activité
    locomotrice générale. Un candidat exige au contraire du mouvement ou des
    transitions disproportionnés aux pas, avec corroboration posturale et
    persistance après la baseline individuelle.
    """

    cfg = config or InstabilityWarningConfig()
    df = interval_df.sort_values("T").copy().reset_index(drop=True)
    minutes = interval_to_minutes(interval)
    window_bins = max(4, int(round(cfg.aggregation_hours * 60 / minutes)))
    persist_bins = max(2, int(round(cfg.persistence_hours * 60 / minutes)))
    cooldown_bins = max(1, int(round(cfg.cooldown_hours * 60 / minutes)))
    day_bins = max(4, int(round(24 * 60 / minutes)))

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

    missing = pd.Series(np.nan, index=df.index, dtype=float)
    steps = pd.to_numeric(df.get("Steps_sum", missing), errors="coerce")
    motion = pd.to_numeric(df.get("Motion Index_sum", missing), errors="coerce")
    transitions = pd.to_numeric(df.get("Transitions_sum", missing), errors="coerce")
    posture_total = pd.to_numeric(df.get("Lying Time_sum", missing), errors="coerce") + pd.to_numeric(
        df.get("Standing Time_sum", missing), errors="coerce"
    )

    raw_ratios: dict[str, pd.Series] = {}
    for name, column in {
        "steps": "Steps_sum",
        "motion": "Motion Index_sum",
        "transitions": "Transitions_sum",
    }.items():
        total = _rolling_total(df, column, window_bins)
        raw_ratios[name] = _positive_ratio_against_references(
            total, slots=slots, baseline_mask=baseline_mask, day_bins=day_bins
        )
        df[f"instability_ratio_{name}"] = raw_ratios[name]

    derived = {
        "restless_motion": _rolling_ratio(motion, steps, window_bins=window_bins),
        "transition_density": _rolling_ratio(transitions, steps, window_bins=window_bins),
        "fragmentation": _rolling_ratio(transitions, posture_total, window_bins=window_bins),
        "posture_volatility": _posture_volatility(df, window_bins),
    }
    derived_ratios: dict[str, pd.Series] = {}
    for name, values in derived.items():
        ratio = _positive_ratio_against_references(
            values, slots=slots, baseline_mask=baseline_mask, day_bins=day_bins
        )
        derived_ratios[name] = ratio
        df[f"instability_ratio_{name}"] = ratio

    excess = {name: (ratio - 1.0).clip(0.0, 1.5) for name, ratio in derived_ratios.items()}
    thresholds = {
        "restless_motion": cfg.restless_motion_min_change,
        "transition_density": cfg.transition_density_min_change,
        "fragmentation": cfg.fragmentation_min_change,
        "posture_volatility": cfg.posture_volatility_min_change,
    }
    family_hits = pd.DataFrame(
        {name: excess[name] >= threshold for name, threshold in thresholds.items()}
    )
    weights = {
        "restless_motion": 0.35,
        "transition_density": 0.30,
        "fragmentation": 0.20,
        "posture_volatility": 0.15,
    }
    score = sum(weights[name] * excess[name].fillna(0.0) for name in weights)
    n_families = family_hits.sum(axis=1).astype(int)

    step_excess = (raw_ratios["steps"] - 1.0).clip(lower=0.0)
    motion_excess = (raw_ratios["motion"] - 1.0).clip(lower=0.0)
    proportional_gap = (motion_excess - step_excess).abs()
    coordinated_activity = (
        (step_excess >= cfg.coordinated_activity_min_change)
        & (motion_excess >= cfg.coordinated_activity_min_change)
        & (proportional_gap <= cfg.coordinated_activity_tolerance)
        & ~family_hits["transition_density"]
        & ~family_hits["fragmentation"]
    )
    core_evidence = family_hits["restless_motion"] & (
        family_hits["transition_density"]
        | family_hits["fragmentation"]
        | family_hits["posture_volatility"]
    )
    point_candidate = (
        (score >= cfg.score_threshold)
        & (n_families >= cfg.min_families)
        & core_evidence
        & ~coordinated_activity
        & valid_cov
        & future_mask
    )
    evidence = score.where(point_candidate, 0.0)
    cusum = _one_sided_cusum(evidence, cfg.cusum_drift)
    persistent = (
        point_candidate.astype(float).rolling(persist_bins, min_periods=persist_bins).mean()
        >= 0.50
    )
    episode = (persistent & (cusum >= cfg.cusum_threshold)).astype(int)
    start = ((episode == 1) & (episode.shift(1, fill_value=0) == 0)).astype(int)

    for family, values in excess.items():
        df[f"instability_change_{family}"] = values
    df["instability_warning_score"] = score.clip(0.0, 1.0)
    df["instability_warning_cusum"] = cusum
    df["instability_warning_families"] = n_families
    df["instability_coordinated_activity"] = coordinated_activity.astype(int)
    df["instability_warning_candidate"] = point_candidate.astype(int)
    df["instability_warning_episode"] = episode
    df["instability_warning_start"] = start
    df["instability_warning_notification"] = _cooldown_notifications(start, cooldown_bins)
    df["instability_warning_scope"] = "instabilite comportementale a verifier, non diagnostique"
    return df


def apply_hybrid_warning(
    interval_df: pd.DataFrame,
    *,
    interval: str,
    hypo_config: EarlyWarningConfig | None = None,
    instability_config: InstabilityWarningConfig | None = None,
    fusion_config: HybridFusionConfig | None = None,
) -> pd.DataFrame:
    """Exécute les deux branches et applique une fusion explicitement choisie."""

    fusion = fusion_config or HybridFusionConfig()
    mode = fusion.mode.upper()
    if mode not in FUSION_MODES:
        raise ValueError(f"Mode de fusion inconnu: {fusion.mode}. Attendus: {sorted(FUSION_MODES)}")

    df = interval_df.copy()
    if "behavioral_warning_episode" not in df:
        df = apply_behavioral_early_warning(df, interval=interval, config=hypo_config)
    df = apply_instability_warning(df, interval=interval, config=instability_config)

    hypo = pd.to_numeric(df["behavioral_warning_episode"], errors="coerce").fillna(0).astype(int)
    instability = pd.to_numeric(df["instability_warning_episode"], errors="coerce").fillna(0).astype(int)
    if "behavioral_warning_start" in df:
        hypo_start = pd.to_numeric(
            df["behavioral_warning_start"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        hypo_start = ((hypo == 1) & (hypo.shift(1, fill_value=0) == 0)).astype(int)
    instability_start = pd.to_numeric(df["instability_warning_start"], errors="coerce").fillna(0).astype(int)

    minutes = interval_to_minutes(interval)
    min_bins = max(0, int(round(fusion.sequence_min_hours * 60 / minutes)))
    max_bins = max(min_bins + 1, int(round(fusion.sequence_max_hours * 60 / minutes)))
    recent_instability = _recent_prior_event(
        instability_start, min_bins=min_bins, max_bins=max_bins
    )
    sequence_start = (hypo_start.eq(1) & recent_instability).astype(int)
    mixed = ((hypo == 1) & (instability == 1)).astype(int)
    mixed_start = ((mixed == 1) & (mixed.shift(1, fill_value=0) == 0)).astype(int)

    if mode == "HYPO_ONLY":
        episode = hypo
        start = hypo_start
    elif mode == "INSTABILITY_ONLY":
        episode = instability
        start = instability_start
    elif mode == "OR":
        episode = ((hypo == 1) | (instability == 1)).astype(int)
        start = ((hypo_start == 1) | (instability_start == 1)).astype(int)
    elif mode == "SEQUENTIAL":
        episode = sequence_start.astype(int)
        start = sequence_start
    else:  # HIERARCHICAL
        episode = hypo
        start = ((hypo_start == 1) | (mixed_start == 1) | (sequence_start == 1)).astype(int)

    surveillance = (instability.eq(1) & hypo.eq(0)).astype(int)
    episode_type = np.select(
        [sequence_start.eq(1), mixed.eq(1), hypo.eq(1), instability.eq(1)],
        ["SEQUENCE", "MIXTE", "HYPO", "INSTABILITE"],
        default="AUCUN",
    )
    priority = np.select(
        [sequence_start.eq(1) | mixed.eq(1), hypo.eq(1), instability.eq(1)],
        [3, 2, 1],
        default=0,
    ).astype(int)
    cooldown_bins = max(1, int(round(fusion.cooldown_hours * 60 / minutes)))

    df["hybrid_warning_score"] = pd.concat(
        [df["behavioral_warning_score"], df["instability_warning_score"]], axis=1
    ).max(axis=1)
    df["hybrid_warning_episode"] = episode.astype(int)
    df["hybrid_warning_surveillance"] = surveillance
    df["hybrid_warning_sequence_start"] = sequence_start
    df["hybrid_warning_start"] = start.astype(int)
    df["hybrid_warning_notification"] = _cooldown_notifications(start, cooldown_bins)
    df["hybrid_warning_type"] = episode_type
    df["hybrid_warning_priority"] = priority
    df["hybrid_warning_fusion_mode"] = mode
    df["hybrid_warning_scope"] = "alerte comportementale a verifier, non diagnostique"
    return df


__all__ = [
    "FUSION_MODES",
    "HybridFusionConfig",
    "InstabilityWarningConfig",
    "apply_hybrid_warning",
    "apply_instability_warning",
]
