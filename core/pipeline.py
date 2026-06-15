"""Pipeline V3: features, comparateur IF, deux branches et fusion.

Le comparateur historique et les sorties bidirectionnelles sont calculés sur
les mêmes intervalles afin de préserver la traçabilité expérimentale.
"""

import pandas as pd
from typing import Dict, Any, Optional, Tuple

from core.io import COW, TIME, available_base_cols
from core.config import DEFAULT_SENSOR_WARMUP_BINS
from core.features import build_interval_features
from core.model_if import run_if_core
from core.alerts import apply_alert_logic
from core.early_warning import EarlyWarningConfig
from core.hybrid_warning import (
    HybridFusionConfig,
    InstabilityWarningConfig,
    apply_hybrid_warning,
)


def run_pipeline_one_cow(
    df_all: pd.DataFrame,
    cow_id: str,
    *,
    interval: str,
    window_baseline: int,
    contamination: float,
    baseline_ratio: Optional[float],
    random_state: int,
    # paramètres des règles métier
    persist_hours: int,
    alert_min: int,
    mix_mode: str,
    mix_rate_thr: float,
    z_low_thr: float,
    z_high_thr: float,
    cooldown_hours: int,
    mi_z_high_thr: float,
    coverage_min_pct: float,
    warning_config: Optional[EarlyWarningConfig] = None,
    instability_config: Optional[InstabilityWarningConfig] = None,
    fusion_config: Optional[HybridFusionConfig] = None,
) -> pd.DataFrame:
    """
    Pipeline complet pour une vache individuelle.
    
    Étapes:
    1. Filtrer données vache
    2. Construction des features (par intervalle)
    3. Isolation Forest (détection d'anomalies)
    4. Règles métier (épisodes, boiterie, notifications)
    
    Retourne:
        DataFrame avec toutes les colonnes calculées
    """
    df_cow = df_all[df_all[COW] == str(cow_id)].copy()
    base_cols = available_base_cols(df_cow)

    # 1. Construction des features
    it = build_interval_features(
        df_cow=df_cow,
        time_col=TIME,
        interval=interval,
        cols=base_cols,
        window_baseline=int(window_baseline),
        mi_name="Motion Index",
    )

    # 2. Isolation Forest (un seul appel)
    it = run_if_core(
        it,
        time_col="T",
        contamination=float(contamination),
        random_state=int(random_state),
        baseline_ratio=baseline_ratio,  # None => fit sur tout
        coverage_min_pct=float(coverage_min_pct),
        sensor_warmup_bins=DEFAULT_SENSOR_WARMUP_BINS,
    )

    # 3. Règles métier (un seul appel)
    it = apply_alert_logic(
        it,
        time_col="T",
        interval=interval,
        persist_hours=int(persist_hours),
        alert_min=int(alert_min),
        mix_mode=mix_mode,
        mix_rate_thr=float(mix_rate_thr),
        z_low_thr=float(z_low_thr),
        z_high_thr=float(z_high_thr),
        cooldown_hours=int(cooldown_hours),
        mi_z_high_thr=float(mi_z_high_thr),
        coverage_min_pct=float(coverage_min_pct),
    )

    # Sortie primaire de MemoireV3: deux branches indépendantes, puis fusion.
    # Les sorties IF historiques restent présentes comme comparateur seulement.
    it = apply_hybrid_warning(
        it,
        interval=interval,
        hypo_config=warning_config,
        instability_config=instability_config,
        fusion_config=fusion_config,
    )

    it[COW] = str(cow_id)
    return it


def summarize_one_cow(it: pd.DataFrame) -> Dict[str, Any]:
    """
    Résumé statistique pour une vache.

    Retourne:
        Dict avec métriques clés (nb anomalies, boiterie, notifs, etc)
    """
    return {
        "n_bins": int(len(it)),
        "if_anomaly_points": int(it["if_anomaly_point"].sum()) if "if_anomaly_point" in it else 0,
        "problem_points": int(it["pred_problem_episode"].sum()) if "pred_problem_episode" in it else 0,
        "lameness_points": int(it["pred_lameness_episode"].sum()) if "pred_lameness_episode" in it else 0,
        "problem_starts": int(it["pred_problem_start"].sum()) if "pred_problem_start" in it else 0,
        "lameness_starts": int(it["pred_lameness_start"].sum()) if "pred_lameness_start" in it else 0,
        "lameness_notifs": int(it["notif_lameness"].sum()) if "notif_lameness" in it else 0,
        "critique_points": int(it["is_critique"].sum()) if "is_critique" in it else 0,
        "behavioral_warning_points": int(it["behavioral_warning_episode"].sum())
        if "behavioral_warning_episode" in it
        else 0,
        "behavioral_warning_notifs": int(it["behavioral_warning_notification"].sum())
        if "behavioral_warning_notification" in it
        else 0,
        "instability_warning_notifs": int(it["instability_warning_notification"].sum())
        if "instability_warning_notification" in it
        else 0,
        "hybrid_warning_notifs": int(it["hybrid_warning_notification"].sum())
        if "hybrid_warning_notification" in it
        else 0,
        "coverage_mean": float(it["coverage_pct"].mean()) if "coverage_pct" in it else float("nan"),
        "coverage_min": float(it["coverage_pct"].min()) if "coverage_pct" in it else float("nan"),
    }


def run_pipeline_herd(
    df_all: pd.DataFrame,
    *,
    interval: str,
    window_baseline: int,
    contamination: float,
    baseline_ratio: Optional[float],
    random_state: int,
    persist_hours: int,
    alert_min: int,
    mix_mode: str,
    mix_rate_thr: float,
    z_low_thr: float,
    z_high_thr: float,
    cooldown_hours: int,
    mi_z_high_thr: float,
    coverage_min_pct: float,
    max_cows: Optional[int] = None,
    warning_config: Optional[EarlyWarningConfig] = None,
    instability_config: Optional[InstabilityWarningConfig] = None,
    fusion_config: Optional[HybridFusionConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline pour tout le troupeau.

    Args:
        df_all: DataFrame brut complet
        max_cows: Limite optionnelle nb vaches (pour tests)
        ... autres params de pipeline

    Retourne:
        (summary_df, out_df):
            - summary_df: 1 ligne par vache avec métriques
            - out_df: tous les intervalles de toutes les vaches
    """
    cows = sorted(df_all[COW].astype(str).unique().tolist())
    if max_cows is not None:
        cows = cows[: int(max_cows)]

    summaries = []
    outs = []

    for cow_id in cows:
        it = run_pipeline_one_cow(
            df_all,
            cow_id,
            interval=interval,
            window_baseline=window_baseline,
            contamination=contamination,
            baseline_ratio=baseline_ratio,
            random_state=random_state,
            persist_hours=persist_hours,
            alert_min=alert_min,
            mix_mode=mix_mode,
            mix_rate_thr=mix_rate_thr,
            z_low_thr=z_low_thr,
            z_high_thr=z_high_thr,
            cooldown_hours=cooldown_hours,
            mi_z_high_thr=mi_z_high_thr,
            coverage_min_pct=coverage_min_pct,
            warning_config=warning_config,
            instability_config=instability_config,
            fusion_config=fusion_config,
        )

        s = summarize_one_cow(it)
        s[COW] = str(cow_id)
        summaries.append(s)
        outs.append(it)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["hybrid_warning_notifs", "behavioral_warning_notifs", "instability_warning_notifs"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    out_df = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
    return summary_df, out_df
