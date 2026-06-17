"""Ablation A/B/C/D post-baseline avec comparaisons regroupées par vache."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from core import config as C
from core.alerts import apply_alert_logic
from core.early_warning import EarlyWarningConfig, apply_behavioral_early_warning
from core.features import build_interval_features
from core.io import COW, TIME, available_base_cols, load_csv
from core.model_if import QUANTILE_RANGE, _default_feature_cols, run_if_core
from validation_hypo.campaign import (
    _evaluate_binary_output,
    _heldout_start_time,
    _monitoring_duration_days,
    final_params,
    has_informative_heldout_signals,
    inject_events_for_cow,
)
from validation_hypo.training import add_production_split_columns

_VARIANT_NAMES = (
    "A. Alerte temporelle multivariée",
    "B. IF + règles de persistance",
    "C. IF seul",
    "D. LOF + règles",
    "E. Comparateur pédométrique (pas seuls)",
)


def _alert_kwargs(params: Dict[str, object]) -> Dict[str, object]:
    return {
        "time_col": TIME,
        "interval": str(params["interval"]),
        "persist_hours": int(params["persist_hours"]),
        "alert_min": int(params["alert_min"]),
        "mix_mode": str(params["mix_mode"]),
        "mix_rate_thr": float(params["mix_rate_thr"]),
        "z_low_thr": float(params["z_low_thr"]),
        "z_high_thr": float(params["z_high_thr"]),
        "cooldown_hours": int(params["cooldown_hours"]),
        "mi_z_high_thr": float(params["mi_z_high_thr"]),
        "coverage_min_pct": float(params["coverage_min_pct"]),
    }


def _run_if(features: pd.DataFrame, params: Dict[str, object]) -> pd.DataFrame:
    return run_if_core(
        features.copy(),
        time_col=TIME,
        contamination=float(params["contamination"]),
        random_state=int(params["random_state"]),
        baseline_ratio=float(params["baseline_ratio"]),
        coverage_min_pct=float(params["coverage_min_pct"]),
        sensor_warmup_bins=C.DEFAULT_SENSOR_WARMUP_BINS,
    )


def _run_variant_a(
    features: pd.DataFrame,
    cow: str,
    params: Dict[str, object],
    warning_config: Optional[EarlyWarningConfig],
) -> pd.DataFrame:
    out = apply_behavioral_early_warning(
        _run_if(features, params),
        interval=str(params["interval"]),
        config=warning_config,
    )
    out["pred_lameness_episode"] = out["behavioral_warning_episode"]
    out["pred_lameness_start"] = out["behavioral_warning_start"]
    out["notif_lameness"] = out["behavioral_warning_notification"]
    out[COW] = str(cow)
    return out


def _pedometric_config(warning_config: Optional[EarlyWarningConfig]) -> EarlyWarningConfig:
    """Restreint le detecteur temporel au seul canal des pas (esprit podometre).

    Comparateur de type Alsaaod et al. (2012) : meme baseline individuelle, meme
    machinerie CUSUM/persistance/cooldown, mais une seule famille de signaux. La
    coherence multi-familles (impossible a un seul canal) est relachee
    (``min_families = 1``) pour ne pas desavantager artificiellement la baseline.
    """
    base = warning_config or EarlyWarningConfig()
    return replace(base, active_families=("steps",), min_families=1)


def _run_variant_e(
    features: pd.DataFrame,
    cow: str,
    params: Dict[str, object],
    warning_config: Optional[EarlyWarningConfig],
) -> pd.DataFrame:
    """Comparateur pedometrique : detecteur temporel restreint aux pas seuls."""
    out = apply_behavioral_early_warning(
        _run_if(features, params),
        interval=str(params["interval"]),
        config=_pedometric_config(warning_config),
    )
    out["pred_lameness_episode"] = out["behavioral_warning_episode"]
    out["pred_lameness_start"] = out["behavioral_warning_start"]
    out["notif_lameness"] = out["behavioral_warning_notification"]
    out[COW] = str(cow)
    return out


def _run_variant_b(features: pd.DataFrame, cow: str, params: Dict[str, object]) -> pd.DataFrame:
    """Comparateur IF suivi de règles de persistance."""
    out = apply_alert_logic(_run_if(features, params), **_alert_kwargs(params))
    out[COW] = str(cow)
    return out


def _run_variant_c(features: pd.DataFrame, cow: str, params: Dict[str, object]) -> pd.DataFrame:
    out = _run_if(features, params)
    out["pred_lameness_episode"] = out["if_anomaly_point"]
    out["pred_lameness_start"] = (
        (out["pred_lameness_episode"] == 1)
        & (out["pred_lameness_episode"].shift(1, fill_value=0) == 0)
    ).astype(int)
    out["notif_lameness"] = out["pred_lameness_start"]
    out[COW] = str(cow)
    return out


def _run_variant_d(features: pd.DataFrame, cow: str, params: Dict[str, object]) -> pd.DataFrame:
    """LOF novelty entraîné sur les mêmes points de baseline que l'IF."""
    out, train_idx = add_production_split_columns(
        features.sort_values(TIME),
        baseline_ratio=float(params["baseline_ratio"]),
        coverage_min_pct=float(params["coverage_min_pct"]),
        sensor_warmup_bins=C.DEFAULT_SENSOR_WARMUP_BINS,
    )
    feature_cols = _default_feature_cols(out)
    x = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = RobustScaler(quantile_range=QUANTILE_RANGE)
    x_train = scaler.fit_transform(x.iloc[train_idx])
    x_all = scaler.transform(x)
    n_neighbors = min(20, max(2, len(train_idx) - 1))
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=float(params["contamination"]),
        novelty=True,
    )
    lof.fit(x_train)
    pred = lof.predict(x_all)
    score = lof.decision_function(x_all)
    out["if_pred"] = pred
    out["if_score"] = score
    out["if_anomaly_point"] = (pred == -1).astype(int)

    if "coverage_pct" in out.columns:
        invalid = (
            pd.to_numeric(out["coverage_pct"], errors="coerce").fillna(0.0)
            < float(params["coverage_min_pct"])
        )
        out.loc[invalid, "if_pred"] = 1
        out.loc[invalid, "if_score"] = np.nan
        out.loc[invalid, "if_anomaly_point"] = 0

    out = apply_alert_logic(out, **_alert_kwargs(params))
    out[COW] = str(cow)
    return out


def _run_variants(
    features: pd.DataFrame,
    cow: str,
    params: Dict[str, object],
    warning_config: Optional[EarlyWarningConfig],
) -> Dict[str, pd.DataFrame]:
    return {
        _VARIANT_NAMES[0]: _run_variant_a(features, cow, params, warning_config),
        _VARIANT_NAMES[1]: _run_variant_b(features, cow, params),
        _VARIANT_NAMES[2]: _run_variant_c(features, cow, params),
        _VARIANT_NAMES[3]: _run_variant_d(features, cow, params),
        _VARIANT_NAMES[4]: _run_variant_e(features, cow, params, warning_config),
    }


def run_clean_ablation(
    raw_csv: str = "data/brut.csv",
    *,
    cows: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (11,),
    scenarios: Sequence[str] = (
        "gradual_mild",
        "gradual_moderate",
        "gradual_marked",
        "isolated_short_variation",
    ),
    params: Optional[Dict[str, object]] = None,
    warning_config: Optional[EarlyWarningConfig] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Évalue les quatre variantes sur les mêmes injections post-baseline."""
    params = params or final_params()
    df_all = load_csv(raw_csv)
    df_all[COW] = df_all[COW].astype(str)
    selected = sorted(df_all[COW].unique()) if cows is None else [str(c) for c in cows]
    rows: List[pd.DataFrame] = []
    done = 0

    eligible = []
    for cow in selected:
        raw_cow = df_all[df_all[COW] == cow]
        span_days = (raw_cow[TIME].max() - raw_cow[TIME].min()).total_seconds() / 86400.0
        if span_days < 14:
            continue
        heldout_start = _heldout_start_time(
            raw_cow,
            interval=str(params["interval"]),
            window_baseline=int(params["window_baseline"]),
            baseline_ratio=float(params["baseline_ratio"]),
            coverage_min_pct=float(params["coverage_min_pct"]),
        )
        if not has_informative_heldout_signals(
            raw_cow,
            heldout_start=heldout_start,
        ):
            continue
        eligible.append(cow)

    total = len(eligible) * len(seeds) * len(scenarios)
    for cow_index, cow in enumerate(eligible):
        raw_cow = df_all[df_all[COW] == cow]
        heldout_start = _heldout_start_time(
            raw_cow,
            interval=str(params["interval"]),
            window_baseline=int(params["window_baseline"]),
            baseline_ratio=float(params["baseline_ratio"]),
            coverage_min_pct=float(params["coverage_min_pct"]),
        )
        clean_features = build_interval_features(
            raw_cow,
            time_col=TIME,
            interval=str(params["interval"]),
            cols=available_base_cols(raw_cow),
            window_baseline=int(params["window_baseline"]),
        )
        clean_variants = _run_variants(
            clean_features,
            cow,
            params,
            warning_config,
        )
        monitoring_days = {
            name: _monitoring_duration_days(
                prediction,
                heldout_start=heldout_start,
                interval=str(params["interval"]),
            )
            for name, prediction in clean_variants.items()
        }
        background_rates = {
            name: (
                float("nan")
                if days <= 0 or "notif_lameness" not in prediction
                else float(
                    pd.to_numeric(
                        prediction.loc[
                            pd.to_datetime(prediction[TIME], errors="coerce")
                            >= pd.Timestamp(heldout_start),
                            "notif_lameness",
                        ],
                        errors="coerce",
                    ).fillna(0).sum()
                    / days
                )
            )
            for name, prediction in clean_variants.items()
            for days in [monitoring_days[name]]
        }
        for seed in seeds:
            for scenario in scenarios:
                done += 1
                injected, events = inject_events_for_cow(
                    raw_cow,
                    cow=cow,
                    scenario=scenario,
                    seed=int(seed),
                    interval=str(params["interval"]),
                    persist_hours=int(params["persist_hours"]),
                    baseline_ratio=float(params["baseline_ratio"]),
                    window_baseline=int(params["window_baseline"]),
                    coverage_min_pct=float(params["coverage_min_pct"]),
                    heldout_start=heldout_start,
                    schedule_rotation=cow_index % 4,
                )
                if events.empty:
                    continue
                features = build_interval_features(
                    injected,
                    time_col=TIME,
                    interval=str(params["interval"]),
                    cols=available_base_cols(injected),
                    window_baseline=int(params["window_baseline"]),
                )
                for name, pred in _run_variants(
                    features,
                    cow,
                    params,
                    warning_config,
                ).items():
                    event = events.iloc[0]
                    metrics = _evaluate_binary_output(
                        pred,
                        event,
                        episode_col="pred_lameness_episode",
                        start_col="pred_lameness_start",
                        score_col=(
                            "behavioral_warning_score"
                            if name in (_VARIANT_NAMES[0], _VARIANT_NAMES[4])
                            else "if_anom_k"
                        ),
                        interval=str(params["interval"]),
                        reference_predictions=clean_variants[name],
                    )
                    result = event.to_dict()
                    result.update(metrics)
                    result["variante"] = name
                    result["scenario"] = scenario
                    result["seed"] = int(seed)
                    result["false_notif_cow_day"] = background_rates[name]
                    result["monitoring_days"] = monitoring_days[name]
                    rows.append(pd.DataFrame([result]))
                if verbose and done % 25 == 0:
                    print(f"  ... {done}/{total}")

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if verbose and not out.empty:
        print(f"Ablation terminée : {out['event_id'].nunique()} événements × 4 variantes.")
    return out


def ablation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Résumé descriptif événementiel; l'inférence est calculée séparément par vache."""
    unique = df.drop_duplicates(subset=["event_id", "variante"])
    return (
        unique.groupby("variante")
        .agg(
            n_events=("event_id", "nunique"),
            n_cows=("cow", "nunique"),
            detect_any=("detected_any_overlap", "mean"),
            iou20=("detected_iou20", "mean"),
            best_iou=("best_iou", "mean"),
            false_notif_cow_day=("false_notif_cow_day", "mean"),
        )
        .round(3)
        .reset_index()
    )


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def ablation_paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Tests appariés sur moyennes PAR VACHE, unité indépendante de l'étude."""
    unique = df.drop_duplicates(subset=["event_id", "variante"])
    per_cow = (
        unique.groupby(["cow", "variante"], as_index=False)
        .agg(
            detection=("detected_any_overlap", "mean"),
            best_iou=("best_iou", "mean"),
        )
    )
    det = per_cow.pivot(index="cow", columns="variante", values="detection")
    iou = per_cow.pivot(index="cow", columns="variante", values="best_iou")
    variants = [name for name in _VARIANT_NAMES if name in det.columns]
    rows = []
    for first, second in combinations(variants, 2):
        paired_det = det[[first, second]].dropna()
        paired_iou = iou[[first, second]].dropna()
        det_first = paired_det[first].to_numpy(float)
        det_second = paired_det[second].to_numpy(float)
        iou_first = paired_iou[first].to_numpy(float)
        iou_second = paired_iou[second].to_numpy(float)
        mean_det_first = float(det_first.mean())
        mean_det_second = float(det_second.mean())
        mean_iou_first = float(iou_first.mean())
        mean_iou_second = float(iou_second.mean())
        rows.append(
            {
                "paire": f"{first[0]} vs {second[0]}",
                "n_cows": int(len(paired_det)),
                "detect_1": round(mean_det_first, 3),
                "detect_2": round(mean_det_second, 3),
                "detect_favori": (
                    first
                    if mean_det_first > mean_det_second
                    else second if mean_det_second > mean_det_first else "égalité"
                ),
                "p_wilcoxon_detection": round(_paired_wilcoxon(det_first, det_second), 5),
                "iou_1": round(mean_iou_first, 3),
                "iou_2": round(mean_iou_second, 3),
                "iou_favori": (
                    first
                    if mean_iou_first > mean_iou_second
                    else second if mean_iou_second > mean_iou_first else "égalité"
                ),
                "p_wilcoxon_iou": round(_paired_wilcoxon(iou_first, iou_second), 5),
            }
        )
    return pd.DataFrame(rows)
