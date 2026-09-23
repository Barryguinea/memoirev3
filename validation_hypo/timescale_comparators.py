"""Isolation Forest et LOF alimentés par les ratios sur 12 heures de HYPO.

Les comparateurs B, C et D de l'ablation reçoivent des variables construites sur
quelques heures (z-scores glissants sur 24 intervalles, différences premières,
écart à une moyenne de sept intervalles). Les dégradations injectées durent 36 à
60 heures. Cette analyse vérifie que l'écart de localisation ne tient pas à cette
seule différence d'échelle : Isolation Forest et LOF reçoivent ici exactement les
ratios à la référence individuelle que lit HYPO. Voir docs/comparateur_echelle_temps.md.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from core.early_warning import apply_behavioral_early_warning
from core.features import build_interval_features, interval_to_minutes
from core.io import COW, TIME, available_base_cols, load_csv
from core.model_if import MAX_FEATURES, N_ESTIMATORS, QUANTILE_RANGE
from validation_hypo.ablation import _paired_resolution, _paired_wilcoxon, _run_if
from validation_hypo.campaign import (
    _evaluate_binary_output,
    _heldout_start_time,
    _monitoring_duration_days,
    final_params,
    has_informative_heldout_signals,
    inject_events_for_cow,
)

RATIO_COLUMNS = tuple(
    f"warning_ratio_{family}"
    for family in ("steps", "motion", "transitions", "lying", "standing")
)
HYPO = "A. Alerte temporelle multivariée"
IF_POINT = "F. IF ponctuel sur ratios 12 h"
IF_PERSISTENT = "G. IF + persistance HYPO sur ratios 12 h"
LOF_POINT = "H. LOF ponctuel sur ratios 12 h"
LOF_PERSISTENT = "I. LOF + persistance HYPO sur ratios 12 h"
VARIANTS = (HYPO, IF_POINT, IF_PERSISTENT, LOF_POINT, LOF_PERSISTENT)
COMPARATORS = VARIANTS[1:]
SCENARIOS = ("gradual_mild", "gradual_moderate", "gradual_marked", "isolated_short_variation")


def _starts(episode: pd.Series) -> pd.Series:
    return ((episode == 1) & (episode.shift(1, fill_value=0) == 0)).astype(int)


def _cooldown(starts: pd.Series, cooldown_bins: int) -> np.ndarray:
    notification = np.zeros(len(starts), dtype=int)
    last = -(10**9)
    for i, start in enumerate(starts.to_numpy(int)):
        if start and i - last > cooldown_bins:
            notification[i] = 1
            last = i
    return notification


def timescale_variants(
    features: pd.DataFrame, params: Dict[str, object]
) -> Dict[str, pd.DataFrame]:
    """HYPO et quatre comparateurs sur les ratios de HYPO.

    Isolation Forest reprend les hyperparamètres figés du comparateur ; LOF est
    réglé comme la variante D de l'ablation (mode nouveauté, au plus 20 voisins,
    même contamination). Les deux apprennent sur les seuls intervalles de
    référence, après la même mise à l'échelle robuste. Les variantes G et I
    appliquent la persistance de HYPO (au moins 45 % d'intervalles anormaux sur
    six heures). Tous notifient au plus une fois par 24 heures, comme HYPO.
    """
    interval = str(params["interval"])
    base = apply_behavioral_early_warning(_run_if(features, params), interval=interval)
    base[TIME] = pd.to_datetime(base[TIME])
    minutes = interval_to_minutes(interval)
    persist_bins = max(2, int(round(6 * 60 / minutes)))
    cooldown_bins = max(1, int(round(24 * 60 / minutes)))

    ratios = base[list(RATIO_COLUMNS)].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    valid = (
        pd.to_numeric(base["coverage_pct"], errors="coerce").fillna(0.0)
        >= float(params["coverage_min_pct"])
    )
    train = base["dataset_split"].eq("baseline") & valid
    future = base["dataset_split"].eq("futur")
    scaler = RobustScaler(quantile_range=QUANTILE_RANGE).fit(ratios[train])
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_features=MAX_FEATURES,
        bootstrap=True,
        contamination=float(params["contamination"]),
        random_state=int(params["random_state"]),
        n_jobs=1,
    ).fit(scaler.transform(ratios[train]))
    anomaly = pd.Series(
        (model.predict(scaler.transform(ratios)) == -1) & valid.to_numpy(),
        index=base.index,
    ).astype(int)
    train_scaled = scaler.transform(ratios[train])
    lof = LocalOutlierFactor(
        n_neighbors=min(20, max(2, len(train_scaled) - 1)),
        contamination=float(params["contamination"]),
        novelty=True,
    ).fit(train_scaled)
    lof_anomaly = pd.Series(
        (lof.predict(scaler.transform(ratios)) == -1) & valid.to_numpy(),
        index=base.index,
    ).astype(int)

    def persistent(points: pd.Series) -> pd.Series:
        return (
            (points.rolling(persist_bins, min_periods=persist_bins).mean() >= 0.45) & future
        ).astype(int)

    episodes = {
        HYPO: base["behavioral_warning_episode"].astype(int),
        IF_POINT: (anomaly.eq(1) & future).astype(int),
        IF_PERSISTENT: persistent(anomaly),
        LOF_POINT: (lof_anomaly.eq(1) & future).astype(int),
        LOF_PERSISTENT: persistent(lof_anomaly),
    }
    outputs = {}
    for name, episode in episodes.items():
        out = base[[TIME]].copy()
        out["pred_lameness_episode"] = episode.to_numpy()
        if name == HYPO:
            out["pred_lameness_start"] = base["behavioral_warning_start"].to_numpy()
            out["notif_lameness"] = base["behavioral_warning_notification"].to_numpy()
        else:
            out["pred_lameness_start"] = _starts(episode).to_numpy()
            out["notif_lameness"] = _cooldown(out["pred_lameness_start"], cooldown_bins)
        outputs[name] = out
    return outputs


def run_timescale_ablation(
    raw_csv: str = "data/brut.csv",
    *,
    params: Optional[Dict[str, object]] = None,
    scenarios: Sequence[str] = SCENARIOS,
    seed: int = 11,
) -> pd.DataFrame:
    """Mêmes onze vaches, mêmes quarante-quatre événements que l'ablation principale."""
    params = params or final_params()
    interval = str(params["interval"])
    raw_all = load_csv(raw_csv)
    raw_all[COW] = raw_all[COW].astype(str)
    eligible = []
    for cow in sorted(raw_all[COW].unique()):
        raw = raw_all[raw_all[COW] == cow]
        if (raw[TIME].max() - raw[TIME].min()).total_seconds() / 86400.0 < 14:
            continue
        heldout = _heldout_start_time(
            raw,
            interval=interval,
            window_baseline=int(params["window_baseline"]),
            baseline_ratio=float(params["baseline_ratio"]),
            coverage_min_pct=float(params["coverage_min_pct"]),
        )
        if has_informative_heldout_signals(raw, heldout_start=heldout):
            eligible.append((cow, heldout))

    rows = []
    for index, (cow, heldout) in enumerate(eligible):
        raw = raw_all[raw_all[COW] == cow]
        clean = timescale_variants(
            build_interval_features(
                raw, time_col=TIME, interval=interval, cols=available_base_cols(raw),
                window_baseline=int(params["window_baseline"]),
            ),
            params,
        )
        background = {}
        for name, prediction in clean.items():
            days = _monitoring_duration_days(prediction, heldout_start=heldout, interval=interval)
            future_notifications = prediction.loc[prediction[TIME] >= heldout, "notif_lameness"]
            background[name] = float(future_notifications.sum()) / days if days > 0 else np.nan
        for scenario in scenarios:
            injected, events = inject_events_for_cow(
                raw, cow=cow, scenario=scenario, seed=seed, interval=interval,
                persist_hours=int(params["persist_hours"]),
                baseline_ratio=float(params["baseline_ratio"]),
                window_baseline=int(params["window_baseline"]),
                coverage_min_pct=float(params["coverage_min_pct"]),
                heldout_start=heldout, schedule_rotation=index % 4,
            )
            if events.empty:
                continue
            event = events.iloc[0]
            injected_outputs = timescale_variants(
                build_interval_features(
                    injected, time_col=TIME, interval=interval,
                    cols=available_base_cols(injected),
                    window_baseline=int(params["window_baseline"]),
                ),
                params,
            )
            for name, prediction in injected_outputs.items():
                metrics = _evaluate_binary_output(
                    prediction, event,
                    episode_col="pred_lameness_episode",
                    start_col="pred_lameness_start",
                    score_col="pred_lameness_episode",
                    interval=interval,
                    reference_predictions=clean[name],
                )
                result = event.to_dict()
                result.update(metrics)
                result.update(
                    variante=name, scenario=scenario, seed=seed,
                    false_notif_cow_day=background[name],
                )
                rows.append(result)
    return pd.DataFrame(rows)


def summarize_timescale(events: pd.DataFrame) -> pd.DataFrame:
    unique = events.drop_duplicates(["event_id", "variante"])
    rows = []
    for name in VARIANTS:
        group = unique[unique["variante"] == name]
        positives = group[group["expected_detected"] == 1]
        negatives = group[group["expected_detected"] == 0]
        tp = int(positives["detected_any_overlap"].sum())
        fp = int(negatives["detected_any_overlap"].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / len(positives) if len(positives) else 0.0
        rows.append({
            "variante": name,
            "n_events": int(group["event_id"].nunique()),
            "n_cows": int(group["cow"].nunique()),
            "nouveau_depart": group["detected_any_overlap"].mean(),
            "couverture_attribuable": group["episode_overlap"].mean(),
            "iou20": group["detected_iou20"].mean(),
            "iou_moyen": group["best_iou"].mean(),
            "fond_par_vache_jour": group["false_notif_cow_day"].mean(),
            "f1_nouveau_depart": (
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            ),
        })
    return pd.DataFrame(rows).round(6)


def paired_tests_timescale(events: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon apparié par vache, HYPO contre chaque comparateur."""
    unique = events.drop_duplicates(["event_id", "variante"])
    per_cow = unique.groupby(["cow", "variante"])[["best_iou", "detected_any_overlap"]].mean()
    rows = []
    for other in COMPARATORS:
        record = {"paire": f"A vs {other[0]}"}
        for metric, label in (("best_iou", "iou"), ("detected_any_overlap", "detection")):
            table = per_cow[metric].unstack()[[HYPO, other]].dropna()
            first, second = table[HYPO].to_numpy(float), table[other].to_numpy(float)
            resolution = _paired_resolution(first, second)
            record.update({
                f"{label}_A": round(float(first.mean()), 6),
                f"{label}_autre": round(float(second.mean()), 6),
                f"p_wilcoxon_{label}": round(_paired_wilcoxon(first, second), 5),
                f"n_informatif_{label}": resolution["n_informatif"],
                f"n_favorables_A_{label}": resolution["n_favorables_premier"],
            })
        rows.append(record)
    return pd.DataFrame(rows)


__all__ = [
    "COMPARATORS",
    "IF_PERSISTENT",
    "IF_POINT",
    "LOF_PERSISTENT",
    "LOF_POINT",
    "RATIO_COLUMNS",
    "paired_tests_timescale",
    "run_timescale_ablation",
    "summarize_timescale",
    "timescale_variants",
]
