"""Campagne technique post-baseline de la branche HYPO.

Chaque execution contient un seul evenement graduel, injecte apres la baseline.
Les profils sont contre-factuels et servent a tester le comportement du code;
ils ne constituent ni des cas cliniques ni une estimation de sensibilite.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core import config as C
from core.early_warning import EarlyWarningConfig
from core.features import build_interval_features, interval_to_minutes
from core.io import (
    COW,
    LYING,
    MI,
    STANDING,
    STEPS,
    TIME,
    TRANSITIONS,
    TR_DOWN,
    TR_UP,
    available_base_cols,
    load_csv,
)
from core.pipeline import run_pipeline_one_cow
from validation_hypo.profiles import PROFILES, SyntheticProfile
from validation_hypo.training import production_split_indices
from scripts.validation_common import iou, overlap_seconds, segments_from_binary


def _stable_seed(cow: str, scenario: str, seed: int) -> int:
    payload = f"{cow}|{scenario}|{int(seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _smooth_envelope(n: int) -> np.ndarray:
    """Montee progressive, plateau, puis retour progressif sans alternance."""
    if n <= 1:
        return np.ones(max(0, n), dtype=float)
    ramp_in = max(2, int(round(0.35 * n)))
    ramp_out = max(2, int(round(0.15 * n)))
    if ramp_in + ramp_out >= n:
        ramp_in = max(1, n // 2)
        ramp_out = n - ramp_in
    envelope = np.ones(n, dtype=float)
    x_in = np.linspace(0.0, 1.0, ramp_in)
    envelope[:ramp_in] = x_in * x_in * (3.0 - 2.0 * x_in)
    if ramp_out:
        x_out = np.linspace(1.0, 0.0, ramp_out)
        envelope[-ramp_out:] = x_out * x_out * (3.0 - 2.0 * x_out)
    return envelope


def _raw_interval_minutes(df: pd.DataFrame) -> float:
    delta = pd.to_datetime(df[TIME], errors="coerce").sort_values().diff().dt.total_seconds() / 60.0
    valid = delta[(delta > 0) & np.isfinite(delta)]
    return float(valid.median()) if len(valid) else 15.0


def _apply_profile(df: pd.DataFrame, idx: np.ndarray, spec: SyntheticProfile) -> None:
    envelope = _smooth_envelope(len(idx))
    changes = {
        STEPS: spec.steps_change,
        MI: spec.motion_change,
        TRANSITIONS: spec.transitions_change,
        TR_UP: spec.transitions_change,
        TR_DOWN: spec.transitions_change,
    }
    for column, change in changes.items():
        if column not in df:
            continue
        values = pd.to_numeric(df.loc[idx, column], errors="coerce").fillna(0.0).to_numpy(float)
        multiplier = 1.0 + float(change) * envelope
        df.loc[idx, column] = np.maximum(0.0, values * multiplier)

    # Le temps couche et le temps debout sont complementaires dans ces donnees.
    # On transfere une part du temps debout vers le temps couche afin de
    # conserver exactement leur somme a chaque intervalle.
    if LYING in df and STANDING in df and spec.posture_shift != 0:
        lying = pd.to_numeric(df.loc[idx, LYING], errors="coerce").fillna(0.0).to_numpy(float)
        standing = pd.to_numeric(df.loc[idx, STANDING], errors="coerce").fillna(0.0).to_numpy(float)
        total = lying + standing
        shifted = lying + float(spec.posture_shift) * envelope * standing
        shifted = np.minimum(np.maximum(shifted, 0.0), total)
        df.loc[idx, LYING] = shifted
        df.loc[idx, STANDING] = total - shifted


def _heldout_start_time(
    df_cow_raw: pd.DataFrame,
    *,
    interval: str,
    window_baseline: int,
    baseline_ratio: float,
    coverage_min_pct: float,
) -> pd.Timestamp:
    features = build_interval_features(
        df_cow_raw,
        time_col=TIME,
        interval=interval,
        cols=available_base_cols(df_cow_raw),
        window_baseline=int(window_baseline),
    )
    _, future_idx, _ = production_split_indices(
        features,
        baseline_ratio=baseline_ratio,
        coverage_min_pct=coverage_min_pct,
        sensor_warmup_bins=C.DEFAULT_SENSOR_WARMUP_BINS,
    )
    if len(future_idx) == 0:
        raise ValueError("Aucun intervalle post-baseline disponible pour cette vache.")
    return pd.Timestamp(features.loc[future_idx[0], TIME])


def has_informative_heldout_signals(
    df_cow_raw: pd.DataFrame,
    *,
    heldout_start: pd.Timestamp,
) -> bool:
    """Vérifie que les trois familles d'activité existent après l'apprentissage."""
    future = df_cow_raw[pd.to_datetime(df_cow_raw[TIME]) >= heldout_start]
    if future.empty:
        return False
    return all(
        column in future
        and float(pd.to_numeric(future[column], errors="coerce").fillna(0.0).sum()) > 0.0
        for column in [STEPS, MI, TRANSITIONS]
    )


def inject_events_for_cow(
    df_cow_raw: pd.DataFrame,
    *,
    cow: str,
    scenario: str,
    seed: int,
    interval: str,
    persist_hours: int,
    baseline_ratio: float = 0.60,
    window_baseline: int = C.DEFAULT_WINDOW_BASELINE,
    coverage_min_pct: float = C.DEFAULT_COVERAGE_MIN_PCT,
    heldout_start: Optional[pd.Timestamp] = None,
    schedule_rotation: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Injecte un evenement graduel unique sans modifier l'entree."""
    del persist_hours  # conserve dans la signature pour compatibilite des appels
    if scenario not in PROFILES:
        raise ValueError(f"Scenario inconnu: {scenario}")
    spec = PROFILES[scenario]
    df = df_cow_raw.sort_values(TIME).copy().reset_index(drop=True)
    for column in [STEPS, MI, LYING, STANDING, TRANSITIONS, TR_UP, TR_DOWN]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    cadence = _raw_interval_minutes(df)
    durations = {
        name: max(2, int(round(profile.duration_hours * 60.0 / cadence)))
        for name, profile in PROFILES.items()
    }
    duration_bins = durations[scenario]
    if heldout_start is None:
        heldout_start = _heldout_start_time(
            df,
            interval=interval,
            window_baseline=window_baseline,
            baseline_ratio=baseline_ratio,
            coverage_min_pct=coverage_min_pct,
        )
    raw_times = pd.to_datetime(df[TIME], errors="coerce")
    lower = int(np.searchsorted(raw_times.to_numpy(), np.datetime64(heldout_start), side="left")) + 4
    names = list(PROFILES)
    rotation = (
        int(schedule_rotation) % len(names)
        if schedule_rotation is not None
        else _stable_seed(cow, "schedule", seed) % len(names)
    )
    order = names[rotation:] + names[:rotation]
    available = len(df) - lower - 4
    total_duration = sum(durations.values())
    if available <= total_duration:
        return df, pd.DataFrame()
    gap = max(4, int((available - total_duration) // (len(order) + 1)))
    cursor = lower + gap
    starts: dict[str, int] = {}
    for name in order:
        starts[name] = cursor
        cursor += durations[name] + gap
    start = starts[scenario]
    if start + duration_bins >= len(df):
        return df, pd.DataFrame()

    idx = np.arange(start, start + duration_bins)
    posture_before = (
        pd.to_numeric(df.loc[idx, LYING], errors="coerce").fillna(0.0)
        + pd.to_numeric(df.loc[idx, STANDING], errors="coerce").fillna(0.0)
        if LYING in df and STANDING in df
        else None
    )
    source_totals = {
        column: float(
            pd.to_numeric(df.loc[idx, column], errors="coerce").fillna(0.0).sum()
        )
        for column in [STEPS, MI, TRANSITIONS]
        if column in df
    }
    _apply_profile(df, idx, spec)
    injected_totals = {
        column: float(
            pd.to_numeric(df.loc[idx, column], errors="coerce").fillna(0.0).sum()
        )
        for column in [STEPS, MI, TRANSITIONS]
        if column in df
    }
    if posture_before is not None:
        posture_after = df.loc[idx, LYING].to_numpy(float) + df.loc[idx, STANDING].to_numpy(float)
        if not np.allclose(posture_before.to_numpy(float), posture_after, atol=1e-8):
            raise RuntimeError("L'injection a modifie le temps postural total.")

    event = {
        "event_id": f"{cow}_{scenario}_s{seed}_{start}",
        "cow": str(cow),
        "start": pd.Timestamp(df.loc[idx[0], TIME]),
        "end": pd.Timestamp(df.loc[idx[-1], TIME]),
        "duration_bins": int(len(idx)),
        "duration_hours": float(spec.duration_hours),
        "profile": scenario,
        "expected_detected": int(spec.expected_alert),
        "design_reason": spec.evidence_basis,
        "steps_change_target": float(spec.steps_change),
        "motion_change_target": float(spec.motion_change),
        "transitions_change_target": float(spec.transitions_change),
        "posture_shift_target": float(spec.posture_shift),
        "schedule_rotation": int(rotation),
        "realized_position_ratio": float(start / len(df)),
        "source_steps_sum": source_totals.get(STEPS, np.nan),
        "source_motion_sum": source_totals.get(MI, np.nan),
        "source_transitions_sum": source_totals.get(TRANSITIONS, np.nan),
        "realized_steps_change": (
            injected_totals[STEPS] / source_totals[STEPS] - 1.0
            if source_totals.get(STEPS, 0.0) > 0.0
            else np.nan
        ),
        "realized_motion_change": (
            injected_totals[MI] / source_totals[MI] - 1.0
            if source_totals.get(MI, 0.0) > 0.0
            else np.nan
        ),
        "realized_transitions_change": (
            injected_totals[TRANSITIONS] / source_totals[TRANSITIONS] - 1.0
            if source_totals.get(TRANSITIONS, 0.0) > 0.0
            else np.nan
        ),
        "informative_source_window": bool(
            source_totals.get(STEPS, 0.0) > 0.0
            and (
                not spec.expected_alert
                or (
                    source_totals.get(MI, 0.0) > 0.0
                    and source_totals.get(TRANSITIONS, 0.0) > 0.0
                )
            )
        ),
    }
    return df, pd.DataFrame([event])


def final_params() -> Dict[str, object]:
    return dict(
        interval=C.DEFAULT_INTERVAL,
        window_baseline=C.DEFAULT_WINDOW_BASELINE,
        contamination=C.DEFAULT_CONTAMINATION,
        baseline_ratio=C.DEFAULT_BASELINE_RATIO,
        random_state=C.DEFAULT_RANDOM_STATE,
        persist_hours=C.DEFAULT_PERSIST_HOURS,
        alert_min=C.DEFAULT_ALERT_MIN,
        mix_mode=C.DEFAULT_MIX_MODE,
        mix_rate_thr=C.DEFAULT_MIX_RATE_THR,
        z_low_thr=C.DEFAULT_Z_LOW_THR,
        z_high_thr=C.DEFAULT_Z_HIGH_THR,
        cooldown_hours=C.DEFAULT_COOLDOWN_HOURS,
        mi_z_high_thr=C.DEFAULT_MI_Z_HIGH_THR,
        coverage_min_pct=C.DEFAULT_COVERAGE_MIN_PCT,
    )


def _evaluate_binary_output(
    predictions: pd.DataFrame,
    event: pd.Series,
    *,
    episode_col: str,
    start_col: str,
    score_col: str,
    interval: str,
    reference_predictions: Optional[pd.DataFrame] = None,
    match_tolerance_hours: float = 1.0,
) -> dict[str, float | int]:
    minutes = interval_to_minutes(interval)
    bin_seconds = int(minutes * 60)
    start = pd.Timestamp(event["start"])
    end = pd.Timestamp(event["end"])
    raw_segs = segments_from_binary(predictions, episode_col)
    raw_best_iou = max(
        [iou((start, end), seg, bin_seconds) for seg in raw_segs],
        default=0.0,
    )
    raw_overlap = int(
        any(overlap_seconds((start, end), seg, bin_seconds) > 0 for seg in raw_segs)
    )

    attributable = predictions.copy()
    attributable[TIME] = pd.to_datetime(attributable[TIME], errors="coerce")
    injected_flag = pd.to_numeric(
        attributable[episode_col], errors="coerce"
    ).fillna(0).astype(int)
    reference_flag = pd.Series(0, index=attributable.index, dtype=int)
    if reference_predictions is not None and episode_col in reference_predictions:
        ref = reference_predictions[[TIME, episode_col]].copy()
        ref[TIME] = pd.to_datetime(ref[TIME], errors="coerce")
        ref_values = (
            ref.dropna(subset=[TIME])
            .drop_duplicates(subset=[TIME], keep="last")
            .set_index(TIME)[episode_col]
        )
        reference_flag = (
            attributable[TIME]
            .map(pd.to_numeric(ref_values, errors="coerce"))
            .fillna(0)
            .astype(int)
        )
    attributable["__attributable_episode"] = (
        (injected_flag == 1) & (reference_flag != 1)
    ).astype(int)
    attributable_segs = segments_from_binary(
        attributable,
        "__attributable_episode",
    )
    best_iou = max(
        [iou((start, end), seg, bin_seconds) for seg in attributable_segs],
        default=0.0,
    )
    overlap = int(
        any(
            overlap_seconds((start, end), seg, bin_seconds) > 0
            for seg in attributable_segs
        )
    )
    win = predictions[(predictions[TIME] >= start) & (predictions[TIME] <= end)]
    starts = win.loc[pd.to_numeric(win[start_col], errors="coerce").fillna(0).astype(int) == 1, TIME]
    novel_starts = starts.copy()
    if reference_predictions is not None and start_col in reference_predictions:
        ref = reference_predictions.copy()
        ref[TIME] = pd.to_datetime(ref[TIME], errors="coerce")
        ref_starts = ref.loc[
            pd.to_numeric(ref[start_col], errors="coerce").fillna(0).astype(int) == 1,
            TIME,
        ]
        tolerance = pd.Timedelta(hours=float(match_tolerance_hours))
        novel_starts = starts[
            [not ((ref_starts - timestamp).abs() <= tolerance).any() for timestamp in starts]
        ]
    delay = (
        float((pd.Timestamp(novel_starts.iloc[0]) - start).total_seconds() / 3600.0)
        if len(novel_starts)
        else np.nan
    )
    score = pd.to_numeric(win.get(score_col, pd.Series(dtype=float)), errors="coerce")
    return {
        # Un episode deja actif avant l'injection n'est pas credite comme une
        # detection de l'evenement. Il faut un nouveau debut dans la fenetre.
        "detected_any_overlap": int(len(novel_starts) > 0),
        "novel_start_count": int(len(novel_starts)),
        "episode_overlap": overlap,
        "detected_iou20": int(best_iou >= 0.20),
        "best_iou": float(best_iou),
        "episode_overlap_raw": raw_overlap,
        "detected_iou20_raw": int(raw_best_iou >= 0.20),
        "best_iou_raw": float(raw_best_iou),
        "attributable_interval_count": int(
            attributable["__attributable_episode"].sum()
        ),
        "onset_delay_hours": delay,
        "score_max": float(score.max()) if score.notna().any() else np.nan,
    }


def _monitoring_duration_days(
    predictions: pd.DataFrame,
    *,
    heldout_start: pd.Timestamp,
    interval: str,
) -> float:
    """Durée réellement surveillée après la fin de l'apprentissage."""
    times = pd.to_datetime(predictions[TIME], errors="coerce")
    future = times[times >= pd.Timestamp(heldout_start)].dropna().sort_values()
    if future.empty:
        return 0.0
    bin_duration = pd.Timedelta(minutes=interval_to_minutes(interval))
    return float((future.iloc[-1] - future.iloc[0] + bin_duration).total_seconds() / 86400.0)


def run_clean_campaign(
    raw_csv: str = "data/brut.csv",
    *,
    cows: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (11,),
    scenarios: Sequence[str] = tuple(PROFILES),
    params: Optional[Dict[str, object]] = None,
    warning_config: Optional[EarlyWarningConfig] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare le détecteur temporel et le comparateur IF sur les mêmes événements."""
    params = params or final_params()
    df_all = load_csv(raw_csv)
    df_all[COW] = df_all[COW].astype(str)
    if cows is None:
        cows = sorted(df_all[COW].unique().tolist())
    cows = [str(cow) for cow in cows]

    interval = str(params["interval"])
    rows: List[dict[str, object]] = []
    done = 0
    eligible: list[str] = []
    for cow in cows:
        raw_cow = df_all[df_all[COW] == cow]
        # Une baseline de plusieurs jours et une partie future de plusieurs jours
        # sont necessaires pour les scenarios de 36 a 60 heures.
        span_days = (raw_cow[TIME].max() - raw_cow[TIME].min()).total_seconds() / 86400.0
        if span_days < 14:
            continue
        heldout_start = _heldout_start_time(
            raw_cow,
            interval=interval,
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
        span_days = (raw_cow[TIME].max() - raw_cow[TIME].min()).total_seconds() / 86400.0
        heldout_start = _heldout_start_time(
            raw_cow,
            interval=interval,
            window_baseline=int(params["window_baseline"]),
            baseline_ratio=float(params["baseline_ratio"]),
            coverage_min_pct=float(params["coverage_min_pct"]),
        )
        clean_pred = run_pipeline_one_cow(
            raw_cow,
            cow,
            **params,
            warning_config=warning_config,
        )
        clean_pred[TIME] = pd.to_datetime(clean_pred[TIME], errors="coerce")
        clean_future = clean_pred[clean_pred[TIME] >= pd.Timestamp(heldout_start)]
        clean_background = int(
            pd.to_numeric(
                clean_future["behavioral_warning_notification"],
                errors="coerce",
            ).fillna(0).sum()
        )
        clean_legacy_background = int(
            pd.to_numeric(
                clean_future["notif_lameness"],
                errors="coerce",
            ).fillna(0).sum()
        )
        monitoring_days = _monitoring_duration_days(
            clean_pred,
            heldout_start=heldout_start,
            interval=interval,
        )
        for seed in seeds:
            for scenario in scenarios:
                done += 1
                injected, events = inject_events_for_cow(
                    raw_cow,
                    cow=cow,
                    scenario=scenario,
                    seed=int(seed),
                    interval=interval,
                    persist_hours=int(params["persist_hours"]),
                    baseline_ratio=float(params["baseline_ratio"]),
                    window_baseline=int(params["window_baseline"]),
                    coverage_min_pct=float(params["coverage_min_pct"]),
                    heldout_start=heldout_start,
                    schedule_rotation=cow_index % len(PROFILES),
                )
                if events.empty:
                    continue
                pred = run_pipeline_one_cow(
                    injected,
                    cow,
                    **params,
                    warning_config=warning_config,
                )
                pred[TIME] = pd.to_datetime(pred[TIME], errors="coerce")
                event = events.iloc[0].copy()
                early = _evaluate_binary_output(
                    pred,
                    event,
                    episode_col="behavioral_warning_episode",
                    start_col="behavioral_warning_start",
                    score_col="behavioral_warning_score",
                    interval=interval,
                    reference_predictions=clean_pred,
                )
                legacy = _evaluate_binary_output(
                    pred,
                    event,
                    episode_col="pred_lameness_episode",
                    start_col="pred_lameness_start",
                    score_col="if_anom_k",
                    interval=interval,
                    reference_predictions=clean_pred,
                )
                result = event.to_dict()
                result.update(early)
                result.update({f"legacy_{key}": value for key, value in legacy.items()})
                result["scenario"] = scenario
                result["seed"] = int(seed)
                result["heldout_start"] = pd.Timestamp(heldout_start)
                result["event_after_heldout"] = pd.Timestamp(event["start"]) >= pd.Timestamp(heldout_start)
                result["protocol_role"] = "primary" if len(seeds) == 1 else "seed_sensitivity"

                win = pred[(pred[TIME] >= event["start"]) & (pred[TIME] <= event["end"])]
                if_score = pd.to_numeric(win.get("if_score", pd.Series(dtype=float)), errors="coerce")
                result["if_score_min"] = float(if_score.min()) if if_score.notna().any() else np.nan
                result["if_anom_k_max"] = float(pd.to_numeric(win["if_anom_k"], errors="coerce").max())
                result["warning_score_max"] = float(early["score_max"])

                result["cow_background_notifs"] = clean_background
                result["legacy_cow_background_notifs"] = clean_legacy_background
                # Alias historique conserve pour les fonctions d'analyse existantes.
                result["cow_false_notifs"] = clean_background
                result["legacy_cow_false_notifs"] = clean_legacy_background
                result["cow_monitoring_days"] = float(monitoring_days)
                result["cow_total_span_days"] = float(span_days)
                # Alias explicite pour les fonctions d'analyse existantes.
                result["cow_days"] = float(monitoring_days)
                rows.append(result)
                if verbose and done % 20 == 0:
                    print(f"  ... {done}/{total} (cow={cow}, scenario={scenario})")

    out = pd.DataFrame(rows)
    if len(out):
        if not out["event_id"].is_unique:
            raise RuntimeError("Les identifiants d'evenement ne sont pas uniques.")
        physical = out[["cow", "start", "end"]].astype(str)
        if len(seeds) == 1 and physical.duplicated().any():
            raise RuntimeError("Des fenetres physiques sont dupliquees dans la campagne primaire.")
        if len(seeds) == 1:
            for cow, group in out.groupby("cow"):
                ordered = group.sort_values("start")
                previous_end = None
                for row in ordered.itertuples(index=False):
                    if previous_end is not None and pd.Timestamp(row.start) <= previous_end:
                        raise RuntimeError(f"Fenetres physiques chevauchantes pour la vache {cow}.")
                    previous_end = pd.Timestamp(row.end)
    if verbose:
        print(f"Campagne terminee: {len(out)} evenements sur {out['cow'].nunique() if len(out) else 0} vaches.")
    return out


__all__ = [
    "PROFILES",
    "final_params",
    "has_informative_heldout_signals",
    "inject_events_for_cow",
    "run_clean_campaign",
]
