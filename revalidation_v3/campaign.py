"""Campagne technique du détecteur bidirectionnel de MemoireV3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from core import config as C
from core.io import COW, LYING, MI, STANDING, STEPS, TIME, TRANSITIONS, TR_DOWN, TR_UP, load_csv
from core.hybrid_warning import HybridFusionConfig, InstabilityWarningConfig
from core.pipeline import run_pipeline_one_cow
from revalidation.campaign import (
    _evaluate_binary_output,
    _heldout_start_time,
    _monitoring_duration_days,
    has_informative_heldout_signals,
)
from revalidation_v3.profiles import PROFILES_V3, HybridSyntheticProfile


def final_params() -> dict[str, object]:
    """Paramètres de production communs; l'IF historique reste comparateur."""
    return {
        "interval": C.DEFAULT_INTERVAL,
        "window_baseline": C.DEFAULT_WINDOW_BASELINE,
        "contamination": C.DEFAULT_CONTAMINATION,
        "baseline_ratio": C.DEFAULT_BASELINE_RATIO,
        "random_state": C.DEFAULT_RANDOM_STATE,
        "persist_hours": C.DEFAULT_PERSIST_HOURS,
        "alert_min": C.DEFAULT_ALERT_MIN,
        "mix_mode": C.DEFAULT_MIX_MODE,
        "mix_rate_thr": C.DEFAULT_MIX_RATE_THR,
        "z_low_thr": C.DEFAULT_Z_LOW_THR,
        "z_high_thr": C.DEFAULT_Z_HIGH_THR,
        "cooldown_hours": C.DEFAULT_COOLDOWN_HOURS,
        "mi_z_high_thr": C.DEFAULT_MI_Z_HIGH_THR,
        "coverage_min_pct": C.DEFAULT_COVERAGE_MIN_PCT,
    }


def _smooth_envelope(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(0, n), dtype=float)
    ramp = max(1, int(round(0.25 * n)))
    envelope = np.ones(n, dtype=float)
    x = np.linspace(0.0, 1.0, ramp)
    envelope[:ramp] = x * x * (3.0 - 2.0 * x)
    envelope[-ramp:] = envelope[:ramp][::-1]
    return envelope


def _apply_changes(
    df: pd.DataFrame,
    idx: np.ndarray,
    *,
    steps_change: float,
    motion_change: float,
    transitions_change: float,
    posture_shift: float,
    posture_oscillation: float,
    oscillation: float,
) -> None:
    envelope = _smooth_envelope(len(idx))
    phase = np.sin(np.linspace(0.0, max(2.0, len(idx) / 6.0) * np.pi, len(idx)))
    for column, change, oscillating in [
        (STEPS, steps_change, False),
        (MI, motion_change, True),
        (TRANSITIONS, transitions_change, True),
        (TR_UP, transitions_change, True),
        (TR_DOWN, transitions_change, True),
    ]:
        if column not in df:
            continue
        values = pd.to_numeric(df.loc[idx, column], errors="coerce").fillna(0.0).to_numpy(float)
        delta = float(change) * envelope
        if oscillating and oscillation:
            delta = delta + float(oscillation) * envelope * phase
        df.loc[idx, column] = np.maximum(0.0, values * (1.0 + delta))

    if LYING in df and STANDING in df and (posture_shift or posture_oscillation):
        lying = pd.to_numeric(df.loc[idx, LYING], errors="coerce").fillna(0.0).to_numpy(float)
        standing = pd.to_numeric(df.loc[idx, STANDING], errors="coerce").fillna(0.0).to_numpy(float)
        total = lying + standing
        posture_delta = float(posture_shift) * envelope * standing
        posture_delta += float(posture_oscillation) * envelope * total * phase
        shifted = np.clip(lying + posture_delta, 0.0, total)
        df.loc[idx, LYING] = shifted
        df.loc[idx, STANDING] = total - shifted


def _apply_profile(df: pd.DataFrame, idx: np.ndarray, spec: HybridSyntheticProfile) -> None:
    primary_bins = len(idx)
    secondary_bins = 0
    gap_bins = 0
    if spec.secondary_duration_hours > 0:
        total_hours = spec.duration_hours + spec.sequence_gap_hours + spec.secondary_duration_hours
        primary_bins = max(1, int(round(len(idx) * spec.duration_hours / total_hours)))
        gap_bins = max(0, int(round(len(idx) * spec.sequence_gap_hours / total_hours)))
        secondary_bins = max(1, len(idx) - primary_bins - gap_bins)

    _apply_changes(
        df,
        idx[:primary_bins],
        steps_change=spec.steps_change,
        motion_change=spec.motion_change,
        transitions_change=spec.transitions_change,
        posture_shift=spec.posture_shift,
        posture_oscillation=spec.posture_oscillation,
        oscillation=spec.oscillation,
    )
    if secondary_bins:
        secondary_idx = idx[-secondary_bins:]
        _apply_changes(
            df,
            secondary_idx,
            steps_change=spec.secondary_steps_change,
            motion_change=spec.secondary_motion_change,
            transitions_change=spec.secondary_transitions_change,
            posture_shift=spec.secondary_posture_shift,
            posture_oscillation=0.0,
            oscillation=0.0,
        )


def inject_profile(
    df_cow_raw: pd.DataFrame,
    *,
    cow: str,
    scenario: str,
    interval: str,
    heldout_start: pd.Timestamp,
    schedule_index: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Injecte un profil dans une fenêtre post-baseline propre au scénario."""
    spec = PROFILES_V3[scenario]
    df = df_cow_raw.sort_values(TIME).copy().reset_index(drop=True)
    for column in [STEPS, MI, LYING, STANDING, TRANSITIONS, TR_UP, TR_DOWN]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    raw_times = pd.to_datetime(df[TIME], errors="coerce")
    cadence = raw_times.sort_values().diff().dt.total_seconds().div(60).dropna()
    cadence = float(cadence[cadence > 0].median()) if (cadence > 0).any() else 15.0
    duration = max(
        1,
        int(
            round(
                (
                    spec.duration_hours
                    + spec.sequence_gap_hours
                    + spec.secondary_duration_hours
                )
                * 60.0
                / cadence
            )
        ),
    )
    lower = int(np.searchsorted(raw_times.to_numpy(), np.datetime64(heldout_start), side="left")) + 4
    available_start_span = len(df) - lower - duration - 4
    if available_start_span <= 0:
        raise ValueError(f"Période future insuffisante pour la vache {cow}.")
    scenario_index = list(PROFILES_V3).index(scenario)
    slot = (scenario_index + schedule_index) % len(PROFILES_V3)
    fraction = (slot + 1) / (len(PROFILES_V3) + 1)
    start = lower + int(round(fraction * available_start_span))
    idx = np.arange(start, start + duration)

    source = {
        column: float(pd.to_numeric(df.loc[idx, column], errors="coerce").fillna(0.0).sum())
        for column in [STEPS, MI, TRANSITIONS]
        if column in df
    }
    posture_before = None
    if LYING in df and STANDING in df:
        posture_before = (
            pd.to_numeric(df.loc[idx, LYING], errors="coerce").fillna(0.0)
            + pd.to_numeric(df.loc[idx, STANDING], errors="coerce").fillna(0.0)
        ).to_numpy(float)
    _apply_profile(df, idx, spec)
    if posture_before is not None:
        posture_after = df.loc[idx, LYING].to_numpy(float) + df.loc[idx, STANDING].to_numpy(float)
        if not np.allclose(posture_before, posture_after, atol=1e-8):
            raise RuntimeError("L'injection a modifié le temps postural total.")

    event = pd.Series(
        {
            "event_id": f"{cow}_{scenario}_{start}",
            "cow": str(cow),
            "scenario": scenario,
            "target_branch": spec.target_branch,
            "start": pd.Timestamp(df.loc[idx[0], TIME]),
            "end": pd.Timestamp(df.loc[idx[-1], TIME]),
            "duration_hours": (
                spec.duration_hours + spec.sequence_gap_hours + spec.secondary_duration_hours
            ),
            "expected_hypo": spec.expected_hypo,
            "expected_instability": spec.expected_instability,
            "expected_hybrid": spec.expected_hybrid,
            "design_reason": spec.evidence_basis,
            "heldout_start": pd.Timestamp(heldout_start),
            "event_after_heldout": pd.Timestamp(df.loc[idx[0], TIME]) >= pd.Timestamp(heldout_start),
            "informative_source_window": all(source.get(col, 0.0) > 0 for col in [STEPS, MI, TRANSITIONS]),
        }
    )
    return df, event


def run_campaign(
    raw_csv: str = "data/brut.csv",
    *,
    cows: Optional[Sequence[str]] = None,
    scenarios: Sequence[str] = tuple(PROFILES_V3),
    instability_config: InstabilityWarningConfig | None = None,
    fusion_config: HybridFusionConfig | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    params = final_params()
    raw = load_csv(raw_csv)
    raw[COW] = raw[COW].astype(str)
    requested = sorted(raw[COW].unique()) if cows is None else [str(cow) for cow in cows]
    eligible: list[str] = []
    heldout_by_cow: dict[str, pd.Timestamp] = {}
    for cow in requested:
        cow_raw = raw[raw[COW] == cow]
        span = (cow_raw[TIME].max() - cow_raw[TIME].min()).total_seconds() / 86400.0
        if span < 14:
            continue
        heldout = _heldout_start_time(
            cow_raw,
            interval=str(params["interval"]),
            window_baseline=int(params["window_baseline"]),
            baseline_ratio=float(params["baseline_ratio"]),
            coverage_min_pct=float(params["coverage_min_pct"]),
        )
        if has_informative_heldout_signals(cow_raw, heldout_start=heldout):
            eligible.append(cow)
            heldout_by_cow[cow] = heldout

    rows: list[dict[str, object]] = []
    total = len(eligible) * len(scenarios)
    done = 0
    for cow_index, cow in enumerate(eligible):
        cow_raw = raw[raw[COW] == cow]
        heldout = heldout_by_cow[cow]
        clean = run_pipeline_one_cow(
            cow_raw,
            cow,
            **params,
            instability_config=instability_config,
            fusion_config=fusion_config,
        )
        clean[TIME] = pd.to_datetime(clean[TIME], errors="coerce")
        clean_future = clean[clean[TIME] >= heldout]
        monitoring_days = _monitoring_duration_days(
            clean, heldout_start=heldout, interval=str(params["interval"])
        )
        backgrounds = {
            "hypo": int(clean_future["behavioral_warning_notification"].sum()),
            "instability": int(clean_future["instability_warning_notification"].sum()),
            "hybrid": int(clean_future["hybrid_warning_notification"].sum()),
        }
        for scenario in scenarios:
            done += 1
            injected, event = inject_profile(
                cow_raw,
                cow=cow,
                scenario=scenario,
                interval=str(params["interval"]),
                heldout_start=heldout,
                schedule_index=cow_index,
            )
            pred = run_pipeline_one_cow(
                injected,
                cow,
                **params,
                instability_config=instability_config,
                fusion_config=fusion_config,
            )
            pred[TIME] = pd.to_datetime(pred[TIME], errors="coerce")
            result = event.to_dict()
            for prefix, episode, start, score in [
                ("hypo", "behavioral_warning_episode", "behavioral_warning_start", "behavioral_warning_score"),
                ("instability", "instability_warning_episode", "instability_warning_start", "instability_warning_score"),
                ("hybrid", "hybrid_warning_episode", "hybrid_warning_start", "hybrid_warning_score"),
            ]:
                metrics = _evaluate_binary_output(
                    pred,
                    event,
                    episode_col=episode,
                    start_col=start,
                    score_col=score,
                    interval=str(params["interval"]),
                    reference_predictions=clean,
                )
                result.update({f"{prefix}_{key}": value for key, value in metrics.items()})
                result[f"{prefix}_background_notifs"] = backgrounds[prefix]
            result["monitoring_days"] = monitoring_days
            rows.append(result)
            if verbose and done % 20 == 0:
                print(f"  ... {done}/{total} (vache={cow}, scénario={scenario})")
    out = pd.DataFrame(rows)
    if verbose:
        print(f"Campagne V3 terminée: {len(out)} événements sur {out['cow'].nunique() if len(out) else 0} vaches.")
    return out


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    return (
        events.groupby(["target_branch", "scenario"], as_index=False)
        .agg(
            n_events=("event_id", "nunique"),
            hypo_detection=("hypo_detected_any_overlap", "mean"),
            instability_detection=("instability_detected_any_overlap", "mean"),
            hybrid_detection=("hybrid_detected_any_overlap", "mean"),
            hybrid_overlap=("hybrid_episode_overlap", "mean"),
            hybrid_iou20=("hybrid_detected_iou20", "mean"),
            hybrid_iou=("hybrid_best_iou", "mean"),
            hybrid_delay_h=("hybrid_onset_delay_hours", "median"),
        )
        .round(3)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Exécuter seulement deux vaches.")
    parser.add_argument("--output-dir", default="data/revalidation/v3_hybrid")
    args = parser.parse_args()
    cows = ["8081", "8147"] if args.smoke else None
    events = run_campaign(cows=cows)
    summary = summarize(events)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "events.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    background = {
        branch: round(
            float(
                events.drop_duplicates("cow")
                .eval(f"{branch}_background_notifs / monitoring_days")
                .mean()
            ),
            4,
        )
        for branch in ["hypo", "instability", "hybrid"]
    } if len(events) else {}
    (output / "background_rates.json").write_text(
        json.dumps(background, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nNotifications de fond par vache-jour:", background)
    print(f"\nSorties: {output.resolve()}")


if __name__ == "__main__":
    main()
