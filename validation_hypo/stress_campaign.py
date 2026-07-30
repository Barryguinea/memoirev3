"""Frozen, dose-matched stress campaign for the HYPO detector.

Each positive scenario receives the same discrete envelope area as the
48-hour gradual-moderate reference, independently for every perturbed signal
family. Detector settings are imported unchanged from the production
configuration.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from core.features import build_interval_features
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
from validation_hypo.ablation import _VARIANT_NAMES, _run_variants
from validation_hypo.campaign import (
    _evaluate_binary_output,
    _heldout_start_time,
    _monitoring_duration_days,
    final_params,
    has_informative_heldout_signals,
)

PROTOCOL_PATH = Path(__file__).with_name("stress_protocol.json")
BASE_CHANGES = {
    "steps": -0.29,
    "motion": -0.25,
    "transitions": -0.25,
}
POSTURE_SHIFT = 0.07


def load_protocol() -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not protocol.get("no_retuning"):
        raise ValueError("The stress protocol must prohibit detector retuning.")
    if "dose_matching" not in protocol:
        raise ValueError("The stress protocol must define dose matching.")
    return protocol


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def _stable_seed(cow: str, scenario: str, placement: int) -> int:
    payload = f"{cow}|{scenario}|{placement}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _smooth_envelope(n: int) -> np.ndarray:
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
    x_out = np.linspace(1.0, 0.0, ramp_out)
    envelope[-ramp_out:] = x_out * x_out * (3.0 - 2.0 * x_out)
    return envelope


def _asymmetric_envelope(n: int) -> np.ndarray:
    if n <= 2:
        return np.ones(max(0, n), dtype=float)
    onset = max(2, int(round(n * 0.10)))
    recovery = max(2, int(round(n * 0.45)))
    plateau = max(0, n - onset - recovery)
    return np.concatenate(
        [
            np.linspace(0.0, 1.0, onset),
            np.ones(plateau, dtype=float),
            np.linspace(1.0, 0.0, recovery),
        ]
    )[:n]


def _shifted(envelope: np.ndarray, lag_bins: int) -> np.ndarray:
    if lag_bins <= 0:
        return envelope.copy()
    shifted = np.zeros_like(envelope)
    shifted[lag_bins:] = envelope[: len(envelope) - lag_bins]
    return shifted


def _raw_interval_minutes(df: pd.DataFrame) -> float:
    times = pd.to_datetime(df[TIME], errors="coerce").sort_values()
    delta = times.diff().dt.total_seconds().div(60)
    valid = delta[(delta > 0) & np.isfinite(delta)]
    return float(valid.median()) if len(valid) else 15.0


def _reference_dose_hours(cadence_minutes: float) -> float:
    protocol = load_protocol()
    duration = float(protocol["dose_matching"]["reference_duration_hours"])
    n = max(2, int(round(duration * 60 / cadence_minutes)))
    return float(_smooth_envelope(n).sum() * cadence_minutes / 60)


def _dropout_mask(n: int, cadence_minutes: float) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    dropout_bins = max(1, int(round(12 * 60 / cadence_minutes)))
    start = int(round(0.45 * max(0, n - dropout_bins)))
    mask[start : start + dropout_bins] = False
    return mask


def _raw_scenario_envelopes(
    scenario: str,
    n: int,
    cadence_minutes: float,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    smooth = _smooth_envelope(n)
    envelopes = {
        "steps": smooth.copy(),
        "motion": smooth.copy(),
        "transitions": smooth.copy(),
        "posture": smooth.copy(),
    }
    retained = np.ones(n, dtype=bool)

    if scenario == "abrupt_persistent":
        envelopes = {name: np.ones(n, dtype=float) for name in envelopes}
    elif scenario == "desynchronized_families":
        lags = {"steps": 0, "motion": 8, "posture": 12, "transitions": 16}
        envelopes = {
            name: _shifted(
                smooth,
                int(round(hours * 60 / cadence_minutes)),
            )
            for name, hours in lags.items()
        }
    elif scenario == "asymmetric_recovery":
        asymmetric = _asymmetric_envelope(n)
        envelopes = {name: asymmetric.copy() for name in envelopes}
    elif scenario == "noisy_gradual":
        common = np.clip(
            smooth * (1.0 + rng.normal(0.0, 0.12, size=n)),
            0.0,
            None,
        )
        envelopes = {
            name: np.clip(
                common + rng.normal(0.0, 0.03, size=n),
                0.0,
                None,
            )
            for name in envelopes
        }
    elif scenario == "contiguous_dropout":
        retained = _dropout_mask(n, cadence_minutes)
    elif scenario == "single_family_only":
        envelopes = {"steps": smooth.copy()}
    else:
        raise ValueError(f"Unknown stress scenario: {scenario}")
    return envelopes, retained


def _dose_match_envelopes(
    envelopes: dict[str, np.ndarray],
    retained: np.ndarray,
    cadence_minutes: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    target = _reference_dose_hours(cadence_minutes)
    matched: dict[str, np.ndarray] = {}
    actual: dict[str, float] = {}
    for family, envelope in envelopes.items():
        raw_area = float(envelope[retained].sum() * cadence_minutes / 60)
        if raw_area <= 0:
            raise ValueError(f"Empty retained envelope for family {family}.")
        scaled = envelope * (target / raw_area)
        matched[family] = scaled
        actual[family] = float(scaled[retained].sum() * cadence_minutes / 60)

    tolerance = float(load_protocol()["dose_matching"]["relative_tolerance"])
    errors = [abs(value - target) / target for value in actual.values()]
    max_error = max(errors, default=0.0)
    if max_error > tolerance:
        raise RuntimeError(
            f"Dose matching failed: relative error {max_error:.3e} > {tolerance:.3e}."
        )
    return matched, actual, target


def _apply_stress(
    df: pd.DataFrame,
    idx: np.ndarray,
    *,
    scenario: str,
    cadence_minutes: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float], float, np.ndarray]:
    raw_envelopes, retained = _raw_scenario_envelopes(
        scenario,
        len(idx),
        cadence_minutes,
        rng,
    )
    envelopes, actual_doses, target_dose = _dose_match_envelopes(
        raw_envelopes,
        retained,
        cadence_minutes,
    )
    family_columns = {
        "steps": [STEPS],
        "motion": [MI],
        "transitions": [TRANSITIONS, TR_UP, TR_DOWN],
    }
    for family, columns in family_columns.items():
        if family not in envelopes:
            continue
        for column in columns:
            if column not in df:
                continue
            values = (
                pd.to_numeric(df.loc[idx, column], errors="coerce")
                .fillna(0.0)
                .to_numpy(float)
            )
            multiplier = 1.0 + BASE_CHANGES[family] * envelopes[family]
            df.loc[idx, column] = np.maximum(0.0, values * multiplier)

    if "posture" in envelopes and LYING in df and STANDING in df:
        lying = (
            pd.to_numeric(df.loc[idx, LYING], errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
        )
        standing = (
            pd.to_numeric(df.loc[idx, STANDING], errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
        )
        total = lying + standing
        shifted = np.clip(
            lying + POSTURE_SHIFT * envelopes["posture"] * standing,
            0.0,
            total,
        )
        df.loc[idx, LYING] = shifted
        df.loc[idx, STANDING] = total - shifted

    if not retained.all():
        df = df.drop(index=idx[~retained]).reset_index(drop=True)
    return df, actual_doses, target_dose, retained


def inject_stress_event(
    df_cow_raw: pd.DataFrame,
    *,
    cow: str,
    scenario: str,
    duration_hours: float,
    expected_alert: int,
    placement_fraction: float,
    placement_index: int,
    heldout_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
    df = df_cow_raw.sort_values(TIME).copy().reset_index(drop=True)
    for column in [STEPS, MI, LYING, STANDING, TRANSITIONS, TR_UP, TR_DOWN]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    cadence = _raw_interval_minutes(df)
    duration_bins = max(2, int(round(duration_hours * 60 / cadence)))
    times = pd.to_datetime(df[TIME], errors="coerce")
    lower = (
        int(
            np.searchsorted(
                times.to_numpy(),
                np.datetime64(heldout_start),
                side="left",
            )
        )
        + 4
    )
    available_start_span = len(df) - lower - duration_bins - 4
    if available_start_span <= 0:
        raise ValueError(f"Insufficient held-out data for cow {cow}.")
    start = lower + int(round(float(placement_fraction) * available_start_span))
    idx = np.arange(start, start + duration_bins)
    start_time = pd.Timestamp(df.loc[idx[0], TIME])
    end_time = pd.Timestamp(df.loc[idx[-1], TIME])

    source = {
        column: float(
            pd.to_numeric(df.loc[idx, column], errors="coerce").fillna(0.0).sum()
        )
        for column in [STEPS, MI, TRANSITIONS]
        if column in df
    }
    posture_before = None
    if LYING in df and STANDING in df:
        posture_before = (
            pd.to_numeric(df.loc[idx, LYING], errors="coerce").fillna(0.0)
            + pd.to_numeric(df.loc[idx, STANDING], errors="coerce").fillna(0.0)
        ).to_numpy(float)

    rng = np.random.default_rng(_stable_seed(cow, scenario, placement_index))
    injected, actual_doses, target_dose, retained = _apply_stress(
        df,
        idx,
        scenario=scenario,
        cadence_minutes=cadence,
        rng=rng,
    )
    if posture_before is not None:
        retained_times = times.iloc[idx[retained]]
        posture_after = (
            injected.set_index(TIME).loc[retained_times, LYING].to_numpy(float)
            + injected.set_index(TIME).loc[retained_times, STANDING].to_numpy(float)
        )
        if not np.allclose(posture_before[retained], posture_after, atol=1e-8):
            raise RuntimeError("The stress injection changed total posture time.")

    dose_errors = {
        family: abs(value - target_dose) / target_dose
        for family, value in actual_doses.items()
    }
    event = pd.Series(
        {
            "event_id": f"{cow}_{scenario}_p{placement_index}_{start}",
            "cow": str(cow),
            "scenario": scenario,
            "expected_alert": int(expected_alert),
            "placement_index": int(placement_index),
            "placement_fraction": float(placement_fraction),
            "start": start_time,
            "end": end_time,
            "duration_hours": float(duration_hours),
            "heldout_start": pd.Timestamp(heldout_start),
            "event_after_heldout": start_time >= pd.Timestamp(heldout_start),
            "informative_source_window": all(
                source.get(column, 0.0) > 0
                for column in [STEPS, MI, TRANSITIONS]
            ),
            "dropout_hours": float((~retained).sum() * cadence / 60),
            "target_dose_hours": target_dose,
            "steps_dose_hours": actual_doses.get("steps"),
            "motion_dose_hours": actual_doses.get("motion"),
            "transitions_dose_hours": actual_doses.get("transitions"),
            "posture_dose_hours": actual_doses.get("posture"),
            "max_relative_dose_error": max(dose_errors.values(), default=0.0),
            "dose_matched": max(dose_errors.values(), default=0.0)
            <= float(load_protocol()["dose_matching"]["relative_tolerance"]),
        }
    )
    return injected, event


def _eligible_cows(
    raw: pd.DataFrame,
    *,
    requested: Sequence[str],
    params: dict[str, object],
) -> tuple[list[str], dict[str, pd.Timestamp]]:
    eligible: list[str] = []
    heldout_by_cow: dict[str, pd.Timestamp] = {}
    for cow in requested:
        cow_raw = raw[raw[COW] == cow]
        span_days = (cow_raw[TIME].max() - cow_raw[TIME].min()).total_seconds() / 86400
        if span_days < 14:
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
    return eligible, heldout_by_cow


def run_stress_campaign(
    raw_csv: str = "data/brut.csv",
    *,
    cows: Optional[Sequence[str]] = None,
    max_cows: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    protocol = load_protocol()
    params = final_params()
    raw = load_csv(raw_csv)
    raw[COW] = raw[COW].astype(str)
    requested = (
        sorted(raw[COW].unique())
        if cows is None
        else [str(cow) for cow in cows]
    )
    eligible, heldout_by_cow = _eligible_cows(raw, requested=requested, params=params)
    if max_cows is not None:
        eligible = eligible[:max_cows]
        heldout_by_cow = {cow: heldout_by_cow[cow] for cow in eligible}
    scenarios = protocol["scenarios"]
    placements = protocol["placement_fractions"]
    total = len(eligible) * len(scenarios) * len(placements)
    done = 0
    rows: list[dict[str, object]] = []

    for cow in eligible:
        cow_raw = raw[raw[COW] == cow]
        heldout = heldout_by_cow[cow]
        clean_features = build_interval_features(
            cow_raw,
            time_col=TIME,
            interval=str(params["interval"]),
            cols=available_base_cols(cow_raw),
            window_baseline=int(params["window_baseline"]),
        )
        clean_variants = _run_variants(clean_features, cow, params, None)
        monitoring_days = {
            name: _monitoring_duration_days(
                predictions,
                heldout_start=heldout,
                interval=str(params["interval"]),
            )
            for name, predictions in clean_variants.items()
        }
        background_rates = {
            name: (
                float("nan")
                if monitoring_days[name] <= 0
                else float(
                    pd.to_numeric(
                        predictions.loc[
                            pd.to_datetime(predictions[TIME], errors="coerce") >= heldout,
                            "notif_lameness",
                        ],
                        errors="coerce",
                    ).fillna(0).sum()
                    / monitoring_days[name]
                )
            )
            for name, predictions in clean_variants.items()
        }

        for scenario_spec, (placement_index, placement_fraction) in product(
            scenarios,
            enumerate(placements),
        ):
            done += 1
            injected, event = inject_stress_event(
                cow_raw,
                cow=cow,
                scenario=str(scenario_spec["name"]),
                duration_hours=float(scenario_spec["duration_hours"]),
                expected_alert=int(scenario_spec["expected_alert"]),
                placement_fraction=float(placement_fraction),
                placement_index=int(placement_index),
                heldout_start=heldout,
            )
            features = build_interval_features(
                injected,
                time_col=TIME,
                interval=str(params["interval"]),
                cols=available_base_cols(injected),
                window_baseline=int(params["window_baseline"]),
            )
            variants = _run_variants(features, cow, params, None)
            for name, predictions in variants.items():
                metrics = _evaluate_binary_output(
                    predictions,
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
                result["variant"] = name
                result["background_notif_per_cow_day"] = background_rates[name]
                result["monitoring_days"] = monitoring_days[name]
                result["protocol_sha256"] = protocol_sha256()
                rows.append(result)
            if verbose and (done % 20 == 0 or done == total):
                print(
                    f"  ... {done}/{total} "
                    f"(cow={cow}, scenario={scenario_spec['name']})"
                )

    events = pd.DataFrame(rows)
    if len(events):
        expected_rows = (
            len(eligible) * len(scenarios) * len(placements) * len(_VARIANT_NAMES)
        )
        if len(events) != expected_rows:
            raise RuntimeError(
                f"Incomplete stress campaign: {len(events)} rows, expected {expected_rows}."
            )
        physical = events.drop_duplicates("event_id")
        if len(physical) != len(eligible) * len(scenarios) * len(placements):
            raise RuntimeError("Stress event identifiers are not unique.")
        if not events["event_after_heldout"].all():
            raise RuntimeError("At least one stress event precedes the held-out period.")
        if not events["informative_source_window"].all():
            raise RuntimeError("At least one stress event has an empty signal family.")
        if not events["dose_matched"].all():
            raise RuntimeError("At least one stress event violates dose matching.")
    return events


def _bootstrap_interval(
    cow_values: pd.Series,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    values = pd.to_numeric(cow_values, errors="coerce").dropna().to_numpy(float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = load_protocol()
    primary = str(protocol["primary_algorithm"])
    primary_events = events[events["variant"].eq(primary)]
    rows = []
    for scenario, group in primary_events.groupby("scenario", sort=False):
        cow_values = group.groupby("cow")["episode_overlap"].mean()
        ci_low, ci_high = _bootstrap_interval(
            cow_values,
            n_resamples=int(protocol["bootstrap_resamples"]),
            seed=_stable_seed("all", scenario, 20260729),
        )
        rows.append(
            {
                "scenario": scenario,
                "expected_alert": int(group["expected_alert"].iloc[0]),
                "n_cows": int(group["cow"].nunique()),
                "n_placements": int(group["placement_index"].nunique()),
                "n_events": int(group["event_id"].nunique()),
                "target_dose_hours": float(group["target_dose_hours"].iloc[0]),
                "max_relative_dose_error": float(
                    group["max_relative_dose_error"].max()
                ),
                "attributable_coverage": float(group["episode_overlap"].mean()),
                "coverage_ci95_low_cow_bootstrap": ci_low,
                "coverage_ci95_high_cow_bootstrap": ci_high,
                "novel_start_rate": float(group["detected_any_overlap"].mean()),
                "iou20_rate": float(group["detected_iou20"].mean()),
                "best_iou_mean": float(group["best_iou"].mean()),
            }
        )

    variant_summary = (
        events.groupby(["variant", "expected_alert"], as_index=False)
        .agg(
            n_events=("event_id", "nunique"),
            attributable_coverage=("episode_overlap", "mean"),
            novel_start_rate=("detected_any_overlap", "mean"),
            iou20_rate=("detected_iou20", "mean"),
            best_iou_mean=("best_iou", "mean"),
            background_notif_per_cow_day=("background_notif_per_cow_day", "mean"),
        )
    )
    return pd.DataFrame(rows), variant_summary


def _exact_sign_flip_p(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = float(values.mean())
    signs = np.asarray(list(product([-1.0, 1.0], repeat=len(values))))
    permuted = (signs * values).mean(axis=1)
    return float(np.mean(permuted >= observed - 1e-12))


def compare_variants(events: pd.DataFrame) -> pd.DataFrame:
    protocol = load_protocol()
    primary = str(protocol["primary_algorithm"])
    positive = events[events["expected_alert"].eq(1)]
    cow_rates = (
        positive.groupby(["cow", "variant"])["episode_overlap"].mean().unstack("variant")
    )
    rows = []
    for comparator in protocol["comparators"]:
        paired = cow_rates[[primary, comparator]].dropna()
        differences = paired[primary].to_numpy() - paired[comparator].to_numpy()
        rows.append(
            {
                "primary": primary,
                "comparator": comparator,
                "n_cows": int(len(paired)),
                "primary_cow_mean": float(paired[primary].mean()),
                "comparator_cow_mean": float(paired[comparator].mean()),
                "mean_paired_difference": float(differences.mean()),
                "exact_sign_flip_p_one_sided": _exact_sign_flip_p(differences),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    events: pd.DataFrame,
    output_dir: str = "data/validation/hypo_stress",
) -> dict[str, object]:
    protocol = load_protocol()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenario_summary, variant_summary = summarize(events)
    comparisons = compare_variants(events)
    events.to_csv(output / "events.csv", index=False)
    scenario_summary.to_csv(output / "scenario_summary.csv", index=False)
    variant_summary.to_csv(output / "variant_summary.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    summary = {
        "protocol_sha256": protocol_sha256(),
        "protocol": protocol,
        "n_cows": int(events["cow"].nunique()) if len(events) else 0,
        "n_physical_events": int(events["event_id"].nunique()) if len(events) else 0,
        "n_result_rows": int(len(events)),
        "dose_integrity": {
            "all_events_matched": bool(events["dose_matched"].all()) if len(events) else False,
            "max_relative_error": (
                float(events["max_relative_dose_error"].max()) if len(events) else None
            ),
        },
        "scenario_summary": scenario_summary.to_dict(orient="records"),
        "paired_comparisons": comparisons.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


__all__ = [
    "compare_variants",
    "inject_stress_event",
    "load_protocol",
    "protocol_sha256",
    "run_stress_campaign",
    "summarize",
    "write_outputs",
]
