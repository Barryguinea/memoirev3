import hashlib

import numpy as np
import pandas as pd

from core import config as C
from core.features import build_interval_features
from core.io import COW, TIME, available_base_cols, load_csv
from core.model_if import run_if_core
from validation_hypo.campaign import (
    _evaluate_binary_output,
    _heldout_start_time,
    _monitoring_duration_days,
    _stable_seed,
    has_informative_heldout_signals,
    inject_events_for_cow,
)
from validation_hypo.qa import assert_campaign_valid
from validation_hypo.training import production_split_indices


def test_stable_seed_uses_process_independent_digest():
    payload = b"8081|gradual_marked|11"
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    assert _stable_seed("8081", "gradual_marked", 11) == expected


def test_training_indices_match_core_model_split():
    raw = load_csv("data/brut.csv")
    raw[COW] = raw[COW].astype(str)
    cow = raw[raw[COW] == "8081"]
    features = build_interval_features(
        cow,
        time_col=TIME,
        interval=C.DEFAULT_INTERVAL,
        cols=available_base_cols(cow),
        window_baseline=C.DEFAULT_WINDOW_BASELINE,
    )
    train_idx, future_idx, _ = production_split_indices(features)
    scored = run_if_core(
        features,
        baseline_ratio=C.DEFAULT_BASELINE_RATIO,
        coverage_min_pct=C.DEFAULT_COVERAGE_MIN_PCT,
        sensor_warmup_bins=C.DEFAULT_SENSOR_WARMUP_BINS,
    )
    assert scored.index[scored["if_train_point"] == 1].tolist() == train_idx.tolist()
    assert scored.index[scored["dataset_split"] == "futur"].tolist() == future_idx.tolist()


def test_injections_are_reproducible_held_out_and_do_not_mutate_input():
    raw = load_csv("data/brut.csv")
    raw[COW] = raw[COW].astype(str)
    cow = raw[raw[COW] == "8081"].copy()
    before = cow.copy(deep=True)
    first, events_first = inject_events_for_cow(
        cow,
        cow="8081",
        scenario="gradual_marked",
        seed=11,
        interval=C.DEFAULT_INTERVAL,
        persist_hours=C.DEFAULT_PERSIST_HOURS,
    )
    second, events_second = inject_events_for_cow(
        cow,
        cow="8081",
        scenario="gradual_marked",
        seed=11,
        interval=C.DEFAULT_INTERVAL,
        persist_hours=C.DEFAULT_PERSIST_HOURS,
    )
    pd.testing.assert_frame_equal(cow, before)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(events_first, events_second)

    event = events_first.iloc[0]
    mask = (pd.to_datetime(first[TIME]) >= event["start"]) & (
        pd.to_datetime(first[TIME]) <= event["end"]
    )
    original = cow.sort_values(TIME).reset_index(drop=True)
    posture_before = original.loc[mask, "Lying Time"] + original.loc[mask, "Standing Time"]
    posture_after = first.loc[mask, "Lying Time"] + first.loc[mask, "Standing Time"]
    assert np.allclose(posture_before, posture_after)
    assert (first.loc[mask, "Steps"] <= original.loc[mask, "Steps"] + 1e-9).all()
    assert (first.loc[mask, "Motion Index"] <= original.loc[mask, "Motion Index"] + 1e-9).all()
    assert bool(event["informative_source_window"])
    assert event["source_steps_sum"] > 0
    assert event["source_motion_sum"] > 0
    assert event["source_transitions_sum"] > 0


def test_zero_core_channels_in_heldout_period_are_not_informative():
    raw = load_csv("data/brut.csv")
    raw[COW] = raw[COW].astype(str)
    cow = raw[raw[COW] == "8144"].copy()
    heldout_start = _heldout_start_time(
        cow,
        interval=C.DEFAULT_INTERVAL,
        window_baseline=C.DEFAULT_WINDOW_BASELINE,
        baseline_ratio=C.DEFAULT_BASELINE_RATIO,
        coverage_min_pct=C.DEFAULT_COVERAGE_MIN_PCT,
    )
    assert not has_informative_heldout_signals(cow, heldout_start=heldout_start)


def test_reference_subtracted_iou_does_not_credit_preexisting_episode():
    times = pd.date_range("2026-01-01", periods=8, freq="15min")
    event = pd.Series({"start": times[2], "end": times[5]})
    injected = pd.DataFrame(
        {
            TIME: times,
            "episode": [0, 0, 1, 1, 1, 1, 0, 0],
            "start": [0, 0, 1, 0, 0, 0, 0, 0],
            "score": np.linspace(0.0, 1.0, len(times)),
        }
    )
    reference = injected.copy()

    result = _evaluate_binary_output(
        injected,
        event,
        episode_col="episode",
        start_col="start",
        score_col="score",
        interval="15T",
        reference_predictions=reference,
    )

    assert result["episode_overlap_raw"] == 1
    assert result["detected_iou20_raw"] == 1
    assert result["episode_overlap"] == 0
    assert result["detected_iou20"] == 0
    assert result["best_iou"] == 0.0
    assert result["attributable_interval_count"] == 0


def test_monitoring_duration_uses_only_post_baseline_period():
    times = pd.date_range("2026-01-01", periods=12, freq="15min")
    predictions = pd.DataFrame({TIME: times})
    duration = _monitoring_duration_days(
        predictions,
        heldout_start=times[8],
        interval="15T",
    )
    assert np.isclose(duration, 1.0 / 24.0)


def test_campaign_checks_accept_valid_minimal_primary_result():
    events = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "cow": ["1", "1", "1", "1"],
            "scenario": [
                "gradual_mild",
                "gradual_moderate",
                "gradual_marked",
                "isolated_short_variation",
            ],
            "seed": [11, 11, 11, 11],
            "start": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]),
            "end": pd.to_datetime(["2026-01-02 01:00", "2026-01-03 01:00", "2026-01-04 01:00", "2026-01-05 01:00"]),
            "heldout_start": pd.to_datetime(["2026-01-01"] * 4),
            "event_after_heldout": [True, True, True, True],
            "informative_source_window": [True, True, True, True],
        }
    )
    checks = assert_campaign_valid(events)
    assert checks["passed"].all()
