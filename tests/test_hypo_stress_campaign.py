from __future__ import annotations

import numpy as np
import pandas as pd

from core.io import COW, LYING, MI, STANDING, STEPS, TIME, TRANSITIONS, TR_DOWN, TR_UP
from validation_hypo.stress_campaign import (
    inject_stress_event,
    load_protocol,
    protocol_sha256,
)


def _raw_cow() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=96 * 20, freq="15min")
    return pd.DataFrame(
        {
            TIME: times,
            COW: "1",
            STEPS: 10.0,
            MI: 5.0,
            LYING: 8.0,
            STANDING: 7.0,
            TRANSITIONS: 2.0,
            TR_UP: 1.0,
            TR_DOWN: 1.0,
        }
    )


def _inject(scenario: str, duration_hours: float = 48) -> tuple[pd.DataFrame, pd.Series]:
    return inject_stress_event(
        _raw_cow(),
        cow="1",
        scenario=scenario,
        duration_hours=duration_hours,
        expected_alert=int(scenario != "single_family_only"),
        placement_fraction=0.5,
        placement_index=1,
        heldout_start=pd.Timestamp("2026-01-10"),
    )


def test_protocol_is_frozen_envelope_area_matched_and_has_three_placements() -> None:
    protocol = load_protocol()
    assert protocol["no_retuning"] is True
    assert protocol["dose_matching"]["reference_scenario"] == "gradual_moderate"
    assert protocol["placement_fractions"] == [0.18, 0.5, 0.82]
    assert len(protocol_sha256()) == 64


def test_stress_placements_are_distinct_and_post_baseline() -> None:
    raw = _raw_cow()
    starts = []
    for placement_index, placement_fraction in enumerate([0.18, 0.5, 0.82]):
        _, event = inject_stress_event(
            raw,
            cow="1",
            scenario="abrupt_persistent",
            duration_hours=48,
            expected_alert=1,
            placement_fraction=placement_fraction,
            placement_index=placement_index,
            heldout_start=pd.Timestamp("2026-01-10"),
        )
        starts.append(event["start"])
        assert event["event_after_heldout"]
        assert event["dose_matched"]
    assert starts == sorted(starts)
    assert len(set(starts)) == 3


def test_all_positive_shapes_receive_the_same_family_dose() -> None:
    for scenario, duration in (
        ("abrupt_persistent", 48),
        ("desynchronized_families", 60),
        ("asymmetric_recovery", 60),
        ("noisy_gradual", 48),
        ("contiguous_dropout", 60),
    ):
        _, event = _inject(scenario, duration)
        target = event["target_dose_hours"]
        for column in (
            "steps_dose_hours",
            "motion_dose_hours",
            "transitions_dose_hours",
            "posture_dose_hours",
        ):
            assert np.isclose(event[column], target, rtol=1e-10, atol=1e-10)
        assert event["max_relative_dose_error"] <= 1e-10


def test_dropout_removes_twelve_hours_and_posture_is_conserved_otherwise() -> None:
    raw = _raw_cow()
    gradual, event = _inject("noisy_gradual")
    before = raw.set_index(TIME)[[LYING, STANDING]].sum(axis=1)
    after = gradual.set_index(TIME)[[LYING, STANDING]].sum(axis=1)
    assert np.allclose(
        before.loc[event["start"] : event["end"]],
        after.loc[event["start"] : event["end"]],
    )

    dropout, event = _inject("contiguous_dropout", 60)
    assert len(raw) - len(dropout) == 48
    assert event["dropout_hours"] == 12.0


def test_single_family_control_changes_steps_only_at_matched_dose() -> None:
    raw = _raw_cow()
    injected, event = _inject("single_family_only")
    window = injected[TIME].between(event["start"], event["end"])
    assert injected.loc[window, STEPS].sum() < raw.loc[window, STEPS].sum()
    for column in [MI, TRANSITIONS, LYING, STANDING]:
        assert np.allclose(injected.loc[window, column], raw.loc[window, column])
    assert np.isclose(event["steps_dose_hours"], event["target_dose_hours"])
    assert pd.isna(event["motion_dose_hours"])
