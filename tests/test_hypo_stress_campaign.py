from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import model_if
from core.features import build_interval_features
from core.io import COW, LYING, MI, STANDING, STEPS, TIME, TRANSITIONS, TR_DOWN, TR_UP
from validation_hypo import stress_campaign
from validation_hypo.ablation import _VARIANT_NAMES, _run_variants
from validation_hypo.campaign import final_params
from validation_hypo.stress_campaign import (
    inject_stress_event,
    load_protocol,
    protocol_sha256,
)
from validation_hypo.training import add_production_split_columns


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


def _check_variants_keep_clean_reference(monkeypatch, scenario_spec):
    monkeypatch.setattr(model_if, "N_ESTIMATORS", 8)
    raw = _raw_cow()
    params = final_params()

    def features(frame):
        return build_interval_features(
            frame, time_col=TIME, interval=str(params["interval"]),
            cols=[STEPS, MI, LYING, STANDING, TRANSITIONS, TR_UP, TR_DOWN],
            window_baseline=int(params["window_baseline"]),
        )

    clean_features = features(raw)
    clean = _run_variants(clean_features, "1", params, None)
    primary = clean[_VARIANT_NAMES[0]]
    reference = pd.DatetimeIndex(primary.loc[primary["if_train_point"].eq(1), TIME])
    explicit_clean = _run_variants(
        clean_features, "1", params, None, reference_times=reference,
    )
    for name in clean:
        pd.testing.assert_frame_equal(clean[name], explicit_clean[name])
    heldout = primary.loc[primary["dataset_split"].eq("futur"), TIME].min()
    injected, _ = inject_stress_event(
        raw, cow="1", scenario=scenario_spec["name"],
        duration_hours=scenario_spec["duration_hours"],
        expected_alert=scenario_spec["expected_alert"],
        placement_fraction=0.5, placement_index=1, heldout_start=heldout,
    )
    injected_features = features(injected)
    fixed = _run_variants(
        injected_features, "1", params, None, reference_times=reference,
    )
    recomputed = _run_variants(injected_features, "1", params, None)
    for name, predictions in fixed.items():
        actual = pd.DatetimeIndex(
            predictions.loc[predictions["if_train_point"].eq(1), TIME]
        )
        assert actual.equals(reference)
        pd.testing.assert_frame_equal(
            clean_features.set_index(TIME).loc[reference],
            injected_features.set_index(TIME).loc[reference],
        )
        if scenario_spec["name"] != "contiguous_dropout":
            pd.testing.assert_frame_equal(predictions, recomputed[name])
        else:
            old_end = recomputed[name].loc[recomputed[name]["if_train_point"].eq(1), TIME].max()
            assert reference[-1] - old_end == pd.Timedelta(hours=7, minutes=15)


def test_all_variants_keep_clean_reference_after_injection(monkeypatch):
    for scenario_spec in load_protocol()["scenarios"]:
        _check_variants_keep_clean_reference(monkeypatch, scenario_spec)


def _check_invalid_reference(defect):
    features = pd.DataFrame({
        TIME: pd.date_range("2026-01-01", periods=100, freq="15min"),
        "coverage_pct": 100.0,
        "sin_hour": np.linspace(0, 1, 100),
    })
    split, train_idx = add_production_split_columns(features)
    reference = pd.DatetimeIndex(split.loc[train_idx, TIME])
    if defect == "missing":
        features = features.drop(index=train_idx[5])
    elif defect == "ineligible":
        features.loc[train_idx[5], "coverage_pct"] = 0
    elif defect == "duplicate":
        reference = reference.append(reference[:1])
    elif defect == "nonprefix":
        reference = reference[1:]
    else:
        reference = reference[:0]
    with pytest.raises(ValueError, match="[Rr]eference"):
        model_if.run_if_core(features, baseline_ratio=0.6, reference_times=reference)
    with pytest.raises(ValueError, match="[Rr]eference"):
        add_production_split_columns(features, reference_times=reference)


def test_fixed_reference_rejects_changed_training_points():
    for defect in ("missing", "ineligible", "duplicate", "nonprefix", "empty"):
        _check_invalid_reference(defect)


def test_stress_campaign_passes_clean_reference_to_every_injected_run(monkeypatch):
    monkeypatch.setattr(model_if, "N_ESTIMATORS", 8)
    monkeypatch.setattr(stress_campaign, "load_csv", lambda _: _raw_cow())
    original = stress_campaign._run_variants
    references = []

    def tracked(*args, **kwargs):
        references.append(kwargs.get("reference_times"))
        return original(*args, **kwargs)

    monkeypatch.setattr(stress_campaign, "_run_variants", tracked)
    events = stress_campaign.run_stress_campaign(verbose=False, fixed_reference=True)
    assert references[0] is None
    assert len(references) == 19
    assert all(ref is not None and ref.equals(references[1]) for ref in references[1:])
    assert len(events) == 90
    assert events["reference_policy"].eq(stress_campaign.FIXED_REFERENCE_POLICY).all()
    assert events["reference_end"].nunique() == 1
    assert events["reference_n_intervals"].nunique() == 1


def test_stress_campaign_recomputes_the_reference_by_default(monkeypatch):
    """Sans drapeau, la campagne conserve exactement le comportement historique."""
    monkeypatch.setattr(model_if, "N_ESTIMATORS", 8)
    monkeypatch.setattr(stress_campaign, "load_csv", lambda _: _raw_cow())
    original = stress_campaign._run_variants
    references = []

    def tracked(*args, **kwargs):
        references.append(kwargs.get("reference_times"))
        return original(*args, **kwargs)

    monkeypatch.setattr(stress_campaign, "_run_variants", tracked)
    events = stress_campaign.run_stress_campaign(verbose=False)
    assert references and all(ref is None for ref in references)
    assert "reference_policy" not in events.columns
    assert stress_campaign.default_output_dir(fixed_reference=False) == (
        stress_campaign.HISTORICAL_OUTPUT_DIR
    )


def test_fixed_reference_output_cannot_overwrite_sealed_artifacts():
    events = pd.DataFrame({"reference_policy": [stress_campaign.FIXED_REFERENCE_POLICY]})
    with pytest.raises(ValueError, match="Preserve the sealed"):
        stress_campaign.write_outputs(events, str(stress_campaign.SEALED_OUTPUT_DIR))
