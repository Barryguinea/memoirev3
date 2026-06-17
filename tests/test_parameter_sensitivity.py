from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from core.early_warning import EarlyWarningConfig
from validation_hypo.sensitivity import (
    add_reference_deltas,
    assert_same_event_schedule,
    build_sensitivity_configurations,
    run_parameter_sensitivity,
    summarize_configuration,
)
from scripts.plot_parameter_sensitivity import build_parameter_effects


def _minimal_events(offset: float = 0.0) -> pd.DataFrame:
    scenarios = [
        "gradual_mild",
        "gradual_moderate",
        "gradual_marked",
        "isolated_short_variation",
    ]
    starts = pd.to_datetime(
        ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
    )
    return pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "cow": ["1"] * 4,
            "scenario": scenarios,
            "seed": [11] * 4,
            "start": starts,
            "end": starts + pd.Timedelta(hours=1),
            "heldout_start": pd.to_datetime(["2026-01-01"] * 4),
            "event_after_heldout": [True] * 4,
            "informative_source_window": [True] * 4,
            "detected_any_overlap": [1, 1, 1, 0],
            "episode_overlap": [1, 1, 1, 0],
            "detected_iou20": [1, 1, 1, 0],
            "best_iou": [0.4 + offset, 0.5 + offset, 0.6 + offset, 0.0],
            "onset_delay_hours": [3.0, 2.0, 1.0, np.nan],
            "cow_false_notifs": [2] * 4,
            "cow_days": [10.0] * 4,
        }
    )


def test_grid_contains_reference_and_twenty_ofat_variants():
    configurations = build_sensitivity_configurations()
    assert len(configurations) == 21
    assert configurations[0].name == "reference"
    reference = asdict(EarlyWarningConfig())
    for configuration in configurations[1:]:
        candidate = asdict(configuration.warning_config)
        changed = [key for key in reference if candidate[key] != reference[key]]
        assert changed == [configuration.parameter]


def test_summary_keeps_progressive_controls_and_background_separate():
    configuration = build_sensitivity_configurations()[0]
    result = summarize_configuration(_minimal_events(), configuration)
    assert result["n_events"] == 4
    assert result["progressive_new_start_rate"] == 1.0
    assert result["progressive_best_iou_mean"] == pytest.approx(0.5)
    assert result["progressive_onset_delay_median_hours"] == 2.0
    assert result["control_new_start_rate"] == 0.0
    assert result["background_per_cow_day_mean"] == 0.2


def test_reference_deltas_do_not_rank_configurations():
    configurations = build_sensitivity_configurations()[:2]
    summary = pd.DataFrame(
        [
            summarize_configuration(_minimal_events(0.0), configurations[0]),
            summarize_configuration(_minimal_events(0.1), configurations[1]),
        ]
    )
    output = add_reference_deltas(summary)
    assert "rank" not in output.columns
    assert output.loc[0, "delta_progressive_best_iou_mean_vs_reference"] == 0.0
    assert output.loc[1, "delta_progressive_best_iou_mean_vs_reference"] == pytest.approx(0.1)


def test_event_schedule_mismatch_is_blocking():
    reference = _minimal_events()
    candidate = _minimal_events()
    candidate.loc[0, "start"] += pd.Timedelta(hours=1)
    with pytest.raises(RuntimeError, match="evenements physiques"):
        assert_same_event_schedule(reference, candidate)


def test_orchestration_writes_reproducible_outputs_without_real_campaign(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text("placeholder\n", encoding="utf-8")

    def fake_runner(*args, warning_config=None, **kwargs):
        assert warning_config is not None
        return _minimal_events(offset=float(warning_config.aggregation_hours) / 1000.0)

    _, summary, by_scenario = run_parameter_sensitivity(
        raw,
        output_dir=tmp_path / "out",
        configuration_names=["aggregation_hours__6p0"],
        verbose=False,
        campaign_runner=fake_runner,
    )

    assert summary["configuration"].tolist() == ["reference", "aggregation_hours__6p0"]
    assert len(by_scenario) == 8
    assert (tmp_path / "out" / "parameter_sensitivity_events.csv").is_file()
    assert (tmp_path / "out" / "parameter_sensitivity_summary.csv").is_file()
    assert (tmp_path / "out" / "parameter_sensitivity_configurations.json").is_file()
    assert (tmp_path / "out" / "parameter_sensitivity.sha256").is_file()


def test_parameter_effects_use_maximum_absolute_delta_per_parameter():
    summary = pd.DataFrame(
        [
            {
                "configuration": "reference",
                "parameter": "reference",
                "delta_progressive_new_start_rate_vs_reference": 0.0,
                "delta_progressive_attributable_coverage_rate_vs_reference": 0.0,
                "delta_progressive_iou20_rate_vs_reference": 0.0,
                "delta_background_per_cow_day_mean_vs_reference": 0.0,
            },
            {
                "configuration": "min_families__2",
                "parameter": "min_families",
                "delta_progressive_new_start_rate_vs_reference": 0.03,
                "delta_progressive_attributable_coverage_rate_vs_reference": 0.06,
                "delta_progressive_iou20_rate_vs_reference": -0.09,
                "delta_background_per_cow_day_mean_vs_reference": 0.12,
            },
            {
                "configuration": "min_families__4",
                "parameter": "min_families",
                "delta_progressive_new_start_rate_vs_reference": -0.12,
                "delta_progressive_attributable_coverage_rate_vs_reference": -0.24,
                "delta_progressive_iou20_rate_vs_reference": -0.03,
                "delta_background_per_cow_day_mean_vs_reference": -0.20,
            },
        ]
    )

    effects = build_parameter_effects(summary)

    assert effects.loc[0, "Nouveau début"] == pytest.approx(12.0)
    assert effects.loc[0, "Couverture"] == pytest.approx(24.0)
    assert effects.loc[0, "IoU20"] == pytest.approx(9.0)
    assert effects.loc[0, "Fond"] == pytest.approx(0.20)
