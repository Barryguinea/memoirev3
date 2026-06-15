from pathlib import Path

import pandas as pd
import pytest

from scripts.compare_robustes import compare_runs, compare_runs_three


def _write_required_base_files(run_dir: Path) -> None:
    pd.DataFrame([{"dummy": 1}]).to_csv(run_dir / "robust_runs.csv", index=False)
    pd.DataFrame([{"dummy": 1}]).to_csv(run_dir / "robust_events_evaluation.csv", index=False)


def _write_v4_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_required_base_files(run_dir)
    pd.DataFrame(
        [
            {
                "contamination": 0.05,
                "persist_hours": 6,
                "alert_min": 2,
                "mix_mode": "MIX",
                "mix_rate_thr": 0.20,
                "robust_pass_v4": 1,
                "selection_score_v4": 0.40,
                "go_reason_v4": "OK",
            },
            {
                "contamination": 0.06,
                "persist_hours": 7,
                "alert_min": 2,
                "mix_mode": "MIX",
                "mix_rate_thr": 0.24,
                "robust_pass_v4": 0,
                "selection_score_v4": 0.20,
                "go_reason_v4": "borderline_any_mean 0.125 < 0.200",
            },
        ]
    ).to_csv(run_dir / "robust_summary_v4.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "detectable_strong",
                "detected_any_overlap_mean": 1.0,
                "detected_iou20_mean": 0.5,
                "best_iou_mean": 0.30,
            },
            {
                "scenario": "detectable_borderline",
                "detected_any_overlap_mean": 0.2,
                "detected_iou20_mean": 0.1,
                "best_iou_mean": 0.15,
            },
        ]
    ).to_csv(run_dir / "robust_events_by_scenario_v4.csv", index=False)
    (run_dir / "run_meta_v4.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_meta.json").write_text(
        '{"base_params": {"mi_z_high_thr": 2.2}, "effective_grid": {"contamination": [0.05, 0.06], "persist_hours": [6, 7], "alert_min": [2], "mix_mode": ["MIX"], "mix_rate_thr": [0.2, 0.24]}}',
        encoding="utf-8",
    )


def _write_v5_1_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_required_base_files(run_dir)
    pd.DataFrame(
        [
            {
                "contamination": 0.05,
                "persist_hours": 6,
                "alert_min": 2,
                "mix_mode": "MIX",
                "mix_rate_thr": 0.20,
                "mi_z_high_thr": 2.2,
                "robust_pass_v5_1": 1,
                "selection_score_v5_1": 0.42,
                "go_reason_v5_1": "OK",
                "quality_pass_v5_1": 1,
                "safety_pass_v5_1": 1,
                "nonregression_pass_v5_1": 1,
            },
            {
                "contamination": 0.06,
                "persist_hours": 7,
                "alert_min": 2,
                "mix_mode": "MIX",
                "mix_rate_thr": 0.24,
                "mi_z_high_thr": 2.2,
                "robust_pass_v5_1": 0,
                "selection_score_v5_1": 0.21,
                "go_reason_v5_1": "borderline_any_mean 0.083 < 0.200",
                "quality_pass_v5_1": 0,
                "safety_pass_v5_1": 1,
                "nonregression_pass_v5_1": 1,
            },
        ]
    ).to_csv(run_dir / "robust_summary_v5_1.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "detectable_strong",
                "detected_any_overlap_mean": 0.95,
                "detected_iou20_mean": 0.55,
                "best_iou_mean": 0.32,
            },
            {
                "scenario": "detectable_borderline",
                "detected_any_overlap_mean": 0.15,
                "detected_iou20_mean": 0.08,
                "best_iou_mean": 0.14,
            },
        ]
    ).to_csv(run_dir / "robust_events_by_scenario_v5_1.csv", index=False)
    (run_dir / "run_meta_v5_1.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_meta.json").write_text(
        '{"effective_grid": {"contamination": [0.05, 0.06], "persist_hours": [6, 7], "alert_min": [2], "mix_mode": ["MIX"], "mix_rate_thr": [0.2, 0.24], "mi_z_high_thr": [2.2]}}',
        encoding="utf-8",
    )


def _write_v5_2_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_required_base_files(run_dir)
    # v5.2 currently reuses v5.1 engine naming.
    pd.DataFrame(
        [
            {
                "contamination": 0.05,
                "persist_hours": 6,
                "alert_min": 2,
                "mix_mode": "MIX",
                "mix_rate_thr": 0.20,
                "mi_z_high_thr": 2.2,
                "robust_pass_v5_1": 1,
                "selection_score_v5_1": 0.44,
                "go_reason_v5_1": "OK",
                "quality_pass_v5_1": 1,
                "safety_pass_v5_1": 1,
                "nonregression_pass_v5_1": 1,
            },
            {
                "contamination": 0.06,
                "persist_hours": 7,
                "alert_min": 2,
                "mix_mode": "MIX",
                "mix_rate_thr": 0.24,
                "mi_z_high_thr": 2.2,
                "robust_pass_v5_1": 0,
                "selection_score_v5_1": 0.25,
                "go_reason_v5_1": "false_notif_p90 0.260 > 0.220",
                "quality_pass_v5_1": 1,
                "safety_pass_v5_1": 0,
                "nonregression_pass_v5_1": 1,
            },
        ]
    ).to_csv(run_dir / "robust_summary_v5_1.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "detectable_strong",
                "detected_any_overlap_mean": 0.98,
                "detected_iou20_mean": 0.58,
                "best_iou_mean": 0.35,
            },
            {
                "scenario": "detectable_borderline",
                "detected_any_overlap_mean": 0.19,
                "detected_iou20_mean": 0.09,
                "best_iou_mean": 0.15,
            },
        ]
    ).to_csv(run_dir / "robust_events_by_scenario_v5_1.csv", index=False)
    (run_dir / "run_meta_v5_1.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_meta.json").write_text(
        '{"effective_grid": {"contamination": [0.05, 0.06], "persist_hours": [6, 7], "alert_min": [2], "mix_mode": ["MIX"], "mix_rate_thr": [0.2, 0.24], "mi_z_high_thr": [2.2]}}',
        encoding="utf-8",
    )


def test_compare_runs_builds_outputs(tmp_path: Path) -> None:
    v4_run = tmp_path / "robust_validation_20260101_000001"
    v5_1_run = tmp_path / "robust_validation_20260101_000002"
    out_root = tmp_path / "out"
    _write_v4_run(v4_run)
    _write_v5_1_run(v5_1_run)

    res = compare_runs(v4_run=v4_run, v5_1_run=v5_1_run, output_root=out_root)

    assert res.output_dir.exists()
    assert (res.output_dir / "config_comparison_v4_vs_v5_1.csv").exists()
    assert (res.output_dir / "scenario_comparison_v4_vs_v5_1.csv").exists()
    assert (res.output_dir / "comparison_summary_v4_vs_v5_1.json").exists()
    assert res.summary["n_go_v4"] == 1
    assert res.summary["n_go_v5_1"] == 1
    assert "mi_z_high_thr" in res.summary["full_config_keys"]
    assert res.summary["n_overlap_cfg"] == 2
    assert res.summary["same_pass_ratio"] == 1.0


def test_compare_runs_requires_complete_v4_run(tmp_path: Path) -> None:
    v4_run = tmp_path / "v4_incomplete"
    v5_1_run = tmp_path / "v5_complete"
    out_root = tmp_path / "out"

    v4_run.mkdir(parents=True, exist_ok=True)
    # robust_runs.csv et robust_events_evaluation.csv manquent volontairement.
    pd.DataFrame([{"robust_pass_v4": 1}]).to_csv(v4_run / "robust_summary_v4.csv", index=False)
    _write_v5_1_run(v5_1_run)

    with pytest.raises(FileNotFoundError):
        compare_runs(v4_run=v4_run, v5_1_run=v5_1_run, output_root=out_root)


def test_compare_runs_three_builds_outputs(tmp_path: Path) -> None:
    v4_run = tmp_path / "robust_validation_20260101_000001"
    v5_1_run = tmp_path / "robust_validation_20260101_000002"
    v5_2_run = tmp_path / "robust_validation_20260101_000003"
    out_root = tmp_path / "out3"
    _write_v4_run(v4_run)
    _write_v5_1_run(v5_1_run)
    _write_v5_2_run(v5_2_run)

    res = compare_runs_three(v4_run=v4_run, v5_1_run=v5_1_run, v5_2_run=v5_2_run, output_root=out_root)

    assert res.output_dir.exists()
    assert (res.output_dir / "config_comparison_v4_vs_v5_1_vs_v5_2.csv").exists()
    assert (res.output_dir / "scenario_comparison_v4_vs_v5_1_vs_v5_2.csv").exists()
    assert (res.output_dir / "comparison_summary_v4_vs_v5_1_vs_v5_2.json").exists()
    assert res.summary["n_go_v4"] == 1
    assert res.summary["n_go_v5_1"] == 1
    assert res.summary["n_go_v5_2"] == 1
    assert "mi_z_high_thr" in res.summary["full_config_keys"]
