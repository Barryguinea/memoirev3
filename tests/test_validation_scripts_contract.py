import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from core.io import COW, LYING, MI, STANDING, STEPS, TIME, TRANSITIONS
import scripts.validation_notebook_utils as vnu
import scripts.validation_notebook_utils_v4 as vnu4
import scripts.validation_notebook_utils_v5_1 as vnu5_1


def _raw_df() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=24, freq="15min")
    rows = []
    for cow in ["8081", "8154"]:
        for t in ts:
            rows.append(
                {
                    COW: cow,
                    TIME: t,
                    STEPS: 10.0,
                    MI: 20.0,
                    LYING: 5.0,
                    STANDING: 5.0,
                    TRANSITIONS: 1.0,
                }
            )
    return pd.DataFrame(rows)


def _summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {COW: "8081", "lameness_notifs": 0, "lameness_starts": 0, "if_anomaly_points": 3},
            {COW: "8154", "lameness_notifs": 0, "lameness_starts": 0, "if_anomaly_points": 2},
        ]
    )


def _pred_df() -> pd.DataFrame:
    base = _raw_df()[[COW, TIME]].copy()
    base["pred_lameness_start"] = 0
    base["notif_lameness"] = 0
    base["alert_level"] = "normal"
    base["if_anom_k"] = 0
    base["anom_rate_k"] = 0.0
    base["coherence_boiterie"] = 0.0
    base["if_anomaly_point"] = 0
    return base


def _events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "inj_detectable_1",
                "cow": "8081",
                "start": pd.Timestamp("2024-01-01 01:00:00"),
                "end": pd.Timestamp("2024-01-01 03:00:00"),
                "duration_bins": 8,
                "profile": "detectable_processed_visible_pattern",
                "expected_detected": 1,
                "design_reason": "test",
            }
        ]
    )


def _eval_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "inj_detectable_1",
                "expected_detected": 1,
                "detected_any_overlap": 1,
                "detected_iou20": 1,
                "best_iou": 0.5,
                "if_anom_k_max": 5,
                "anom_rate_k_max": 0.4,
                "coherence_boiterie_mean": 1.0,
                "cooldown_ratio": 0.5,
                "why": "ok",
            }
        ]
    )


class TestManualValidationArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        self.params = dict(
            interval="15T",
            window_baseline=24,
            contamination=0.05,
            baseline_ratio=0.6,
            random_state=42,
            persist_hours=6,
            alert_min=2,
            mix_mode="MIX",
            mix_rate_thr=0.34,
            z_low_thr=-2.0,
            z_high_thr=2.0,
            cooldown_hours=12,
            mi_z_high_thr=2.5,
            coverage_min_pct=25.0,
            max_cows=None,
        )

    @patch("scripts.validation_notebook_utils.extract_detected_episodes", return_value=pd.DataFrame())
    @patch("scripts.validation_notebook_utils.evaluate_injected_events", return_value=_eval_df())
    @patch("scripts.validation_notebook_utils.inject_manual_plan_raw", return_value=(_raw_df(), _events_df()))
    @patch("scripts.validation_notebook_utils.run_pipeline_herd", side_effect=[(_summary_df(), _pred_df()), (_summary_df(), _pred_df())])
    @patch("scripts.validation_notebook_utils.load_csv", return_value=_raw_df())
    def test_raw_mode_skips_duplicate_dataset_file_by_default(self, *_mocks) -> None:
        with tempfile.TemporaryDirectory() as td:
            res = vnu.run_manual_validation(
                input_path="dummy.csv",
                output_root=Path(td),
                params=self.params,
                input_mode="raw",
            )
            run_dir = Path(res["run_dir"])
            assert (run_dir / "streamlit_brut_with_injections.csv").exists()
            assert not (run_dir / "dataset_with_injections.csv").exists()

    @patch("scripts.validation_notebook_utils.extract_detected_episodes", return_value=pd.DataFrame())
    @patch("scripts.validation_notebook_utils.evaluate_injected_events", return_value=_eval_df())
    @patch("scripts.validation_notebook_utils.inject_manual_plan_raw", return_value=(_raw_df(), _events_df()))
    @patch("scripts.validation_notebook_utils.run_pipeline_herd", side_effect=[(_summary_df(), _pred_df()), (_summary_df(), _pred_df())])
    @patch("scripts.validation_notebook_utils.load_csv", return_value=_raw_df())
    def test_raw_mode_can_save_dataset_when_explicitly_requested(self, *_mocks) -> None:
        with tempfile.TemporaryDirectory() as td:
            res = vnu.run_manual_validation(
                input_path="dummy.csv",
                output_root=Path(td),
                params=self.params,
                input_mode="raw",
                save_dataset_with_injections=True,
            )
            run_dir = Path(res["run_dir"])
            assert (run_dir / "streamlit_brut_with_injections.csv").exists()
            assert (run_dir / "dataset_with_injections.csv").exists()


class TestRobustWrappersArtifacts(unittest.TestCase):
    def test_v4_wrapper_disables_base_generic_best_streamlit_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run_v4"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"dummy": 1}]).to_csv(run_dir / "robust_summary.csv", index=False)
            base_return = {
                "run_dir": run_dir,
                "runs": pd.DataFrame(),
                "events": pd.DataFrame(),
                "summary": pd.DataFrame(),
                "baseline_summary": pd.DataFrame(),
                "baseline_episodes": pd.DataFrame(),
            }
            with patch("scripts.validation_notebook_utils_v4.base_utils.run_robust_validation", return_value=base_return) as mocked:
                vnu4.run_robust_validation_v4(
                    raw_input_path="dummy.csv",
                    output_root=Path(td),
                    base_params={},
                    seeds=[11],
                    scenarios=["detectable_strong"],
                    grid={
                        "contamination": [0.05],
                        "persist_hours": [6],
                        "alert_min": [2],
                        "mix_mode": ["MIX"],
                        "mix_rate_thr": [0.2],
                    },
                )
                assert mocked.call_args.kwargs["write_best_streamlit_artifacts"] is False
                assert not (run_dir / "robust_summary.csv").exists()
                assert (run_dir / "robust_summary_base_engine.csv").exists()

    def test_v5_1_wrapper_disables_base_generic_best_streamlit_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run_v5_1"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"dummy": 1}]).to_csv(run_dir / "robust_summary.csv", index=False)
            base_return = {
                "run_dir": run_dir,
                "runs": pd.DataFrame(),
                "events": pd.DataFrame(),
                "summary": pd.DataFrame(),
                "baseline_summary": pd.DataFrame(),
                "baseline_episodes": pd.DataFrame(),
            }
            with patch("scripts.validation_notebook_utils_v5_1.base_utils.run_robust_validation", return_value=base_return) as mocked:
                vnu5_1.run_robust_validation_v5_1(
                    raw_input_path="dummy.csv",
                    output_root=Path(td),
                    base_params={},
                    seeds=[11],
                    scenarios=["detectable_strong"],
                    grid={
                        "contamination": [0.05],
                        "persist_hours": [6],
                        "alert_min": [2],
                        "mix_mode": ["MIX"],
                        "mix_rate_thr": [0.2],
                        "mi_z_high_thr": [2.2],
                    },
                )
                assert mocked.call_args.kwargs["write_best_streamlit_artifacts"] is False
                assert not (run_dir / "robust_summary.csv").exists()
                assert (run_dir / "robust_summary_base_engine.csv").exists()

    def test_v4_wrapper_accepts_new_base_summary_filename_without_legacy_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run_v4_new_base"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"dummy": 1}]).to_csv(run_dir / "robust_summary_base_engine.csv", index=False)
            base_return = {
                "run_dir": run_dir,
                "runs": pd.DataFrame(),
                "events": pd.DataFrame(),
                "summary": pd.DataFrame(),
                "baseline_summary": pd.DataFrame(),
                "baseline_episodes": pd.DataFrame(),
            }
            with patch("scripts.validation_notebook_utils_v4.base_utils.run_robust_validation", return_value=base_return):
                vnu4.run_robust_validation_v4(
                    raw_input_path="dummy.csv",
                    output_root=Path(td),
                    base_params={},
                    seeds=[11],
                    scenarios=["detectable_strong"],
                    grid={
                        "contamination": [0.05],
                        "persist_hours": [6],
                        "alert_min": [2],
                        "mix_mode": ["MIX"],
                        "mix_rate_thr": [0.2],
                    },
                )
                assert (run_dir / "robust_summary_base_engine.csv").exists()
                assert not (run_dir / "robust_summary.csv").exists()

    def test_v5_1_wrapper_accepts_new_base_summary_filename_without_legacy_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run_v5_1_new_base"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"dummy": 1}]).to_csv(run_dir / "robust_summary_base_engine.csv", index=False)
            base_return = {
                "run_dir": run_dir,
                "runs": pd.DataFrame(),
                "events": pd.DataFrame(),
                "summary": pd.DataFrame(),
                "baseline_summary": pd.DataFrame(),
                "baseline_episodes": pd.DataFrame(),
            }
            with patch("scripts.validation_notebook_utils_v5_1.base_utils.run_robust_validation", return_value=base_return):
                vnu5_1.run_robust_validation_v5_1(
                    raw_input_path="dummy.csv",
                    output_root=Path(td),
                    base_params={},
                    seeds=[11],
                    scenarios=["detectable_strong"],
                    grid={
                        "contamination": [0.05],
                        "persist_hours": [6],
                        "alert_min": [2],
                        "mix_mode": ["MIX"],
                        "mix_rate_thr": [0.2],
                        "mi_z_high_thr": [2.2],
                    },
                )
                assert (run_dir / "robust_summary_base_engine.csv").exists()
                assert not (run_dir / "robust_summary.csv").exists()

    def test_v4_wrapper_forwards_injection_controls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run_v4_fixed_inj"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"dummy": 1}]).to_csv(run_dir / "robust_summary_base_engine.csv", index=False)
            base_return = {
                "run_dir": run_dir,
                "runs": pd.DataFrame(),
                "events": pd.DataFrame(),
                "summary": pd.DataFrame(),
                "baseline_summary": pd.DataFrame(),
                "baseline_episodes": pd.DataFrame(),
            }
            with patch("scripts.validation_notebook_utils_v4.base_utils.run_robust_validation", return_value=base_return) as mocked:
                vnu4.run_robust_validation_v4(
                    raw_input_path="dummy.csv",
                    output_root=Path(td),
                    base_params={},
                    seeds=[11],
                    scenarios=["detectable_strong"],
                    grid={
                        "contamination": [0.05],
                        "persist_hours": [6],
                        "alert_min": [2],
                        "mix_mode": ["MIX"],
                        "mix_rate_thr": [0.2],
                    },
                    injection_mode="fixed",
                    fixed_detect_cow="8081",
                    fixed_hidden_cow="8154",
                )
                assert mocked.call_args.kwargs["injection_mode"] == "fixed"
                assert mocked.call_args.kwargs["fixed_detect_cow"] == "8081"
                assert mocked.call_args.kwargs["fixed_hidden_cow"] == "8154"

    def test_v5_1_wrapper_forwards_injection_controls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run_v5_1_fixed_inj"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"dummy": 1}]).to_csv(run_dir / "robust_summary_base_engine.csv", index=False)
            base_return = {
                "run_dir": run_dir,
                "runs": pd.DataFrame(),
                "events": pd.DataFrame(),
                "summary": pd.DataFrame(),
                "baseline_summary": pd.DataFrame(),
                "baseline_episodes": pd.DataFrame(),
            }
            with patch("scripts.validation_notebook_utils_v5_1.base_utils.run_robust_validation", return_value=base_return) as mocked:
                vnu5_1.run_robust_validation_v5_1(
                    raw_input_path="dummy.csv",
                    output_root=Path(td),
                    base_params={},
                    seeds=[11],
                    scenarios=["detectable_strong"],
                    grid={
                        "contamination": [0.05],
                        "persist_hours": [6],
                        "alert_min": [2],
                        "mix_mode": ["MIX"],
                        "mix_rate_thr": [0.2],
                        "mi_z_high_thr": [2.2],
                    },
                    injection_mode="fixed",
                    fixed_detect_cow="8081",
                    fixed_hidden_cow="8154",
                )
                assert mocked.call_args.kwargs["injection_mode"] == "fixed"
                assert mocked.call_args.kwargs["fixed_detect_cow"] == "8081"
                assert mocked.call_args.kwargs["fixed_hidden_cow"] == "8154"

    def test_v4_wrapper_is_compatible_with_older_base_signature(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run_v4_oldsig"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"dummy": 1}]).to_csv(run_dir / "robust_summary.csv", index=False)
            base_return = {
                "run_dir": run_dir,
                "runs": pd.DataFrame(),
                "events": pd.DataFrame(),
                "summary": pd.DataFrame(),
                "baseline_summary": pd.DataFrame(),
                "baseline_episodes": pd.DataFrame(),
            }

            def old_base_run(
                *,
                raw_input_path,
                output_root,
                base_params,
                seeds,
                scenarios,
                grid,
                raw_source_for_streamlit=None,
                verbose=False,
                progress_every=1,
                detectable_events_per_run=1,
                go_mode="legacy",
                go_thresholds=None,
                checkpoint_every=25,
            ):
                return base_return

            with patch("scripts.validation_notebook_utils_v4.base_utils.run_robust_validation", new=old_base_run):
                out = vnu4.run_robust_validation_v4(
                    raw_input_path="dummy.csv",
                    output_root=Path(td),
                    base_params={},
                    seeds=[11],
                    scenarios=["detectable_strong"],
                    grid={
                        "contamination": [0.05],
                        "persist_hours": [6],
                        "alert_min": [2],
                        "mix_mode": ["MIX"],
                        "mix_rate_thr": [0.2],
                    },
                )
                assert "summary_v4" in out
                assert (run_dir / "robust_summary_v4.csv").exists()

    def test_v5_1_wrapper_is_compatible_with_older_base_signature(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run_v5_1_oldsig"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"dummy": 1}]).to_csv(run_dir / "robust_summary.csv", index=False)
            base_return = {
                "run_dir": run_dir,
                "runs": pd.DataFrame(),
                "events": pd.DataFrame(),
                "summary": pd.DataFrame(),
                "baseline_summary": pd.DataFrame(),
                "baseline_episodes": pd.DataFrame(),
            }

            def old_base_run(
                *,
                raw_input_path,
                output_root,
                base_params,
                seeds,
                scenarios,
                grid,
                raw_source_for_streamlit=None,
                verbose=False,
                progress_every=1,
                detectable_events_per_run=1,
                go_mode="legacy",
                go_thresholds=None,
                checkpoint_every=25,
            ):
                return base_return

            with patch("scripts.validation_notebook_utils_v5_1.base_utils.run_robust_validation", new=old_base_run):
                out = vnu5_1.run_robust_validation_v5_1(
                    raw_input_path="dummy.csv",
                    output_root=Path(td),
                    base_params={},
                    seeds=[11],
                    scenarios=["detectable_strong"],
                    grid={
                        "contamination": [0.05],
                        "persist_hours": [6],
                        "alert_min": [2],
                        "mix_mode": ["MIX"],
                        "mix_rate_thr": [0.2],
                        "mi_z_high_thr": [2.2],
                    },
                )
                assert "summary_v5_1" in out
                assert (run_dir / "robust_summary_v5_1.csv").exists()


if __name__ == "__main__":
    unittest.main()
