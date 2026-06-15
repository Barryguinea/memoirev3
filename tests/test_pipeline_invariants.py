import unittest
from pathlib import Path

from core.io import load_csv
from core.pipeline import run_pipeline_herd


class TestPipelineInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        data_path = root / "data" / "brut.csv"
        df = load_csv(str(data_path))
        cls.summary, cls.out = run_pipeline_herd(
            df,
            interval="15T",
            window_baseline=24,
            contamination=0.08,
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
            max_cows=5,
        )

    def test_output_non_empty(self):
        self.assertGreater(len(self.out), 0)
        self.assertGreater(len(self.summary), 0)

    def test_notification_only_on_lameness_start(self):
        bad = ((self.out["notif_lameness"] == 1) & (self.out["pred_lameness_start"] != 1)).sum()
        self.assertEqual(int(bad), 0)

    def test_lameness_start_implies_episode(self):
        bad = ((self.out["pred_lameness_start"] == 1) & (self.out["pred_lameness_episode"] != 1)).sum()
        self.assertEqual(int(bad), 0)

    def test_low_coverage_gates_signals(self):
        low = self.out["coverage_pct"] < 25.0
        if_anom_bad = ((low) & (self.out["if_anomaly_point"] == 1)).sum()
        lame_bad = ((low) & (self.out["pred_lameness_episode"] == 1)).sum()
        self.assertEqual(int(if_anom_bad), 0)
        self.assertEqual(int(lame_bad), 0)

    def test_train_point_is_candidate(self):
        bad = ((self.out["if_train_point"] == 1) & (self.out["if_train_candidate"] != 1)).sum()
        self.assertEqual(int(bad), 0)

    def test_expected_columns_exist(self):
        required = {
            "dataset_split",
            "if_train_candidate",
            "if_train_point",
            "if_anomaly_point",
            "pred_lameness_episode",
            "notif_lameness",
        }
        self.assertTrue(required.issubset(set(self.out.columns)))

    def test_behavioral_warning_is_held_out_and_non_diagnostic(self):
        required = {
            "behavioral_warning_score",
            "behavioral_warning_episode",
            "behavioral_warning_start",
            "behavioral_warning_notification",
            "behavioral_warning_scope",
        }
        self.assertTrue(required.issubset(set(self.out.columns)))
        baseline = self.out["dataset_split"].astype(str) == "baseline"
        self.assertEqual(int(self.out.loc[baseline, "behavioral_warning_episode"].sum()), 0)
        self.assertTrue(
            self.out["behavioral_warning_scope"].astype(str).str.contains("non diagnostique").all()
        )


if __name__ == "__main__":
    unittest.main()
