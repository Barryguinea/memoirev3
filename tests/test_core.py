"""
Tests unitaires pour le projet Lameness Detection.
Donnees synthetiques uniquement (pas de dependance a brut.csv).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd

from core.features import interval_to_minutes, robust_z, rolling_robust_z, _pandas_freq
from core.alerts import apply_alert_logic, _feature_family
from core.io import normalize_columns, _duration_to_minutes, COW, TIME


# ============================================================
# interval_to_minutes
# ============================================================
class TestIntervalToMinutes(unittest.TestCase):
    def test_minutes_T(self):
        self.assertEqual(interval_to_minutes("15T"), 15)
        self.assertEqual(interval_to_minutes("30T"), 30)

    def test_minutes_MIN(self):
        self.assertEqual(interval_to_minutes("15MIN"), 15)
        self.assertEqual(interval_to_minutes("30min"), 30)

    def test_hours(self):
        self.assertEqual(interval_to_minutes("1H"), 60)
        self.assertEqual(interval_to_minutes("2H"), 120)
        self.assertEqual(interval_to_minutes("4H"), 240)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            interval_to_minutes("abc")
        with self.assertRaises(ValueError):
            interval_to_minutes("15D")


# ============================================================
# _pandas_freq
# ============================================================
class TestPandasFreq(unittest.TestCase):
    def test_T_to_min(self):
        self.assertEqual(_pandas_freq("15T"), "15min")
        self.assertEqual(_pandas_freq("30T"), "30min")

    def test_H_unchanged(self):
        self.assertEqual(_pandas_freq("1H"), "1H")

    def test_already_min(self):
        # "15min" ne finit pas par T majuscule, donc retourne tel quel
        self.assertEqual(_pandas_freq("15min"), "15min")


# ============================================================
# robust_z
# ============================================================
class TestRobustZ(unittest.TestCase):
    def test_constant_returns_zero(self):
        x = np.array([5.0] * 20)
        z = robust_z(x)
        np.testing.assert_array_almost_equal(z, np.zeros(20))

    def test_known_outlier_detected(self):
        x = np.array([0.0] * 19 + [100.0])
        z = robust_z(x)
        self.assertGreater(z[-1], 2.0, "L'outlier doit avoir un z-score eleve")

    def test_nan_produces_finite_output(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        z = robust_z(x)
        self.assertTrue(np.all(np.isfinite(z)), "robust_z ne doit jamais retourner NaN/Inf")

    def test_symmetry(self):
        x = np.array([0.0] * 10 + [10.0] * 10)
        z = robust_z(x)
        # Les deux groupes doivent avoir des z-scores opposes
        self.assertGreater(z[-1], 0)
        self.assertLess(z[0], 0)


# ============================================================
# rolling_robust_z
# ============================================================
class TestRollingRobustZ(unittest.TestCase):
    def test_output_length_preserved(self):
        s = pd.Series(np.random.randn(100))
        z = rolling_robust_z(s, window_size=20)
        self.assertEqual(len(z), 100)

    def test_no_nan_in_output(self):
        s = pd.Series(np.random.randn(100))
        z = rolling_robust_z(s, window_size=20)
        self.assertFalse(z.isna().any(), "rolling_robust_z ne doit pas contenir de NaN")

    def test_small_window(self):
        s = pd.Series(np.random.randn(30))
        z = rolling_robust_z(s, window_size=10)
        self.assertEqual(len(z), 30)
        self.assertTrue(np.all(np.isfinite(z)))


# ============================================================
# _feature_family
# ============================================================
class TestFeatureFamily(unittest.TestCase):
    def test_activity(self):
        self.assertEqual(_feature_family("Motion Index_sum_rrz"), "activity")
        self.assertEqual(_feature_family("Steps_sum_rrz"), "activity")

    def test_rest(self):
        self.assertEqual(_feature_family("Lying Time_sum_rrz"), "rest")
        self.assertEqual(_feature_family("Standing Time_sum_rrz"), "rest")

    def test_transitions(self):
        self.assertEqual(_feature_family("Transitions_sum_rrz"), "transitions")

    def test_other(self):
        self.assertEqual(_feature_family("unknown_col"), "other")
        self.assertEqual(_feature_family("random_metric_rrz"), "other")


# ============================================================
# _duration_to_minutes
# ============================================================
class TestDurationToMinutes(unittest.TestCase):
    def test_pure_numeric(self):
        s = pd.Series([10.0, 20.5, 0.0])
        result = _duration_to_minutes(s)
        pd.testing.assert_series_equal(result, pd.Series([10.0, 20.5, 0.0]))

    def test_hms_format(self):
        s = pd.Series(["1:30:00", "0:15:00", "2:00:30"])
        result = _duration_to_minutes(s)
        self.assertAlmostEqual(result.iloc[0], 90.0)       # 1h30m
        self.assertAlmostEqual(result.iloc[1], 15.0)       # 0h15m
        self.assertAlmostEqual(result.iloc[2], 120.5)      # 2h00m30s

    def test_ms_format(self):
        s = pd.Series(["15:30", "5:00"])
        result = _duration_to_minutes(s)
        self.assertAlmostEqual(result.iloc[0], 15.5)       # 15m30s
        self.assertAlmostEqual(result.iloc[1], 5.0)        # 5m00s

    def test_nan_passthrough(self):
        s = pd.Series([np.nan, "1:00:00", np.nan])
        result = _duration_to_minutes(s)
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertAlmostEqual(result.iloc[1], 60.0)
        self.assertTrue(pd.isna(result.iloc[2]))


# ============================================================
# normalize_columns
# ============================================================
class TestNormalizeColumns(unittest.TestCase):
    def test_synonym_rename(self):
        df = pd.DataFrame({
            "ID": ["cow1", "cow1"],
            "Timestamp": ["2024-01-01 00:00", "2024-01-01 01:00"],
            "Steps": [100, 200],
        })
        out = normalize_columns(df)
        self.assertIn(COW, out.columns)
        self.assertIn(TIME, out.columns)

    def test_negative_clamp(self):
        df = pd.DataFrame({
            "Cow": ["c1"],
            "T": ["2024-01-01"],
            "Steps": [-5],
        })
        out = normalize_columns(df)
        self.assertEqual(out["Steps"].iloc[0], 0)

    def test_transitions_computed_from_up_down(self):
        df = pd.DataFrame({
            "Cow": ["c1", "c1"],
            "T": ["2024-01-01 00:00", "2024-01-01 01:00"],
            "Transitions Up": [3, 5],
            "Transitions Down": [2, 4],
        })
        out = normalize_columns(df)
        self.assertIn("Transitions", out.columns)
        self.assertEqual(out["Transitions"].iloc[0], 5)
        self.assertEqual(out["Transitions"].iloc[1], 9)

    def test_missing_cow_raises(self):
        df = pd.DataFrame({"T": ["2024-01-01"], "Steps": [10]})
        with self.assertRaises(ValueError):
            normalize_columns(df)


# ============================================================
# BUG 1 — Tests de régression : contamination de la fenêtre glissante
# ============================================================
class TestCoverageGatingBug1(unittest.TestCase):
    """Verifie que les anomalies sur intervalles faible couverture
    ne contaminent PAS les rolling windows des intervalles normaux."""

    def _make_df(self, n=50):
        np.random.seed(42)
        return pd.DataFrame({
            "T": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "if_anomaly_point": np.zeros(n, dtype=int),
            "if_score": np.random.randn(n) * 0.1,
            "if_score_pct": np.random.uniform(10, 90, n),
            "coverage_pct": np.full(n, 80.0),
            "Motion Index_sum_log_rrz": np.random.randn(n) * 0.5,
            "Steps_sum_rrz": np.random.randn(n) * 0.5,
            "Lying Time_sum_rrz": np.random.randn(n) * 0.5,
            "Standing Time_sum_rrz": np.random.randn(n) * 0.5,
            "Transitions_sum_rrz": np.random.randn(n) * 0.5,
        })

    def test_low_cov_anomaly_does_not_contaminate_rolling(self):
        """Une anomalie IF sur un intervalle faible couverture ne doit PAS
        incrementer if_anom_k sur les intervalles normaux adjacents."""
        df = self._make_df(50)
        # Row 10: faible couverture mais marquee anomalie
        df.loc[10, "coverage_pct"] = 5.0
        df.loc[10, "if_anomaly_point"] = 1

        result = apply_alert_logic(
            df, time_col="T", interval="1H",
            persist_hours=6, alert_min=2,
            coverage_min_pct=25.0,
        )

        # Row 10 elle-meme doit etre a zero
        self.assertEqual(result.loc[10, "if_anomaly_point"], 0)
        # Row 11 ne doit PAS voir cette anomalie dans son rolling
        self.assertEqual(result.loc[11, "if_anom_k"], 0)
        # Row 15 non plus (dans la fenetre de 6h)
        self.assertEqual(result.loc[15, "if_anom_k"], 0)

    def test_low_cov_family_does_not_contaminate_rolling(self):
        """Les signaux familles sur faible couverture ne doivent pas
        contaminer n_families_k."""
        df = self._make_df(50)
        # Row 5: faible couverture avec z-scores extremes
        df.loc[5, "coverage_pct"] = 5.0
        df.loc[5, "Motion Index_sum_log_rrz"] = 10.0  # spike enorme
        df.loc[5, "Lying Time_sum_rrz"] = -10.0

        result = apply_alert_logic(
            df, time_col="T", interval="1H",
            persist_hours=6, alert_min=2,
            coverage_min_pct=25.0,
        )

        # Row 5: familles doivent etre a zero
        self.assertEqual(result.loc[5, "fam_activity"], 0)
        self.assertEqual(result.loc[5, "fam_rest"], 0)

    def test_low_cov_mi_spike_does_not_contaminate_rolling(self):
        """mi_spike sur faible couverture ne doit pas contaminer mi_spike_k."""
        df = self._make_df(50)
        df.loc[8, "coverage_pct"] = 5.0
        df.loc[8, "Motion Index_sum_log_rrz"] = 10.0  # spike

        result = apply_alert_logic(
            df, time_col="T", interval="1H",
            persist_hours=6, alert_min=2,
            coverage_min_pct=25.0, mi_z_high_thr=2.5,
        )

        # Row 8: mi_spike doit etre a zero
        self.assertEqual(result.loc[8, "mi_spike"], 0)
        # Row 9: mi_spike_k ne doit pas etre contaminee
        self.assertEqual(result.loc[9, "mi_spike_k"], 0)


# ============================================================
# lame_confidence
# ============================================================
class TestLameConfidence(unittest.TestCase):
    def _make_df(self, n=50):
        np.random.seed(42)
        return pd.DataFrame({
            "T": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "if_anomaly_point": np.zeros(n, dtype=int),
            "if_score": np.random.randn(n) * 0.1,
            "if_score_pct": np.random.uniform(10, 90, n),
            "coverage_pct": np.full(n, 80.0),
            "Motion Index_sum_log_rrz": np.random.randn(n) * 0.5,
            "Steps_sum_rrz": np.random.randn(n) * 0.5,
            "Lying Time_sum_rrz": np.random.randn(n) * 0.5,
            "Standing Time_sum_rrz": np.random.randn(n) * 0.5,
            "Transitions_sum_rrz": np.random.randn(n) * 0.5,
        })

    def test_no_nan_in_lame_confidence(self):
        """lame_confidence ne doit jamais contenir NaN."""
        df = self._make_df(50)
        df.loc[5, "if_score"] = np.nan
        df.loc[5, "if_score_pct"] = np.nan
        df.loc[10, "coverage_pct"] = 5.0

        result = apply_alert_logic(
            df, time_col="T", interval="1H",
            coverage_min_pct=25.0,
        )

        self.assertFalse(
            result["lame_confidence"].isna().any(),
            f"NaN trouve dans lame_confidence aux index: "
            f"{result[result['lame_confidence'].isna()].index.tolist()}"
        )

    def test_zero_on_low_coverage(self):
        """lame_confidence doit etre 0.0 sur les intervalles faible couverture."""
        df = self._make_df(50)
        df.loc[5, "coverage_pct"] = 5.0
        df.loc[20, "coverage_pct"] = 0.0

        result = apply_alert_logic(
            df, time_col="T", interval="1H",
            coverage_min_pct=25.0,
        )

        self.assertEqual(result.loc[5, "lame_confidence"], 0.0)
        self.assertEqual(result.loc[20, "lame_confidence"], 0.0)

    def test_range_0_100(self):
        """lame_confidence doit etre dans [0, 100]."""
        df = self._make_df(100)
        result = apply_alert_logic(
            df, time_col="T", interval="1H",
            coverage_min_pct=25.0,
        )

        self.assertTrue((result["lame_confidence"] >= 0).all())
        self.assertTrue((result["lame_confidence"] <= 100).all())


if __name__ == "__main__":
    unittest.main()
