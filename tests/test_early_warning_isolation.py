import numpy as np
import pandas as pd

from core.early_warning import EarlyWarningConfig, apply_behavioral_early_warning


def _warning_frame(*, hidden_activity_drop: bool, declared_split: bool = True) -> pd.DataFrame:
    periods = 192
    split = int(round(0.60 * periods))
    future = np.arange(periods) >= split
    activity = np.full(periods, 100.0)
    if hidden_activity_drop:
        activity[future] = 5.0
    frame = pd.DataFrame(
        {
            "T": pd.date_range("2024-01-01", periods=periods, freq="15min"),
            "Steps_sum": activity,
            "Motion Index_sum": activity * 2,
            "Transitions_sum": activity / 20,
            "Lying Time_sum": np.where(future, 50.0, 20.0),
            "Standing Time_sum": np.where(future, 50.0, 80.0),
            "coverage_pct": 100.0,
        }
    )
    if declared_split:
        frame["dataset_split"] = np.where(future, "futur", "baseline")
    else:
        frame["dataset_split"] = "all"
    return frame


def _posture_config() -> EarlyWarningConfig:
    return EarlyWarningConfig(
        aggregation_hours=2.0,
        persistence_hours=1.0,
        cooldown_hours=2.0,
        posture_min_change=0.01,
        score_threshold=0.01,
        cusum_drift=0.0,
        cusum_threshold=0.10,
        min_families=1,
        active_families=("posture",),
    )


def test_posture_only_ablation_ignores_hidden_activity_channels():
    stable = apply_behavioral_early_warning(
        _warning_frame(hidden_activity_drop=False), interval="15T", config=_posture_config()
    )
    dropped = apply_behavioral_early_warning(
        _warning_frame(hidden_activity_drop=True), interval="15T", config=_posture_config()
    )
    for column in (
        "behavioral_warning_score",
        "behavioral_warning_cusum",
        "behavioral_warning_episode",
        "behavioral_warning_notification",
    ):
        np.testing.assert_allclose(stable[column], dropped[column])


def test_missing_split_labels_fall_back_to_chronological_reference():
    explicit = apply_behavioral_early_warning(
        _warning_frame(hidden_activity_drop=False), interval="15T", config=_posture_config()
    )
    fallback = apply_behavioral_early_warning(
        _warning_frame(hidden_activity_drop=False, declared_split=False),
        interval="15T",
        config=_posture_config(),
    )
    np.testing.assert_allclose(
        explicit["behavioral_warning_episode"], fallback["behavioral_warning_episode"]
    )
