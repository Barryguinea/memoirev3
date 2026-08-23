"""Invariants qui protegent la separation entre periode de reference et periode future.

Ces deux proprietes sont au fondement du protocole d'evaluation : la reference doit
ignorer la periode future, et le cumul CUSUM doit rester unilateral. Elles etaient
verifiables par lecture du code mais aucun test ne les exercait, de sorte qu'une
regression y serait passee inapercue.
"""

import numpy as np
import pandas as pd

from core.early_warning import (
    EarlyWarningConfig,
    _baseline_expected_by_slot,
    _one_sided_cusum,
    apply_behavioral_early_warning,
)


def _serie(future_value: float) -> pd.DataFrame:
    """Serie de 192 intervalles dont seules les valeurs futures changent."""
    periods = 192
    split = int(round(0.60 * periods))
    future = np.arange(periods) >= split
    activity = np.where(future, future_value, 100.0)
    return pd.DataFrame(
        {
            "T": pd.date_range("2024-01-01", periods=periods, freq="15min"),
            "Steps_sum": activity,
            "Motion Index_sum": activity * 2,
            "Transitions_sum": activity / 20,
            "Lying Time_sum": np.where(future, 50.0, 20.0),
            "Standing Time_sum": np.where(future, 50.0, 80.0),
            "coverage_pct": 100.0,
            "dataset_split": np.where(future, "futur", "baseline"),
        }
    )


def test_reference_ignores_future_values() -> None:
    """La mediane de reference ne doit pas bouger quand la periode future change."""
    attendus = []
    for future_value in (100.0, 1.0, 10_000.0):
        frame = _serie(future_value)
        times = pd.to_datetime(frame["T"])
        slots = times.dt.hour * 60 + times.dt.minute
        mask = frame["dataset_split"].eq("baseline")
        attendus.append(
            _baseline_expected_by_slot(frame["Steps_sum"], slots, mask).loc[mask].to_numpy()
        )
    reference = attendus[0]
    for autre in attendus[1:]:
        np.testing.assert_allclose(autre, reference)


def test_pipeline_outputs_on_baseline_do_not_depend_on_the_future() -> None:
    """Bout en bout : les sorties de la periode de reference sont insensibles au futur."""
    config = EarlyWarningConfig()
    sorties = []
    for future_value in (100.0, 1.0, 10_000.0):
        frame = apply_behavioral_early_warning(
            _serie(future_value), interval="15min", config=config
        )
        mask = frame["dataset_split"].eq("baseline")
        sorties.append(frame.loc[mask, "warning_ratio_steps"].fillna(-1.0).to_numpy())
    for autre in sorties[1:]:
        np.testing.assert_allclose(autre, sorties[0])


def test_cusum_stays_one_sided() -> None:
    """Le cumul part de zero, n'y descend jamais et se reinitialise apres une baisse."""
    evidence = pd.Series([0.0, 0.0, -5.0, 0.0, 0.30, 0.30, 0.0, -2.0, 0.0])
    cumul = _one_sided_cusum(evidence, drift=0.07)
    assert (cumul >= 0.0).all(), "le CUSUM unilateral ne doit jamais devenir negatif"
    assert cumul.iloc[3] == 0.0, "une evidence tres negative doit ramener le cumul a zero"
    assert cumul.iloc[5] > cumul.iloc[4] > 0.0, "le cumul doit croitre sur une evidence positive"
