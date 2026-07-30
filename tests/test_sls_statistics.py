import pandas as pd
import pytest

from validation_hybrid.mcgill_sls_validation import (
    _exact_permutation_auc,
    treatment_sensitivity,
)


def test_exact_permutation_auc_with_ties() -> None:
    frame = pd.DataFrame(
        {
            "sls_ge_2": [1, 1, 1, 0, 0, 0],
            "pre7_hybrid_notifs": [7, 7, 6, 3, 7, 3],
        }
    )
    result = _exact_permutation_auc(frame, score="pre7_hybrid_notifs")

    assert result["auc"] == pytest.approx(7 / 9)
    assert result["exact_permutation_p_one_sided"] == pytest.approx(0.20)
    assert result["exact_permutation_p_two_sided"] == pytest.approx(0.40)
    assert result["n_permutations"] == 20


def test_treatment_sensitivity_reports_unestimable_stratum() -> None:
    frame = pd.DataFrame(
        {
            "sls_ge_2": [1, 1, 1, 0, 0, 0, 0],
            "pre7_hybrid_notifs": [7, 7, 6, 3, 7, 3, 4],
            "treatment": [
                "Exercise",
                "Exercise",
                "Exercise",
                "Exercise",
                "Exercise",
                "Exercise",
                "No_Exercise",
            ],
        }
    )
    sensitivity, diagnostics = treatment_sensitivity(frame)
    no_exercise = sensitivity[sensitivity["analysis"].eq("no_exercise_only")].iloc[0]

    assert pd.isna(no_exercise["auc"])
    assert diagnostics["treatment_sls_fisher_p_two_sided"] is not None
