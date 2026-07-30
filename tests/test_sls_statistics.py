import pandas as pd
import pytest

from validation_hybrid.mcgill_sls_validation import (
    _exact_permutation_auc,
    multiplicity_sensitivity,
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


def test_multiplicity_sensitivity_reports_both_families() -> None:
    rows = []
    variants = ["hypo_only", "instability_only", "or", "hierarchical", "sequential_24_72h"]
    labels = [1, 1, 0, 0, 0, 0]
    for variant_index, variant in enumerate(variants):
        for cow, label in enumerate(labels):
            rows.append(
                {
                    "variant": variant,
                    "cow": str(cow),
                    "sls_ge_2": label,
                    "pre7_hybrid_notifs": 6 - cow + variant_index,
                    "pre7_hybrid_frac_time": (6 - cow) / 10,
                    "pre7_hybrid_score_max": (cow + variant_index) / 10,
                    "pre7_instability_surveillance_frac": cow / 10,
                }
            )

    result = multiplicity_sensitivity(pd.DataFrame(rows))

    assert result["family"].tolist() == [
        "notifications_5_variantes",
        "toutes_20_combinaisons",
    ]
    assert result["n_tests"].tolist() == [5, 20]
    assert result["n_permutations"].tolist() == [15, 15]
    assert result["exact_maxstat_p_one_sided"].between(0, 1).all()
