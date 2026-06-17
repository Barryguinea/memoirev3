from validation_hybrid.sensitivity import FUSION_VARIANTS, PARAMETER_CONFIGURATIONS


def test_sensitivity_configurations_are_named_and_prespecified():
    assert list(FUSION_VARIANTS) == [
        "hypo_only",
        "instability_only",
        "or",
        "hierarchical",
        "sequential_24_72h",
    ]
    assert list(PARAMETER_CONFIGURATIONS) == [
        "reference",
        "persistence_3h",
        "thresholds_strict",
        "aggregation_4h",
    ]
    for instability in PARAMETER_CONFIGURATIONS.values():
        assert instability.persistence_hours > 0
        assert instability.min_families >= 2
    assert FUSION_VARIANTS["hierarchical"].mode == "HIERARCHICAL"
    assert FUSION_VARIANTS["sequential_24_72h"].sequence_min_hours == 24.0
