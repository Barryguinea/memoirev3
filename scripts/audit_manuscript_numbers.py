"""Verify the principal numerical claims reported in the manuscript.

The script reads the final CSV/JSON artifacts, compares them with the rounded values
printed in the manuscript, and writes a traceable audit table. It is intentionally
limited to empirical results; numbers taken from the literature remain controlled by
their cited sources.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/revalidation/manuscript_number_audit.csv"


def close(actual: float, expected: float, tolerance: float = 5e-4) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


records: list[dict[str, object]] = []


def check(claim: str, actual: float, expected: float, source: str, tolerance: float = 5e-4) -> None:
    passed = close(actual, expected, tolerance)
    records.append(
        {
            "claim": claim,
            "manuscript_value": expected,
            "artifact_value": float(actual),
            "source": source,
            "passed": passed,
        }
    )
    if not passed:
        raise AssertionError(f"{claim}: manuscript={expected}, artifact={actual}")


# Corpus and descriptive statistics.
raw_path = ROOT / "data/brut.csv"
raw = pd.read_csv(raw_path)
check("Corpus rows", len(raw), 37839, "data/brut.csv", 0)
check("Corpus cows", raw["Cow"].nunique(), 28, "data/brut.csv", 0)
start = pd.to_datetime(raw["Start"])
check("Corpus duration days", (start.max() - start.min()).total_seconds() / 86400, 41.2, "data/brut.csv", 0.051)
for column, mean, std, minimum, maximum in (
    ("Steps", 4.0, 15.4, 0, 387),
    ("Motion Index", 25.5, 103.7, 0, 3351),
    ("Transitions", 0.16, 0.40, 0, 6),
):
    check(f"{column} mean", raw[column].mean(), mean, "data/brut.csv", 0.051)
    check(f"{column} standard deviation", raw[column].std(), std, "data/brut.csv", 0.051)
    check(f"{column} minimum", raw[column].min(), minimum, "data/brut.csv", 0)
    check(f"{column} maximum", raw[column].max(), maximum, "data/brut.csv", 0)

provenance = pd.read_csv(ROOT / "data/revalidation/hypo_module/cowalert_provenance.csv").set_index("variable")
for claim, variable, column, expected in (
    ("CowAlert matched rows", "n_observations_appariees", "n", 7631),
    ("CowAlert source rows", "n_observations_appariees", "n_source", 7823),
    ("CowAlert matched percentage", "n_observations_appariees", "appariement_pct", 97.5),
    ("CowAlert matched cows", "n_observations_appariees", "n_vaches", 5),
    ("CowAlert Steps correlation", "Steps", "correlation", 1.0),
    ("CowAlert Motion Index correlation", "Motion Index", "correlation", 1.0),
    ("CowAlert Transitions correlation", "Transitions", "correlation", 1.0),
):
    check(claim, provenance.loc[variable, column], expected, "hypo_module/cowalert_provenance.csv", 5e-4)

# HYPO primary campaign.
hypo_dir = ROOT / "data/revalidation/hypo_module"
detection = pd.read_csv(hypo_dir / "detection_summary.csv").set_index("scenario")
for scenario, expected in {
    "gradual_mild": (11, 0.455, 0.727, 0.273, 0.087),
    "gradual_moderate": (11, 0.636, 0.909, 0.364, 0.192),
    "gradual_marked": (11, 0.636, 1.000, 0.545, 0.269),
    "isolated_short_variation": (11, 0.000, 0.000, 0.000, 0.000),
}.items():
    row = detection.loc[scenario]
    for column, value in zip(
        ("n_events", "new_warning_start_mean", "warning_coverage_mean", "detect_iou20_mean", "best_iou_mean"),
        expected,
        strict=True,
    ):
        check(f"HYPO {scenario} {column}", row[column], value, "hypo_module/detection_summary.csv", 5e-4)

events = pd.read_csv(hypo_dir / "events_primary.csv")
check("HYPO eligible cows", events["cow"].nunique(), 11, "hypo_module/events_primary.csv", 0)
check("HYPO events", events["event_id"].nunique(), 44, "hypo_module/events_primary.csv", 0)

config_path = hypo_dir / "parameter_sensitivity/parameter_sensitivity_configurations.json"
config = json.loads(config_path.read_text())
reference_config = config["configurations"][0]
for claim, actual, expected in (
    ("Baseline ratio", config["campaign_params"]["baseline_ratio"], 0.60),
    ("HYPO aggregation hours", reference_config["aggregation_hours"], 12.0),
    ("HYPO persistence hours", reference_config["persistence_hours"], 6.0),
    ("HYPO refractory period hours", reference_config["cooldown_hours"], 24.0),
    ("HYPO activity minimum change", reference_config["family_min_change"], 0.10),
    ("HYPO posture minimum change", reference_config["posture_min_change"], 0.05),
    ("HYPO score threshold", reference_config["score_threshold"], 0.12),
    ("HYPO CUSUM drift", reference_config["cusum_drift"], 0.07),
    ("HYPO CUSUM threshold", reference_config["cusum_threshold"], 1.20),
    ("HYPO minimum families", reference_config["min_families"], 3),
    ("HYPO minimum coverage percentage", reference_config["coverage_min_pct"], 25.0),
):
    check(claim, actual, expected, "hypo_module/parameter_sensitivity/parameter_sensitivity_configurations.json")

sensitivity = pd.read_csv(hypo_dir / "parameter_sensitivity/parameter_sensitivity_summary.csv")
reference = sensitivity.loc[sensitivity["configuration"] == "reference"].iloc[0]
for claim, column, expected in (
    ("HYPO new onset", "progressive_new_start_rate", 0.5758),
    ("HYPO attributable coverage", "progressive_attributable_coverage_rate", 0.8788),
    ("HYPO IoU20", "progressive_iou20_rate", 0.3939),
    ("HYPO mean IoU", "progressive_best_iou_mean", 0.1827),
    ("HYPO median onset delay hours", "progressive_onset_delay_median_hours", 20.75),
    ("HYPO background per cow-day", "background_per_cow_day_mean", 0.4016),
    ("HYPO background P90", "background_per_cow_day_p90", 0.4913),
):
    check(claim, reference[column], expected, "hypo_module/parameter_sensitivity/parameter_sensitivity_summary.csv")

check("OFAT configurations including reference", len(sensitivity), 21, "hypo_module/parameter_sensitivity/parameter_sensitivity_summary.csv", 0)
check("OFAT event evaluations", int(sensitivity["n_events"].sum()), 924, "hypo_module/parameter_sensitivity/parameter_sensitivity_summary.csv", 0)
for claim, column, expected_min, expected_max in (
    ("OFAT new onset range", "progressive_new_start_rate", 0.455, 0.667),
    ("OFAT coverage range", "progressive_attributable_coverage_rate", 0.636, 0.909),
    ("OFAT IoU20 range", "progressive_iou20_rate", 0.273, 0.455),
    ("OFAT delay range", "progressive_onset_delay_median_hours", 15.5, 24.5),
    ("OFAT background range", "background_per_cow_day_mean", 0.187, 0.553),
):
    check(f"{claim} minimum", sensitivity[column].min(), expected_min, "hypo_module/parameter_sensitivity/parameter_sensitivity_summary.csv", 5e-4)
    check(f"{claim} maximum", sensitivity[column].max(), expected_max, "hypo_module/parameter_sensitivity/parameter_sensitivity_summary.csv", 5e-4)
check("OFAT brief control maximum", sensitivity["control_attributable_coverage_rate"].max(), 0.0, "hypo_module/parameter_sensitivity/parameter_sensitivity_summary.csv", 0)

bootstrap = pd.read_csv(hypo_dir / "bootstrap_ci.csv").set_index(["analyse", "cible"])
for target, point, low, high in (
    ("couverture_attribuable", 0.8788, 0.7879, 0.9697),
    ("nouveau_depart", 0.5758, 0.4242, 0.7576),
    ("iou20", 0.3939, 0.2424, 0.5455),
):
    row = bootstrap.loc[("hypo_metrique", target)]
    check(f"Bootstrap {target} point", row["valeur"], point, "hypo_module/bootstrap_ci.csv")
    check(f"Bootstrap {target} lower", row["ic95_bas"], low, "hypo_module/bootstrap_ci.csv")
    check(f"Bootstrap {target} upper", row["ic95_haut"], high, "hypo_module/bootstrap_ci.csv")

calibration = pd.read_csv(hypo_dir / "per_cow_calibration.csv").set_index("indicateur")
for indicator, reference_value, calibrated_value in (
    ("Fond moyen / vache-jour", 0.4016, 0.4016),
    ("Fond CV inter-vache", 0.2437, 0.2437),
    ("Couverture (graduels)", 0.8788, 0.8788),
    ("Nouveau depart (graduels)", 0.5758, 0.6061),
    ("IoU20 (graduels)", 0.3939, 0.3333),
):
    row = calibration.loc[indicator]
    check(f"Calibration {indicator} reference", row["reference"], reference_value, "hypo_module/per_cow_calibration.csv")
    check(f"Calibration {indicator} calibrated", row["calibre"], calibrated_value, "hypo_module/per_cow_calibration.csv")
calibration_cows = pd.read_csv(hypo_dir / "per_cow_calibration_by_cow.csv")
check("Calibration factor minimum", calibration_cows["calibration_factor"].min(), 0.98, "hypo_module/per_cow_calibration_by_cow.csv", 0.011)
check("Calibration factor maximum", calibration_cows["calibration_factor"].max(), 1.03, "hypo_module/per_cow_calibration_by_cow.csv", 0.011)

curve = pd.read_csv(hypo_dir / "detection_background_curve/detection_background_curve.csv").set_index("score_threshold")
for threshold, background, coverage, onset, iou20 in (
    (0.04, 0.419487, 0.878788, 0.575758, 0.393939),
    (0.12, 0.401636, 0.878788, 0.575758, 0.393939),
    (0.14, 0.374861, 0.878788, 0.575758, 0.454545),
    (0.20, 0.330234, 0.787879, 0.545455, 0.303030),
):
    row = curve.loc[threshold]
    for column, expected in (
        ("background_per_cow_day", background),
        ("attributable_coverage", coverage),
        ("new_start_rate", onset),
        ("iou20_rate", iou20),
        ("brief_control_rejection", 1.0),
    ):
        check(f"Detection-background threshold {threshold} {column}", row[column], expected, "hypo_module/detection_background_curve/detection_background_curve.csv")

failure = pd.read_csv(hypo_dir / "failure_modes.csv").set_index("mode")
for mode, numerator, denominator in (
    ("gradual_uncovered", 4, 33),
    ("gradual_poorly_localized", 16, 33),
    ("covered_without_novel_start", 10, 29),
    ("nonlocomotor_hypoactivity_alerted", 9, 11),
    ("pure_instability_in_surveillance", 22, 33),
    ("pure_instability_actionable_episode", 0, 33),
    ("brief_control_alerted", 0, 11),
):
    check(f"Failure mode {mode} numerator", failure.loc[mode, "numerator"], numerator, "hypo_module/failure_modes.csv", 0)
    check(f"Failure mode {mode} denominator", failure.loc[mode, "denominator"], denominator, "hypo_module/failure_modes.csv", 0)

ablation = pd.read_csv(hypo_dir / "ablation_summary.csv").set_index("variante")
for name, expected in {
    "A. Alerte temporelle multivariée": (0.432, 0.295, 0.137, 0.402),
    "B. IF + règles de persistance": (0.159, 0.000, 0.004, 0.152),
    "C. IF seul": (0.136, 0.000, 0.002, 7.658),
    "D. LOF + règles": (0.159, 0.000, 0.005, 0.107),
    "E. Comparateur pédométrique (pas seuls)": (0.273, 0.318, 0.141, 0.518),
}.items():
    row = ablation.loc[name]
    for column, value in zip(("detect_any", "iou20", "best_iou", "false_notif_cow_day"), expected, strict=True):
        check(f"Ablation {name} {column}", row[column], value, "hypo_module/ablation_summary.csv")

ablation_tests = pd.read_csv(hypo_dir / "ablation_tests_by_cow.csv").set_index("paire")
for pair, p_detection, p_iou in (
    ("A vs B", 0.04297, 0.00098),
    ("A vs C", 0.01562, 0.00098),
    ("A vs D", 0.04297, 0.00098),
    ("A vs E", 0.06250, 0.96582),
):
    check(f"Ablation {pair} detection p", ablation_tests.loc[pair, "p_wilcoxon_detection"], p_detection, "hypo_module/ablation_tests_by_cow.csv")
    check(f"Ablation {pair} IoU p", ablation_tests.loc[pair, "p_wilcoxon_iou"], p_iou, "hypo_module/ablation_tests_by_cow.csv")

# Extended bidirectional campaign.
fusion = pd.read_csv(ROOT / "data/revalidation/v3_refined_full/comparison_summary.csv")
fusion = fusion.loc[fusion["experiment"] == "fusion"].set_index("configuration")
hierarchical = fusion.loc["hierarchical"]
for claim, column, expected in (
    ("Hierarchical actionable coverage", "actionable_detection", 0.7273),
    ("Hierarchical instability surveillance", "instability_surveillance_detection", 0.6667),
    ("Hierarchical sequence coverage", "sequence_actionable_detection", 0.9091),
    ("Hierarchical confound alert rate", "confound_alert_rate", 0.4091),
    ("Hierarchical IoU20", "actionable_iou20", 0.2955),
    ("Hierarchical background", "background_per_cow_day", 0.5711),
):
    check(claim, hierarchical[column], expected, "v3_refined_full/comparison_summary.csv")

for name, expected in {
    "hypo_only": (0.6136, 0.6667, 0.4091, 0.2955, 0.4461),
    "instability_only": (0.3864, 0.6667, 0.0000, 0.0000, 0.8748),
    "or": (0.6591, 0.6667, 0.3182, 0.1136, 0.9819),
    "hierarchical": (0.7273, 0.6667, 0.4091, 0.2955, 0.5711),
    "sequential_24_72h": (0.5682, 0.6667, 0.3636, 0.0000, 0.3658),
}.items():
    row = fusion.loc[name]
    for column, value in zip(
        ("actionable_detection", "instability_surveillance_detection", "confound_alert_rate", "actionable_iou20", "background_per_cow_day"),
        expected,
        strict=True,
    ):
        check(f"Fusion {name} {column}", row[column], value, "v3_refined_full/comparison_summary.csv")
check("Hierarchical utility", hierarchical["technical_utility"], -0.0884, "v3_refined_full/comparison_summary.csv")
check("Hierarchical median delay", hierarchical["actionable_delay_h_median"], 19.875, "v3_refined_full/comparison_summary.csv")

hierarchical_events = pd.read_csv(ROOT / "data/revalidation/v3_refined_full/events_fusion_hierarchical.csv")
hierarchical_events["hybrid_new_start"] = (hierarchical_events["hybrid_novel_start_count"] > 0).astype(float)
hierarchical_events["instability_new_start"] = (hierarchical_events["instability_novel_start_count"] > 0).astype(float)
scenario_rates = hierarchical_events.groupby("scenario")[["hybrid_new_start", "instability_new_start"]].mean()
for scenario, verification, surveillance in (
    ("hypo_mild", 0.455, 0.000),
    ("hypo_moderate", 0.727, 0.182),
    ("hypo_marked", 0.818, 0.545),
    ("instability_mild", 0.091, 0.455),
    ("instability_moderate", 0.091, 0.545),
    ("instability_marked", 0.182, 1.000),
    ("instability_then_hypo", 0.909, 0.818),
    ("isolated_sensor_spike", 0.000, 0.000),
    ("short_exercise", 0.000, 0.000),
    ("handling_manipulation", 0.000, 0.000),
    ("estrus_like_activity", 0.000, 0.000),
    ("nonlocomotor_hypoactivity", 0.818, 0.000),
):
    check(f"Scenario {scenario} verification", scenario_rates.loc[scenario, "hybrid_new_start"], verification, "v3_refined_full/events_fusion_hierarchical.csv", 5e-4)
    check(f"Scenario {scenario} surveillance", scenario_rates.loc[scenario, "instability_new_start"], surveillance, "v3_refined_full/events_fusion_hierarchical.csv", 5e-4)
brief_controls = hierarchical_events.loc[
    hierarchical_events["scenario"].isin(["isolated_sensor_spike", "short_exercise", "handling_manipulation"])
]
check("Brief controls verification rate", brief_controls["hybrid_new_start"].mean(), 0.000, "v3_refined_full/events_fusion_hierarchical.csv", 0)
check("Brief controls surveillance rate", brief_controls["instability_new_start"].mean(), 0.000, "v3_refined_full/events_fusion_hierarchical.csv", 0)
background_by_cow = (
    hierarchical_events.groupby("cow").first()["hybrid_background_notifs"]
    / hierarchical_events.groupby("cow").first()["monitoring_days"]
)
check("Hierarchical background minimum", background_by_cow.min(), 0.295, "v3_refined_full/events_fusion_hierarchical.csv", 5e-4)
check("Hierarchical background maximum", background_by_cow.max(), 0.884, "v3_refined_full/events_fusion_hierarchical.csv", 5e-4)

fusion_sensitivity = pd.read_csv(ROOT / "data/revalidation/v3_refined_full/comparison_summary.csv")
fusion_sensitivity = fusion_sensitivity.loc[fusion_sensitivity["experiment"] == "parameter_sensitivity"]
for claim, column, expected_min, expected_max in (
    ("Fusion sensitivity verification range", "actionable_detection", 0.727, 0.750),
    ("Fusion sensitivity surveillance range", "instability_surveillance_detection", 0.545, 0.667),
    ("Fusion sensitivity background range", "background_per_cow_day", 0.518, 0.571),
):
    check(f"{claim} minimum", fusion_sensitivity[column].min(), expected_min, "v3_refined_full/comparison_summary.csv", 5e-4)
    check(f"{claim} maximum", fusion_sensitivity[column].max(), expected_max, "v3_refined_full/comparison_summary.csv", 5e-4)

# Observation against SLS scores.
sls = json.loads((ROOT / "data/revalidation/v3_mcgill_sls/mcgill_v3_summary.json").read_text())
check("SLS evaluable cows", sls["cohort"]["n_evaluable"], 14, "v3_mcgill_sls/mcgill_v3_summary.json", 0)
check("SLS cows >=2", sls["cohort"]["n_sls_ge_2"], 3, "v3_mcgill_sls/mcgill_v3_summary.json", 0)
primary = sls["primary_metrics"][0]
for claim, key, expected in (
    ("SLS notifications mean >=2", "mean_sls_ge_2", 6.6667),
    ("SLS notifications mean <2", "mean_sls_lt_2", 4.4545),
    ("SLS exploratory AUC", "auc", 0.9242),
    ("SLS Mann-Whitney p", "mann_whitney_p", 0.0306),
    ("SLS Spearman rho", "spearman_rho", 0.5044),
    ("SLS Spearman p", "spearman_p", 0.0659),
):
    check(claim, primary[key], expected, "v3_mcgill_sls/mcgill_v3_summary.json")
for index, label, expected_auc in (
    (1, "SLS time fraction AUC", 0.7121),
    (2, "SLS maximum score AUC", 0.6061),
    (3, "SLS instability surveillance AUC", 0.3485),
):
    check(label, sls["primary_metrics"][index]["auc"], expected_auc, "v3_mcgill_sls/mcgill_v3_summary.json")
sls_cohort = pd.read_csv(ROOT / "data/revalidation/v3_mcgill_sls/mcgill_v3_cohort_all_variants.csv")
sls_hierarchical = sls_cohort.loc[sls_cohort["variant"] == "hierarchical"]
check("SLS future notification load", sls_hierarchical["future_hybrid_notif_per_cow_day"].mean(), 0.7551, "v3_mcgill_sls/mcgill_v3_cohort_all_variants.csv")

# Runtime benchmark.
perf = json.loads((ROOT / "data/revalidation/performance_v3_full_corpus.json").read_text())
for claim, key, expected in (
    ("Benchmark cows", "dataset_cows", 28),
    ("Benchmark rows", "dataset_rows", 37839),
    ("Benchmark repeats", "repeats", 5),
    ("Benchmark median seconds", "total_seconds_median", 10.64),
    ("Benchmark minimum seconds", "total_seconds_min", 10.58),
    ("Benchmark maximum seconds", "total_seconds_max", 10.83),
    ("Benchmark rows per second", "rows_per_sec_median", 3557),
    ("Feature engineering share", "features_share_of_core_median", 0.082),
    ("IF share of instrumented core", "if_share_of_core_median", 0.828),
    ("Hybrid share of instrumented core", "hybrid_share_of_core_median", 0.064),
    ("Comparative IF rules share", "legacy_alerts_share_of_core_median", 0.025),
):
    tolerance = 1.0 if key == "rows_per_sec_median" else 0.0051
    check(claim, perf[key], expected, "performance_v3_full_corpus.json", tolerance)

audit = pd.DataFrame(records)
OUT.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUT, index=False)
print(f"Verified {len(audit)} manuscript values; audit written to {OUT.relative_to(ROOT)}")
