"""Validation observationnelle exploratoire du prototype final sur McGill Winter 2019.

Les signaux IceTag sont strictement antérieurs au score SLS du 12 mars 2019.
Le SLS n'est utilisé ni pour entraîner le détecteur ni pour régler ses seuils.
L'unité statistique est la vache. La petite cohorte et la confusion avec le
traitement Exercise interdisent toute estimation clinique de sensibilité ou de
spécificité; cette analyse teste uniquement la concordance observationnelle.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

from core.hybrid_warning import HybridFusionConfig, InstabilityWarningConfig
from core.io import COW, TIME, normalize_columns
from core.pipeline import run_pipeline_one_cow
from validation_hypo.mcgill_sync_validation import (
    EXPECTED_BINS_PER_DAY,
    MIN_BASELINE_DAYS,
    MIN_PRIMARY_COVERAGE,
    PRIMARY_WINDOW_DAYS,
    SCORE_TIME,
    SENSORS,
    SLS_EXPLORATORY_THRESHOLD,
    load_sls,
    load_treatment,
)
from validation_hybrid.campaign import final_params
from validation_hybrid.sensitivity import FUSION_VARIANTS


def _clean_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".0", "", regex=False).str.strip()


def _safe_auc(frame: pd.DataFrame, score: str) -> float | None:
    clean = frame.dropna(subset=["sls_ge_2", score])
    if clean["sls_ge_2"].nunique() < 2:
        return None
    return float(roc_auc_score(clean["sls_ge_2"].astype(int), clean[score]))


def _group_metric(frame: pd.DataFrame, score: str) -> dict[str, object]:
    positive = frame.loc[frame["sls_ge_2"] == 1, score].dropna()
    negative = frame.loc[frame["sls_ge_2"] == 0, score].dropna()
    rho, rho_p = spearmanr(frame["sls_mar"], frame[score], nan_policy="omit")
    if positive.empty or negative.empty:
        p_value = None
    else:
        p_value = float(
            mannwhitneyu(positive, negative, alternative="two-sided", method="auto").pvalue
        )
    return {
        "metric": score,
        "n_sls_ge_2": int(len(positive)),
        "n_sls_lt_2": int(len(negative)),
        "mean_sls_ge_2": float(positive.mean()) if len(positive) else None,
        "mean_sls_lt_2": float(negative.mean()) if len(negative) else None,
        "auc": _safe_auc(frame, score),
        "mann_whitney_p": p_value,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
    }


def _exact_permutation_auc(
    frame: pd.DataFrame,
    *,
    score: str,
) -> dict[str, float | int | None]:
    clean = frame.dropna(subset=["sls_ge_2", score]).copy()
    labels = clean["sls_ge_2"].astype(int).to_numpy()
    values = pd.to_numeric(clean[score], errors="coerce").to_numpy(float)
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return {
            "n": int(len(labels)),
            "n_sls_ge_2": n_positive,
            "n_sls_lt_2": n_negative,
            "auc": None,
            "exact_permutation_p_one_sided": None,
            "exact_permutation_p_two_sided": None,
            "n_permutations": 0,
        }

    observed = float(roc_auc_score(labels, values))
    permutation_aucs = []
    for positive_indices in combinations(range(len(labels)), n_positive):
        permuted = np.zeros(len(labels), dtype=int)
        permuted[list(positive_indices)] = 1
        permutation_aucs.append(float(roc_auc_score(permuted, values)))
    permutation_aucs_array = np.asarray(permutation_aucs, dtype=float)
    return {
        "n": int(len(labels)),
        "n_sls_ge_2": n_positive,
        "n_sls_lt_2": n_negative,
        "auc": observed,
        "exact_permutation_p_one_sided": float(
            np.mean(permutation_aucs_array >= observed - 1e-12)
        ),
        "exact_permutation_p_two_sided": float(
            np.mean(
                np.abs(permutation_aucs_array - 0.5)
                >= abs(observed - 0.5) - 1e-12
            )
        ),
        "n_permutations": int(len(permutation_aucs_array)),
    }


def treatment_sensitivity(
    primary: pd.DataFrame,
    *,
    score: str = "pre7_hybrid_notifs",
) -> tuple[pd.DataFrame, dict[str, object]]:
    analyses = []
    for analysis, subset in (
        ("global", primary),
        ("exercise_only", primary[primary["treatment"].eq("Exercise")]),
        ("no_exercise_only", primary[primary["treatment"].eq("No_Exercise")]),
    ):
        analyses.append(
            {"analysis": analysis, **_exact_permutation_auc(subset, score=score)}
        )

    known_treatment = primary.dropna(subset=["treatment", "sls_ge_2"])
    treatment_table = pd.crosstab(
        known_treatment["treatment"],
        known_treatment["sls_ge_2"],
    ).reindex(index=["Exercise", "No_Exercise"], columns=[0, 1], fill_value=0)
    fisher_p = (
        float(fisher_exact(treatment_table.to_numpy(), alternative="two-sided").pvalue)
        if treatment_table.shape == (2, 2)
        else None
    )

    leave_one_out_aucs = []
    for index in primary.index[primary["sls_ge_2"].eq(1)]:
        auc = _safe_auc(primary.drop(index=index), score)
        if auc is not None:
            leave_one_out_aucs.append(auc)
    diagnostics = {
        "score": score,
        "known_treatment_n": int(len(known_treatment)),
        "treatment_sls_fisher_p_two_sided": fisher_p,
        "leave_one_positive_out_auc_min": (
            float(min(leave_one_out_aucs)) if leave_one_out_aucs else None
        ),
        "leave_one_positive_out_auc_max": (
            float(max(leave_one_out_aucs)) if leave_one_out_aucs else None
        ),
        "interpretation": (
            "Signal descriptif global positif; après restriction au groupe Exercise, "
            "la direction persiste mais l'inférence est non concluante."
        ),
    }
    return pd.DataFrame(analyses), diagnostics


def multiplicity_sensitivity(
    all_cohorts: pd.DataFrame,
    *,
    primary_variant: str = "hierarchical",
    primary_metric: str = "pre7_hybrid_notifs",
) -> pd.DataFrame:
    """Test max-stat exact pour les variantes et métriques observées dans la cohorte SLS."""
    metrics = [
        "pre7_hybrid_notifs",
        "pre7_hybrid_frac_time",
        "pre7_hybrid_score_max",
        "pre7_instability_surveillance_frac",
    ]
    primary = (
        all_cohorts[all_cohorts["variant"].eq(primary_variant)]
        .sort_values("cow")
        .reset_index(drop=True)
    )
    cows = primary["cow"].astype(str).tolist()
    labels = primary["sls_ge_2"].astype(int).to_numpy()
    n_positive = int(labels.sum())
    if n_positive == 0 or n_positive == len(labels):
        raise ValueError("Le test max-stat exige deux classes SLS.")

    score_vectors: dict[tuple[str, str], np.ndarray] = {}
    for variant, cohort in all_cohorts.groupby("variant"):
        indexed = cohort.assign(cow=cohort["cow"].astype(str)).set_index("cow").reindex(cows)
        for metric in metrics:
            values = pd.to_numeric(indexed[metric], errors="raise").to_numpy(float)
            score_vectors[(str(variant), metric)] = values

    primary_values = score_vectors[(primary_variant, primary_metric)]
    observed_primary_auc = float(roc_auc_score(labels, primary_values))
    families = {
        "notifications_5_variantes": [
            key for key in score_vectors if key[1] == "pre7_hybrid_notifs"
        ],
        "toutes_20_combinaisons": list(score_vectors),
    }
    rows: list[dict[str, object]] = []
    for family, keys in families.items():
        max_one_sided: list[float] = []
        max_two_sided: list[float] = []
        for positive_indices in combinations(range(len(labels)), n_positive):
            permuted = np.zeros(len(labels), dtype=int)
            permuted[list(positive_indices)] = 1
            aucs = np.asarray(
                [roc_auc_score(permuted, score_vectors[key]) for key in keys],
                dtype=float,
            )
            max_one_sided.append(float(aucs.max()))
            max_two_sided.append(float(np.abs(aucs - 0.5).max()))
        unique_vectors = {
            tuple(np.round(score_vectors[key], 12).tolist())
            for key in keys
        }
        rows.append(
            {
                "family": family,
                "n_tests": len(keys),
                "n_unique_score_vectors": len(unique_vectors),
                "observed_primary_auc": observed_primary_auc,
                "exact_maxstat_p_one_sided": float(
                    np.mean(np.asarray(max_one_sided) >= observed_primary_auc - 1e-12)
                ),
                "exact_maxstat_p_two_sided": float(
                    np.mean(
                        np.asarray(max_two_sided)
                        >= abs(observed_primary_auc - 0.5) - 1e-12
                    )
                ),
                "n_permutations": len(max_one_sided),
            }
        )
    return pd.DataFrame(rows)


def run_variant(
    name: str,
    fusion: HybridFusionConfig,
    *,
    instability: InstabilityWarningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = final_params()
    sensors = normalize_columns(pd.read_csv(SENSORS))
    sensors[COW] = _clean_id(sensors[COW])
    sensors[TIME] = pd.to_datetime(sensors[TIME], errors="raise")
    sensors = sensors[sensors[TIME] < SCORE_TIME].copy()
    primary_start = SCORE_TIME - pd.Timedelta(days=PRIMARY_WINDOW_DAYS)

    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for cow, raw_cow in sensors.groupby(COW):
        first = raw_cow[TIME].min()
        last = raw_cow[TIME].max()
        baseline_days = (primary_start - first).total_seconds() / 86400.0
        primary_raw = raw_cow[raw_cow[TIME] >= primary_start]
        primary_coverage = len(primary_raw) / (PRIMARY_WINDOW_DAYS * EXPECTED_BINS_PER_DAY)
        reasons: list[str] = []
        if baseline_days < MIN_BASELINE_DAYS:
            reasons.append("baseline_trop_courte")
        if primary_coverage < MIN_PRIMARY_COVERAGE:
            reasons.append("couverture_pre7_insuffisante")
        if reasons:
            exclusions.append(
                {
                    "cow": cow,
                    "variant": name,
                    "first_sensor_time": first,
                    "last_sensor_time_before_score": last,
                    "baseline_days_before_pre7": round(baseline_days, 3),
                    "primary_coverage": round(primary_coverage, 4),
                    "exclusion_reason": ";".join(reasons),
                }
            )
            continue

        prediction = run_pipeline_one_cow(
            sensors,
            cow,
            **params,
            instability_config=instability,
            fusion_config=fusion,
        )
        prediction[TIME] = pd.to_datetime(prediction[TIME], errors="raise")
        primary = prediction[
            (prediction[TIME] >= primary_start) & (prediction[TIME] < SCORE_TIME)
        ]
        future = prediction[prediction["dataset_split"].astype(str).eq("futur")]
        future_days = max(1 / EXPECTED_BINS_PER_DAY, len(future) / EXPECTED_BINS_PER_DAY)
        rows.append(
            {
                "variant": name,
                "cow": cow,
                "first_sensor_time": first,
                "last_sensor_time_before_score": last,
                "primary_coverage": primary_coverage,
                "pre7_hybrid_notifs": int(primary["hybrid_warning_notification"].sum()),
                "pre7_hybrid_frac_time": float(primary["hybrid_warning_episode"].mean()),
                "pre7_hybrid_score_max": float(primary["hybrid_warning_score"].max()),
                "pre7_instability_surveillance_frac": float(
                    primary["hybrid_warning_surveillance"].mean()
                ),
                "pre7_hypo_notifs": int(primary["behavioral_warning_notification"].sum()),
                "pre7_instability_notifs": int(
                    primary["instability_warning_notification"].sum()
                ),
                "future_hybrid_notif_per_cow_day": float(
                    future["hybrid_warning_notification"].sum() / future_days
                ),
            }
        )

    cohort = pd.DataFrame(rows).merge(load_sls(), on="cow", how="left")
    missing = cohort[cohort["sls_mar"].isna()]["cow"].tolist()
    if missing:
        exclusions.extend(
            {"cow": cow, "variant": name, "exclusion_reason": "score_sls_12_mars_absent"}
            for cow in missing
        )
        cohort = cohort[cohort["sls_mar"].notna()].copy()
    cohort["treatment"] = cohort["cow"].map(load_treatment())
    cohort["sls_ge_2"] = (cohort["sls_mar"] >= SLS_EXPLORATORY_THRESHOLD).astype(int)
    return cohort, pd.DataFrame(exclusions)


def run_all(output_dir: str = "data/validation/mcgill_sls") -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cohorts: list[pd.DataFrame] = []
    exclusions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    for name, fusion in FUSION_VARIANTS.items():
        print(f"McGill SLS: {name}")
        cohort, excluded = run_variant(name, fusion)
        cohorts.append(cohort)
        exclusions.append(excluded)
        for metric in [
            "pre7_hybrid_notifs",
            "pre7_hybrid_frac_time",
            "pre7_hybrid_score_max",
            "pre7_instability_surveillance_frac",
        ]:
            row = _group_metric(cohort, metric)
            row["variant"] = name
            metrics.append(row)

    all_cohorts = pd.concat(cohorts, ignore_index=True)
    all_exclusions = pd.concat(exclusions, ignore_index=True, sort=False)
    metric_frame = pd.DataFrame(metrics)
    all_cohorts.to_csv(output / "mcgill_cohort_all_variants.csv", index=False)
    all_exclusions.to_csv(output / "mcgill_exclusions.csv", index=False)
    metric_frame.to_csv(output / "mcgill_metrics.csv", index=False)

    primary = all_cohorts[all_cohorts["variant"] == "hierarchical"]
    treatment_frame, treatment_diagnostics = treatment_sensitivity(primary)
    treatment_frame.to_csv(output / "mcgill_treatment_sensitivity.csv", index=False)
    multiplicity_frame = multiplicity_sensitivity(all_cohorts)
    multiplicity_frame.to_csv(output / "mcgill_multiplicity_sensitivity.csv", index=False)
    treatment_records = (
        treatment_frame.astype(object)
        .where(pd.notna(treatment_frame), None)
        .to_dict(orient="records")
    )
    summary = {
        "protocol": {
            "endpoint": str(SCORE_TIME),
            "sensor_window": "strictement antérieure au score SLS",
            "primary_window_days": PRIMARY_WINDOW_DAYS,
            "statistical_unit": "vache",
            "primary_variant": (
                "hierarchical, retenue comme analyse principale; valeur p globale non ajustée, "
                "avec sensibilité max-stat rapportée séparément"
            ),
            "interpretation": "concordance observationnelle exploratoire, non validation diagnostique",
        },
        "cohort": {
            "n_evaluable": int(len(primary)),
            "n_sls_ge_2": int(primary["sls_ge_2"].sum()),
            "n_sls_lt_2": int((primary["sls_ge_2"] == 0).sum()),
            "sls_distribution": {
                str(int(score)): int(count)
                for score, count in primary["sls_mar"].value_counts().sort_index().items()
            },
            "treatment_by_sls": pd.crosstab(
                primary["sls_ge_2"], primary["treatment"]
            ).to_dict(),
        },
        "primary_metrics": metric_frame[
            metric_frame["variant"] == "hierarchical"
        ].to_dict(orient="records"),
        "treatment_sensitivity": treatment_records,
        "treatment_diagnostics": treatment_diagnostics,
        "multiplicity_sensitivity": multiplicity_frame.to_dict(orient="records"),
        "verdict_rule": (
            "Le classement global et sa stabilité au retrait d'une vache positive sont "
            "encourageants. La sensibilité max-stat et la restriction au groupe Exercise "
            "bornent toutefois l'inférence: aucune variante n'est cliniquement validée."
        ),
    }
    (output / "mcgill_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/validation/mcgill_sls")
    args = parser.parse_args()
    run_all(args.output_dir)


if __name__ == "__main__":
    main()
