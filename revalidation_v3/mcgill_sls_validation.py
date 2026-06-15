"""Validation observationnelle exploratoire du prototype V3 sur McGill Winter 2019.

Les signaux IceTag sont strictement antérieurs au score SLS du 12 mars 2019.
Le SLS n'est utilisé ni pour entraîner le détecteur ni pour régler ses seuils.
L'unité statistique est la vache. La petite cohorte et la confusion avec le
traitement Exercise interdisent toute estimation clinique de sensibilité ou de
spécificité; cette analyse teste uniquement la concordance observationnelle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

from core.hybrid_warning import HybridFusionConfig, InstabilityWarningConfig
from core.io import COW, TIME, normalize_columns
from core.pipeline import run_pipeline_one_cow
from revalidation.mcgill_sync_validation import (
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
from revalidation_v3.campaign import final_params
from revalidation_v3.sensitivity import FUSION_VARIANTS


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


def run_all(output_dir: str = "data/revalidation/v3_mcgill_sls") -> dict[str, object]:
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
    all_cohorts.to_csv(output / "mcgill_v3_cohort_all_variants.csv", index=False)
    all_exclusions.to_csv(output / "mcgill_v3_exclusions.csv", index=False)
    metric_frame.to_csv(output / "mcgill_v3_metrics.csv", index=False)

    primary = all_cohorts[all_cohorts["variant"] == "hierarchical"]
    summary = {
        "protocol": {
            "endpoint": str(SCORE_TIME),
            "sensor_window": "strictement antérieure au score SLS",
            "primary_window_days": PRIMARY_WINDOW_DAYS,
            "statistical_unit": "vache",
            "primary_variant": "hierarchical, prédéfinie sans consulter les SLS",
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
        "verdict_rule": (
            "Aucune variante ne peut être déclarée cliniquement validée avec trois "
            "SLS>=2 et un entrelacement du traitement Exercise."
        ),
    }
    (output / "mcgill_v3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/revalidation/v3_mcgill_sls")
    args = parser.parse_args()
    run_all(args.output_dir)


if __name__ == "__main__":
    main()
