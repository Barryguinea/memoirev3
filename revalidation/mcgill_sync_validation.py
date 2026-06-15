"""Evaluation externe exploratoire synchronisee sur McGill Winter 2019.

Le score SLS du 12 mars 2019 est l'unique point d'evaluation. Le detecteur est
execute sur des donnees strictement anterieures a cette date, avec sa
configuration gelee et sans utiliser le SLS pour l'entrainement ou le reglage
des seuils. La fenetre primaire est constituee des sept jours precedant le
score. L'unite statistique est la vache.

Cette analyse mesure une concordance exploratoire avec un score observationnel
SLS. Elle ne constitue ni une validation diagnostique, ni une estimation de
sensibilite ou de specificite clinique.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

from core.io import COW, TIME, normalize_columns
from core.pipeline import run_pipeline_one_cow
from revalidation.campaign import final_params


MEMOIREV2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MEMOIREV2_ROOT.parent
MCGILL_ROOT = PROJECT_ROOT / "mcgill_iot_cattle"
SLS_XLSX = MCGILL_ROOT / (
    "Données completes/Données accelerometres/Winter 2019/Icetag/"
    "IceTags_Data/IceTags-issues and reports/Exercise Study - SLS Scores.xlsx"
)
SENSORS = (
    MCGILL_ROOT
    / "reports/objective1_pipeline_icetag/winter_2019_pipeline_input_15min.csv"
)
PUBLISHED = MCGILL_ROOT / "data-movement-behavior-main/dataset/Winter2019.csv"
OUT = MEMOIREV2_ROOT / "data/revalidation/mcgill"

SCORE_TIME = pd.Timestamp("2019-03-12 00:00:00")
PRIMARY_WINDOW_DAYS = 7
MIN_BASELINE_DAYS = 20
MIN_PRIMARY_COVERAGE = 0.70
EXPECTED_BINS_PER_DAY = 96
SLS_EXPLORATORY_THRESHOLD = 2
SLS_COMPONENTS = ["Edge", "Rest", "Shiftwt", "Uneven"]


def _clean_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".0", "", regex=False).str.strip()


def _load_one_sls_sheet(sheet: str) -> pd.Series:
    data = pd.read_excel(SLS_XLSX, sheet_name=sheet)
    data = data[["Cow", *SLS_COMPONENTS]].dropna(subset=["Cow"]).copy()
    data["Cow"] = _clean_id(data["Cow"])
    for column in SLS_COMPONENTS:
        data[column] = pd.to_numeric(data[column], errors="raise")
        invalid = ~data[column].isin([0, 1])
        if invalid.any():
            raise ValueError(f"Composante SLS non binaire dans {sheet}: {column}")

    duplicate_counts = data.groupby("Cow")[SLS_COMPONENTS].nunique(dropna=False)
    if (duplicate_counts > 1).any().any():
        raise ValueError(f"Evaluations SLS dupliquees et discordantes dans {sheet}")

    data["sls"] = data[SLS_COMPONENTS].sum(axis=1)
    return data.groupby("Cow")["sls"].first()


def load_sls() -> pd.DataFrame:
    """Charge les scores SLS 0--4, sans transformer le seuil en diagnostic."""
    scores = pd.DataFrame(
        {
            "sls_jan": _load_one_sls_sheet("Baseline - 15JAN19"),
            "sls_mar": _load_one_sls_sheet("Midway - 12MAR19"),
        }
    )
    scores.index.name = "cow"
    return scores.reset_index()


def load_treatment() -> pd.Series:
    data = pd.read_csv(PUBLISHED, sep=";")
    data["Cow_ID"] = _clean_id(data["Cow_ID"])
    treatment = data.groupby("Cow_ID")["Trt"].agg(
        lambda values: values.mode().iloc[0] if len(values.mode()) else values.iloc[0]
    )
    treatment.index.name = "cow"
    return treatment


def _safe_auc(frame: pd.DataFrame, score: str) -> float | None:
    clean = frame.dropna(subset=["sls_ge_2", score])
    labels = clean["sls_ge_2"].astype(int)
    if labels.nunique() < 2:
        return None
    return float(roc_auc_score(labels, clean[score]))


def _group_test(frame: pd.DataFrame, score: str) -> Dict[str, float | int | None]:
    positive = frame.loc[frame["sls_ge_2"] == 1, score].dropna()
    negative = frame.loc[frame["sls_ge_2"] == 0, score].dropna()
    if positive.empty or negative.empty:
        return {
            "metric": score,
            "n_sls_ge_2": int(len(positive)),
            "n_sls_lt_2": int(len(negative)),
            "mean_sls_ge_2": None,
            "mean_sls_lt_2": None,
            "auc": None,
            "mann_whitney_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }

    mw = mannwhitneyu(positive, negative, alternative="two-sided", method="auto")
    rho, rho_p = spearmanr(frame["sls_mar"], frame[score], nan_policy="omit")
    return {
        "metric": score,
        "n_sls_ge_2": int(len(positive)),
        "n_sls_lt_2": int(len(negative)),
        "mean_sls_ge_2": float(positive.mean()),
        "mean_sls_lt_2": float(negative.mean()),
        "auc": _safe_auc(frame, score),
        "mann_whitney_p": float(mw.pvalue),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
    }


def run_detector_per_cow(
    params: Dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute le detecteur sur la periode strictement anterieure au score."""
    params = params or final_params()
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
        primary_coverage = len(primary_raw) / (
            PRIMARY_WINDOW_DAYS * EXPECTED_BINS_PER_DAY
        )
        reasons: list[str] = []
        if baseline_days < MIN_BASELINE_DAYS:
            reasons.append("baseline_trop_courte")
        if primary_coverage < MIN_PRIMARY_COVERAGE:
            reasons.append("couverture_pre7_insuffisante")
        if reasons:
            exclusions.append(
                {
                    "cow": cow,
                    "first_sensor_time": first,
                    "last_sensor_time_before_score": last,
                    "baseline_days_before_pre7": round(baseline_days, 3),
                    "primary_coverage": round(primary_coverage, 4),
                    "exclusion_reason": ";".join(reasons),
                }
            )
            continue

        prediction = run_pipeline_one_cow(sensors, cow, **params)
        prediction[TIME] = pd.to_datetime(prediction[TIME], errors="raise")
        future = prediction[prediction["dataset_split"].astype(str).eq("futur")]
        primary = prediction[
            (prediction[TIME] >= primary_start) & (prediction[TIME] < SCORE_TIME)
        ]
        future_days = max(
            1.0 / EXPECTED_BINS_PER_DAY,
            len(future) / EXPECTED_BINS_PER_DAY,
        )

        rows.append(
            {
                "cow": cow,
                "first_sensor_time": first,
                "last_sensor_time_before_score": last,
                "baseline_end": prediction.loc[
                    prediction["dataset_split"].astype(str).eq("baseline"), TIME
                ].max(),
                "primary_coverage": primary_coverage,
                "pre7_warning_notifs": int(
                    pd.to_numeric(
                        primary["behavioral_warning_notification"], errors="coerce"
                    ).fillna(0).sum()
                ),
                "pre7_warning_frac_time": float(
                    pd.to_numeric(
                        primary["behavioral_warning_episode"], errors="coerce"
                    ).fillna(0).mean()
                ),
                "pre7_warning_score_max": float(
                    pd.to_numeric(
                        primary["behavioral_warning_score"], errors="coerce"
                    ).max()
                ),
                "future_notif_per_cow_day": float(
                    pd.to_numeric(
                        future["behavioral_warning_notification"], errors="coerce"
                    ).fillna(0).sum()
                    / future_days
                ),
                "future_warning_frac_time": float(
                    pd.to_numeric(
                        future["behavioral_warning_episode"], errors="coerce"
                    ).fillna(0).mean()
                ),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(exclusions)


def build_cohort(
    params: Dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detector, exclusions = run_detector_per_cow(params)
    cohort = detector.merge(load_sls(), on="cow", how="left")
    missing_sls = cohort[cohort["sls_mar"].isna()]["cow"].tolist()
    if missing_sls:
        extra = pd.DataFrame(
            {
                "cow": missing_sls,
                "exclusion_reason": "score_sls_12_mars_absent",
            }
        )
        exclusions = pd.concat([exclusions, extra], ignore_index=True, sort=False)
        cohort = cohort[cohort["sls_mar"].notna()].copy()

    treatment = load_treatment()
    cohort["treatment"] = cohort["cow"].map(treatment)
    cohort["sls_ge_2"] = (
        cohort["sls_mar"] >= SLS_EXPLORATORY_THRESHOLD
    ).astype(int)
    return cohort.sort_values(["sls_mar", "cow"], ascending=[False, True]), exclusions


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort, exclusions = build_cohort()
    cohort.to_csv(OUT / "mcgill_cohort.csv", index=False)
    exclusions.to_csv(OUT / "mcgill_exclusions.csv", index=False)

    labelled_treatment = cohort.dropna(subset=["treatment"])
    confound = pd.crosstab(
        labelled_treatment["sls_ge_2"], labelled_treatment["treatment"]
    ).reindex(index=[0, 1], columns=["Exercise", "No_Exercise"], fill_value=0)
    confound.index = ["SLS<2", "SLS>=2"]
    confound.to_csv(OUT / "mcgill_confound.csv")
    fisher = fisher_exact(confound.to_numpy(), alternative="two-sided")

    metric_names = [
        "pre7_warning_notifs",
        "pre7_warning_frac_time",
        "pre7_warning_score_max",
    ]
    metrics = pd.DataFrame([_group_test(cohort, metric) for metric in metric_names])
    metrics.to_csv(OUT / "mcgill_metrics.csv", index=False)

    summary = {
        "protocol": {
            "score_endpoint": str(SCORE_TIME),
            "sensor_data_used": "strictement avant le 12 mars 2019",
            "primary_window_days": PRIMARY_WINDOW_DAYS,
            "minimum_baseline_days": MIN_BASELINE_DAYS,
            "minimum_primary_coverage": MIN_PRIMARY_COVERAGE,
            "detector_configuration": "gelee; aucun reglage sur les SLS",
            "statistical_unit": "vache",
        },
        "cohort": {
            "n_evaluable": int(len(cohort)),
            "n_sls_ge_2": int(cohort["sls_ge_2"].sum()),
            "n_sls_lt_2": int((cohort["sls_ge_2"] == 0).sum()),
            "sls_distribution": {
                str(int(score)): int(count)
                for score, count in cohort["sls_mar"].value_counts().sort_index().items()
            },
            "n_excluded": int(len(exclusions)),
            "sls_scale": "0..4; somme de Edge, Rest, Shiftwt et Uneven binaires",
            "threshold_status": (
                "SLS>=2 utilise uniquement pour une comparaison exploratoire; "
                "seuil clinique non documente dans le classeur"
            ),
        },
        "primary_metric": metrics.iloc[0].to_dict(),
        "secondary_metrics": metrics.iloc[1:].to_dict(orient="records"),
        "treatment_confounding": {
            "table": {
                str(index): {str(column): int(value) for column, value in row.items()}
                for index, row in confound.iterrows()
            },
            "fisher_exact_odds_ratio": float(fisher.statistic),
            "fisher_exact_p": float(fisher.pvalue),
            "interpretation": (
                "Les trois SLS>=2 appartiennent au bras Exercise; aucun SLS>=2 "
                "n'est observe dans No_Exercise. Le traitement est donc fortement "
                "entremele au statut SLS, sans etre une correspondance parfaite."
            ),
        },
        "background": {
            "future_notif_per_cow_day_mean": float(
                cohort["future_notif_per_cow_day"].mean()
            ),
            "future_warning_frac_time_mean": float(
                cohort["future_warning_frac_time"].mean()
            ),
        },
        "verdict": (
            "Concordance exploratoire non concluante: seulement trois SLS>=2, "
            "tous dans Exercise, seuil SLS non valide cliniquement et taux "
            "d'alerte de fond eleve. Les resultats ne valident pas la detection "
            "clinique de boiterie."
        ),
    }
    (OUT / "mcgill_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
