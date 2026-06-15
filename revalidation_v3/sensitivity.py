"""Comparaison des fusions et sensibilité du prototype bidirectionnel V3.

Les scénarios synthétiques vérifient le comportement technique du système. Ils
ne servent pas à estimer une sensibilité clinique. La sélection descriptive
utilise une fonction d'utilité qui pénalise fortement les contrôles, les
facteurs confondants et les notifications de fond.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from core.hybrid_warning import HybridFusionConfig, InstabilityWarningConfig
from revalidation_v3.campaign import run_campaign


FUSION_VARIANTS: dict[str, HybridFusionConfig] = {
    "hypo_only": HybridFusionConfig(mode="HYPO_ONLY"),
    "instability_only": HybridFusionConfig(mode="INSTABILITY_ONLY"),
    "or": HybridFusionConfig(mode="OR"),
    "hierarchical": HybridFusionConfig(mode="HIERARCHICAL"),
    "sequential_24_72h": HybridFusionConfig(
        mode="SEQUENTIAL", sequence_min_hours=24.0, sequence_max_hours=72.0
    ),
}


PARAMETER_CONFIGURATIONS: dict[str, InstabilityWarningConfig] = {
    "reference": InstabilityWarningConfig(),
    "persistence_3h": InstabilityWarningConfig(persistence_hours=3.0),
    "thresholds_strict": InstabilityWarningConfig(
        restless_motion_min_change=0.30,
        transition_density_min_change=0.30,
        fragmentation_min_change=0.30,
        posture_volatility_min_change=0.30,
        score_threshold=0.25,
        cusum_threshold=1.00,
    ),
    "aggregation_4h": InstabilityWarningConfig(
        aggregation_hours=4.0,
        persistence_hours=3.0,
    ),
}


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].mean()) if len(frame) else float("nan")


def _metrics(events: pd.DataFrame, name: str, experiment: str) -> dict[str, object]:
    hypo = events[events["target_branch"] == "HYPO"]
    instability = events[events["target_branch"] == "INSTABILITE"]
    sequence = events[events["target_branch"] == "SEQUENCE"]
    controls = events[events["target_branch"] == "CONTROLE"]
    confounds = events[events["target_branch"] == "CONFONDANT"]
    actionable = events[events["target_branch"].isin(["HYPO", "SEQUENCE"])]
    per_cow = events.drop_duplicates("cow").copy()
    background = float(
        (per_cow["hybrid_background_notifs"] / per_cow["monitoring_days"]).mean()
    )
    row = {
        "experiment": experiment,
        "configuration": name,
        "n_cows": int(events["cow"].nunique()),
        "n_events": int(len(events)),
        "hypo_actionable_detection": _safe_mean(hypo, "hybrid_detected_any_overlap"),
        "instability_surveillance_detection": _safe_mean(
            instability, "instability_detected_any_overlap"
        ),
        "instability_actionable_detection": _safe_mean(
            instability, "hybrid_detected_any_overlap"
        ),
        "sequence_actionable_detection": _safe_mean(
            sequence, "hybrid_detected_any_overlap"
        ),
        "actionable_detection": _safe_mean(actionable, "hybrid_detected_any_overlap"),
        "benign_control_alert_rate": _safe_mean(
            controls, "hybrid_detected_any_overlap"
        ),
        "confound_alert_rate": _safe_mean(confounds, "hybrid_detected_any_overlap"),
        "actionable_iou20": _safe_mean(actionable, "hybrid_detected_iou20"),
        "actionable_delay_h_median": float(actionable["hybrid_onset_delay_hours"].median()),
        "background_per_cow_day": background,
    }
    # Les pénalités dominent volontairement un faible gain de détection.
    row["technical_utility"] = (
        row["actionable_detection"]
        + 0.15 * row["instability_surveillance_detection"]
        + 0.25 * row["actionable_iou20"]
        - 1.50 * row["benign_control_alert_rate"]
        - 2.00 * row["confound_alert_rate"]
        - 0.30 * min(1.0, background)
    )
    row["eligible_full_scope"] = bool(
        row["actionable_detection"] >= 0.60
        and row["instability_surveillance_detection"] >= 0.50
        and row["actionable_iou20"] >= 0.20
    )
    return row


def _run_one(
    *,
    name: str,
    experiment: str,
    instability: InstabilityWarningConfig,
    fusion: HybridFusionConfig,
    cows: list[str] | None,
    output: Path,
) -> dict[str, object]:
    print(f"\n=== {experiment}: {name} ===")
    events = run_campaign(
        cows=cows,
        instability_config=instability,
        fusion_config=fusion,
        verbose=False,
    )
    events.insert(0, "configuration", name)
    events.insert(0, "experiment", experiment)
    events.to_csv(output / f"events_{experiment}_{name}.csv", index=False)
    row = _metrics(events, name, experiment)
    row.update({f"instability_{key}": value for key, value in asdict(instability).items()})
    row.update({f"fusion_{key}": value for key, value in asdict(fusion).items()})
    print(pd.Series(row).to_string())
    return row


def run_sensitivity(
    *,
    smoke: bool = False,
    output_dir: str = "data/revalidation/v3_refined",
) -> pd.DataFrame:
    cows = ["8081", "8147"] if smoke else None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    reference = InstabilityWarningConfig()
    for name, fusion in FUSION_VARIANTS.items():
        rows.append(
            _run_one(
                name=name,
                experiment="fusion",
                instability=reference,
                fusion=fusion,
                cows=cows,
                output=output,
            )
        )

    hierarchical = FUSION_VARIANTS["hierarchical"]
    for name, instability in PARAMETER_CONFIGURATIONS.items():
        rows.append(
            _run_one(
                name=name,
                experiment="parameter_sensitivity",
                instability=instability,
                fusion=hierarchical,
                cows=cows,
                output=output,
            )
        )

    summary = pd.DataFrame(rows).sort_values(
        ["experiment", "technical_utility"], ascending=[True, False]
    )
    summary.to_csv(output / "comparison_summary.csv", index=False)
    print(f"\nSorties: {output.resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", default="data/revalidation/v3_refined")
    args = parser.parse_args()
    run_sensitivity(smoke=args.smoke, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
