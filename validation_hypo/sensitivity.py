"""Analyse de sensibilite locale du detecteur temporel.

Cette analyse fait varier un seul parametre a la fois autour de la
configuration documentee. Elle sert a quantifier la robustesse des resultats;
elle ne selectionne pas une nouvelle configuration et ne constitue pas une
validation clinique.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from core.early_warning import EarlyWarningConfig
from validation_hypo.analysis import background_notification_rate
from validation_hypo.campaign import final_params, run_clean_campaign
from validation_hypo.profiles import PROFILES
from validation_hypo.qa import assert_campaign_valid, write_json, write_sha256_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "validation" / "parameter_sensitivity"
MODULE_VERSION = 1


@dataclass(frozen=True)
class SensitivityConfiguration:
    """Une configuration OFAT (one factor at a time)."""

    name: str
    parameter: str
    direction: str
    reference_value: float | int | None
    tested_value: float | int | None
    warning_config: EarlyWarningConfig

    def as_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "parameter": self.parameter,
            "direction": self.direction,
            "reference_value": self.reference_value,
            "tested_value": self.tested_value,
        }
        record.update(asdict(self.warning_config))
        return record


# Valeurs choisies a priori pour encadrer la configuration documentee.
# Elles ne sont pas issues d'une recherche de performance sur les resultats.
PARAMETER_VARIANTS: tuple[tuple[str, tuple[float | int, ...]], ...] = (
    ("aggregation_hours", (6.0, 24.0)),
    ("persistence_hours", (3.0, 12.0)),
    ("cooldown_hours", (12.0, 48.0)),
    ("family_min_change", (0.05, 0.15)),
    ("posture_min_change", (0.025, 0.10)),
    ("score_threshold", (0.08, 0.16)),
    ("cusum_drift", (0.04, 0.10)),
    ("cusum_threshold", (0.80, 1.60)),
    ("min_families", (2, 4)),
    ("coverage_min_pct", (50.0, 75.0)),
)


def _value_token(value: float | int) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def build_sensitivity_configurations() -> list[SensitivityConfiguration]:
    """Construit la reference et 20 variantes a un seul parametre."""
    reference = EarlyWarningConfig()
    configurations = [
        SensitivityConfiguration(
            name="reference",
            parameter="reference",
            direction="reference",
            reference_value=None,
            tested_value=None,
            warning_config=reference,
        )
    ]
    reference_values = asdict(reference)
    for parameter, values in PARAMETER_VARIANTS:
        reference_value = reference_values[parameter]
        for tested_value in values:
            direction = "lower" if tested_value < reference_value else "higher"
            configurations.append(
                SensitivityConfiguration(
                    name=f"{parameter}__{_value_token(tested_value)}",
                    parameter=parameter,
                    direction=direction,
                    reference_value=reference_value,
                    tested_value=tested_value,
                    warning_config=replace(reference, **{parameter: tested_value}),
                )
            )
    validate_sensitivity_configurations(configurations)
    return configurations


def validate_sensitivity_configurations(
    configurations: Sequence[SensitivityConfiguration],
) -> None:
    """Verifie que chaque variante differe de la reference sur un seul champ."""
    if not configurations or configurations[0].name != "reference":
        raise ValueError("La premiere configuration doit etre la reference.")
    names = [configuration.name for configuration in configurations]
    if len(names) != len(set(names)):
        raise ValueError("Les noms de configuration doivent etre uniques.")

    reference = asdict(configurations[0].warning_config)
    for configuration in configurations[1:]:
        candidate = asdict(configuration.warning_config)
        changed = [key for key in reference if candidate[key] != reference[key]]
        if changed != [configuration.parameter]:
            raise ValueError(
                f"{configuration.name}: variation non OFAT, champs modifies={changed}."
            )
        if candidate[configuration.parameter] != configuration.tested_value:
            raise ValueError(f"{configuration.name}: valeur testee incoherente.")


def _resolve_from_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_hash(
    configuration: SensitivityConfiguration,
    *,
    raw_sha256: str,
    seed: int,
    cows: Sequence[str] | None,
) -> str:
    payload = {
        "module_version": MODULE_VERSION,
        "configuration": configuration.as_record(),
        "raw_sha256": raw_sha256,
        "seed": int(seed),
        "cows": list(cows) if cows is not None else None,
        "scenarios": list(PROFILES),
        "campaign_params": final_params(),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_EVENT_SIGNATURE_COLUMNS = (
    "event_id",
    "cow",
    "scenario",
    "seed",
    "start",
    "end",
    "heldout_start",
)


def event_signature(events: pd.DataFrame) -> pd.DataFrame:
    """Normalise l'identite et le placement des evenements physiques."""
    missing = [column for column in _EVENT_SIGNATURE_COLUMNS if column not in events]
    if missing:
        raise ValueError(f"Colonnes de signature absentes: {missing}")
    signature = events.loc[:, _EVENT_SIGNATURE_COLUMNS].copy()
    signature["cow"] = signature["cow"].astype(str)
    for column in ("start", "end", "heldout_start"):
        signature[column] = pd.to_datetime(signature[column], errors="raise")
    return signature.sort_values("event_id").reset_index(drop=True)


def assert_same_event_schedule(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    """Bloque l'analyse si une variante ne porte pas sur les memes evenements."""
    left = event_signature(reference)
    right = event_signature(candidate)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise RuntimeError(
            "Les evenements physiques ont change entre la reference et une variante. "
            "Les metriques ne sont donc pas comparables."
        ) from exc


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def _median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.median()) if values.notna().any() else float("nan")


def summarize_configuration(
    events: pd.DataFrame,
    configuration: SensitivityConfiguration,
) -> dict[str, object]:
    """Produit les indicateurs principaux sans melanger profils et lignes capteur."""
    unique = events.drop_duplicates(subset=["event_id"]).copy()
    progressive = unique[unique["scenario"].astype(str).str.startswith("gradual_")]
    control = unique[unique["scenario"] == "isolated_short_variation"]
    background = background_notification_rate(unique)
    return {
        "configuration": configuration.name,
        "parameter": configuration.parameter,
        "direction": configuration.direction,
        "reference_value": configuration.reference_value,
        "tested_value": configuration.tested_value,
        "n_cows": int(unique["cow"].nunique()),
        "n_events": int(unique["event_id"].nunique()),
        "progressive_new_start_rate": _mean(progressive["detected_any_overlap"]),
        "progressive_attributable_coverage_rate": _mean(progressive["episode_overlap"]),
        "progressive_iou20_rate": _mean(progressive["detected_iou20"]),
        "progressive_best_iou_mean": _mean(progressive["best_iou"]),
        "progressive_onset_delay_median_hours": _median(progressive["onset_delay_hours"]),
        "control_new_start_rate": _mean(control["detected_any_overlap"]),
        "control_attributable_coverage_rate": _mean(control["episode_overlap"]),
        "control_iou20_rate": _mean(control["detected_iou20"]),
        **background,
    }


def summarize_by_scenario(
    events: pd.DataFrame,
    configuration: SensitivityConfiguration,
) -> pd.DataFrame:
    """Conserve les resultats separes par profil synthetique."""
    unique = events.drop_duplicates(subset=["event_id"]).copy()
    rows = []
    for scenario, group in unique.groupby("scenario", sort=False):
        rows.append(
            {
                "configuration": configuration.name,
                "parameter": configuration.parameter,
                "direction": configuration.direction,
                "reference_value": configuration.reference_value,
                "tested_value": configuration.tested_value,
                "scenario": scenario,
                "n_cows": int(group["cow"].nunique()),
                "n_events": int(group["event_id"].nunique()),
                "new_start_rate": _mean(group["detected_any_overlap"]),
                "attributable_coverage_rate": _mean(group["episode_overlap"]),
                "iou20_rate": _mean(group["detected_iou20"]),
                "best_iou_mean": _mean(group["best_iou"]),
                "onset_delay_median_hours": _median(group["onset_delay_hours"]),
            }
        )
    return pd.DataFrame(rows)


_DELTA_METRICS = (
    "progressive_new_start_rate",
    "progressive_attributable_coverage_rate",
    "progressive_iou20_rate",
    "progressive_best_iou_mean",
    "progressive_onset_delay_median_hours",
    "control_new_start_rate",
    "control_attributable_coverage_rate",
    "control_iou20_rate",
    "background_per_cow_day_mean",
    "background_per_cow_day_p90",
)


def add_reference_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des ecarts descriptifs; aucun classement n'est calcule."""
    reference = summary.loc[summary["configuration"] == "reference"]
    if len(reference) != 1:
        raise ValueError("Le resume doit contenir exactement une reference.")
    output = summary.copy()
    baseline = reference.iloc[0]
    for metric in _DELTA_METRICS:
        output[f"delta_{metric}_vs_reference"] = (
            pd.to_numeric(output[metric], errors="coerce") - float(baseline[metric])
        )
    return output


def _select_configurations(names: Sequence[str] | None) -> list[SensitivityConfiguration]:
    available = build_sensitivity_configurations()
    if not names:
        return available
    requested = list(dict.fromkeys(str(name) for name in names))
    known = {configuration.name: configuration for configuration in available}
    unknown = sorted(set(requested).difference(known))
    if unknown:
        raise ValueError(f"Configurations inconnues: {unknown}")
    selected_names = ["reference", *[name for name in requested if name != "reference"]]
    return [known[name] for name in selected_names]


def _load_reusable_run(path: Path, expected_hash: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    events = pd.read_csv(path)
    if "sensitivity_run_hash" not in events or events.empty:
        return None
    hashes = events["sensitivity_run_hash"].dropna().astype(str).unique().tolist()
    if hashes != [expected_hash]:
        return None
    assert_campaign_valid(events)
    return events


def run_parameter_sensitivity(
    raw_csv: str | Path = "data/brut.csv",
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed: int = 11,
    cows: Sequence[str] | None = None,
    configuration_names: Sequence[str] | None = None,
    force: bool = False,
    verbose: bool = True,
    campaign_runner: Callable[..., pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute ou reprend l'analyse et ecrit les artefacts reproductibles."""
    raw_path = _resolve_from_root(raw_csv)
    if not raw_path.is_file():
        raise FileNotFoundError(f"CSV source introuvable: {raw_path}")
    target = _resolve_from_root(output_dir)
    runs_dir = target / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    selected_cows = [str(cow) for cow in cows] if cows is not None else None
    configurations = _select_configurations(configuration_names)
    raw_hash = _sha256(raw_path)
    runner = campaign_runner or run_clean_campaign

    all_events: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    scenario_summaries: list[pd.DataFrame] = []
    reference_events: pd.DataFrame | None = None
    run_paths: list[Path] = []

    for index, configuration in enumerate(configurations, start=1):
        run_hash = _configuration_hash(
            configuration,
            raw_sha256=raw_hash,
            seed=seed,
            cows=selected_cows,
        )
        run_path = runs_dir / f"{configuration.name}.csv"
        events = None if force else _load_reusable_run(run_path, run_hash)
        if events is not None:
            if verbose:
                print(f"[{index}/{len(configurations)}] reprise: {configuration.name}")
        else:
            if verbose:
                print(f"[{index}/{len(configurations)}] execution: {configuration.name}")
            events = runner(
                str(raw_path),
                cows=selected_cows,
                seeds=(int(seed),),
                scenarios=tuple(PROFILES),
                params=final_params(),
                warning_config=configuration.warning_config,
                verbose=verbose,
            )
            assert_campaign_valid(events)
            events = events.copy()
            events["sensitivity_run_hash"] = run_hash
            events["sensitivity_module_version"] = MODULE_VERSION
            events["sensitivity_configuration"] = configuration.name
            events["sensitivity_parameter"] = configuration.parameter
            events["sensitivity_tested_value"] = configuration.tested_value
            events.to_csv(run_path, index=False)

        if reference_events is None:
            reference_events = events
        else:
            assert_same_event_schedule(reference_events, events)

        run_paths.append(run_path)
        all_events.append(events)
        summaries.append(summarize_configuration(events, configuration))
        scenario_summaries.append(summarize_by_scenario(events, configuration))

    events_long = pd.DataFrame.from_records(
        record
        for frame in all_events
        for record in frame.to_dict(orient="records")
    )
    summary = add_reference_deltas(pd.DataFrame(summaries))
    by_scenario = pd.DataFrame.from_records(
        record
        for frame in scenario_summaries
        for record in frame.to_dict(orient="records")
    )

    events_path = target / "parameter_sensitivity_events.csv"
    summary_path = target / "parameter_sensitivity_summary.csv"
    scenario_path = target / "parameter_sensitivity_by_scenario.csv"
    config_path = target / "parameter_sensitivity_configurations.json"
    manifest_path = target / "parameter_sensitivity.sha256"

    events_long.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_scenario.to_csv(scenario_path, index=False)
    write_json(
        config_path,
        {
            "module_version": MODULE_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "descriptive_one_factor_at_a_time_not_model_selection",
            "raw_csv": str(raw_path),
            "raw_sha256": raw_hash,
            "seed": int(seed),
            "requested_cows": selected_cows,
            "scenarios": list(PROFILES),
            "campaign_params": final_params(),
            "configurations": [configuration.as_record() for configuration in configurations],
        },
    )
    write_sha256_manifest(
        [*run_paths, events_path, summary_path, scenario_path, config_path],
        manifest_path,
    )
    return events_long, summary, by_scenario


def _parse_csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return values or None


def _print_configurations(configurations: Iterable[SensitivityConfiguration]) -> None:
    records = [configuration.as_record() for configuration in configurations]
    columns = ["name", "parameter", "direction", "reference_value", "tested_value"]
    print(pd.DataFrame(records)[columns].to_string(index=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sensibilite OFAT du detecteur temporel, sans selection de modele."
    )
    parser.add_argument("--raw-csv", default="data/brut.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--cows",
        help="Liste optionnelle separee par des virgules (ex.: 8081,8165).",
    )
    parser.add_argument(
        "--configurations",
        help="Sous-ensemble separe par des virgules; la reference est toujours ajoutee.",
    )
    parser.add_argument("--force", action="store_true", help="Recalculer les runs existants.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--list-configurations", action="store_true")
    args = parser.parse_args(argv)

    if args.list_configurations:
        _print_configurations(build_sensitivity_configurations())
        return 0

    _, summary, _ = run_parameter_sensitivity(
        args.raw_csv,
        output_dir=args.output_dir,
        seed=args.seed,
        cows=_parse_csv_list(args.cows),
        configuration_names=_parse_csv_list(args.configurations),
        force=args.force,
        verbose=not args.quiet,
    )
    print("\nAnalyse terminee. Resultats descriptifs (aucun classement):")
    print(summary.to_string(index=False))
    print(f"\nSorties: {_resolve_from_root(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PARAMETER_VARIANTS",
    "SensitivityConfiguration",
    "add_reference_deltas",
    "assert_same_event_schedule",
    "build_sensitivity_configurations",
    "event_signature",
    "run_parameter_sensitivity",
    "summarize_by_scenario",
    "summarize_configuration",
    "validate_sensitivity_configurations",
]
