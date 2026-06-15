"""CLI du comparateur inter-versions de runs robustes."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._compare_runs.compare_three import compare_runs_three
from scripts._compare_runs.helpers import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VALIDATION_ROOT,
    REQUIRED_V4,
    REQUIRED_V5_BASE,
    SUMMARY_CANDIDATES_V5_1,
    SUMMARY_CANDIDATES_V5_2,
    _discover_latest_run,
)

def main() -> None:
    """Point d'entrée CLI : découvre/accepte les chemins de run puis lance la comparaison à trois voies."""
    parser = argparse.ArgumentParser(
        description="Compare trois runs complets: robuste v4 vs robuste v5.1 vs robuste v5.2",
    )
    parser.add_argument("--v4-run", type=Path, default=None, help="Chemin du dossier run v4")
    parser.add_argument("--v5-1-run", type=Path, default=None, help="Chemin du dossier run v5.1")
    parser.add_argument("--v5-2-run", type=Path, default=None, help="Chemin du dossier run v5.2")
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=DEFAULT_VALIDATION_ROOT,
        help="Dossier racine des runs robustes (auto-discovery si --v4-run/--v5-1-run/--v5-2-run absents)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Dossier de sortie pour les fichiers de comparaison",
    )
    args = parser.parse_args()

    validation_root = args.validation_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    v4_run = args.v4_run.resolve() if args.v4_run else _discover_latest_run(validation_root, REQUIRED_V4)
    v5_1_run = (
        args.v5_1_run.resolve()
        if args.v5_1_run
        else _discover_latest_run(
            validation_root,
            REQUIRED_V5_BASE,
            summary_candidates=SUMMARY_CANDIDATES_V5_1,
            exclude_dirs=[v4_run] if v4_run else [],
        )
    )
    v5_2_run = (
        args.v5_2_run.resolve()
        if args.v5_2_run
        else _discover_latest_run(
            validation_root,
            REQUIRED_V5_BASE,
            summary_candidates=SUMMARY_CANDIDATES_V5_2,
            exclude_dirs=[p for p in [v4_run, v5_1_run] if p is not None],
        )
    )

    if v4_run is None:
        raise FileNotFoundError(
            f"Aucun run v4 complet detecte dans {validation_root}. "
            "Il faut au moins: robust_summary_v4.csv, robust_runs.csv, robust_events_evaluation.csv"
        )
    if v5_1_run is None:
        raise FileNotFoundError(
            f"Aucun run v5.1 complet detecte dans {validation_root}. "
            "Il faut au moins: robust_summary_v5_1.csv (ou v5_2), robust_runs.csv, robust_events_evaluation.csv"
        )
    if v5_2_run is None:
        raise FileNotFoundError(
            f"Aucun run v5.2 complet detecte dans {validation_root}. "
            "Passe --v5-2-run explicitement ou lance un run v5.2 complet."
        )

    res = compare_runs_three(v4_run=v4_run, v5_1_run=v5_1_run, v5_2_run=v5_2_run, output_root=output_root)

    print(f"V4 run: {v4_run}")
    print(f"V5.1 run: {v5_1_run}")
    print(f"V5.2 run: {v5_2_run}")
    print(f"Output dir: {res.output_dir}")
    print(f"GO v4: {res.summary['n_go_v4']}/{res.summary['n_cfg_v4']}")
    print(f"GO v5.1: {res.summary['n_go_v5_1']}/{res.summary['n_cfg_v5_1']}")
    print(f"GO v5.2: {res.summary['n_go_v5_2']}/{res.summary['n_cfg_v5_2']}")
    print(f"Same pass v4/v5.1: {res.summary['same_pass_ratio_v4_vs_v5_1']:.3f}")
    print(f"Same pass v4/v5.2: {res.summary['same_pass_ratio_v4_vs_v5_2']:.3f}")
    print(f"Same pass v5.1/v5.2: {res.summary['same_pass_ratio_v5_1_vs_v5_2']:.3f}")
    print(f"Config comparison: {res.summary['files']['config_comparison']}")
    if res.summary["files"]["scenario_comparison"]:
        print(f"Scenario comparison: {res.summary['files']['scenario_comparison']}")
    print(f"Summary JSON: {res.output_dir / 'comparison_summary_v4_vs_v5_1_vs_v5_2.json'}")


if __name__ == "__main__":
    main()
