# ruff: noqa: E402
"""Test de stress sous le MAD standard, seul puis combiné à la nouvelle gestion des trous.

Les notes docs/politique_mad.md et docs/politique_trous_de_donnees.md évaluent
chaque correction séparément. Ce script rejoue le test de stress à référence fixe
(198 événements, cinq variantes) avec le MAD standard pour Isolation Forest et
LOF, d'abord avec l'ancienne gestion des trous, puis avec la politique
``coverage_aware``. Les deux autres combinaisons figurent déjà dans
data/validation/gap_policy_sensitivity/.

Les résultats sont écrits dans un dossier séparé muni de sa propre provenance et
de son propre manifeste ; les artefacts scellés ne sont jamais modifiés.

Usage : ``python scripts/compute_combined_corrections_stress.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

# Permet l'exécution directe : `python scripts/compute_combined_corrections_stress.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from validation_hypo.stress_campaign import compare_variants, run_stress_campaign, summarize

ROOT = PROJECT_ROOT
RAW_CSV = ROOT / "data/brut.csv"
OUTPUT = ROOT / "data/validation/combined_corrections_stress"
SEALED = (
    ROOT / "data/validation/hypo_module",
    ROOT / "data/validation/hypo_stress",
    ROOT / "data/validation/stress_fixed_reference",
    ROOT / "data/validation/derived_metrics",
)
CONFIGURATIONS = (
    ("window", "historical"),
    ("window", "coverage_aware"),
)
ARTIFACTS = (
    "stress_scenario_summary.csv",
    "stress_variant_summary.csv",
    "stress_paired_comparisons.csv",
    "provenance.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if any(output == sealed.resolve() for sealed in SEALED):
        raise SystemExit(f"Refus d'écrire dans un dossier scellé : {output}")
    if not RAW_CSV.exists():
        raise SystemExit("data/brut.csv est requis ; le corpus confidentiel n'est pas versionné.")
    output.mkdir(parents=True, exist_ok=True)

    scenarios, variants, comparisons = [], [], []
    for mad_mode, gap_policy in CONFIGURATIONS:
        events = run_stress_campaign(
            str(RAW_CSV), verbose=False, fixed_reference=True,
            gap_policy=gap_policy, mad_mode=mad_mode,
        )
        scenario_summary, variant_summary = summarize(events)
        labels = {"mad_mode": mad_mode, "gap_policy": gap_policy}
        scenarios.append(scenario_summary.assign(**labels))
        variants.append(variant_summary.assign(**labels))
        comparisons.append(compare_variants(events).assign(**labels))

    for frames, name in (
        (scenarios, "stress_scenario_summary.csv"),
        (variants, "stress_variant_summary.csv"),
        (comparisons, "stress_paired_comparisons.csv"),
    ):
        table = pd.concat(frames, ignore_index=True)
        table.insert(0, "gap_policy", table.pop("gap_policy"))
        table.insert(0, "mad_mode", table.pop("mad_mode"))
        table.to_csv(output / name, index=False)

    sources = sorted({
        *ROOT.glob("core/*.py"),
        *ROOT.glob("validation_hypo/*.py"),
        ROOT / "validation_hypo/stress_protocol.json",
        Path(__file__).resolve(),
    })
    provenance = {
        "scope": "Test de stress sous le MAD standard, seul et combiné à coverage_aware ; "
                 "les artefacts scellés sont intacts.",
        "configurations": [
            {"mad_mode": mad, "gap_policy": gap} for mad, gap in CONFIGURATIONS
        ],
        "stress_reference_policy": "clean_training_timestamps_fixed",
        "python_version": platform.python_version(),
        "packages": {
            name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "input_sha256": hashlib.sha256(RAW_CSV.read_bytes()).hexdigest(),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "artifacts.sha256").write_text(
        "".join(
            f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
            for name in ARTIFACTS
        ),
        encoding="utf-8",
    )
    print(pd.concat(comparisons, ignore_index=True).to_string(index=False))
    print(f"Écrit : {output}")


if __name__ == "__main__":
    main()
