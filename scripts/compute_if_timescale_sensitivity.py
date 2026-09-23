# ruff: noqa: E402
"""Isolation Forest et LOF alimentes par les ratios sur 12 heures de HYPO.

Verifie que l'avantage de localisation de HYPO sur Isolation Forest et LOF ne
tient pas seulement a l'echelle de temps des variables fournies aux comparateurs. Memes onze
vaches, memes quarante-quatre evenements que l'ablation principale. Resultats
ecrits dans un dossier separe muni de sa propre provenance et de son propre
manifeste ; les artefacts scelles ne sont jamais modifies. Voir
docs/comparateur_echelle_temps.md.

Usage : ``python scripts/compute_if_timescale_sensitivity.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

# Permet l'execution directe: `python scripts/compute_if_timescale_sensitivity.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation_hypo.timescale_comparators import (
    RATIO_COLUMNS,
    paired_tests_timescale,
    run_timescale_ablation,
    summarize_timescale,
)

ROOT = PROJECT_ROOT
RAW_CSV = ROOT / "data/brut.csv"
OUTPUT = ROOT / "data/validation/if_timescale_sensitivity"
SEALED = (
    ROOT / "data/validation/hypo_module",
    ROOT / "data/validation/hypo_stress",
    ROOT / "data/validation/stress_fixed_reference",
    ROOT / "data/validation/derived_metrics",
)
ARTIFACTS = ("events.csv", "variant_summary.csv", "paired_tests.csv", "provenance.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if any(output == sealed.resolve() for sealed in SEALED):
        raise SystemExit(f"Refus d'ecrire dans un dossier scelle : {output}")
    if not RAW_CSV.exists():
        raise SystemExit("data/brut.csv est requis ; le corpus confidentiel n'est pas versionne.")
    output.mkdir(parents=True, exist_ok=True)

    events = run_timescale_ablation(str(RAW_CSV))
    events.to_csv(output / "events.csv", index=False)
    summary = summarize_timescale(events)
    summary.to_csv(output / "variant_summary.csv", index=False)
    tests = paired_tests_timescale(events)
    tests.to_csv(output / "paired_tests.csv", index=False)

    sources = sorted({
        *ROOT.glob("core/*.py"),
        *ROOT.glob("validation_hypo/*.py"),
        Path(__file__).resolve(),
    })
    provenance = {
        "scope": "IF et LOF sur les ratios de HYPO ; les artefacts scelles sont intacts.",
        "features": list(RATIO_COLUMNS),
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
    print(summary.to_string(index=False))
    print(tests.to_string(index=False))
    print(f"Ecrit : {output}")


if __name__ == "__main__":
    main()
