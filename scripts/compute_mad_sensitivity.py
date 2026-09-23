# ruff: noqa: E402
"""Sensibilite de l'ablation a la definition du z-score robuste glissant.

Le Tableau 4.5 du manuscrit decrit un ecart absolu median (MAD) calcule sur une
fenetre mobile de 24 intervalles. Le code du manuscrit v3 calcule une variante :
chaque ecart est pris a la mediane mobile de son propre intervalle, puis la
mediane de ces ecarts est prise sur 24 intervalles. Seules les colonnes ``_rrz``
en dependent ; elles alimentent Isolation Forest, LOF et leurs regles (variantes
B, C et D). HYPO et le comparateur pedometrique ne les lisent pas.

Ce script rejoue l'ablation complete avec les deux definitions, sur les memes
onze vaches et les memes quarante-quatre evenements, puis ecrit les resultats
dans un dossier separe muni de sa propre provenance et de son propre manifeste.
Les artefacts scelles du manuscrit ne sont jamais modifies. Voir
docs/politique_mad.md.

Usage : ``python scripts/compute_mad_sensitivity.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

# Permet l'execution directe: `python scripts/compute_mad_sensitivity.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from core.features import ROLLING_MAD_MODES
from validation_hypo.ablation import (
    ablation_paired_tests,
    ablation_summary,
    run_clean_ablation,
)

ROOT = PROJECT_ROOT
RAW_CSV = ROOT / "data/brut.csv"
OUTPUT = ROOT / "data/validation/mad_sensitivity"
SEALED = (
    ROOT / "data/validation/hypo_module",
    ROOT / "data/validation/hypo_stress",
    ROOT / "data/validation/derived_metrics",
)
ARTIFACTS = (
    "events.csv",
    "variant_summary.csv",
    "paired_tests.csv",
    "provenance.json",
)


def event_f1(events: pd.DataFrame) -> pd.Series:
    """F1 evenementiel sur le critere de nouveau depart, comme au Tableau 6.5."""
    unique = events.drop_duplicates(["event_id", "variante"])
    scores = {}
    for variante, group in unique.groupby("variante"):
        positives = group[group["expected_detected"] == 1]
        negatives = group[group["expected_detected"] == 0]
        tp = int(positives["detected_any_overlap"].sum())
        fn = len(positives) - tp
        fp = int(negatives["detected_any_overlap"].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores[variante] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return pd.Series(scores, name="f1_nouveau_depart")


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    unique = events.drop_duplicates(["event_id", "variante"])
    summary = ablation_summary(events).set_index("variante")
    summary["couverture_attribuable"] = unique.groupby("variante")["episode_overlap"].mean()
    summary["f1_nouveau_depart"] = event_f1(events)
    return summary.reset_index().round(6)


def write_provenance(output: Path) -> None:
    sources = sorted({
        *ROOT.glob("core/*.py"),
        *ROOT.glob("validation_hypo/*.py"),
        Path(__file__).resolve(),
    })
    provenance = {
        "scope": "Sensibilite a la definition du MAD glissant ; les artefacts scelles sont intacts.",
        "modes": list(ROLLING_MAD_MODES),
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

    events, summaries, tests = [], [], []
    for mode in ROLLING_MAD_MODES:
        run = run_clean_ablation(str(RAW_CSV), verbose=False, mad_mode=mode)
        run.insert(0, "mad_mode", mode)
        events.append(run)
        summaries.append(summarize(run).assign(mad_mode=mode))
        tests.append(ablation_paired_tests(run).assign(mad_mode=mode))

    pd.concat(events, ignore_index=True).to_csv(output / "events.csv", index=False)
    summary = pd.concat(summaries, ignore_index=True)
    summary.insert(0, "mad_mode", summary.pop("mad_mode"))
    summary.to_csv(output / "variant_summary.csv", index=False)
    paired = pd.concat(tests, ignore_index=True)
    paired.insert(0, "mad_mode", paired.pop("mad_mode"))
    paired.to_csv(output / "paired_tests.csv", index=False)
    write_provenance(output)
    (output / "artifacts.sha256").write_text(
        "".join(
            f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
            for name in ARTIFACTS
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Ecrit : {output}")


if __name__ == "__main__":
    main()
