"""Regenere le manifeste des artefacts scientifiques utilises par mémoire final."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/validation_artifacts.sha256"
RAW_INPUT = ROOT / "data/brut.csv"
INPUT_PROVENANCE = ROOT / "data/validation/input_provenance.json"

PATTERNS = (
    "data/validation/bibliography_audit.csv",
    "data/validation/input_provenance.json",
    "data/validation/manuscript_number_audit.csv",
    "data/validation/derived_metrics/*.csv",
    "data/validation/hypo_module/*.csv",
    "data/validation/hypo_module/detection_background_curve/*.csv",
    "data/validation/hypo_module/parameter_sensitivity/*.csv",
    "data/validation/hypo_module/parameter_sensitivity/*.json",
    "data/validation/hypo_module/parameter_sensitivity/runs/*.csv",
    "data/validation/hypo_stress/*.csv",
    "data/validation/hypo_stress/*.json",
    "data/validation/performance_full_corpus.json",
    "data/validation/mcgill_sls/*.csv",
    "data/validation/mcgill_sls/*.json",
    "data/validation/hybrid_refined_full/*.csv",
    "data/validation/hybrid_sensitivity_full/*.csv",
    "validation_hypo/stress_protocol.json",
)


def update_input_provenance() -> None:
    """Consigne l'empreinte du corpus confidentiel sans en publier le contenu."""
    if not RAW_INPUT.exists():
        if not INPUT_PROVENANCE.exists():
            raise FileNotFoundError(
                "data/brut.csv absent et aucune provenance d'entree n'est disponible"
            )
        print("Corpus brut absent; provenance d'entree existante conservee")
        return

    digest = hashlib.sha256(RAW_INPUT.read_bytes()).hexdigest()
    rows = 0
    cows: set[str] = set()
    with RAW_INPUT.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            if row.get("Cow"):
                cows.add(str(row["Cow"]))
    provenance = {
        "path": "data/brut.csv",
        "confidential": True,
        "sha256": digest,
        "size_bytes": RAW_INPUT.stat().st_size,
        "rows": rows,
        "cows": len(cows),
    }
    INPUT_PROVENANCE.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    update_input_provenance()
    paths = sorted({path for pattern in PATTERNS for path in ROOT.glob(pattern) if path.is_file()})
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT)}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Ecrit: {OUTPUT} ({len(paths)} fichiers)")


if __name__ == "__main__":
    main()
