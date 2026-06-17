"""Regenere le manifeste des artefacts scientifiques utilises par mémoire final."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/validation_artifacts.sha256"

PATTERNS = (
    "data/validation/manuscript_number_audit.csv",
    "data/validation/hypo_module/*.csv",
    "data/validation/hypo_module/detection_background_curve/*.csv",
    "data/validation/hypo_module/parameter_sensitivity/*.csv",
    "data/validation/hypo_module/parameter_sensitivity/*.json",
    "data/validation/hypo_module/parameter_sensitivity/runs/*.csv",
    "data/validation/performance_full_corpus.json",
    "data/validation/mcgill_sls/*.csv",
    "data/validation/mcgill_sls/*.json",
    "data/validation/hybrid_refined_full/*.csv",
    "data/validation/hybrid_sensitivity_full/*.csv",
)


def main() -> None:
    paths = sorted({path for pattern in PATTERNS for path in ROOT.glob(pattern) if path.is_file()})
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT)}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Ecrit: {OUTPUT} ({len(paths)} fichiers)")


if __name__ == "__main__":
    main()
