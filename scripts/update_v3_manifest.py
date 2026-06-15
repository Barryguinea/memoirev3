"""Regenere le manifeste des artefacts scientifiques utilises par MemoireV3."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/revalidation/v3_artifacts.sha256"

PATTERNS = (
    "data/revalidation/manuscript_number_audit.csv",
    "data/revalidation/hypo_module/*.csv",
    "data/revalidation/hypo_module/detection_background_curve/*.csv",
    "data/revalidation/hypo_module/parameter_sensitivity/*.csv",
    "data/revalidation/hypo_module/parameter_sensitivity/*.json",
    "data/revalidation/hypo_module/parameter_sensitivity/runs/*.csv",
    "data/revalidation/performance_v3_full_corpus.json",
    "data/revalidation/v3_mcgill_sls/*.csv",
    "data/revalidation/v3_mcgill_sls/*.json",
    "data/revalidation/v3_refined_full/*.csv",
    "data/revalidation/v3_sensitivity_full/*.csv",
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
