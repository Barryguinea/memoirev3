#!/usr/bin/env python3
# ruff: noqa: E402
"""Reproduit la validation technique autonome de la branche HYPO de la validation finale."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_hypo.ablation import (
    ablation_paired_tests,
    ablation_summary,
    run_clean_ablation,
)
from validation_hypo.analysis import detection_summary
from validation_hypo.campaign import run_clean_campaign
from validation_hypo.qa import assert_campaign_valid


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validation post-baseline et ablation de la branche HYPO"
    )
    parser.add_argument("--raw-csv", default="data/brut.csv")
    parser.add_argument(
        "--output-dir", default="data/validation/hypo_module"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    events = run_clean_campaign(
        args.raw_csv,
        seeds=(11,),
        verbose=verbose,
    )
    checks = assert_campaign_valid(events)
    events.to_csv(output / "events_primary.csv", index=False)
    checks.to_csv(output / "protocol_checks.csv", index=False)
    detection_summary(events).to_csv(
        output / "detection_summary.csv", index=False
    )

    ablation = run_clean_ablation(
        args.raw_csv,
        seeds=(11,),
        verbose=verbose,
    )
    ablation.to_csv(output / "ablation_primary.csv", index=False)
    ablation_summary(ablation).to_csv(
        output / "ablation_summary.csv", index=False
    )
    ablation_paired_tests(ablation).to_csv(
        output / "ablation_tests_by_cow.csv", index=False
    )

    if verbose:
        print(f"Artefacts HYPO écrits dans {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
