"""Execute the frozen dose-matched HYPO stress campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation_hypo.stress_campaign import (
    default_output_dir,
    run_stress_campaign,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        help="Default: the directory associated with the selected reference policy.",
    )
    parser.add_argument(
        "--fixed-reference",
        action="store_true",
        help=(
            "Reuse the clean run training timestamps in the injected runs "
            "instead of recomputing the reference ratio on each run."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the first two eligible cows only.",
    )
    args = parser.parse_args()
    # Une execution partielle n'ecrit jamais dans un dossier de resultats complets.
    base = Path(default_output_dir(fixed_reference=args.fixed_reference))
    if args.smoke:
        base = Path("data/validation/stress_smoke") / base.name
    output_dir = args.output_dir or str(base)
    events = run_stress_campaign(
        max_cows=2 if args.smoke else None,
        fixed_reference=args.fixed_reference,
    )
    summary = write_outputs(events, output_dir, raw_csv="data/brut.csv")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
