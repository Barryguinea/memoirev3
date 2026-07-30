"""Execute the frozen dose-matched HYPO stress campaign."""

from __future__ import annotations

import argparse
import json

from validation_hypo.stress_campaign import run_stress_campaign, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/validation/hypo_stress")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the first two eligible cows only.",
    )
    args = parser.parse_args()
    events = run_stress_campaign(
        max_cows=2 if args.smoke else None,
    )
    summary = write_outputs(events, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
