"""Finalize the reproducibility snapshot after notebook execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from validation_hypo.analysis import background_notification_rate
from validation_hypo.qa import write_json, write_sha256_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


def _git_diff_sha256(root: Path) -> str:
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", "."],
        cwd=root,
    )
    return hashlib.sha256(diff).hexdigest()


def finalize(root: Path, executed_notebook: Path) -> tuple[Path, Path]:
    output = root / "data" / "validation" / "hypo_module"
    events = pd.read_csv(output / "events_primary.csv")
    ablation = pd.read_csv(output / "ablation_summary.csv")
    review = pd.read_csv(output / "manual_review_sample.csv")
    protocol = json.loads((output / "protocol_configuration.json").read_text(encoding="utf-8"))

    progressive = events[events["scenario"].str.startswith("gradual_")].copy()
    control = events[events["scenario"] == "isolated_short_variation"].copy()
    background = background_notification_rate(events)
    realized = (
        events.groupby("scenario")
        .agg(
            steps_change_mean=("realized_steps_change", "mean"),
            motion_change_mean=("realized_motion_change", "mean"),
            transitions_change_mean=("realized_transitions_change", "mean"),
        )
        .to_dict(orient="index")
    )

    now = datetime.now(ZoneInfo("America/Toronto"))
    snapshot = {
        "snapshot_date": now.date().isoformat(),
        "generated_at": now.isoformat(),
        "timezone": "America/Toronto",
        "scope": "benchmark technique post-baseline d'une alerte comportementale precoce",
        "clinical_validation": False,
        "repository_commit": _git_output(root, "rev-parse", "HEAD"),
        "repository_branch": _git_output(root, "branch", "--show-current"),
        "repository_dirty": bool(_git_output(root, "status", "--short")),
        "project_diff_sha256": _git_diff_sha256(root),
        "raw_csv_sha256": _sha256(root / "data" / "brut.csv"),
        "executed_notebook": executed_notebook.relative_to(root).as_posix(),
        "executed_notebook_sha256": _sha256(executed_notebook),
        "protocol_configuration": protocol,
        "n_cows": int(events["cow"].nunique()),
        "n_events": int(events["event_id"].nunique()),
        "n_events_per_profile": {
            str(key): int(value)
            for key, value in events.groupby("scenario")["event_id"].nunique().items()
        },
        "progressive_events": {
            "n": int(progressive["event_id"].nunique()),
            "new_warning_start_rate": float(progressive["detected_any_overlap"].mean()),
            "attributable_coverage_rate": float(progressive["episode_overlap"].mean()),
            "attributable_iou20_rate": float(progressive["detected_iou20"].mean()),
            "attributable_mean_best_iou": float(progressive["best_iou"].mean()),
            "raw_coverage_rate": float(progressive["episode_overlap_raw"].mean()),
            "raw_iou20_rate": float(progressive["detected_iou20_raw"].mean()),
            "raw_mean_best_iou": float(progressive["best_iou_raw"].mean()),
        },
        "isolated_short_control": {
            "n": int(control["event_id"].nunique()),
            "new_warning_start_rate": float(control["detected_any_overlap"].mean()),
            "attributable_coverage_rate": float(control["episode_overlap"].mean()),
            "attributable_iou20_rate": float(control["detected_iou20"].mean()),
            "attributable_mean_best_iou": float(control["best_iou"].mean()),
            "raw_coverage_rate": float(control["episode_overlap_raw"].mean()),
            "raw_iou20_rate": float(control["detected_iou20_raw"].mean()),
        },
        "background_notifications": {
            **background,
            "denominator": "post-baseline monitoring duration",
            "interpretation": "notifications de fond non adjudicables",
        },
        "realized_profile_changes": realized,
        "ablation": ablation.to_dict(orient="records"),
        "manual_review": {
            "n": int(len(review)),
            "status_counts": {
                str(key): int(value)
                for key, value in review["manual_status"].value_counts(dropna=False).items()
            },
        },
    }

    snapshot_path = output / f"snapshot_{now.date().isoformat()}.json"
    write_json(snapshot_path, snapshot)

    artifact_paths = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "validation.sha256"
    ]
    provenance_paths = [
        root / "data" / "brut.csv",
        root / "requirements.txt",
        root / "app.py",
        root / "memoire" / "main.pdf",
        root / "notebooks" / "validation_complete.ipynb",
        executed_notebook,
    ]
    for pattern in (
        "core/*.py",
        "validation_hypo/*.py",
        "ui/*.py",
        "memoire/*.tex",
        "memoire/*.bib",
        "memoire/*.cls",
        "memoire/figures/*.png",
        "docs/*.md",
    ):
        provenance_paths.extend(path for path in root.glob(pattern) if path.is_file())
    manifest_path = write_sha256_manifest(
        sorted(set(artifact_paths + provenance_paths)),
        output / "validation.sha256",
    )
    return snapshot_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        default="notebooks/validation_complete_executed_2026-06-12.ipynb",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    snapshot, manifest = finalize(root, root / args.notebook)
    print(snapshot)
    print(manifest)


if __name__ == "__main__":
    main()
