"""Contrôles de protocole et artefacts reproductibles de la revalidation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from revalidation.profiles import PROFILES


def campaign_checks(events: pd.DataFrame) -> pd.DataFrame:
    """Retourne une table de contrôles bloquants pour la campagne primaire."""
    required = {
        "event_id",
        "cow",
        "scenario",
        "seed",
        "start",
        "end",
        "heldout_start",
        "event_after_heldout",
    }
    missing = sorted(required.difference(events.columns))
    physical = events[["cow", "start", "end"]] if not missing else pd.DataFrame()
    overlaps = 0
    if len(physical):
        for _, group in physical.assign(
            start=pd.to_datetime(physical["start"]),
            end=pd.to_datetime(physical["end"]),
        ).groupby("cow"):
            ordered = group.sort_values("start")
            previous_end = None
            for row in ordered.itertuples(index=False):
                if previous_end is not None and row.start <= previous_end:
                    overlaps += 1
                previous_end = max(previous_end, row.end) if previous_end is not None else row.end
    rows = [
        {
            "check": "required_columns",
            "passed": not missing,
            "detail": "ok" if not missing else f"missing={missing}",
        },
        {
            "check": "non_empty",
            "passed": len(events) > 0,
            "detail": f"n_events={len(events)}",
        },
        {
            "check": "unique_event_ids",
            "passed": bool(len(events) and events["event_id"].is_unique) if "event_id" in events else False,
            "detail": f"duplicates={int(events['event_id'].duplicated().sum()) if 'event_id' in events else 'NA'}",
        },
        {
            "check": "unique_physical_windows",
            "passed": bool(len(physical) and not physical.duplicated().any()),
            "detail": f"duplicates={int(physical.duplicated().sum()) if len(physical) else 'NA'}",
        },
        {
            "check": "non_overlapping_physical_windows",
            "passed": bool(len(physical) and overlaps == 0),
            "detail": f"overlaps={overlaps}" if len(physical) else "NA",
        },
        {
            "check": "strictly_post_baseline",
            "passed": bool(events["event_after_heldout"].astype(bool).all())
            if "event_after_heldout" in events and len(events)
            else False,
            "detail": (
                f"violations={int((~events['event_after_heldout'].astype(bool)).sum())}"
                if "event_after_heldout" in events
                else "NA"
            ),
        },
        {
            "check": "single_primary_seed",
            "passed": bool(events["seed"].nunique() == 1) if "seed" in events and len(events) else False,
            "detail": f"n_seeds={events['seed'].nunique() if 'seed' in events else 'NA'}",
        },
        {
            "check": "all_scenarios_present",
            "passed": (
                set(events["scenario"].unique()) == set(PROFILES)
            )
            if "scenario" in events and len(events)
            else False,
            "detail": (
                ",".join(sorted(events["scenario"].astype(str).unique()))
                if "scenario" in events
                else "NA"
            ),
        },
        {
            "check": "informative_source_windows",
            "passed": bool(
                len(events)
                and "informative_source_window" in events
                and events["informative_source_window"].astype(bool).all()
            ),
            "detail": (
                f"violations={int((~events['informative_source_window'].astype(bool)).sum())}"
                if "informative_source_window" in events
                else "missing_column"
            ),
        },
    ]
    return pd.DataFrame(rows)


def assert_campaign_valid(events: pd.DataFrame) -> pd.DataFrame:
    """Lève une erreur avant analyses si un contrôle de protocole échoue."""
    checks = campaign_checks(events)
    failures = checks.loc[~checks["passed"]]
    if not failures.empty:
        raise RuntimeError(f"Contrôles de campagne en échec:\n{failures.to_string(index=False)}")
    return checks


def manual_review_sample(events: pd.DataFrame, per_scenario: int = 3) -> pd.DataFrame:
    """Prépare un échantillon stratifié à vérifier humainement, sans simuler cette revue."""
    unique = events.drop_duplicates(subset=["event_id"]).copy()
    samples = []
    for _, group in unique.groupby("scenario", sort=True):
        ordered = group.sort_values(
            ["detected_any_overlap", "best_iou", "cow"],
            ascending=[True, True, True],
        )
        samples.append(ordered.head(int(per_scenario)))
    out = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    keep = [
        "event_id",
        "cow",
        "scenario",
        "start",
        "end",
        "expected_detected",
        "detected_any_overlap",
        "episode_overlap",
        "detected_iou20",
        "best_iou",
        "onset_delay_hours",
        "warning_score_max",
        "if_score_min",
        "source_steps_sum",
        "source_motion_sum",
        "source_transitions_sum",
        "realized_steps_change",
        "realized_motion_change",
        "realized_transitions_change",
        "informative_source_window",
    ]
    out = out[[column for column in keep if column in out.columns]].copy()
    out["manual_status"] = "A_REVOIR"
    out["manual_comment"] = ""
    return out


def write_json(path: str | Path, payload: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_sha256_manifest(paths: Iterable[str | Path], output: str | Path) -> Path:
    """Gèle les hashes des artefacts produits par la revalidation."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for raw_path in sorted(Path(path) for path in paths):
        digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {raw_path.as_posix()}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
