"""Gestion des artefacts V4 (archivage résumé base, exports Streamlit best sample)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from core.io import load_csv
from scripts import validation_notebook_utils as base_utils


def _archive_base_summary_file(run_dir: Path) -> Path | None:
    """Normalise le nom du résumé de base en `robust_summary_base_engine.csv`."""
    archived = run_dir / "robust_summary_base_engine.csv"
    legacy = run_dir / "robust_summary.csv"

    if archived.exists():
        if legacy.exists():
            legacy.unlink()
        return archived

    if not legacy.exists():
        return None

    legacy.replace(archived)
    return archived


def _extract_best_streamlit_sample_v4(
    *,
    run_dir: Path,
    summary_v4: pd.DataFrame,
    runs_df: pd.DataFrame,
    events_df: pd.DataFrame,
    raw_input_path: str | Path,
    grid: Dict[str, List[object]],
) -> Dict[str, str]:
    """Exporte des artefacts CSV prêts pour Streamlit pour la meilleure configuration V4."""
    if len(summary_v4) == 0 or len(runs_df) == 0 or len(events_df) == 0:
        return {}

    group_cols = [k for k in grid.keys() if k in runs_df.columns]
    if not group_cols:
        group_cols = [k for k in ["contamination", "persist_hours", "alert_min", "mix_mode", "mix_rate_thr"] if k in runs_df.columns]

    best_cfg = summary_v4.iloc[0]
    sel = runs_df.copy()
    for k in group_cols:
        if k in sel.columns:
            sel = sel[sel[k] == best_cfg[k]]

    if len(sel) == 0:
        return {}

    pick = sel.sort_values(["detectable_recall_iou20", "detectable_recall_any"], ascending=[False, False]).iloc[0]

    chosen = events_df.copy()
    for k in group_cols + ["seed", "scenario"]:
        if k in chosen.columns:
            chosen = chosen[chosen[k] == pick[k]]

    if len(chosen) == 0:
        return {}

    keep = ["event_id", "cow", "start", "end", "duration_bins", "profile", "expected_detected", "design_reason"]
    keep = [c for c in keep if c in chosen.columns]
    chosen_manifest = chosen[keep].drop_duplicates().copy()

    raw_df = load_csv(str(raw_input_path))
    streamlit_best, applied = base_utils.build_streamlit_raw_with_events(raw_df, chosen_manifest, seed=int(pick["seed"]))

    brut_file = run_dir / "streamlit_brut_with_best_injections_v4.csv"
    win_file = run_dir / "streamlit_best_injection_windows_v4.csv"
    streamlit_best.to_csv(brut_file, index=False)
    applied.to_csv(win_file, index=False)

    return {
        "streamlit_brut_with_best_injections_v4": str(brut_file.resolve()),
        "streamlit_best_injection_windows_v4": str(win_file.resolve()),
    }


__all__ = ["_archive_base_summary_file", "_extract_best_streamlit_sample_v4"]
