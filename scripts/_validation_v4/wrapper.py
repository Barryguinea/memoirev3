"""Wrapper de validation robuste V4 (orchestration + compatibilité)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from scripts import validation_notebook_utils as base_utils
from scripts._validation_v4.artifacts import _archive_base_summary_file, _extract_best_streamlit_sample_v4
from scripts._validation_v4.summary import (
    _build_v4_summary,
    _events_scenario_report,
    _filter_cfg_grid,
    _scenario_slice,
    _series_stats,
)


def _call_base_run_robust_validation_compat(**kwargs):
    """Appel rétrocompatible du moteur de base (tolère d'anciens modules sans nouveaux kwargs)."""
    try:
        sig = inspect.signature(base_utils.run_robust_validation)
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not accepts_var_kw:
            allowed = set(sig.parameters.keys())
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        pass
    return base_utils.run_robust_validation(**kwargs)


def run_robust_validation_v4(
    *,
    raw_input_path: str | Path,
    output_root: str | Path,
    base_params: Dict[str, object],
    seeds: list[int],
    scenarios: list[str],
    grid: Dict[str, list[object]],
    raw_source_for_streamlit: str | Path | None = None,
    verbose: bool = False,
    progress_every: int = 1,
    detectable_events_per_run: int = 3,
    checkpoint_every: int = 25,
    go_thresholds_v4: Optional[Dict[str, float]] = None,
    injection_mode: str = "auto",
    fixed_detect_cow: Optional[str] = None,
    fixed_hidden_cow: Optional[str] = None,
) -> Dict[str, object]:
    """Extension non-destructive de la robuste de base avec scoring V4 et artefacts spécifiques."""
    effective_grid = _filter_cfg_grid(grid)

    thresholds = {
        "strong_iou20_p10_min": 0.25,
        "strong_any_mean_min": 0.90,
        "borderline_any_mean_min": 0.20,
        "borderline_any_p10_min": 0.00,
        "hidden_leak_any_p90_max": 0.10,
        "false_notif_p90_max": 0.25,
    }
    if go_thresholds_v4:
        thresholds.update({k: float(v) for k, v in go_thresholds_v4.items()})

    legacy_go = {
        "strong_iou20_p10_min": float(thresholds["strong_iou20_p10_min"]),
        "borderline_any_mean_min": float(thresholds["borderline_any_mean_min"]),
        "hidden_leak_any_p90_max": float(thresholds["hidden_leak_any_p90_max"]),
        "false_notif_p90_max": float(thresholds["false_notif_p90_max"]),
    }

    base_res = _call_base_run_robust_validation_compat(
        raw_input_path=raw_input_path,
        output_root=output_root,
        base_params=base_params,
        seeds=seeds,
        scenarios=scenarios,
        grid=effective_grid,
        raw_source_for_streamlit=raw_source_for_streamlit,
        verbose=verbose,
        progress_every=progress_every,
        detectable_events_per_run=detectable_events_per_run,
        go_mode="scenario_aware",
        go_thresholds=legacy_go,
        checkpoint_every=checkpoint_every,
        write_best_streamlit_artifacts=False,
        injection_mode=injection_mode,
        fixed_detect_cow=fixed_detect_cow,
        fixed_hidden_cow=fixed_hidden_cow,
    )

    run_dir = Path(base_res["run_dir"])
    archived_base_summary = _archive_base_summary_file(run_dir)
    runs_df = base_res["runs"].copy() if "runs" in base_res else pd.DataFrame()
    events_df = base_res["events"].copy() if "events" in base_res else pd.DataFrame()

    summary_v4 = _build_v4_summary(runs_df, grid=effective_grid, thresholds=thresholds)
    scenario_report = _events_scenario_report(events_df)

    summary_v4_file = run_dir / "robust_summary_v4.csv"
    scenario_report_file = run_dir / "robust_events_by_scenario_v4.csv"
    summary_v4.to_csv(summary_v4_file, index=False)
    scenario_report.to_csv(scenario_report_file, index=False)

    best_files = _extract_best_streamlit_sample_v4(
        run_dir=run_dir,
        summary_v4=summary_v4,
        runs_df=runs_df,
        events_df=events_df,
        raw_input_path=raw_input_path,
        grid=effective_grid,
    )

    meta_v4 = {
        "raw_input_path": str(Path(raw_input_path).resolve()),
        "run_dir": str(run_dir.resolve()),
        "scenarios": [str(s) for s in scenarios],
        "seeds": [int(s) for s in seeds],
        "effective_grid": {k: [x for x in v] for k, v in effective_grid.items()},
        "detectable_events_per_run": int(max(1, detectable_events_per_run)),
        "go_thresholds_v4": {k: float(v) for k, v in thresholds.items()},
        "n_runs": int(len(runs_df)),
        "n_events": int(len(events_df)),
        "n_summary_v4": int(len(summary_v4)),
        "n_go_v4": int((summary_v4["robust_pass_v4"] == 1).sum()) if "robust_pass_v4" in summary_v4.columns else 0,
        "base_summary_archived_file": str(archived_base_summary.resolve()) if archived_base_summary else "",
        "injection_mode": str(injection_mode),
        "fixed_detect_cow_requested": "" if fixed_detect_cow is None else str(fixed_detect_cow),
        "fixed_hidden_cow_requested": "" if fixed_hidden_cow is None else str(fixed_hidden_cow),
    }
    meta_v4.update(best_files)
    (run_dir / "run_meta_v4.json").write_text(json.dumps(meta_v4, indent=2), encoding="utf-8")

    out = dict(base_res)
    out.update(
        {
            "summary_v4": summary_v4,
            "scenario_report_v4": scenario_report,
            "go_thresholds_v4": thresholds,
            "summary_v4_file": summary_v4_file,
            "scenario_report_v4_file": scenario_report_file,
            "base_summary_archived_file": archived_base_summary,
        }
    )
    out.update(best_files)
    return out


__all__ = [
    "base_utils",
    "_call_base_run_robust_validation_compat",
    "_archive_base_summary_file",
    "_series_stats",
    "_filter_cfg_grid",
    "_scenario_slice",
    "_build_v4_summary",
    "_events_scenario_report",
    "_extract_best_streamlit_sample_v4",
    "run_robust_validation_v4",
]
