"""Façade publique du wrapper de validation robuste V4."""

from __future__ import annotations

from scripts import validation_notebook_utils as base_utils
from scripts._validation_v4.wrapper import (
    _archive_base_summary_file,
    _build_v4_summary,
    _call_base_run_robust_validation_compat,
    _events_scenario_report,
    _extract_best_streamlit_sample_v4,
    _filter_cfg_grid,
    _scenario_slice,
    _series_stats,
    run_robust_validation_v4,
)

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
