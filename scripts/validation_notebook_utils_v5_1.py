"""Façade de validation robuste V5.1/V5.2."""

from __future__ import annotations

from scripts._validation_v5_1.engine import run_robust_validation_v5_1
from scripts._validation_v5_1.helpers import (
    base_utils,
    _archive_base_summary_file,
    _build_summary_v5_1,
    _call_base_run_robust_validation_compat,
    _choose_reference_row,
    _clean_grid,
    _is_dominated,
    _is_reference_balanced,
    _json_safe,
    _match_cfg_row,
    _pareto_frontier,
    _pick_best_streamlit,
    _scenario,
    _scenario_report,
    _to_stats,
)

__all__ = [
    "base_utils",
    "_call_base_run_robust_validation_compat",
    "_archive_base_summary_file",
    "_json_safe",
    "_to_stats",
    "_scenario",
    "_clean_grid",
    "_match_cfg_row",
    "_is_reference_balanced",
    "_choose_reference_row",
    "_build_summary_v5_1",
    "_scenario_report",
    "_is_dominated",
    "_pareto_frontier",
    "_pick_best_streamlit",
    "run_robust_validation_v5_1",
]
