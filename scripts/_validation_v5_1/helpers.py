"""Hub helpers du wrapper robuste V5.1/V5.2."""

from __future__ import annotations

from scripts._validation_v5_1.base import (
    base_utils,
    _archive_base_summary_file,
    _call_base_run_robust_validation_compat,
    _clean_grid,
    _json_safe,
    _scenario,
    _to_stats,
)
from scripts._validation_v5_1.exports import (
    _is_dominated,
    _pareto_frontier,
    _pick_best_streamlit,
    _scenario_report,
)
from scripts._validation_v5_1.summary import (
    _build_summary_v5_1,
    _choose_reference_row,
    _is_reference_balanced,
    _match_cfg_row,
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
]
