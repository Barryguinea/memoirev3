"""Façade publique de la validation robuste."""

from __future__ import annotations

from scripts._validation_robust.run import run_robust_validation
from scripts._validation_robust.scenarios import _inject_scenario_raw

__all__ = ["_inject_scenario_raw", "run_robust_validation"]
