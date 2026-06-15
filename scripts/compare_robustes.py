#!/usr/bin/env python3
"""Façade publique du comparateur de runs robustes."""

from __future__ import annotations

from scripts._compare_runs.cli import main as _impl_main
from scripts._compare_runs.compare_pair import compare_runs
from scripts._compare_runs.compare_three import compare_runs_three
from scripts._compare_runs.helpers import (
    ComparisonResult,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VALIDATION_ROOT,
    PREFERRED_CFG_KEYS,
    REQUIRED_V4,
    REQUIRED_V5_BASE,
    ROOT,
    SCENARIO_CANDIDATES_V5_1,
    SCENARIO_CANDIDATES_V5_2,
    SUMMARY_CANDIDATES_V5_1,
    SUMMARY_CANDIDATES_V5_2,
)

__all__ = [
    "ROOT",
    "DEFAULT_VALIDATION_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "REQUIRED_V4",
    "REQUIRED_V5_BASE",
    "SUMMARY_CANDIDATES_V5_1",
    "SUMMARY_CANDIDATES_V5_2",
    "SCENARIO_CANDIDATES_V5_1",
    "SCENARIO_CANDIDATES_V5_2",
    "PREFERRED_CFG_KEYS",
    "ComparisonResult",
    "compare_runs",
    "compare_runs_three",
    "main",
]


def main() -> None:
    """Point d'entrée CLI conservé pour compatibilité."""
    _impl_main()


if __name__ == "__main__":
    main()
