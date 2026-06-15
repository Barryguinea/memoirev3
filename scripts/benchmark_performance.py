#!/usr/bin/env python3
"""Façade CLI du benchmark de performance (implémentation dans `scripts._benchmark_performance.cli`)."""

from __future__ import annotations

import sys
from pathlib import Path

# Permet l'execution directe: `python scripts/benchmark_performance.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._benchmark_performance.cli import PipelineParams, main, run_benchmark_herd

__all__ = ["PipelineParams", "run_benchmark_herd", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
