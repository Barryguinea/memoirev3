"""Façade publique des helpers de sélection/résumé pour validations."""

from __future__ import annotations

from scripts._validation_selection.planning import (
    _manual_injection_plan,
    _pick_window_for_cow,
    _resolve_target_cow,
    pick_cows_for_injection,
)
from scripts._validation_selection.summary import (
    _augment_summary_with_injectability,
    extract_detected_episodes,
    summarize_from_predictions,
)

__all__ = [
    "summarize_from_predictions",
    "extract_detected_episodes",
    "_augment_summary_with_injectability",
    "pick_cows_for_injection",
    "_pick_window_for_cow",
    "_resolve_target_cow",
    "_manual_injection_plan",
]
