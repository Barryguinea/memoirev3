"""Façade publique des injections raw (implémentation interne package `_validation_injection_raw`)."""

from __future__ import annotations

from scripts._validation_injection_raw.injection import inject_manual_plan_raw, inject_two_events_raw
from scripts._validation_injection_raw.profiles import (
    InjectedEvent,
    _profile_to_raw_multipliers,
    apply_event_on_raw,
    choose_start_index,
)
from scripts._validation_injection_raw.streamlit import build_streamlit_raw_with_events

__all__ = [
    "InjectedEvent",
    "choose_start_index",
    "_profile_to_raw_multipliers",
    "apply_event_on_raw",
    "inject_two_events_raw",
    "inject_manual_plan_raw",
    "build_streamlit_raw_with_events",
]
