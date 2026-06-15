"""Façade publique des règles métier d'alertes (compatibilité imports/tests)."""

from __future__ import annotations

from core._alerts_engine import apply_alert_logic
from core._alerts_helpers import _family_presence, _feature_family, _mi_spike_series, _pick_rrz_candidates

__all__ = [
    "apply_alert_logic",
    "_feature_family",
    "_pick_rrz_candidates",
    "_family_presence",
    "_mi_spike_series",
]
