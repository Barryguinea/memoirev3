"""Alias de lisibilite pour la validation robuste V5.2 (logique mutualisee)."""

from __future__ import annotations

from scripts.validation_notebook_utils_v5_1 import *  # noqa: F401,F403
from scripts import validation_notebook_utils_v5_1 as _v5_1

# Alias explicite pour rendre le point d'entree v5.2 visible dans `scripts/`.
run_robust_validation_v5_2 = _v5_1.run_robust_validation_v5_1

__all__ = list(getattr(_v5_1, "__all__", []))
if "run_robust_validation_v5_2" not in __all__:
    __all__.append("run_robust_validation_v5_2")
