"""API publique minimale du coeur pipeline.

Ce package reexporte les fonctions et constantes utiles pour lire le projet
sans entrer d'abord dans tous les sous-modules.
"""

from .io import (
    load_csv,
    normalize_columns,
    available_base_cols,
    COW,
    TIME,
    STEPS,
    MI,
    LYING,
    STANDING,
    TRANSITIONS,
)

# Features
from .features import build_interval_features

# Modèle
from .model_if import run_if_core

# Alertes / règles
from .alerts import apply_alert_logic
from .early_warning import EarlyWarningConfig, apply_behavioral_early_warning
from .hybrid_warning import (
    HybridFusionConfig,
    InstabilityWarningConfig,
    apply_hybrid_warning,
    apply_instability_warning,
)

# Pipeline (orchestration)
from .pipeline import (
    run_pipeline_one_cow,
    run_pipeline_herd,
)

__all__ = [
    "COW",
    "TIME",
    "STEPS",
    "MI",
    "LYING",
    "STANDING",
    "TRANSITIONS",
    "EarlyWarningConfig",
    "HybridFusionConfig",
    "InstabilityWarningConfig",
    "apply_alert_logic",
    "apply_behavioral_early_warning",
    "apply_hybrid_warning",
    "apply_instability_warning",
    "available_base_cols",
    "build_interval_features",
    "load_csv",
    "normalize_columns",
    "run_if_core",
    "run_pipeline_herd",
    "run_pipeline_one_cow",
]
