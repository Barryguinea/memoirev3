"""Primitives d'injection raw: profils, choix d'indices et application in-place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from core.io import LYING, MI, STANDING, STEPS, TRANSITIONS, TR_DOWN, TR_UP


@dataclass
class InjectedEvent:
    """Métadonnées d'un événement synthétique de boiterie injecté dans les données capteurs."""

    event_id: str
    cow: str
    start: pd.Timestamp
    end: pd.Timestamp
    duration_bins: int
    profile: str
    expected_detected: int
    design_reason: str


def choose_start_index(n: int, duration: int, ratio: float, rng: np.random.Generator) -> int:
    """Choisit un indice de départ proche de *n * ratio* avec un léger bruit aléatoire."""
    max_start = max(0, n - duration)
    center = int(round(n * float(ratio)))
    jitter = max(1, int(0.06 * n))
    guess = center + int(rng.integers(-jitter, jitter + 1))
    return int(np.clip(guess, 0, max_start))


def _profile_to_raw_multipliers(profile: str, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    phase = np.arange(n) % 4
    if profile == "detectable_strong":
        # Scénario strong ajusté pour améliorer la robustesse IoU au niveau
        # événement : blocs cohérents plus larges et amplitude plus élevée.
        phase6 = np.arange(n) % 6
        return {
            "steps": np.where(phase6 < 3, rng.uniform(0.02, 0.10, size=n), rng.uniform(1.80, 2.80, size=n)),
            "mi": np.where(phase6 < 3, rng.uniform(0.01, 0.08, size=n), rng.uniform(2.00, 3.40, size=n)),
            "lying": np.where(phase6 < 3, rng.uniform(2.60, 4.20, size=n), rng.uniform(0.65, 1.10, size=n)),
            "standing": np.where(phase6 < 3, rng.uniform(0.04, 0.20, size=n), rng.uniform(1.25, 2.10, size=n)),
            "trans": np.where((phase6 % 3) == 0, rng.uniform(0.03, 0.15, size=n), rng.uniform(3.00, 5.20, size=n)),
        }
    if profile in {"detectable_multisignal_persistent", "detectable_processed_visible_pattern"}:
        return {
            "steps": np.where(phase < 2, rng.uniform(0.03, 0.15, size=n), rng.uniform(1.40, 2.20, size=n)),
            "mi": np.where(phase < 2, rng.uniform(0.02, 0.12, size=n), rng.uniform(1.60, 2.80, size=n)),
            "lying": np.where(phase < 2, rng.uniform(2.20, 3.80, size=n), rng.uniform(0.70, 1.30, size=n)),
            "standing": np.where(phase < 2, rng.uniform(0.05, 0.35, size=n), rng.uniform(1.10, 1.90, size=n)),
            "trans": np.where((phase % 2) == 0, rng.uniform(0.04, 0.18, size=n), rng.uniform(2.50, 4.80, size=n)),
        }
    if profile in {"borderline", "detectable_borderline"}:
        # Scénario borderline ajusté pour être "détectable mais plus faible
        # que strong" : motif multi-familles visible + quelques pics MI,
        # sans atteindre les amplitudes du scénario strong.
        phase6 = np.arange(n) % 6
        return {
            "steps": np.where(phase6 < 3, rng.uniform(0.12, 0.30, size=n), rng.uniform(1.35, 1.95, size=n)),
            "mi": np.where(phase6 < 3, rng.uniform(0.08, 0.24, size=n), rng.uniform(1.70, 2.45, size=n)),
            "lying": np.where(phase6 < 3, rng.uniform(1.75, 2.55, size=n), rng.uniform(0.62, 0.98, size=n)),
            "standing": np.where(phase6 < 3, rng.uniform(0.16, 0.44, size=n), rng.uniform(1.15, 1.70, size=n)),
            "trans": np.where((phase6 % 3) == 0, rng.uniform(0.14, 0.42, size=n), rng.uniform(1.85, 3.05, size=n)),
        }
    # Cas caché / non détectable
    return {
        "steps": np.clip(rng.normal(loc=0.86, scale=0.04, size=n), 0.70, 0.98),
        "mi": np.clip(rng.normal(loc=0.86, scale=0.04, size=n), 0.70, 0.98),
        "lying": np.ones(n, dtype=float),
        "standing": np.ones(n, dtype=float),
        "trans": np.ones(n, dtype=float),
    }


def apply_event_on_raw(df: pd.DataFrame, idx: np.ndarray, profile: str, rng: np.random.Generator) -> None:
    """Multiplie les colonnes capteurs brutes à *idx* par des multiplicateurs propres au profil (in-place)."""
    n = len(idx)
    m = _profile_to_raw_multipliers(profile, n, rng)
    if STEPS in df.columns:
        df.loc[idx, STEPS] = np.maximum(0.0, df.loc[idx, STEPS].to_numpy() * m["steps"])
    if MI in df.columns:
        df.loc[idx, MI] = np.maximum(0.0, df.loc[idx, MI].to_numpy() * m["mi"])
    if LYING in df.columns:
        df.loc[idx, LYING] = np.maximum(0.0, df.loc[idx, LYING].to_numpy() * m["lying"])
    if STANDING in df.columns:
        df.loc[idx, STANDING] = np.maximum(0.0, df.loc[idx, STANDING].to_numpy() * m["standing"])
    if TRANSITIONS in df.columns:
        df.loc[idx, TRANSITIONS] = np.maximum(0.0, df.loc[idx, TRANSITIONS].to_numpy() * m["trans"])
    if TR_UP in df.columns:
        df.loc[idx, TR_UP] = np.maximum(0.0, df.loc[idx, TR_UP].to_numpy() * m["trans"])
    if TR_DOWN in df.columns:
        df.loc[idx, TR_DOWN] = np.maximum(0.0, df.loc[idx, TR_DOWN].to_numpy() * m["trans"])

