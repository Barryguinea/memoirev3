"""Garantit qu'une injection hybride modifie effectivement au moins un signal.

Regression : l'injection est multiplicative et ponderee par une enveloppe. Sur
une fenetre ou le signal cible est nul (ou nul la ou l'enveloppe est non nulle),
elle ne changeait rien, rendant certains controles vides. La selection de fenetre
doit desormais glisser vers la position effective la plus proche.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.io import COW, TIME, STEPS, MI, TRANSITIONS, LYING, STANDING, TR_UP, TR_DOWN
from validation_hybrid.campaign import inject_profile
from validation_hybrid.profiles import HYBRID_PROFILES


def _sparse_cow(active_bins: range, n: int = 500) -> pd.DataFrame:
    """Vache dont l'activite est nulle partout sauf sur une fenetre tardive."""
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    steps = np.zeros(n)
    motion = np.zeros(n)
    trans = np.zeros(n)
    steps[active_bins.start : active_bins.stop] = 20.0
    motion[active_bins.start : active_bins.stop] = 60.0
    trans[active_bins.start : active_bins.stop] = 2.0
    return pd.DataFrame(
        {
            COW: "9999",
            TIME: ts,
            STEPS: steps,
            MI: motion,
            TRANSITIONS: trans,
            TR_UP: trans / 2.0,
            TR_DOWN: trans / 2.0,
            LYING: np.full(n, 7.0),
            STANDING: np.full(n, 8.0),
        }
    )


def _realized_change(injected: pd.DataFrame, source: pd.DataFrame, event: pd.Series) -> float:
    mask = (injected[TIME] >= event["start"]) & (injected[TIME] <= event["end"])
    total = 0.0
    for column in [STEPS, MI, TRANSITIONS, LYING, STANDING]:
        a = pd.to_numeric(injected.loc[mask, column], errors="coerce").fillna(0.0).to_numpy(float)
        b = pd.to_numeric(source.loc[mask.values, column], errors="coerce").fillna(0.0).to_numpy(float)
        total += float(np.abs(a - b).sum())
    return total


def test_sensor_spike_injection_slides_to_an_effective_window():
    # Activite concentree en fin de periode future, loin de la fenetre par defaut.
    source = _sparse_cow(active_bins=range(480, 495)).sort_values(TIME).reset_index(drop=True)
    heldout = source[TIME].iloc[300]
    injected, event = inject_profile(
        source,
        cow="9999",
        scenario="isolated_sensor_spike",
        interval="15T",
        heldout_start=heldout,
        schedule_index=0,
    )
    assert bool(event["injection_effective"])
    assert _realized_change(injected, source, event) > 0.0


def test_every_hybrid_profile_injects_a_real_change_on_an_active_cow():
    # n large pour accueillir le profil sequentiel (72 h) dans la periode future.
    source = _sparse_cow(active_bins=range(310, 795), n=800).sort_values(TIME).reset_index(drop=True)
    heldout = source[TIME].iloc[300]
    for index, scenario in enumerate(HYBRID_PROFILES):
        injected, event = inject_profile(
            source,
            cow="9999",
            scenario=scenario,
            interval="15T",
            heldout_start=heldout,
            schedule_index=index,
        )
        assert bool(event["injection_effective"]), scenario
        assert _realized_change(injected, source, event) > 0.0, scenario
