"""Definition du z-score robuste glissant : mode historique et MAD standard.

Voir docs/politique_mad.md. Ces tests n'exigent pas le corpus confidentiel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.features import (
    MAD_FACTOR,
    MIN_PERIODS_ROLLING,
    build_interval_features,
    rolling_robust_z,
)

WINDOW = 24


def _serie() -> pd.Series:
    rng = np.random.default_rng(7)
    values = rng.gamma(2.0, 5.0, size=200)
    values[[10, 55, 56, 120]] = np.nan
    values[80:90] += np.linspace(0.0, 60.0, 10)
    return pd.Series(values)


def test_le_mode_historique_reste_le_defaut() -> None:
    serie = _serie()
    pd.testing.assert_series_equal(
        rolling_robust_z(serie, WINDOW),
        rolling_robust_z(serie, WINDOW, mad_mode="historical"),
    )


def test_le_mode_window_suit_la_definition_du_mad_de_fenetre() -> None:
    serie = _serie()
    obtenu = rolling_robust_z(serie, WINDOW, mad_mode="window")
    minimum = MIN_PERIODS_ROLLING(WINDOW)
    verifies = 0
    for t in range(len(serie)):
        fenetre = serie.iloc[max(0, t - WINDOW + 1): t + 1].to_numpy()
        fenetre = fenetre[np.isfinite(fenetre)]
        if fenetre.size < minimum or not np.isfinite(serie.iloc[t]):
            continue
        mediane = np.median(fenetre)
        mad = np.median(np.abs(fenetre - mediane))
        if MAD_FACTOR * mad < 1e-9:
            continue
        attendu = (serie.iloc[t] - mediane) / (MAD_FACTOR * mad)
        assert obtenu.iloc[t] == pytest.approx(attendu, rel=1e-12, abs=1e-12)
        verifies += 1
    assert verifies > 150
    assert not np.allclose(obtenu, rolling_robust_z(serie, WINDOW, mad_mode="historical"))


def test_un_mode_inconnu_est_refuse() -> None:
    with pytest.raises(ValueError):
        rolling_robust_z(_serie(), WINDOW, mad_mode="global")


def test_le_mode_ne_modifie_que_les_z_scores_glissants() -> None:
    temps = pd.date_range("2023-10-01", periods=300, freq="15min")
    rng = np.random.default_rng(11)
    brut = pd.DataFrame({
        "Date": temps,
        "Steps": rng.poisson(4.0, size=len(temps)).astype(float),
        "Motion Index": rng.gamma(2.0, 12.0, size=len(temps)),
    })
    historique = build_interval_features(
        brut, time_col="Date", interval="15T", cols=["Steps", "Motion Index"], window_baseline=WINDOW
    )
    standard = build_interval_features(
        brut, time_col="Date", interval="15T", cols=["Steps", "Motion Index"],
        window_baseline=WINDOW, mad_mode="window",
    )
    glissants = [c for c in historique.columns if c.endswith("_rrz")]
    autres = [c for c in historique.columns if c not in glissants]
    assert glissants
    pd.testing.assert_frame_equal(historique[autres], standard[autres])
    assert not historique[glissants].equals(standard[glissants])
