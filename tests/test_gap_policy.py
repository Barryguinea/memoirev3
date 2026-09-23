"""Politique des trous de donnees de HYPO et comparateur a l'echelle de 12 heures.

Voir docs/politique_trous_de_donnees.md et docs/comparateur_echelle_temps.md.
Ces tests reposent sur des donnees synthetiques et n'exigent pas le corpus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import config as C
from core.early_warning import (
    _rolling_total,
    _rolling_total_coverage_aware,
    apply_behavioral_early_warning,
)
from core.features import build_interval_features
from core.model_if import run_if_core
from validation_hypo.campaign import final_params
from validation_hypo.timescale_comparators import (
    IF_PERSISTENT,
    IF_POINT,
    timescale_variants,
)

COLONNES = ["Steps", "Motion Index", "Transitions", "Lying Time", "Standing Time"]


def _serie_brute(jours: int = 20) -> pd.DataFrame:
    temps = pd.date_range("2023-10-01", periods=jours * 96, freq="15min")
    heure = temps.hour.to_numpy() + temps.minute.to_numpy() / 60.0
    cycle = 1.0 + 0.5 * np.sin(2 * np.pi * heure / 24.0)
    couche = 7.5 + 3.0 * np.cos(2 * np.pi * heure / 24.0)
    return pd.DataFrame({
        "Date": temps,
        "Steps": 10.0 * cycle,
        "Motion Index": 20.0 * cycle,
        "Transitions": 0.5 * cycle,
        "Lying Time": couche,
        "Standing Time": 15.0 - couche,
    })


def _intervalles(brut: pd.DataFrame) -> pd.DataFrame:
    features = build_interval_features(
        brut, time_col="Date", interval="15T", cols=COLONNES, window_baseline=24
    )
    return run_if_core(
        features, time_col="T", baseline_ratio=0.6,
        sensor_warmup_bins=C.DEFAULT_SENSOR_WARMUP_BINS,
    )


def _brut_avec_trou() -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    brut = _serie_brute()
    debut = pd.Timestamp("2023-10-17 06:00")
    fin = debut + pd.Timedelta(hours=12)
    return brut[~((brut["Date"] >= debut) & (brut["Date"] < fin))], debut, fin


def test_la_politique_historique_reste_le_defaut() -> None:
    intervalles = _intervalles(_brut_avec_trou()[0])
    pd.testing.assert_frame_equal(
        apply_behavioral_early_warning(intervalles, interval="15T"),
        apply_behavioral_early_warning(intervalles, interval="15T", gap_policy="historical"),
    )


def test_un_intervalle_vide_ne_compte_plus_comme_une_activite_nulle() -> None:
    df = pd.DataFrame({"x": np.ones(200)})
    valide = pd.Series(True, index=df.index)
    valide.iloc[100:110] = False
    complet = _rolling_total_coverage_aware(df, "x", 48, pd.Series(True, index=df.index))
    pd.testing.assert_series_equal(complet, _rolling_total(df, "x", 48), check_names=False)
    avec_trou = _rolling_total_coverage_aware(df, "x", 48, valide)
    # Au-dela de 4 intervalles vides sur 48, la fenetre est mesuree a moins de 90 %.
    assert avec_trou.iloc[104:153].isna().all()
    assert avec_trou.iloc[100:104].eq(48.0).all()
    assert avec_trou.iloc[153:].eq(48.0).all()


def test_une_politique_inconnue_est_refusee() -> None:
    intervalles = _intervalles(_serie_brute(16))
    with pytest.raises(ValueError):
        apply_behavioral_early_warning(intervalles, interval="15T", gap_policy="zero")


def test_un_trou_seul_ne_declenche_hypo_qu_en_mode_historique() -> None:
    brut, debut, fin = _brut_avec_trou()
    intervalles = _intervalles(brut)
    temps = pd.to_datetime(intervalles["T"])
    apres = (temps >= debut) & (temps <= fin + pd.Timedelta(hours=24))
    departs = {}
    for politique in ("historical", "coverage_aware"):
        sortie = apply_behavioral_early_warning(intervalles, interval="15T", gap_policy=politique)
        departs[politique] = int(sortie.loc[apres.to_numpy(), "behavioral_warning_start"].sum())
    assert departs["historical"] >= 1
    assert departs["coverage_aware"] == 0


def test_le_comparateur_a_12_heures_n_alerte_que_dans_la_periode_future() -> None:
    brut = _serie_brute()
    features = build_interval_features(
        brut, time_col="Date", interval="15T", cols=COLONNES, window_baseline=24
    )
    sorties = timescale_variants(features, final_params())
    reference = _intervalles(brut)["dataset_split"].ne("futur").to_numpy()
    for nom in (IF_POINT, IF_PERSISTENT):
        assert sorties[nom].loc[reference, "pred_lameness_episode"].sum() == 0
