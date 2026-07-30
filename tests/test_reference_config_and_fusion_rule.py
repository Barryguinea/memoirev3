"""Verrous sur les valeurs de reference et sur la regle de fusion hierarchique.

Ces tests ne verifient pas un comportement statistique: ils epinglent les
constantes publiees dans le manuscrit et la semantique de la fusion. Ils
existent pour qu'une modification du code de production qui contredirait le
Tableau 3.2 ou la section 3.4 fasse echouer la suite, et non pour mesurer une
performance.
"""

import pandas as pd

from core.early_warning import EarlyWarningConfig
from core.hybrid_warning import (
    HybridFusionConfig,
    InstabilityWarningConfig,
    apply_hybrid_warning,
)

DAY_BINS = 96  # nombre d'intervalles de 15 minutes dans une journee


def test_reference_configs_match_manuscript_table_3_2():
    """Tableau 3.2 « Parametres de reference des deux branches »."""
    hypo = EarlyWarningConfig()
    instability = InstabilityWarningConfig()

    # Colonne HYPO du tableau, ligne par ligne.
    assert hypo.aggregation_hours == 12.0
    assert hypo.persistence_hours == 6.0
    assert hypo.cooldown_hours == 24.0
    assert hypo.score_threshold == 0.12
    assert hypo.cusum_drift == 0.07
    assert hypo.cusum_threshold == 1.20
    assert hypo.min_families == 3
    assert hypo.coverage_min_pct == 25.0

    # Colonne INSTABILITE du tableau, ligne par ligne.
    assert instability.aggregation_hours == 3.0
    assert instability.persistence_hours == 2.0
    assert instability.cooldown_hours == 12.0
    assert instability.score_threshold == 0.18
    assert instability.cusum_drift == 0.06
    assert instability.cusum_threshold == 0.70
    assert instability.min_families == 2
    assert instability.coverage_min_pct == 25.0

    # Seuils de changement par famille, decrits en section 3.4.
    assert hypo.family_min_change == 0.10
    assert hypo.posture_min_change == 0.05
    assert hypo.active_families == ("steps", "motion", "transitions", "posture")
    assert instability.restless_motion_min_change == 0.20
    assert instability.transition_density_min_change == 0.20
    assert instability.fragmentation_min_change == 0.20
    assert instability.posture_volatility_min_change == 0.20
    assert instability.coordinated_activity_min_change == 0.15
    assert instability.coordinated_activity_tolerance == 0.20

    # Configuration de fusion de reference: hierarchique, refractaire 12 h,
    # fenetre de sequence bornee a 72 h.
    fusion = HybridFusionConfig()
    assert fusion.mode == "HIERARCHICAL"
    assert fusion.cooldown_hours == 12.0
    assert fusion.sequence_max_hours == 72.0


def _synthetic_frame(*, instability_window=None, hypo_window=None):
    """Trame de dix jours: sept de baseline, trois d'evaluation.

    Les signaux sont plats par defaut. Une fenetre d'instabilite eleve le
    Motion Index et les transitions en laissant les pas inchanges: cela touche
    restless_motion, transition_density et fragmentation sans activer le filtre
    coordinated_activity, qui exige une hausse conjointe des pas.
    """
    total = 10 * DAY_BINS
    baseline = 7 * DAY_BINS
    frame = pd.DataFrame(
        {
            "T": pd.date_range("2026-01-01", periods=total, freq="15min"),
            "dataset_split": ["baseline"] * baseline + ["futur"] * (total - baseline),
            "coverage_pct": [100.0] * total,
            "Steps_sum": [40.0] * total,
            "Motion Index_sum": [60.0] * total,
            "Transitions_sum": [6.0] * total,
            "Lying Time_sum": [30.0] * total,
            "Standing Time_sum": [30.0] * total,
            "behavioral_warning_episode": [0] * total,
            "behavioral_warning_start": [0] * total,
            "behavioral_warning_score": [0.0] * total,
        }
    )
    if instability_window is not None:
        first, last = instability_window
        frame.loc[first:last, "Motion Index_sum"] = 150.0
        frame.loc[first:last, "Transitions_sum"] = 15.0
    if hypo_window is not None:
        first, last = hypo_window
        frame.loc[first:last, "behavioral_warning_episode"] = 1
        frame.loc[first, "behavioral_warning_start"] = 1
        frame.loc[first:last, "behavioral_warning_score"] = 0.40
    return frame


def _fuse(frame):
    return apply_hybrid_warning(
        frame,
        interval="15T",
        fusion_config=HybridFusionConfig(mode="HIERARCHICAL"),
    )


def test_isolated_instability_is_surveillance_and_never_an_episode():
    """Une instabilite seule ne doit jamais produire de notification."""
    out = _fuse(_synthetic_frame(instability_window=(864, 911)))

    # Le fixture doit reellement declencher la branche, sans quoi le test
    # verifierait seulement que rien ne se passe quand rien ne se passe.
    assert out["instability_warning_episode"].sum() > 0

    assert out["hybrid_warning_episode"].max() == 0
    assert out["hybrid_warning_surveillance"].max() == 1
    assert out["hybrid_warning_notification"].sum() == 0
    assert out["hybrid_warning_priority"].max() == 1
    assert set(out["hybrid_warning_type"]) == {"AUCUN", "INSTABILITE"}


def test_isolated_hypoactivity_produces_an_episode_at_priority_two():
    """Une hypoactivite seule declenche la verification."""
    out = _fuse(_synthetic_frame(hypo_window=(864, 911)))

    assert out["instability_warning_episode"].sum() == 0
    assert out["hybrid_warning_episode"].max() == 1
    assert out["hybrid_warning_surveillance"].max() == 0
    assert out["hybrid_warning_notification"].sum() == 1
    assert out["hybrid_warning_priority"].max() == 2
    assert set(out["hybrid_warning_type"]) == {"AUCUN", "HYPO"}


def test_instability_followed_by_hypoactivity_raises_priority_to_three():
    """Un depart HYPO precede d'un depart INSTABILITE dans les 72 h."""
    out = _fuse(
        _synthetic_frame(instability_window=(768, 815), hypo_window=(864, 911))
    )

    assert out["instability_warning_episode"].sum() > 0
    assert out["hybrid_warning_sequence_start"].sum() == 1
    assert out["hybrid_warning_priority"].max() == 3
    assert "SEQUENCE" in set(out["hybrid_warning_type"])


def test_surveillance_and_episode_are_mutually_exclusive():
    """L'asymetrie de la regle: aucun intervalle ne peut porter les deux."""
    out = _fuse(
        _synthetic_frame(instability_window=(768, 815), hypo_window=(864, 911))
    )
    overlap = out["hybrid_warning_episode"].astype(bool) & out[
        "hybrid_warning_surveillance"
    ].astype(bool)
    assert not overlap.any()
