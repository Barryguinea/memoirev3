"""Profils synthétiques graduels pour la validation technique.

Les amplitudes ne sont pas des seuils cliniques. Le profil modere est ancre sur
Paudyal et al. (2021), qui rapportent environ 29 % de pas en moins et 9 % de
temps couche en plus la veille d'un parage therapeutique. Les autres niveaux
encadrent cette valeur pour tester la sensibilite du systeme.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticProfile:
    duration_hours: float
    steps_change: float
    motion_change: float
    transitions_change: float
    posture_shift: float
    expected_alert: int
    evidence_basis: str


PROFILES: dict[str, SyntheticProfile] = {
    "gradual_mild": SyntheticProfile(
        duration_hours=36,
        steps_change=-0.12,
        motion_change=-0.10,
        transitions_change=-0.12,
        posture_shift=0.03,
        expected_alert=1,
        evidence_basis="Niveau de sensibilite conservateur, inferieur a l'effet publie.",
    ),
    "gradual_moderate": SyntheticProfile(
        duration_hours=48,
        steps_change=-0.29,
        motion_change=-0.25,
        transitions_change=-0.25,
        posture_shift=0.07,
        expected_alert=1,
        evidence_basis="Pas ancres sur Paudyal et al. 2021; autres signaux = hypotheses coherentes.",
    ),
    "gradual_marked": SyntheticProfile(
        duration_hours=60,
        steps_change=-0.42,
        motion_change=-0.38,
        transitions_change=-0.40,
        posture_shift=0.12,
        expected_alert=1,
        evidence_basis="Scenario marque de sensibilite, non calibre cliniquement.",
    ),
    "isolated_short_variation": SyntheticProfile(
        duration_hours=3,
        steps_change=0.15,
        motion_change=0.00,
        transitions_change=0.00,
        posture_shift=0.00,
        expected_alert=0,
        evidence_basis="Controle negatif: variation isolee et breve, sans coherence multivariee.",
    ),
}


__all__ = ["PROFILES", "SyntheticProfile"]
