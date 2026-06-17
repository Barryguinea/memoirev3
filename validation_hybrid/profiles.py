"""Profils synthétiques prudents pour le prototype bidirectionnel.

Ces profils testent le comportement du code. Ils ne représentent ni une
prévalence, ni une sensibilité clinique, ni une chronologie universelle de la
boiterie. Les contrôles adversariaux documentent les confusions attendues.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HybridSyntheticProfile:
    duration_hours: float
    target_branch: str
    steps_change: float = 0.0
    motion_change: float = 0.0
    transitions_change: float = 0.0
    posture_shift: float = 0.0
    posture_oscillation: float = 0.0
    oscillation: float = 0.0
    sequence_gap_hours: float = 0.0
    secondary_duration_hours: float = 0.0
    secondary_steps_change: float = 0.0
    secondary_motion_change: float = 0.0
    secondary_transitions_change: float = 0.0
    secondary_posture_shift: float = 0.0
    expected_hypo: int = 0
    expected_instability: int = 0
    expected_hybrid: int = 0
    evidence_basis: str = ""


HYBRID_PROFILES: dict[str, HybridSyntheticProfile] = {
    "hypo_mild": HybridSyntheticProfile(
        duration_hours=24,
        target_branch="HYPO",
        steps_change=-0.12,
        motion_change=-0.10,
        transitions_change=-0.12,
        posture_shift=0.03,
        expected_hypo=1,
        expected_hybrid=1,
        evidence_basis="Dégradation graduelle conservatrice sous l'effet publié sur les pas.",
    ),
    "hypo_moderate": HybridSyntheticProfile(
        duration_hours=36,
        target_branch="HYPO",
        steps_change=-0.29,
        motion_change=-0.25,
        transitions_change=-0.25,
        posture_shift=0.07,
        expected_hypo=1,
        expected_hybrid=1,
        evidence_basis="Pas ancrés sur Paudyal et al.; autres signaux = hypothèses techniques.",
    ),
    "hypo_marked": HybridSyntheticProfile(
        duration_hours=48,
        target_branch="HYPO",
        steps_change=-0.42,
        motion_change=-0.38,
        transitions_change=-0.40,
        posture_shift=0.12,
        expected_hypo=1,
        expected_hybrid=1,
        evidence_basis="Scénario marqué de sensibilité, non calibré cliniquement.",
    ),
    "instability_mild": HybridSyntheticProfile(
        duration_hours=8,
        target_branch="INSTABILITE",
        steps_change=0.00,
        motion_change=0.25,
        transitions_change=0.30,
        posture_oscillation=0.06,
        oscillation=0.08,
        expected_instability=1,
        expected_hybrid=1,
        evidence_basis="Hausse multivariée modeste et répétée; hypothèse exploratoire.",
    ),
    "instability_moderate": HybridSyntheticProfile(
        duration_hours=12,
        target_branch="INSTABILITE",
        steps_change=0.05,
        motion_change=0.45,
        transitions_change=0.50,
        posture_oscillation=0.10,
        oscillation=0.12,
        expected_instability=1,
        expected_hybrid=1,
        evidence_basis="Instabilité multivariée persistante, sans pic extrême.",
    ),
    "instability_marked": HybridSyntheticProfile(
        duration_hours=16,
        target_branch="INSTABILITE",
        steps_change=0.08,
        motion_change=0.65,
        transitions_change=0.70,
        posture_oscillation=0.15,
        oscillation=0.18,
        expected_instability=1,
        expected_hybrid=1,
        evidence_basis="Scénario technique marqué, inférieur aux multiplicateurs de l'ancien protocole.",
    ),
    "isolated_sensor_spike": HybridSyntheticProfile(
        duration_hours=0.25,
        target_branch="CONTROLE",
        motion_change=2.00,
        expected_hybrid=0,
        evidence_basis="Contrôle négatif: un seul pic de Motion Index doit être rejeté.",
    ),
    "short_exercise": HybridSyntheticProfile(
        duration_hours=2,
        target_branch="CONTROLE",
        steps_change=0.45,
        motion_change=0.45,
        transitions_change=0.45,
        oscillation=0.05,
        expected_hybrid=0,
        evidence_basis="Contrôle négatif: hausse cohérente mais trop brève pour un épisode persistant.",
    ),
    "estrus_like_activity": HybridSyntheticProfile(
        duration_hours=6,
        target_branch="CONFONDANT",
        steps_change=0.40,
        motion_change=0.40,
        transitions_change=0.40,
        oscillation=0.10,
        expected_hybrid=0,
        evidence_basis="Contrôle adversarial: activité de type œstrus, indiscernable sans contexte externe.",
    ),
    "nonlocomotor_hypoactivity": HybridSyntheticProfile(
        duration_hours=24,
        target_branch="CONFONDANT",
        steps_change=-0.30,
        motion_change=-0.28,
        transitions_change=-0.25,
        posture_shift=0.08,
        expected_hybrid=0,
        evidence_basis="Contrôle adversarial: hypoactivité non locomotrice, causalité indiscernable par capteur seul.",
    ),
    "handling_manipulation": HybridSyntheticProfile(
        duration_hours=1,
        target_branch="CONTROLE",
        steps_change=0.10,
        motion_change=0.55,
        transitions_change=0.70,
        posture_oscillation=0.18,
        oscillation=0.20,
        expected_hybrid=0,
        evidence_basis="Contrôle négatif: manipulation intense mais trop brève pour une alerte persistante.",
    ),
    "instability_then_hypo": HybridSyntheticProfile(
        duration_hours=12,
        target_branch="SEQUENCE",
        steps_change=0.03,
        motion_change=0.50,
        transitions_change=0.55,
        posture_oscillation=0.12,
        oscillation=0.14,
        sequence_gap_hours=24,
        secondary_duration_hours=36,
        secondary_steps_change=-0.32,
        secondary_motion_change=-0.28,
        secondary_transitions_change=-0.30,
        secondary_posture_shift=0.08,
        expected_hypo=1,
        expected_instability=1,
        expected_hybrid=1,
        evidence_basis="Séquence technique: instabilité, délai de 24 h, puis hypoactivité persistante.",
    ),
}


__all__ = ["HybridSyntheticProfile", "HYBRID_PROFILES"]
