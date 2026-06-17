"""Validation technique de la branche HYPO de l'alerte comportementale.

Ce package contient la campagne principale de validation technique du module
HYPO, sans modifier le coeur ``core/`` qui reste la source de vérité :

1. Séparation temporelle — toutes les injections sont placées après la période
   de référence, donc strictement dans la partie post-baseline.

2. Unité statistique — la campagne primaire utilise une configuration gelée.
   Les intervalles de confiance, la calibration et l'ablation regroupent les
   résultats au niveau de la vache.

3. Couverture multi-vaches — les événements sont répartis sur les vaches
   admissibles du corpus.

Les profils graduels et le détecteur temporel servent à mesurer une alerte
comportementale attribuable, non une sensibilité clinique.
"""

from validation_hypo.campaign import (
    inject_events_for_cow,
    run_clean_campaign,
)
from validation_hypo.ablation import (
    ablation_paired_tests,
    ablation_summary,
    run_clean_ablation,
)
from validation_hypo.qa import assert_campaign_valid, campaign_checks, manual_review_sample

__all__ = [
    "inject_events_for_cow",
    "run_clean_campaign",
    "run_clean_ablation",
    "ablation_summary",
    "ablation_paired_tests",
    "campaign_checks",
    "assert_campaign_valid",
    "manual_review_sample",
]
