"""Revalidation technique de l'alerte comportementale (memoirev2).

Ce package corrige les failles méthodologiques identifiées dans la campagne
d'origine, sans modifier le coeur ``core/`` (qui reste la source de vérité) :

1. Fuite train/éval — les injections étaient placées à 20 %, 54 % et 58 % de la
   série, donc DANS les 60 % utilisés comme baseline d'entraînement de l'IF.
   Ici, toutes les injections sont placées APRÈS la baseline (fenêtre 66–90 %),
   donc strictement dans la partie post-baseline.

2. Pseudo-réplication — la campagne primaire utilise une seed et une seule
   configuration. Les intervalles de confiance, la calibration et l'ablation
   utilisent la vache comme unité de regroupement.

3. Couverture multi-vaches — au lieu d'une seule vache cible par scénario, les
   événements sont répartis sur toutes les vaches du corpus → beaucoup plus de
   fenêtres physiques indépendantes.

4. SHAP aligné production — l'analyse reprend aussi les mêmes indices de
   baseline après filtrage de couverture et warm-up capteur.

Les profils graduels et le détecteur temporel sont propres à ``memoirev2``.
Les sorties historiques restent disponibles comme comparateurs.
"""

from revalidation.campaign import (
    inject_events_for_cow,
    run_clean_campaign,
)
from revalidation.ablation import (
    ablation_paired_tests,
    ablation_summary,
    run_clean_ablation,
)
from revalidation.qa import assert_campaign_valid, campaign_checks, manual_review_sample

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
