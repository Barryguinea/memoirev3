# Interprétation de l'ablation actuelle

## Métriques

Les métriques sont calculées après retrait des intervalles déjà en alerte dans l'exécution
propre de la même vache:

- `detection_any`: au moins un intervalle attribuable recouvre l'événement;
- `detection_iou20`: meilleur IoU attribuable supérieur ou égal à 0,20;
- `best_iou`: meilleur IoU attribuable;
- fond par vache-jour: notifications de l'exécution propre pendant la période post-baseline.

## Résultats

| Méthode | Recouvrement | IoU20 | IoU moyen | Fond/vache-jour |
|---|---:|---:|---:|---:|
| Alerte temporelle | 43,2 % | 29,5 % | 0,137 | 0,402 |
| IF + règles | 15,9 % | 0,0 % | 0,004 | 0,152 |
| IF ponctuel | 13,6 % | 0,0 % | 0,002 | 7,658 |
| LOF + règles | 15,9 % | 0,0 % | 0,005 | 0,107 |

L'alerte temporelle localise mieux les dégradations graduelles, au prix d'une charge de
fond plus élevée que les variantes à règles. IF ponctuel génère une charge incompatible
avec une utilisation opérationnelle. Aucun contrôle bref n'obtient de recouvrement
attribuable.

Les tests de Wilcoxon sont agrégés par vache. Les tests de McNemar des anciennes campagnes
ne soutiennent pas cette ablation.

Source: `data/revalidation/ablation_summary.csv` et
`data/revalidation/ablation_tests_by_cow.csv`.
