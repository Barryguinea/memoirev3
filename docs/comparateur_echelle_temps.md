# Isolation Forest à l'échelle de temps de HYPO

## La question

Dans l'ablation du chapitre 6, Isolation Forest et LOF (variantes B, C et D)
reçoivent des variables construites sur quelques heures : z-scores robustes
glissants sur 24 intervalles, soit 6 heures, différences premières et écart à
une moyenne de sept intervalles. Les dégradations injectées durent de 36 à
60 heures. Leur échec de localisation pourrait donc tenir à l'échelle de temps
de leurs variables plutôt qu'à leur principe de détection.

Cette analyse répond à cette objection en donnant à Isolation Forest exactement
les variables que lit HYPO.

## Les deux variantes ajoutées

Les deux variantes reçoivent les cinq ratios de HYPO : pas, Motion Index,
transitions, temps couché et temps debout, chacun sur un total glissant de
12 heures rapporté à la référence individuelle du même créneau horaire. Isolation
Forest reprend les hyperparamètres figés du comparateur (Tableau 3.2) et apprend
sur les seuls intervalles de la période de référence.

- **F. IF ponctuel sur ratios 12 h** : chaque intervalle jugé anormal pendant la
  période future forme un épisode.
- **G. IF + persistance HYPO sur ratios 12 h** : un épisode exige au moins 45 %
  d'intervalles anormaux sur six heures, comme la persistance de HYPO.

Les deux notifient au plus une fois par 24 heures, comme HYPO. Mêmes onze
vaches, mêmes quarante-quatre événements, même attribution par soustraction de
l'exécution propre que l'ablation principale.

## Résultats

| Variante | Nouveau départ | Couverture attribuable | IoU20 | IoU moyen | Fond / vache-jour | F1 |
|---|---|---|---|---|---|---|
| A. HYPO | 43,2 % | 65,9 % | 29,5 % | 0,137 | 0,402 | 0,73 |
| F. IF ponctuel, ratios 12 h | 22,7 % | 20,5 % | 0,0 % | 0,002 | 0,330 | 0,47 |
| G. IF + persistance, ratios 12 h | 6,8 % | 11,4 % | 0,0 % | 0,001 | 0,143 | 0,17 |

Tests de Wilcoxon appariés par vache :

| Comparaison | IoU | Vaches favorisant HYPO | Nouveau départ |
|---|---|---|---|
| A contre F | p = 0,00098 | 11 sur 11 | p = 0,03125 |
| A contre G | p = 0,00098 | 11 sur 11 | p = 0,00195 |

## Lecture

Même alimenté par les ratios sur 12 heures de HYPO, Isolation Forest ne localise
aucun événement au seuil IoU20, et son IoU moyen reste inférieur à 0,003.
L'avantage de localisation de HYPO tient chez les onze vaches. Il ne s'explique
donc pas par la seule échelle de temps des variables : il tient à la manière de
les exploiter, c'est-à-dire à la direction du changement recherchée, à la
concordance entre familles et à l'accumulation par CUSUM.

La conclusion porte sur ces deux variantes, avec les hyperparamètres figés du
comparateur. Elle ne couvre pas toutes les configurations possibles
d'Isolation Forest.

## Commande

```bash
python scripts/compute_if_timescale_sensitivity.py
```

Le script exige le corpus confidentiel `data/brut.csv`. Il écrit dans
`data/validation/if_timescale_sensitivity/` les événements, le résumé par
variante, les tests appariés et une provenance, avec un manifeste propre
vérifiable depuis ce dossier :

```bash
shasum -a 256 -c artifacts.sha256
```

Il refuse d'écrire dans un dossier scellé. Le code est dans
`validation_hypo/timescale_comparators.py` ; l'ablation scellée n'est pas
modifiée.
