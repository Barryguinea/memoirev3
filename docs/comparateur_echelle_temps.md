# Isolation Forest et LOF à l'échelle de temps de HYPO

## La question

Dans l'ablation du chapitre 6, Isolation Forest et LOF (variantes B, C et D)
reçoivent des variables construites sur quelques heures : z-scores robustes
glissants sur 24 intervalles, soit 6 heures, différences premières et écart à
une moyenne de sept intervalles. Les dégradations injectées durent de 36 à
60 heures. Leur échec de localisation pourrait donc tenir à l'échelle de temps
de leurs variables plutôt qu'à leur principe de détection.

Cette analyse teste cette objection en donnant à Isolation Forest et à LOF
exactement les variables que lit HYPO.

## Les quatre variantes ajoutées

Les quatre variantes reçoivent les cinq ratios de HYPO : pas, Motion Index,
transitions, temps couché et temps debout, chacun sur un total glissant de
12 heures rapporté à la référence individuelle du même créneau horaire. Les
ratios sont mis à l'échelle comme dans l'ablation (RobustScaler, intervalle
10-90), et les modèles apprennent sur les seuls intervalles de la période de
référence.

- **F. IF ponctuel** et **G. IF + persistance** : Isolation Forest avec les
  hyperparamètres figés du comparateur (Tableau 3.2).
- **H. LOF ponctuel** et **I. LOF + persistance** : LOF réglé comme la variante D
  de l'ablation (mode nouveauté, au plus 20 voisins, contamination de 0,06).
- Les variantes ponctuelles forment un épisode de chaque intervalle jugé
  anormal pendant la période future ; les variantes avec persistance exigent au
  moins 45 % d'intervalles anormaux sur six heures, comme HYPO.

Toutes notifient au plus une fois par 24 heures, comme HYPO. Mêmes onze vaches,
mêmes quarante-quatre événements, même attribution par soustraction de
l'exécution propre que l'ablation principale.

## Résultats

| Variante | Nouveau départ | Couverture attribuable | IoU20 | IoU moyen | Fond / vache-jour | F1 |
|---|---|---|---|---|---|---|
| A. HYPO | 43,2 % | 65,9 % | 29,5 % | 0,137 | 0,402 | 0,73 |
| F. IF ponctuel, ratios 12 h | 22,7 % | 20,5 % | 0,0 % | 0,002 | 0,330 | 0,47 |
| G. IF + persistance, ratios 12 h | 6,8 % | 11,4 % | 0,0 % | 0,001 | 0,143 | 0,17 |
| H. LOF ponctuel, ratios 12 h | 45,5 % | 47,7 % | 0,0 % | 0,009 | 0,669 | 0,75 |
| I. LOF + persistance, ratios 12 h | 20,5 % | 15,9 % | 0,0 % | 0,012 | 0,286 | 0,43 |

Tests de Wilcoxon appariés par vache :

| Comparaison | IoU | Vaches favorisant HYPO (IoU) | Nouveau départ |
|---|---|---|---|
| A contre F | p = 0,00098 | 11 sur 11 | p = 0,03125 |
| A contre G | p = 0,00098 | 11 sur 11 | p = 0,00195 |
| A contre H | p = 0,00098 | 11 sur 11 | p = 1,0 |
| A contre I | p = 0,00098 | 11 sur 11 | p = 0,03125 |

## Lecture

Même alimentés par les ratios sur 12 heures de HYPO, Isolation Forest et LOF ne
localisent aucun événement au seuil IoU20, et leur IoU moyen reste inférieur à
0,013. L'avantage de localisation de HYPO sur ces quatre variantes tient chez les
onze vaches. Fournir aux comparateurs des variables à la même échelle de temps ne
suffit donc pas à combler cet écart.

L'avantage ne porte pas sur la détection. LOF ponctuel ouvre autant de nouveaux
départs que HYPO (45,5 contre 43,2 %, p = 1,0) et atteint un F1 comparable (0,75
contre 0,73), au prix d'une charge de fond plus élevée (0,669 contre 0,402
notification par vache-jour) et sans localiser les événements. Ce constat rejoint
celui de `docs/politique_mad.md` : le résultat le mieux soutenu est l'avantage de
localisation, non une supériorité de HYPO sur toutes les métriques.

## Portée

L'analyse porte sur ces quatre variantes, avec les hyperparamètres figés des
comparateurs. Elle ne couvre pas toutes les configurations possibles
d'Isolation Forest ou de LOF. Elle ne dit pas non plus quelle part de l'avantage
de HYPO revient à la direction du changement recherchée, à la concordance entre
familles ou à l'accumulation par CUSUM : ces ingrédients n'ont pas été isolés.

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
