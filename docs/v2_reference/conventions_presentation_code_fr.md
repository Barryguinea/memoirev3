# Conventions de présentation du code et des résultats

Ce document fixe le vocabulaire de la version reproductible. Les noms historiques
`lameness_*` peuvent encore apparaître dans des modules archivés ou de compatibilité; ils ne
doivent pas être repris dans le mémoire ni interprétés comme des sorties cliniques.

## 1. Principe général

- Les identifiants Python et les colonnes CSV restent en anglais.
- Le mémoire décrit leur sens en français sans élargir leur portée.
- La sortie principale est une **alerte comportementale précoce** destinée à prioriser une
  vérification terrain.
- Le système ne pose pas de diagnostic de boiterie et n'estime pas une probabilité clinique.

## 2. Dictionnaire courant

| Code ou terme | Formulation française |
|---|---|
| `behavioral_warning_candidate` | intervalle candidat |
| `behavioral_warning_episode` | épisode d'alerte comportementale |
| `behavioral_warning_notification` | notification comportementale |
| `behavioral_warning_score` | score temporel multivarié |
| `if_anomaly_point` | anomalie ponctuelle Isolation Forest |
| `if_score` | score Isolation Forest |
| `cooldown` | période réfractaire entre notifications |
| `persist_hours` | durée minimale de persistance |
| `episode_overlap` | couverture attribuable de la fenêtre injectée |
| `detected_iou20` | recouvrement attribuable avec IoU supérieur ou égal à 0,20 |
| `background_notification_rate` | notifications de fond par vache-jour post-baseline |
| `CUSUM` | somme cumulative unilatérale |
| `LOF` | Local Outlier Factor, comparateur |
| `SHAP` | contribution des variables du comparateur IF |

## 3. Validation technique

La campagne reproductible utilise une configuration fixée avant l'évaluation. Les événements
synthétiques sont injectés uniquement après la baseline, sur onze vaches admissibles et sans
chevauchement physique. Chaque vache est aussi exécutée sans injection afin de soustraire les
alertes préexistantes.

Présenter séparément:

1. le taux de nouveaux débuts attribuables;
2. la couverture et l'IoU attribuables après soustraction de la référence;
3. les métriques brutes, uniquement comme diagnostic de confusion;
4. les notifications de fond sur la période post-baseline;
5. le contrôle bref isolé;
6. les résultats par vache ou par événement indépendant.

Ne pas appeler ces quantités sensibilité, spécificité, vrais positifs ou faux positifs. Aucun
label clinique contemporain n'est disponible dans le corpus principal.

## 4. Analyse IceTag--SLS

La confrontation de l'hiver 2019 est une analyse exploratoire synchronisée, non une validation
externe. Elle porte sur quatorze vaches, dont seulement trois ont un SLS supérieur ou égal à 2;
ces trois vaches appartiennent toutes au groupe `Exercise`. Toute AUC doit donc être accompagnée
de ce confond et du taux de notifications de fond.

## 5. Ablation et comparateurs

Les variantes sont évaluées sur les mêmes événements indépendants. La variante temporelle est
comparée à IF avec persistance, IF ponctuel et LOF avec règles. Les tests appariés portent sur
les événements; ils ne prouvent pas une supériorité clinique.

## 6. Formulations à éviter

- détection automatique ou diagnostic de boiterie;
- probabilité de boiterie;
- validation clinique ou externe;
- sensibilité et spécificité sans labels contemporains;
- faux positif pour une notification de fond non adjudiquée;
- meilleure configuration parmi une grille si cette sélection n'a pas été reproduite sur un
  corpus de développement séparé.

## 7. Formulation de référence

« Le détecteur temporel à configuration fixe repère des dégradations comportementales
persistantes par rapport à la baseline individuelle. La campagne post-baseline mesure une
capacité de récupération technique sur des événements synthétiques plausibles. Elle ne permet
pas d'inférer une performance diagnostique de la boiterie. »
