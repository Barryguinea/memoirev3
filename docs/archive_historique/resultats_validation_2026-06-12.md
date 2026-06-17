# Résultats archivés de la validation du 12 juin 2026

## Portée

Cette campagne évalue techniquement un outil d'alerte comportementale précoce. Elle
n'estime ni sensibilité, ni spécificité, ni probabilité clinique de boiterie.

## Protocole

- 11 vaches possédant une baseline exploitable et une période future informative;
- 44 événements physiques uniques: 33 dégradations graduelles et 11 contrôles brefs;
- quatre fenêtres non chevauchantes par vache, toutes postérieures à la baseline;
- configuration du détecteur fixée avant l'évaluation et enregistrée dans
  `data/validation/protocol_configuration.json`;
- comparaison, à chaque instant, avec l'exécution propre de la même vache;
- unité statistique des tests appariés: la vache.

La campagne constitue un benchmark technique post-baseline à configuration fixe. Le même
corpus a servi à la conception du détecteur; il ne s'agit donc pas d'une validation externe
indépendante.

## Perturbations réalisées

| Profil | n | Pas | Motion Index | Transitions |
|---|---:|---:|---:|---:|
| Léger | 11 | -9,25 % | -7,77 % | -9,22 % |
| Modéré | 11 | -22,33 % | -19,16 % | -18,26 % |
| Marqué | 11 | -34,41 % | -31,08 % | -29,79 % |
| Contrôle bref | 11 | +11,93 % | 0,00 % | 0,00 % |

Ces valeurs sont les changements moyens effectivement obtenus après application des
profils aux fenêtres réelles. Les amplitudes cibles restent documentées séparément.

## Résultats attribuables

Un intervalle n'est crédité à l'injection que s'il est en alerte dans l'exécution injectée
et ne l'est pas au même instant dans l'exécution propre.

| Profil | n | Nouveau début | Couverture attribuable | IoU20 attribuable | IoU moyen attribuable |
|---|---:|---:|---:|---:|---:|
| Léger | 11 | 45,5 % | 72,7 % | 27,3 % | 0,087 |
| Modéré | 11 | 63,6 % | 90,9 % | 36,4 % | 0,192 |
| Marqué | 11 | 63,6 % | 100,0 % | 54,5 % | 0,269 |
| Contrôle bref | 11 | 0,0 % | 0,0 % | 0,0 % | 0,000 |

Pour les 33 dégradations graduelles:

- nouveau début: 19/33, soit 57,6 %;
- couverture attribuable: 29/33, soit 87,9 %;
- IoU20 attribuable: 13/33, soit 39,4 %;
- IoU moyen attribuable: 0,183;
- délai moyen lorsqu'un nouveau début existe: 20,3 h; médiane: 20,8 h.

Les métriques brutes, conservées comme diagnostic, sont plus élevées parce qu'elles
incluent les alertes déjà présentes sur les données propres: couverture 93,9 %, IoU20
81,8 % et IoU moyen 0,412.

## Charge de fond

- moyenne: 0,402 notification par vache-jour post-baseline;
- 90e percentile: 0,491 notification par vache-jour;
- ordre de grandeur: environ 40 notifications quotidiennes pour 100 vaches.

Ces notifications ne sont pas adjudicables en l'absence d'examen clinique contemporain.
Elles ne doivent pas être nommées faux positifs.

## Ablation

| Méthode | Recouvrement attribuable | IoU20 attribuable | IoU moyen attribuable | Fond/vache-jour |
|---|---:|---:|---:|---:|
| Alerte temporelle multivariée | 43,2 % | 29,5 % | 0,137 | 0,402 |
| IF + règles | 15,9 % | 0,0 % | 0,004 | 0,152 |
| IF ponctuel | 13,6 % | 0,0 % | 0,002 | 7,658 |
| LOF + règles | 15,9 % | 0,0 % | 0,005 | 0,107 |

Tests de Wilcoxon après agrégation par vache:

- alerte temporelle contre IF + règles: recouvrement p=0,04297; IoU p=0,00098;
- alerte temporelle contre IF ponctuel: recouvrement p=0,01562; IoU p=0,00098;
- alerte temporelle contre LOF + règles: recouvrement p=0,04297; IoU p=0,00098.

Les quatre variantes obtiennent un recouvrement attribuable nul sur les contrôles brefs.

## Revue manuelle

Douze cas ont été inspectés visuellement, trois par profil. Les neuf cas graduels montrent
des réponses cohérentes mais parfois tardives ou faiblement localisées; les trois contrôles
brefs ne produisent aucune réponse attribuable. Les annotations sont dans
`data/validation/manual_review_annotations.csv` et
`data/validation/manual_review_sample.csv`.

## Analyse synchronisée avec les scores SLS

L'analyse Winter 2019 porte sur 14 vaches et utilise uniquement les données antérieures au
score du 12 mars. L'AUC exploratoire du nombre de notifications sur sept jours est 0,955,
mais les trois vaches SLS >= 2 appartiennent toutes au groupe `Exercise`. Ce confondant,
la petite taille d'échantillon et l'instabilité des mesures secondaires rendent le résultat
non concluant pour la validation clinique.

## Formulation autorisée

> Le benchmark technique post-baseline montre une couverture attribuable de 87,9 %, un
> IoU20 attribuable de 39,4 % et un nouveau début d'alerte dans 57,6 % des dégradations
> graduelles. Les contrôles brefs ne produisent aucun recouvrement attribuable. La charge
> de fond atteint 0,402 notification par vache-jour. Ces résultats caractérisent un outil
> de priorisation comportementale et non un diagnostic clinique de boiterie.

## Artefacts de référence

- notebook exécuté: `notebooks/validation_complete_executed_2026-06-12.ipynb`;
- instantané: `data/validation/snapshot_2026-06-12.json`;
- configuration: `data/validation/protocol_configuration.json`;
- manifeste: `data/validation/validation_hypo.sha256`.
