# Résultats corrigés du prototype mémoire final

Date d'exécution: 12 juin 2026.

## Corrections appliquées

1. L'instabilité est définie par le Motion Index rapporté aux pas, les
   transitions rapportées aux pas, la fragmentation posturale et la volatilité
   couché/debout. Une activité coordonnée n'est plus assimilée à une alerte.
2. Les branches HYPO et INSTABILITE conservent des persistances distinctes.
3. Cinq fusions sont comparées: HYPO seule, INSTABILITE seule, `OR`,
   hiérarchique et séquentielle 24--72 h.
4. Les contrôles couvrent le pic capteur, l'exercice, la manipulation,
   l'activité de type œstrus et l'hypoactivité non locomotrice.
5. Les injections sont post-baseline, exécutées séparément et évaluées au
   niveau de la vache et de l'événement.
6. Les performances, délais, recouvrements, contrôles, confondants et
   notifications de fond sont mesurés conjointement.
7. Une sensibilité prédéfinie porte sur la persistance, les seuils et la fenêtre
   d'agrégation. Le SLS n'est jamais utilisé pour régler ces paramètres.

## Validation technique

La campagne comprend 11 vaches admissibles et 12 scénarios par vache, soit 132
événements indépendants. Les valeurs ci-dessous sont des performances
techniques sur injections, pas une sensibilité clinique.

| Fusion | HYPO actionnable | Instabilité surveillée | Séquence actionnable | Actionnable global | Contrôles | Confondants | IoU >= 0,20 | Fond/vache-jour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HYPO seule | 54,5 % | 66,7 % | 81,8 % | 61,4 % | 0,0 % | 40,9 % | 29,5 % | 0,446 |
| INSTABILITE seule | 24,2 % | 66,7 % | 81,8 % | 38,6 % | 0,0 % | 0,0 % | 0,0 % | 0,875 |
| `OR` | 57,6 % | 66,7 % | 90,9 % | 65,9 % | 0,0 % | 31,8 % | 11,4 % | 0,982 |
| Hiérarchique | 66,7 % | 66,7 % | 90,9 % | 72,7 % | 0,0 % | 40,9 % | 29,5 % | 0,571 |
| Séquentielle 24--72 h | 48,5 % | 66,7 % | 81,8 % | 56,8 % | 0,0 % | 36,4 % | 0,0 % | 0,366 |

La fusion hiérarchique est la seule variante qui satisfait les exigences
techniques prédéfinies de portée complète: au moins 60 % d'événements
actionnables, 50 % d'instabilités surveillées et 20 % d'IoU >= 0,20. Son score
d'utilité demeure toutefois négatif à cause des confondants et du fond.

## Lecture par scénario

- Aucun des 33 contrôles bénins n'a déclenché d'alerte attribuable.
- Aucun des 11 scénarios d'activité de type œstrus n'a déclenché la fusion
  hiérarchique.
- L'hypoactivité non locomotrice déclenche dans 9 cas sur 11 (81,8 %).
- L'instabilité est reconnue comme surveillance dans 45,5 % des profils légers,
  54,5 % des profils modérés et 100 % des profils marqués.
- La séquence instabilité puis hypoactivité est détectée dans 10 cas sur 11.
- L'hypoactivité légère demeure fragile: 45,5 % de détection.

La confusion restante n'est pas un simple défaut de seuil. Avec les variables
IceQube disponibles, une diminution persistante des pas, du Motion Index et des
transitions peut être compatible avec une douleur locomotrice, une maladie non
locomotrice ou un autre état physiologique. Le système doit donc rester un outil
de priorisation pour vérification terrain.

## Sensibilité des paramètres

Le relèvement des seuils d'instabilité améliore légèrement l'utilité technique
(-0,083 contre -0,088) et réduit le fond de 0,571 à 0,553 notification par
vache-jour, sans réduire l'alerte sur l'hypoactivité non locomotrice. Une
persistance de 3 h augmente au contraire les confondants à 45,5 %. Une fenêtre
de 4 h réduit la surveillance des instabilités à 54,5 %. Les résultats ne
justifient donc pas un changement de paramètres présenté comme cliniquement
optimal.

## Concordance SLS synchronisée

L'analyse McGill Winter 2019 utilise les signaux strictement antérieurs au SLS
du 12 mars 2019 et une fenêtre principale de sept jours. Quatorze vaches sont
évaluables: 3 ont un SLS >= 2 et 11 un SLS < 2.

Pour la fusion hiérarchique:

- notifications moyennes pré-7 jours: 6,67 pour SLS >= 2 contre 4,45;
- AUC descriptive: 0,924;
- test de Mann--Whitney: p = 0,0306;
- corrélation de Spearman avec le SLS: rho = 0,504, p = 0,0659;
- fond moyen sur la période future: 0,755 notification par vache-jour.

Ces nombres ne valident pas la détection clinique. Les trois vaches SLS >= 2
appartiennent toutes au bras `Exercise`, contre aucune dans `No_Exercise`. Le
traitement et le statut SLS sont donc fortement entrelacés, l'effectif positif
est de trois et le taux d'alerte de fond reste élevé. L'AUC est instable et peut
refléter l'exercice plutôt que le statut locomoteur.

## Conclusion décisionnelle

La correction rend la branche d'instabilité plus spécifique que la première
fusion `OR`: les hausses coordonnées et brèves sont rejetées, et l'instabilité
isolée n'est plus présentée comme une alerte de boiterie. La fusion hiérarchique
est la meilleure variante couvrant les deux signatures, mais elle n'est pas
prête à remplacer version antérieure comme résultat principal. Une validation clinique
plus large, des événements contextuels datés et des données permettant de
distinguer les maladies non locomotrices restent nécessaires.

## Sorties reproductibles

- `data/validation/hybrid_refined_full/comparison_summary.csv`;
- `data/validation/hybrid_refined_full/events_*.csv`;
- `data/validation/mcgill_sls/mcgill_cohort_all_variants.csv`;
- `data/validation/mcgill_sls/mcgill_metrics.csv`;
- `data/validation/mcgill_sls/mcgill_summary.json`;
- `logs/hybrid_refined_full_2026-06-12.log`.
