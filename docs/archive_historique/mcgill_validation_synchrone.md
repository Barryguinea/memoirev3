# Confrontation exploratoire synchronisée aux scores SLS — Winter 2019

**Statut : reproductible, mais non concluante sur le plan clinique.** Cette analyse utilise
des données capteurs strictement antérieures au score SLS du 12 mars 2019. Elle mesure une
concordance exploratoire entre les alertes comportementales et un score observationnel; elle
n'estime ni une sensibilité ni une spécificité diagnostique de la boiterie.

Commande de reproduction :

```bash
cd /Users/alioubarry/PROJECT/version_anterieure
python -m validation_hypo.mcgill_sync_validation
```

Les sorties sont écrites dans `data/validation/mcgill/`.

## Protocole figé

- Point d'évaluation unique : **12 mars 2019 à 00 h 00**.
- Données utilisées : mesures IceTag Winter 2019 strictement antérieures à ce point.
- Fenêtre d'analyse principale : les **sept jours précédents**.
- Antériorité minimale exigée avant cette fenêtre : **20 jours**.
- Couverture minimale de la fenêtre principale : **70 %** des intervalles attendus.
- Configuration du détecteur : gelée avant l'analyse, sans ajustement sur les SLS.
- Unité statistique : **la vache**.

Le classeur SLS contient quatre composantes binaires (`Edge`, `Rest`, `Shiftwt`, `Uneven`).
Le score vérifié varie donc de **0 à 4**, et non de 0 à 8. Le seuil SLS >= 2 est utilisé
uniquement pour une comparaison exploratoire; sa signification clinique n'est pas documentée
dans le classeur fourni.

## Cohorte analysée

Après application des critères temporels et de couverture, 14 vaches sont évaluables :
2 avec SLS=0, 9 avec SLS=1 et 3 avec SLS=2. Trois animaux sont exclus en raison d'une
couverture insuffisante ou de l'absence du score du 12 mars. Aucun cas SLS 3 ou 4 n'est
disponible.

## Résultats

| Mesure sur les sept jours précédents | AUC exploratoire | Test de Mann–Whitney | Corrélation de Spearman avec le SLS |
|---|---:|---:|---:|
| Nombre de notifications | 0,955 | p=0,020 | rho=0,699; p=0,005 |
| Fraction du temps en épisode | 0,712 | p=0,311 | rho=0,217; p=0,456 |
| Score d'alerte maximal | 0,788 | p=0,170 | rho=0,351; p=0,219 |

Le nombre moyen de notifications est de 5,0 chez les trois SLS >= 2 et de 3,18 chez les
onze SLS < 2. Cette association ne suffit pas à valider le système : elle repose sur trois
positifs seulement, sur une fenêtre déjà identifiée au cours de l'exploration, et les autres
mesures ne montrent pas une séparation stable.

Le taux de fond sur la partie future du pipeline atteint **0,511 notification par
vache-jour**, avec **54,6 %** du temps en épisode. Ce régime d'alerte élevé réduit fortement
la portée opérationnelle du résultat.

## Déséquilibre expérimental

Les trois SLS >= 2 appartiennent au bras `Exercise`; aucun n'appartient au bras
`No_Exercise`. Parmi les SLS < 2 dont le traitement est connu, trois sont dans `Exercise` et
sept dans `No_Exercise` (test exact de Fisher p=0,070). Le statut SLS est donc fortement
entremêlé au traitement. Les différences observées peuvent refléter l'accès au mouvement ou
d'autres effets du protocole, plutôt qu'une dégradation locomotrice spécifique.

## Interprétation autorisée

L'analyse montre qu'une concordance exploratoire peut apparaître entre le nombre d'alertes
et le SLS dans une fenêtre synchronisée. Elle ne démontre pas la détection clinique de la
boiterie, car l'effectif positif est trop faible, le seuil SLS n'est pas documenté comme seuil
diagnostique, le traitement est déséquilibré et le taux d'alerte de fond est élevé.

Le benchmark synthétique reste donc l'évaluation technique principale. Cette confrontation
sur une autre période instrumentée de la même ferme est utile pour révéler les limites de
transférabilité temporelle et la faible spécificité étiologique des signaux d'activité;
elle ne constitue pas une validation externe inter-fermes.

## Corrections par rapport aux anciennes analyses

1. Les scores de janvier et mars 2019 ne sont plus associés aux capteurs de novembre et
   décembre 2019.
2. La feuille du 12 mars est chargée explicitement; `read_excel()` ne choisit plus la feuille
   de janvier par défaut.
3. Aucune mesure postérieure au score n'entre dans les variables évaluées.
4. L'échelle SLS est vérifiée comme une somme de quatre composantes binaires, soit 0 à 4.
5. Le seuil SLS >= 2 demeure exploratoire et n'est pas transformé en diagnostic clinique.
6. Les anciens chiffres issus du corpus désynchronisé sont retirés du mémoire principal.
