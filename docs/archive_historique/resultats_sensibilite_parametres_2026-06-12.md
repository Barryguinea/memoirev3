# Resultats de l'analyse de sensibilite des parametres - 12 juin 2026

## Portee

Cette analyse fait varier un seul parametre a la fois autour de la configuration
documentee du detecteur temporel. Elle mesure la robustesse technique du
benchmark synthetique post-baseline. Elle ne selectionne pas une nouvelle
configuration et ne constitue pas une validation clinique de la boiterie.

## Integrite et population

- 21 configurations: une reference et 20 variantes OFAT;
- 11 vaches et 44 evenements physiques par configuration;
- 33 degradations progressives et 11 variations breves de controle;
- meme graine (`11`) et memes fenetres physiques pour toutes les variantes;
- 924 evaluations evenement-configuration au total;
- manifeste SHA-256 verifie sans erreur pour tous les artefacts.

## Configuration de reference

| Indicateur | Resultat |
|---|---:|
| Nouveau debut sur les profils progressifs | 57,6 % (19/33) |
| Couverture attribuable | 87,9 % (29/33) |
| IoU >= 0,20 attribuable | 39,4 % (13/33) |
| IoU moyen attribuable | 0,183 |
| Delai median lorsqu'un nouveau debut existe | 20,75 h |
| Controle bref: nouveau debut / couverture / IoU20 | 0 % / 0 % / 0 % |
| Notifications de fond moyennes | 0,402 par vache-jour |
| Notifications de fond, 90e percentile | 0,491 par vache-jour |

## Robustesse globale

Les 20 variantes conservent toutes un resultat nul sur les controles brefs.
Douze variantes sur vingt conservent exactement le taux de nouveaux debuts de
la reference; quinze restent dans un ecart maximal de deux evenements sur 33
(6,1 points de pourcentage). Pour l'IoU20, quinze variantes sur vingt restent
dans un ecart maximal d'un evenement (3,0 points).

Sur l'ensemble de la grille, les indicateurs progressifs varient dans les
plages suivantes:

| Indicateur | Minimum | Maximum |
|---|---:|---:|
| Nouveau debut | 45,5 % | 66,7 % |
| Couverture attribuable | 63,6 % | 90,9 % |
| IoU20 attribuable | 27,3 % | 45,5 % |
| IoU moyen | 0,125 | 0,195 |
| Delai median | 15,5 h | 24,5 h |
| Fond moyen par vache-jour | 0,187 | 0,553 |

Ces plages montrent une stabilite partielle, mais aussi un compromis reel entre
reactivite, localisation temporelle et charge de fond.

## Parametres les plus influents

### Nombre minimal de familles concordantes

La valeur de reference est trois familles.

- Deux familles augmentent l'IoU20 et la couverture de 3,0 points, mais
  augmentent le fond de 0,107 notification par vache-jour (+26,7 %).
- Quatre familles diminuent les nouveaux debuts et l'IoU20 de 12,1 points,
  la couverture de 24,2 points et le fond de 53,4 %.

Trois familles constituent donc un compromis defensible entre sensibilite aux
degradations et charge de fond. La variante a quatre familles penalise surtout
le profil leger.

### Fenetre d'agregation

La valeur de reference est 12 heures.

- Six heures augmentent les nouveaux debuts de 9,1 points et reduisent le
  delai de 1,88 h, mais diminuent l'IoU20 de 12,1 points et augmentent le fond
  de 37,8 %. Le signal devient plus reactif, mais plus fragmente et plus bruyant.
- Vingt-quatre heures diminuent les nouveaux debuts de 9,1 points, la
  couverture de 18,2 points et le fond de 24,4 %, avec un delai accru de 3,75 h.

La fenetre de 12 heures est un compromis plausible entre reactivite et lissage;
elle n'est pas demontree comme optimum unique.

## Parametres a influence intermediaire

### Persistance

- Trois heures conservent le taux de nouveaux debuts et la couverture, reduisent
  le delai median de 5,25 h, mais augmentent le fond de 20,0 % et diminuent
  legerement l'IoU20.
- Douze heures diminuent les nouveaux debuts de 6,1 points et retardent de
  2,50 h, mais ameliorent l'IoU20 de 6,1 points et reduisent le fond de 13,3 %.

La valeur de six heures represente un compromis operationnel; l'analyse ne
permet pas d'affirmer qu'elle est statistiquement optimale.

### Baisse minimale d'activite

- Le seuil de 5 % augmente legerement la couverture, mais produit davantage de
  fond et moins de nouveaux debuts attribuables, notamment parce que des
  episodes peuvent deja etre actifs avant l'injection.
- Le seuil de 15 % reduit les nouveaux debuts, la couverture et l'IoU20.

Le seuil de 10 % est mieux equilibre dans ce benchmark que les deux valeurs
encadrantes.

### Seuil du score

Le seuil 0,08 est presque equivalent a 0,12, avec une faible hausse du fond.
Le seuil 0,16 diminue les nouveaux debuts et la couverture de 6,1 points. Les
resultats soutiennent surtout l'idee de ne pas durcir le seuil au-dela de 0,12;
ils ne distinguent pas nettement 0,08 de 0,12.

## Parametres peu discriminants sur ce corpus

- Le seuil postural modifie peu les indicateurs globaux, mais peut modifier le
  delai et certains evenements individuels.
- Les variantes de derive et de seuil CUSUM produisent des changements nuls ou
  tres faibles. L'analyse ne prouve donc pas que 0,07 et 1,20 sont optimaux;
  elle montre que le benchmark est peu sensible a ces valeurs dans les plages
  testees.
- Les couvertures minimales de 25 %, 50 % et 75 % donnent exactement les memes
  resultats, car le corpus utilise presente une couverture elevee. Cela ne
  permet pas de generaliser a des donnees plus incompletes.
- Le cooldown ne modifie pas les episodes ni leur localisation; il agit sur la
  frequence des notifications. Douze heures augmentent le fond de 11,1 % et
  48 heures le reduisent de 20,0 %. La campagne ne contient toutefois pas de
  recidives cliniquement distinctes rapprochees permettant d'evaluer le risque
  de supprimer une notification utile avec un cooldown de 48 heures.

## Conclusion defendable

La configuration actuelle n'est pas un point unique dont tous les resultats
s'effondrent au moindre changement. Les controles brefs restent correctement
rejetes dans toutes les variantes et la majorite des modifications moderees
conservent des performances proches de la reference. La robustesse est cependant
incomplete: l'agregation et le nombre de familles concordantes modifient
substantiellement le compromis entre nouveaux debuts, IoU et charge de fond.

La formulation correcte est donc:

> L'analyse de sensibilite OFAT montre que la configuration documentee est un
> compromis techniquement stable dans les plages testees, particulierement pour
> les seuils CUSUM, posturaux et de couverture. La fenetre d'agregation et le
> nombre minimal de familles sont les choix les plus structurants. Ces resultats
> ne demontrent ni optimalite globale ni validite clinique.

## Limites a conserver dans le memoire

- une seule graine et onze vaches;
- benchmark synthetique construit sur le meme corpus que le developpement;
- analyse OFAT ne mesurant pas les interactions entre parametres;
- pas de correction ou de test inferentiel pour comparer les 20 variantes;
- charge de fond elevee: environ 0,402 notification par vache-jour;
- delai median de 20,75 h, qui limite l'emploi non nuance du terme « precoce »;
- absence de reference clinique independante pour les evenements injectes.

## Artefacts

- `data/validation/parameter_sensitivity/parameter_sensitivity_summary.csv`;
- `data/validation/parameter_sensitivity/parameter_sensitivity_by_scenario.csv`;
- `data/validation/parameter_sensitivity/parameter_sensitivity_events.csv`;
- `data/validation/parameter_sensitivity/parameter_sensitivity_configurations.json`;
- `data/validation/parameter_sensitivity/parameter_sensitivity.sha256`;
- `logs/parameter_sensitivity_2026-06-12.log`.
