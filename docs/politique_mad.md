# Définition du z-score robuste glissant

Le Tableau 4.5 du manuscrit décrit le z-score robuste glissant comme le même
principe que le z-score robuste global, appliqué sur une fenêtre mobile de
24 intervalles : centrage par la médiane de la fenêtre, échelle par l'écart
absolu médian (MAD) de cette même fenêtre.

Le code du manuscrit v3 calcule une variante. La médiane mobile est bien celle
de la fenêtre, mais chaque écart est pris à la médiane mobile de son propre
intervalle, puis la médiane de ces écarts est prise sur 24 intervalles
(`core/features.py`, fonction `rolling_robust_z`). Les écarts ne sont donc pas
mesurés par rapport à un même centre, et l'échelle dépend en pratique d'environ
47 intervalles. Cette variante n'était pas documentée.

## Ce qui est concerné

Seules les colonnes `*_rrz` dépendent de ce calcul. Elles alimentent Isolation
Forest, LOF et leurs règles de persistance, c'est-à-dire les variantes B, C et D
de l'ablation. HYPO, INSTABILITÉ, la fusion et le comparateur pédométrique ne
les lisent pas : leurs résultats sont identiques sous les deux définitions.

## Deux modes

`historical` est le comportement par défaut. Il reproduit le calcul du manuscrit
v3 et tous ses artefacts scellés.

`window` applique le MAD standard de la fenêtre courante, conformément à la
description du Tableau 4.5.

Le mode est un paramètre de `rolling_robust_z` et de `build_interval_features`
(`mad_mode`), transmis par `run_clean_ablation`.

## Effet mesuré sur l'ablation

Mêmes onze vaches, mêmes quarante-quatre événements, mêmes paramètres ; seule la
définition du MAD change. Les taux portent sur les 44 événements, comme au
Tableau 6.5 ; le F1 suit le critère de nouveau départ.

| Variante | Nouveau départ | IoU20 | IoU moyen | Fond / vache-jour | F1 |
|---|---|---|---|---|---|
| A. HYPO (les deux modes) | 43,2 % | 29,5 % | 0,137 | 0,402 | 0,73 |
| B. IF + persistance, `historical` | 15,9 % | 0,0 % | 0,004 | 0,152 | 0,35 |
| B. IF + persistance, `window` | 29,5 % | 0,0 % | 0,010 | 0,134 | 0,57 |
| C. IF ponctuel, `historical` | 13,6 % | 0,0 % | 0,002 | 7,658 | 0,31 |
| C. IF ponctuel, `window` | 45,5 % | 0,0 % | 0,005 | 7,711 | 0,75 |
| D. LOF + règles, `historical` | 15,9 % | 0,0 % | 0,005 | 0,107 | 0,35 |
| D. LOF + règles, `window` | 43,2 % | 0,0 % | 0,024 | 0,116 | 0,73 |
| E. Pédométrique (les deux modes) | 27,3 % | 31,8 % | 0,141 | 0,518 | 0,53 |

Tests appariés de Wilcoxon, par vache, de HYPO contre B, C et D :

| Comparaison | IoU, `historical` | IoU, `window` | Nouveau départ, `historical` | Nouveau départ, `window` |
|---|---|---|---|---|
| A contre B | p = 0,00098 | p = 0,00098 | p = 0,04297 | p = 0,28516 |
| A contre C | p = 0,00098 | p = 0,00098 | p = 0,01562 | p = 0,75 |
| A contre D | p = 0,00098 | p = 0,00098 | p = 0,04297 | p = 1,0 |

## Ce qui tient, ce qui change

Ce qui tient sous les deux définitions :

- l'avantage de localisation de HYPO sur IF et LOF, chez les onze vaches sur onze (p = 0,00098) ;
- l'absence de tout événement à IoU20 pour IF et LOF ;
- tous les résultats de HYPO, du comparateur pédométrique, de la fusion et de l'analyse SLS.

Ce qui change avec le MAD standard :

- IF et LOF ouvrent davantage de nouveaux départs, et leur F1 rejoint ou dépasse
  celui de HYPO. Les deux phrases du chapitre 6 qui présentent HYPO comme
  supérieur à IF et LOF en F1 et en taux de nouveau départ ne tiennent pas sous
  la méthode décrite ;
- la comparaison exploratoire A contre C sur le nouveau départ perd sa
  significativité après correction de Holm.

La sensibilité à la contamination et le test de stress n'ont pas été rejoués
ici avec le mode `window` : ils le seront lors de la mise en conformité du
manuscrit.

## Ce qui n'est pas modifié

Le mode par défaut reste `historical` : le code de `main` reproduit à
l'identique les 220 lignes de `data/validation/hypo_module/ablation_primary.csv`
et les tests appariés scellés. Les artefacts scellés et leur manifeste
`data/validation/validation_artifacts.sha256` sont inchangés. L'étiquette
`manuscrit-v3` ne bouge pas.

Le manuscrit sera mis en conformité lors de sa phase de corrections : soit en
décrivant exactement la variante et sa sensibilité, soit en adoptant le MAD
standard et en régénérant les valeurs concernées.

## Commande

```bash
python scripts/compute_mad_sensitivity.py
```

Le script exige le corpus confidentiel `data/brut.csv`. Il écrit dans
`data/validation/mad_sensitivity/` les événements, le résumé par variante, les
tests appariés et une provenance (empreintes des sources et du corpus, versions
des bibliothèques), avec un manifeste propre vérifiable depuis ce dossier :

```bash
shasum -a 256 -c artifacts.sha256
```

Il refuse d'écrire dans un dossier scellé.

## Tests

Quatre tests, qui n'exigent pas le corpus, couvrent ces modes : le mode
historique reste le défaut, le mode `window` suit la définition du MAD de
fenêtre, un mode inconnu est refusé, et le mode ne modifie que les z-scores
glissants.
