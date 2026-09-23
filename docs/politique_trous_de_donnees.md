# Trous de données dans les totaux glissants de HYPO

## Le comportement du manuscrit

Après l'agrégation à 15 minutes, un intervalle sans aucune mesure reçoit une
somme nulle (`core/features.py`, `resample().agg("sum")`), avec une couverture de
0 %. La couverture empêche cet intervalle de devenir lui-même candidat, mais les
totaux glissants de 12 heures de HYPO (`core/early_warning.py`,
`_rolling_total`) l'additionnent comme une activité nulle. Pendant les 12 heures
qui suivent un trou, les pas, le Motion Index, les transitions et le temps
debout paraissent donc en baisse : un trou de transmission ressemble à une
hypoactivité.

Les périodes surveillées du corpus ne contiennent aucun trou : la charge de fond
de 0,402 notification par vache-jour et les résultats de la campagne principale
n'en dépendent pas. Le comportement concerne un usage réel, où les pertes de
transmission sont fréquentes, et la forme « bloc manquant » du test de stress.

## Deux politiques

`historical` est le comportement par défaut. Il reproduit le manuscrit v3 et
tous ses artefacts scellés.

`coverage_aware` calcule le total glissant sur les seuls intervalles mesurés,
ramené au nombre d'intervalles présents dans la fenêtre, et ne produit aucun
total tant que moins de 90 % de la fenêtre est mesurée. Un trou suspend donc la
décision au lieu d'être lu comme une baisse.

La politique est un paramètre de `apply_behavioral_early_warning`
(`gap_policy`), transmis par `run_clean_ablation` et `run_stress_campaign`. Elle
ne concerne que HYPO et le comparateur pédométrique, qui partagent ce calcul ;
INSTABILITÉ, Isolation Forest et LOF ne sont pas modifiés.

## Effet mesuré

### Un trou seul, sans aucune baisse

Un bloc de 12 heures de mesures est retiré à trois positions de la période
surveillée de chacune des onze vaches, soit 33 blocs, sans aucune autre
modification. La référence d'apprentissage est figée sur celle de l'exécution
propre. On compte les nouveaux départs HYPO entre le début du trou et 24 heures
après sa fin.

| Politique | Blocs suivis d'un nouveau départ HYPO | Délai médian après le trou |
|---|---|---|
| `historical` | 30 sur 33 | 2,6 heures |
| `coverage_aware` | 0 sur 33 | sans objet |

### Campagne principale et ablation

Les 44 événements et les cinq variantes donnent des résultats identiques sous
les deux politiques : HYPO conserve 43,2 % de nouveaux départs, 29,5 % à IoU20,
un IoU moyen de 0,137 et un fond de 0,402 notification par vache-jour.

### Test de stress à référence fixe

Seule la forme `contiguous_dropout`, une dégradation graduelle qui contient un
trou de 12 heures, change pour HYPO :

| Forme `contiguous_dropout` | `historical` | `coverage_aware` |
|---|---|---|
| Couverture attribuable | 100,0 % | 75,8 % |
| Nouveau départ | 100,0 % | 54,5 % |
| IoU20 | 24,2 % | 0,0 % |
| IoU moyen | 0,171 | 0,057 |

Sous la politique du manuscrit, une partie de la réponse à cette forme venait du
trou lui-même. Sur les 165 événements positifs, HYPO passe ainsi de 97,0 à
92,1 % de couverture attribuable, de 71,5 à 62,4 % de nouveaux départs et de
30,9 à 26,1 % à IoU20. Le comparateur pédométrique, qui partage ce calcul, passe
de 98,8 à 93,9 % de couverture et de 35,2 à 30,9 % à IoU20.

Les comparaisons appariées conservent leur verdict :

| HYPO contre | Différence de couverture, `historical` | `coverage_aware` | p unilatéral, `coverage_aware` |
|---|---|---|---|
| IF + persistance | +0,842 | +0,794 | 0,000488 |
| IF ponctuel | +0,527 | +0,479 | 0,000488 |
| LOF + persistance | +0,691 | +0,642 | 0,000488 |
| Pédométrique | −0,018 | −0,018 | 0,766 |

Les formes sans trou et le contrôle pas-seuls sont inchangés.

## Ce qui n'est pas modifié

La politique par défaut reste `historical`. Les artefacts scellés, leur manifeste
`data/validation/validation_artifacts.sha256`, le dossier
`data/validation/stress_fixed_reference/` et l'étiquette `manuscrit-v3` sont
inchangés. Le manuscrit sera mis en conformité lors de sa phase de corrections.

## Commande

```bash
python scripts/compute_gap_policy_sensitivity.py
```

Le script exige le corpus confidentiel `data/brut.csv`. Il écrit dans
`data/validation/gap_policy_sensitivity/` l'effet des blocs seuls, le résumé de
l'ablation, les résumés et comparaisons appariées du test de stress sous les
deux politiques, et une provenance, avec un manifeste propre vérifiable depuis ce
dossier :

```bash
shasum -a 256 -c artifacts.sha256
```

Il refuse d'écrire dans un dossier scellé.

## Tests

Quatre tests, qui n'exigent pas le corpus, couvrent cette politique : la
politique historique reste le défaut, un intervalle vide ne compte plus comme
une activité nulle, une politique inconnue est refusée, et un trou seul ne
déclenche HYPO qu'en mode historique. Un cinquième vérifie que le comparateur de
`docs/comparateur_echelle_temps.md` n'alerte que dans la période future.
