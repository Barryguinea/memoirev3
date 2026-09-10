# Politique de référence du test de stress

Le test de stress compare, pour chaque vache, une exécution propre et une
exécution dans laquelle un événement synthétique a été injecté. Les métriques
attribuables sont obtenues en soustrayant la première de la seconde. Cette
soustraction n'isole la perturbation que si les deux exécutions apprennent leur
comportement de référence sur les mêmes intervalles.

## Deux politiques

`ratio_recomputed_per_run` est demandé avec `--historical-reference` sur `main`.
Il reste le comportement par défaut de la version `manuscrit-v3`. La référence est
constituée des premiers 60 % des intervalles admissibles de l'exécution
considérée. C'est la politique qui produit les artefacts scellés de
`data/validation/hypo_stress/` et les chiffres du manuscrit.

`clean_training_timestamps_fixed` est le comportement par défaut sur `main`.
Il transmet à l'exécution injectée les
horodatages d'entraînement retenus par l'exécution propre. Isolation Forest et
LOF apprennent alors sur ces mêmes points, et le détecteur temporel ainsi que le
comparateur pédométrique utilisent les étiquettes de référence correspondantes.

## Ce que la seconde politique corrige

Le nombre d'intervalles de référence est calculé sur les intervalles qui
passent le filtre de couverture, et non sur la durée calendaire. Dans le
scénario `contiguous_dropout`, la suppression de douze heures de mesures met
48 intervalles à une couverture nulle. Ils sortent du décompte des candidats, le
seuil des 60 % recule d'autant, et la fin de la référence se déplace de
7 h 15. Sur chacune des onze vaches, la référence perd 29 intervalles :
1465 deviennent 1436 pour dix vaches et 1480 deviennent 1451 pour la vache 8147.

Le bloc supprimé se situe entièrement après la frontière de référence, donc le
contenu de la période de référence n'est pas altéré : elle est seulement
tronquée par la règle de comptage. La comparaison propre contre injectée
combinait néanmoins la perturbation et un déplacement de référence.

Ce comportement reste celui d'une chaîne de production, où une coupure de
transmission réduit effectivement la référence apprenable. Ce qui relève du
protocole d'évaluation, c'est que l'exécution soustraite n'utilise plus alors la
même référence.

## Effet mesuré

Sur les 165 événements à alerte attendue, la politique à référence fixe donne :

| Variante | Métrique | Référence recalculée | Référence fixe |
|---|---|---|---|
| A. Alerte temporelle multivariée | couverture attribuable | 0,969697 | 0,969697 |
| A | taux de départ nouveau | 0,715152 | 0,715152 |
| A | taux IoU >= 0,20 | 0,309091 | 0,309091 |
| A | IoU moyen | 0,164642 | 0,165017 |
| E. Comparateur pédométrique | couverture attribuable | 0,981818 | 0,987879 |
| E | taux de départ nouveau | 0,660606 | 0,660606 |
| E | taux IoU >= 0,20 | 0,339394 | 0,351515 |
| E | IoU moyen | 0,156602 | 0,157284 |

Les quatre tests appariés conservent leur verdict. Les comparaisons de A contre
B, C et D restent à p unilatéral de 0,000488. La comparaison de A contre le
comparateur pédométrique passe de 0,844 à 0,875 et demeure non significative.

Un déplacement de référence ne constitue pas en soi un test plus sévère ni plus
indulgent : son effet dépend du détecteur et de la métrique.

## Ce qui n'est pas modifié

Le ratio de référence initial, les seuils, la contamination, les graines, les
enveloppes d'injection et les métriques restent identiques. Le protocole
`validation_hypo/stress_protocol.json` est inchangé, empreinte comprise. Les
autres campagnes conservent leur comportement. La politique employée est
enregistrée dans les sorties de la seconde politique et dans leur provenance.

## Commandes

```bash
python -m scripts.run_hypo_stress_validation
python -m scripts.run_hypo_stress_validation --historical-reference
```

Sur `main`, la première commande utilise la référence fixe et écrit dans
`data/validation/stress_fixed_reference/` un jeu complet accompagné d'un fichier
de provenance et d'un manifeste propre, vérifiable depuis ce dossier :

```bash
shasum -a 256 -c artifacts.sha256
```

La seconde commande reproduit `data/validation/hypo_stress/` et reste vérifiable
par `data/validation/validation_artifacts.sha256`. L'option `--fixed-reference`
reste acceptée comme sélection explicite du mode par défaut. Elle ne peut pas
être combinée avec `--historical-reference`.

La version `manuscrit-v3` reste inchangée : sa commande sans option reproduit les
résultats du manuscrit. L'option `--historical-reference` concerne la branche
`main`, pas cette version archivée.

Le programme refuse d'écrire des résultats à référence fixe dans le dossier
scellé, et refuse une référence manquante, dupliquée ou qui ne correspond plus
au préfixe admissible. Chaque exécution à référence fixe vérifie l'égalité des
horodatages d'entraînement entre les cinq variantes.

La provenance consigne les empreintes des sources et du corpus ainsi que les
versions des bibliothèques, sans diffuser les données brutes confidentielles.

## Tests

Six tests couvrent cette politique : les six formes d'injection sur les cinq
variantes, cinq références invalides, la transmission de la référence dans une
campagne complète par défaut, la reproduction historique explicite, la sélection
du mode et du dossier en ligne de commande, et la protection du dossier scellé.

Le compteur du chapitre 3 décrit la suite scellée. Le compteur du README décrit
la suite courante. Le contrôle documentaire vérifie les deux séparément.
