# Test de stress sous les corrections combinées

Les notes `docs/politique_mad.md` et `docs/politique_trous_de_donnees.md` évaluent
chaque correction séparément. Cette note rapporte le test de stress à référence
fixe (198 événements, cinq variantes) avec les deux corrections appliquées
ensemble : le MAD standard pour Isolation Forest et LOF (`mad_mode="window"`) et
la politique `coverage_aware` pour HYPO et le comparateur pédométrique.

Toutes les configurations suivent la référence fixe, comportement par défaut de
`main` (voir `docs/politique_reference_stress.md`). Le manuscrit déposé suit la
référence historique : HYPO, IF + persistance et IF ponctuel y ont les mêmes
valeurs, mais LOF + persistance y recouvre 26,7 % et le comparateur pédométrique
98,2 %, pour des écarts de HYPO de +0,703 et −0,012 (p = 0,844).

## Recouvrement attribuable sur les 165 événements positifs

| Configuration | HYPO | IF + persistance | IF ponctuel | LOF + persistance | Pédométrique |
|---|---|---|---|---|---|
| MAD et trous historiques | 97,0 % | 12,7 % | 44,2 % | 27,9 % | 98,8 % |
| MAD standard seul | 97,0 % | 35,8 % | 70,3 % | 60,0 % | 98,8 % |
| Trous corrigés seuls | 92,1 % | 12,7 % | 44,2 % | 27,9 % | 93,9 % |
| **Les deux corrections** | **92,1 %** | **35,8 %** | **70,3 %** | **60,0 %** | **93,9 %** |

Aucune variante d'Isolation Forest ou de LOF n'atteint IoU20 sur un seul
événement, quelle que soit la configuration. Avec les deux corrections, HYPO
atteint 26,1 % à IoU20 et le comparateur pédométrique 30,9 %.

## Comparaisons appariées de HYPO (différence moyenne de recouvrement par vache)

| HYPO contre | MAD et trous historiques | MAD standard seul | Les deux corrections | p unilatéral exact, deux corrections |
|---|---|---|---|---|
| IF + persistance | +0,842 | +0,612 | +0,564 | 0,000488 |
| IF ponctuel | +0,527 | +0,267 | +0,218 | 0,000977 |
| LOF + persistance | +0,691 | +0,370 | +0,321 | 0,000488 |
| Pédométrique | −0,018 | −0,018 | −0,018 | 0,766 |

## Lecture

Avec les deux corrections, HYPO conserve un avantage de recouvrement sur les trois
variantes d'Isolation Forest et de LOF (p ≤ 0,001, test exact par vache), mais
nettement réduit, surtout face à IF ponctuel (de +0,527 à +0,218). Il n'a toujours
aucun avantage sur le comparateur pédométrique. Sur les 33 événements du contrôle
pas-seuls, HYPO reste à 78,8 % de recouvrement dans toutes les configurations ;
avec le MAD standard, IF ponctuel passe de 12,1 % à 0 %, et IF + persistance et
LOF restent à 0 %.

## Ce qui n'est pas couvert

La sensibilité des comparateurs à leur paramètre de contamination n'a pas été
rejouée sous le MAD standard. Le test de stress porte sur la référence fixe ; il
n'a pas été rejoué sous la politique de référence historique.

## Commande

```bash
python scripts/compute_combined_corrections_stress.py
```

Le script exige le corpus confidentiel `data/brut.csv` et écrit dans
`data/validation/combined_corrections_stress/` les résumés et comparaisons
appariées des deux configurations sous MAD standard, avec une provenance et un
manifeste propre (`shasum -a 256 -c artifacts.sha256`). Les deux configurations
sous MAD historique figurent dans `data/validation/gap_policy_sensitivity/`. Le
comportement par défaut du code n'est pas modifié.
