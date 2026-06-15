# Analyse de sensibilite des parametres du detecteur temporel

## Objectif

Le module `revalidation.sensitivity` mesure la variation des resultats lorsque
chaque parametre du detecteur temporel est modifie separement. Cette analyse est
descriptive. Elle ne sert ni a choisir la configuration qui maximise les
performances, ni a revendiquer une validation clinique.

La reference reste la configuration declaree dans
`core.early_warning.EarlyWarningConfig`. Les memes vaches, la meme graine et les
memes fenetres d'evenements physiques sont imposes a toutes les variantes. Le
programme s'arrete si le placement d'un evenement differe de la reference.

## Grille definie avant execution

| Parametre | Reference | Variantes |
|---|---:|---:|
| Agregation | 12 h | 6 h; 24 h |
| Persistance | 6 h | 3 h; 12 h |
| Periode refractaire | 24 h | 12 h; 48 h |
| Baisse minimale d'activite | 10 % | 5 %; 15 % |
| Changement postural minimal | 5 % | 2,5 %; 10 % |
| Seuil du score | 0,12 | 0,08; 0,16 |
| Derive CUSUM | 0,07 | 0,04; 0,10 |
| Seuil CUSUM | 1,20 | 0,80; 1,60 |
| Familles concordantes | 3 | 2; 4 |
| Couverture minimale | 25 % | 50 %; 75 % |

Cela produit 21 configurations: une reference et 20 variantes OFAT.

## Commandes

Lister les configurations sans lancer la campagne:

```bash
cd /Users/alioubarry/PROJECT/memoirev2
source .venv/bin/activate
python -m revalidation.sensitivity --list-configurations
```

Executer ensuite l'analyse complete:

```bash
cd /Users/alioubarry/PROJECT/memoirev2
source .venv/bin/activate
python -m revalidation.sensitivity
```

Les executions terminees sont conservees individuellement dans
`data/revalidation/parameter_sensitivity/runs/`. Une relance reprend les runs
compatibles. L'option `--force` impose un recalcul complet.

Pour un essai technique limite, sans produire de resultats a reporter:

```bash
python -m revalidation.sensitivity \
  --cows 8081 \
  --configurations aggregation_hours__6p0
```

## Sorties

- `parameter_sensitivity_events.csv`: resultats au niveau evenement;
- `parameter_sensitivity_summary.csv`: indicateurs globaux et ecarts a la reference;
- `parameter_sensitivity_by_scenario.csv`: resultats separes par profil;
- `parameter_sensitivity_configurations.json`: provenance et configurations;
- `parameter_sensitivity.sha256`: controle d'integrite.

Le tableau principal distingue les profils progressifs, le controle bref et le
taux de notifications de fond. Aucun rang ni score composite de « meilleure
configuration » n'est calcule.
