# Cartographie du projet révisé

## Éléments actifs

| Dossier ou fichier | Fonction actuelle |
|---|---|
| `app.py` | point d'entrée Streamlit |
| `core/` | ingestion, variables, comparateurs et détecteur temporel |
| `ui/` | vues individuelle, troupeau, journalière et export |
| `revalidation/` | benchmark post-baseline, ablation, SLS et provenance |
| `notebooks/revalidation_complete.ipynb` | orchestration reproductible de la campagne |
| `tests/` | contrats, invariants, UI et protocole |
| `data/brut.csv` | corpus IceQube de référence |
| `data/revalidation/` | résultats et manifestes actuels |
| `memoire/` | sources LaTeX et PDF |

## Cœur du pipeline

| Fichier | Rôle |
|---|---|
| `core/io.py` | normalisation des colonnes et dates |
| `core/features.py` | rééchantillonnage et variables dérivées |
| `core/model_if.py` | Isolation Forest conservé comme comparateur |
| `core/alerts.py` | règles de persistance conservées comme comparateur |
| `core/early_warning.py` | déficit multivarié, comparaison horaire, CUSUM et persistance |
| `core/pipeline.py` | exécution cohérente des deux branches |

## Revalidation

| Fichier | Rôle |
|---|---|
| `revalidation/injection.py` | profils graduels et contrôle bref |
| `revalidation/campaign.py` | exécutions propre/injectée et métriques attribuables |
| `revalidation/analysis.py` | synthèses et statistiques |
| `revalidation/ablation.py` | comparaison des quatre méthodes |
| `revalidation/manual_review.py` | échantillon visuel et annotations |
| `revalidation/mcgill_sync_validation.py` | confrontation Winter 2019 avec SLS |
| `revalidation/finalize_artifacts.py` | instantané et manifeste SHA-256 |

## Interface

| Fichier | Rôle |
|---|---|
| `ui/panels_individual.py` | chronologie et signaux d'une vache |
| `ui/panels_herd.py` | classement, animaux prioritaires, carte journalière et export |
| `ui/plots.py` | graphiques Plotly |
| `ui/presentation.py` | libellés et tables |

## Éléments historiques

Le dossier `scripts/`, les notebooks `validation_manuelle_*` et
`validation_finale_robuste_*`, ainsi que les sorties de calibration, ROC, SHAP et
grille v4--v5.2, sont conservés pour comprendre l'évolution du projet et vérifier la
non-régression. Ils ne doivent pas être exécutés pour reproduire les résultats du mémoire
révisé et ne constituent pas ses preuves principales.

## Données et population

- `data/brut.csv`: 28 vaches disponibles pour l'application.
- Benchmark injecté: 11 vaches satisfaisant les critères de baseline, durée future et
  signaux informatifs.
- Winter 2019: 14 vaches avec données synchronisables avant le score SLS du 12 mars.

Ces trois effectifs répondent à des objectifs différents et ne doivent pas être confondus.
