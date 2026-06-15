# Guide de lecture du code actuel

## Périmètre scientifique

Le système priorise des animaux dont les signaux comportementaux présentent une
dégradation persistante. Il ne pose pas un diagnostic de boiterie.

## Ordre de lecture

1. `core/early_warning.py`: détecteur temporel principal.
2. `core/pipeline.py`: orchestration par vache et par troupeau.
3. `revalidation/campaign.py`: injections post-baseline et métriques attribuables.
4. `revalidation/ablation.py`: comparaison avec IF, IF + règles et LOF + règles.
5. `revalidation/mcgill_sync_validation.py`: analyse synchronisée Winter 2019.
6. `app.py` et `ui/`: interface Streamlit.
7. `notebooks/revalidation_complete.ipynb`: exécution reproductible.

## Fichiers qui soutiennent le mémoire

| Élément | Rôle |
|---|---|
| `data/brut.csv` | corpus IceQube principal, 28 vaches |
| `core/early_warning.py` | alerte temporelle multivariée |
| `revalidation/campaign.py` | benchmark technique post-baseline |
| `revalidation/ablation.py` | ablation à unité statistique vache |
| `revalidation/manual_review.py` | sélection et planches de revue manuelle |
| `revalidation/mcgill_sync_validation.py` | confrontation exploratoire aux scores SLS |
| `data/revalidation/protocol_configuration.json` | configuration explicitement figée |
| `data/revalidation/revalidation.sha256` | contrôle d'intégrité |

## Rôle des anciens fichiers

Les fichiers de `scripts/` et les notebooks `validation_manuelle_*` ou
`validation_finale_robuste_*` documentent l'évolution du projet et servent à des tests
de non-régression. Ils ne produisent pas les chiffres principaux du mémoire révisé. En
particulier, les campagnes v4--v5.2, les grilles de 336 configurations, la calibration
isotonique et les analyses SHAP de l'Isolation Forest ne doivent pas être présentées comme
validation du détecteur temporel.

## Pourquoi 11 vaches

Les 28 vaches restent analysables par l'application. Le benchmark injecté retient 11 vaches
qui possèdent simultanément une baseline suffisante, une période future assez longue et
des fenêtres où les pas, le Motion Index et les transitions sont informatifs. Ce filtrage
évite d'évaluer une perturbation sur un signal nul ou incomplet. Il limite toutefois la
validité externe et doit être mentionné.

## Reproduction

```bash
cd /Users/alioubarry/PROJECT/memoirev2
source .venv/bin/activate
pytest -q
jupyter nbconvert \
  --to notebook \
  --execute notebooks/revalidation_complete.ipynb \
  --output revalidation_complete_executed_2026-06-12.ipynb \
  --output-dir notebooks \
  --ExecutePreprocessor.timeout=1800
python -m revalidation.finalize_artifacts \
  --notebook notebooks/revalidation_complete_executed_2026-06-12.ipynb
python -m revalidation.mcgill_sync_validation
```

## Vocabulaire

À employer: alerte comportementale, dégradation persistante, notification, exécution
propre, métrique attribuable, benchmark post-baseline, confrontation exploratoire SLS.

À éviter sans labels cliniques contemporains: diagnostic, sensibilité clinique,
spécificité clinique, faux positif, probabilité de boiterie.
