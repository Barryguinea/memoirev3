# Architecture scientifique actuelle

```mermaid
flowchart TD
    A["Données IceQube brutes"] --> B["Normalisation et agrégation 15 min"]
    B --> C["Baseline individuelle et comparaison horaire"]
    C --> D["Déficits multivariés et accumulation unilatérale"]
    D --> E["Persistance, cohérence et notifications"]
    E --> F["Dashboard Streamlit et exports"]

    A --> G["Injections graduelles post-baseline"]
    A --> H["Exécution propre de référence"]
    G --> I["Exécution injectée"]
    H --> J["Soustraction temporelle"]
    I --> J
    J --> K["Couverture et IoU attribuables"]

    A --> L["IF / IF + règles / LOF + règles"]
    L --> M["Ablation agrégée par vache"]
    E --> M

    N["IceTag Winter 2019"] --> O["Fenêtres antérieures au SLS"]
    P["Scores SLS du 12 mars"] --> O
    Q["Traitement Exercise"] --> O
    O --> R["Confrontation exploratoire avec confondant"]
```

## Séparation des responsabilités

| Bloc | Fichiers principaux |
|---|---|
| Ingestion et variables | `core/io.py`, `core/features.py` |
| Comparateur historique | `core/model_if.py`, `core/alerts.py` |
| Détecteur principal | `core/early_warning.py` |
| Orchestration | `core/pipeline.py` |
| Benchmark | `revalidation/campaign.py` |
| Ablation | `revalidation/ablation.py` |
| Revue manuelle | `revalidation/manual_review.py` |
| Analyse SLS | `revalidation/mcgill_sync_validation.py` |
| Provenance | `revalidation/finalize_artifacts.py` |
| Interface | `app.py`, `ui/` |

Le notebook appelle ces modules; il ne contient pas la logique scientifique principale.
Les notebooks et scripts historiques restent archivés pour la traçabilité, mais ne
soutiennent pas les résultats actuels.
