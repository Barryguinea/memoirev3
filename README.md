# Alerte comportementale précoce chez la vache laitière par capteurs IoT

Système reproductible d'**alerte comportementale précoce** à partir de données
d'accéléromètres IceQube (pas, *Motion Index*, transitions, temps couché/debout).
Le système compare chaque vache à sa **propre période de référence** et signale
une dégradation comportementale persistante **à vérifier sur le terrain** ; il ne
pose **jamais** un diagnostic de boiterie ni une classification de stade clinique.

La contribution principale est un détecteur temporel par vache (**HYPO**),
complété par une extension bidirectionnelle exploratoire (**INSTABILITÉ** +
fusion hiérarchique). La validation est **post‑baseline et attribuable** : aucune
sélection de paramètres n'est faite sur le jeu d'évaluation, et chaque chiffre du
mémoire est régénérable par script et scellé par un manifeste SHA‑256.

---

## Pipeline

```
Données capteurs (CSV, par vache)
  -> Intervalles de 15 min + couverture + variables dérivées (z-scores robustes / MAD)
  -> Période de référence individuelle par créneau horaire
  -> Branche HYPO         : baisses persistantes et concordantes (CUSUM unilatéral, Page 1954)
  -> Branche INSTABILITÉ  : mouvement disproportionné aux pas (exploratoire)
  -> Fusion hiérarchique  : HYPO = notification de vérification ; instabilité = surveillance
  -> Notifications priorisées
  -> Tableau de bord Streamlit
```

`core/model_if.py` (Isolation Forest) et LOF sont conservés **uniquement comme
comparateurs**, pas comme détecteur principal.

---

## Structure du projet

```
projet-memoire/
├── app.py                       # Tableau de bord Streamlit (point d'entrée)
├── requirements.txt             # Dépendances figées (reproductibilité)
├── core/                        # Pipeline de détection
│   ├── io.py                    #   Chargement et normalisation CSV
│   ├── features.py              #   Agrégation 15 min, couverture, variables dérivées
│   ├── early_warning.py         #   Branche HYPO (référence, score, CUSUM, persistance)
│   ├── hybrid_warning.py        #   Branche INSTABILITÉ + cinq règles de fusion
│   ├── model_if.py              #   Isolation Forest (comparateur)
│   ├── alerts.py                #   Règles de persistance des comparateurs
│   └── pipeline.py              #   Orchestration vache par vache
├── ui/                          # Interface (fonctions pures de présentation)
├── validation/                # Validation technique du module HYPO
│   ├── campaign.py              #   Campagne post-baseline (44 événements)
│   ├── ablation.py              #   Ablation A-E (HYPO vs IF, LOF, pédométrique)
│   └── sensitivity.py           #   Analyse OFAT (robustesse locale, sans sélection)
├── validation_hybrid/             # Extension bidirectionnelle
│   ├── profiles.py              #   Douze scénarios synthétiques + contrôles
│   ├── campaign.py              #   Injections post-baseline + métriques attribuables
│   ├── sensitivity.py           #   Comparaison des fusions
│   └── mcgill_sls_validation.py #   Concordance exploratoire IceTag–SLS (synchronisée)
├── scripts/                     # Orchestrateurs et analyses reproductibles
│   ├── run_hypo_module_validation.py   # Campagne + ablation + tests appariés
│   ├── compute_bootstrap_ci.py         # IC95 (bootstrap par vache) + tailles d'effet
│   ├── compute_failure_modes.py        # Typologie des modes de défaillance
│   ├── detection_background_curve.py   # Courbe détection–charge (robustesse)
│   ├── audit_manuscript_numbers.py     # Vérifie 248 valeurs du mémoire ↔ sources
│   └── update_validation_manifest.py           # (Re)génère le manifeste SHA-256
├── tests/                       # 98 tests (unitaires, invariants, non-régression)
├── data/
│   ├── brut.csv                 # Données capteurs brutes (confidentiel, non versionné)
│   └── validation/            # Artefacts + manifeste validation_artifacts.sha256
├── memoire/                     # Mémoire LaTeX (sources, figures, bibliographie)
├── docs/                        # Documentation (architecture, lecture du code, audit)
└── notebooks/                   # Notebook de reproduction
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

Les données capteurs brutes (`data/brut.csv`) sont **confidentielles** et ne sont
pas diffusées ; elles doivent être placées localement dans `data/`.

---

## Utilisation

### Tableau de bord

```bash
python -m streamlit run app.py          # http://localhost:8501
```

Analyse individuelle, classement du troupeau, comparaison journalière et export.
Les marqueurs signalent une **vérification à effectuer**, pas une boiterie confirmée.

### Tests

```bash
pytest -q                                # 98 tests
```

---

## Stratégie de validation

La validation est **honnête par construction** : pas de fuite, pas de sélection
de paramètres sur le jeu d'évaluation.

1. **Validation technique du module HYPO** — onze vaches admissibles, quatre
   événements par vache (44 au total) :
   ```bash
   python scripts/run_hypo_module_validation.py --output-dir data/validation/hypo_module
   ```
   - Injections placées **strictement après** la période de référence (aucune
     contamination de la baseline).
   - **Métriques attribuables** : chaque vache est exécutée *sans* et *avec*
     injection ; l'exécution propre est soustraite, on ne crédite que ce que
     l'injection ajoute.
   - **Paramètres gelés** ; l'analyse OFAT teste la robustesse locale et **ne
     sélectionne pas** de nouvelle configuration.

2. **Ablation A–E** — HYPO comparé à : Isolation Forest + règles, IF ponctuel,
   LOF + règles, et un **comparateur pédométrique** (pas seuls, esprit
   Alsaaod 2012). Tests de Wilcoxon appariés au niveau vache, tailles d'effet
   rang‑bisériale, intervalles de confiance par bootstrap regroupé par vache :
   ```bash
   python scripts/compute_bootstrap_ci.py
   ```

3. **Extension bidirectionnelle** — douze scénarios (hypoactivité, instabilité,
   séquence) plus des contrôles et confondants (pic capteur, exercice, œstrus) :
   ```bash
   python -m validation_hybrid.sensitivity --output-dir data/validation/hybrid_refined_full
   ```

4. **Concordance clinique exploratoire** — cohorte IceTag–SLS (hiver 2019),
   mesures capteurs **strictement antérieures** au score locomoteur :
   ```bash
   python -m validation_hybrid.mcgill_sls_validation
   ```
   Explicitement **non concluante** (trois SLS ≥ 2, tous dans le bras *Exercise*).

### Limites assumées

Validation sur **injections synthétiques** (pas de gold standard vétérinaire
synchrone), **une ferme**, **pas de jeu de test indépendant**, branche
d'instabilité **exploratoire**. Ces limites sont énoncées dans le mémoire.

---

## Reproductibilité

- Chaque résultat est régénéré par un script ; **aucun chiffre n'est saisi à la main**.
- Le manifeste **SHA‑256** scelle les artefacts :
  ```bash
  python scripts/update_validation_manifest.py
  shasum -a 256 -c data/validation/validation_artifacts.sha256
  ```
- Un script d'audit confronte **248 valeurs** du manuscrit à leurs CSV/JSON sources :
  ```bash
  python scripts/audit_manuscript_numbers.py
  ```

---

## Mémoire

Sources LaTeX dans `memoire/` (compile avec `latexmk -lualatex main.tex`, PDF/A‑1b).

```bash
cd memoire && latexmk -lualatex -interaction=nonstopmode main.tex
```

---

## Documentation

- `docs/archive_historique/architecture_memoire.md` — architecture détaillée
- `docs/archive_historique/cartographie_projet_fr.md` — rôle de chaque dossier et fichier
- `docs/archive_historique/guide_lecture_code_fr.md` — ordre de lecture du code
- `docs/source_audit/selection_sources.md` — provenance et intégrité des sources
