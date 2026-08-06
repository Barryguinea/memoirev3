"""Correction de Holm des comparaisons appariees de l'ablation HYPO.

Le chapitre 6 rapporte des valeurs de p ajustees pour la multiplicite des trois
comparaisons de HYPO face a IF, IF ponctuel et LOF. Ces valeurs etaient calculees
a la main et ne figuraient dans aucun artefact, contrairement au principe de
tracabilite du chapitre 3. Ce script les derive de
``hypo_module/ablation_tests_by_cow.csv`` et les ecrit dans un artefact auditable.

La procedure descendante de Holm ordonne les p bruts par valeur croissante,
multiplie le rang i par (m - i + 1), puis impose la monotonie en propageant le
maximum courant, ce qui garantit un controle du taux d'erreur par famille.

Usage : ``python scripts/compute_holm_adjustment.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/validation/hypo_module/ablation_tests_by_cow.csv"
OUTPUT = ROOT / "data/validation/derived_metrics/holm_adjusted_pvalues.csv"

# Les trois comparaisons de HYPO face aux comparateurs forment une famille; les
# paires entre comparateurs ne sont pas des tests d'hypothese du memoire.
FAMILLE = ("A vs B", "A vs C", "A vs D")
METRIQUES = (("iou", "p_wilcoxon_iou"), ("nouveau_depart", "p_wilcoxon_detection"))


def holm(p_values: list[float]) -> np.ndarray:
    """Valeurs ajustees de Holm, monotonie imposee et bornees a 1."""
    valeurs = np.asarray(p_values, dtype=float)
    m = len(valeurs)
    ordre = np.argsort(valeurs, kind="stable")
    ajustees = np.empty(m, dtype=float)
    courant = 0.0
    for rang, indice in enumerate(ordre):
        courant = max(courant, (m - rang) * valeurs[indice])
        ajustees[indice] = min(1.0, courant)
    return ajustees


def main() -> None:
    tests = pd.read_csv(SOURCE).set_index("paire")
    lignes: list[dict[str, object]] = []
    for metrique, colonne in METRIQUES:
        bruts = [float(tests.loc[paire, colonne]) for paire in FAMILLE]
        ajustees = holm(bruts)
        for paire, brut, ajustee in zip(FAMILLE, bruts, ajustees, strict=True):
            lignes.append(
                {
                    "metrique": metrique,
                    "paire": paire,
                    "p_brut": round(brut, 6),
                    "p_holm": round(float(ajustee), 6),
                    "significatif_5pct": int(ajustee < 0.05),
                }
            )

    frame = pd.DataFrame(lignes)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(f"Ecrit: {OUTPUT} ({len(frame)} lignes)")
    for metrique, _ in METRIQUES:
        sous = frame[frame["metrique"].eq(metrique)]
        print(f"  {metrique}: p ajustes {sorted(sous['p_holm'].unique())}, "
              f"{int(sous['significatif_5pct'].sum())}/{len(sous)} significatifs a 5 %")


if __name__ == "__main__":
    main()
