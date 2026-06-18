"""F1 evenementiel des variantes d'ablation HYPO (positifs: 33 graduels; negatifs: 11 controles).

Lit ``data/validation/hypo_module/ablation_primary.csv`` et calcule, pour chaque
variante et trois definitions de detection (couverture, nouveau depart / recouvrement,
IoU>=0,20), la precision, le rappel et le F1 evenementiels. Donne aussi un IC95 par
vache (bootstrap par grappe) pour HYPO. Aucune ecriture: l'utilite est la reproductibilite
du chiffre rapporte au Tableau d'ablation du manuscrit.

Usage : ``PYTHONPATH=. python scripts/compute_event_f1.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CSV = "data/validation/hypo_module/ablation_primary.csv"
SEED = 11
N_BOOT = 10000

# Colonne CSV -> libelle (et chiffre de couverture correspondant au manuscrit).
DEFINITIONS = [
    ("episode_overlap", "couverture (87,9%)"),
    ("detected_any_overlap", "recouvrement / nouveau depart (57,6%)"),
    ("detected_iou20", "IoU>=0,20 (39,4%)"),
]


def _prf(sub: pd.DataFrame, col: str) -> tuple[int, int, int, int, float, float, float]:
    tp = int(((sub.pos) & (sub[col] == 1)).sum())
    fn = int(((sub.pos) & (sub[col] == 0)).sum())
    fp = int(((~sub.pos) & (sub[col] == 1)).sum())
    tn = int(((~sub.pos) & (sub[col] == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, tn, precision, recall, f1


def main(csv: str = CSV) -> None:
    df = pd.read_csv(csv).drop_duplicates(["event_id", "variante"])
    df["pos"] = df["scenario"].str.startswith("gradual")
    print(f"positifs (graduels): {int(df[df.variante.str.startswith('A')].pos.sum())} "
          f"| negatifs (controles): {int((~df[df.variante.str.startswith('A')].pos).sum())}")

    for col, label in DEFINITIONS:
        print(f"\n=== F1 evenementiel -- detection = {label} ===")
        print(f"  {'variante':40}{'Prec':>7}{'Rappel':>8}{'F1':>7}")
        for v in sorted(df.variante.unique()):
            *_, p, r, f1 = _prf(df[df.variante == v], col)
            print(f"  {v:40}{p:>7.3f}{r:>8.3f}{f1:>7.3f}")

    a = df[df.variante.str.startswith("A")]
    cows = a.cow.unique()
    print(f"\n=== HYPO : F1 + IC95 par vache (bootstrap par grappe, {N_BOOT}, graine {SEED}) ===")
    for col, label in DEFINITIONS:
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(N_BOOT):
            draw = rng.choice(cows, size=len(cows), replace=True)
            sample = pd.concat([a[a.cow == c] for c in draw], ignore_index=True)
            boot.append(_prf(sample, col)[6])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {label:38}: F1={_prf(a, col)[6]:.3f}  IC95 [{lo:.3f} ; {hi:.3f}]")


if __name__ == "__main__":
    main()
