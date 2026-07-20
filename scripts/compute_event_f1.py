"""F1 evenementiel des variantes d'ablation HYPO (positifs: 33 graduels; negatifs: 11 controles).

Lit ``data/validation/hypo_module/ablation_primary.csv`` et calcule, pour chaque
variante et trois definitions de detection (couverture attribuable, nouveau depart
attribuable, IoU>=0,20), la precision, le rappel et le F1 evenementiels. Donne aussi un IC95 par
vache (bootstrap par grappe) pour HYPO et ecrit les resultats dans un artefact CSV auditable.

Usage : ``python scripts/compute_event_f1.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/validation/hypo_module/ablation_primary.csv"
OUTPUT = ROOT / "data/validation/derived_metrics/event_f1.csv"
SEED = 11
N_BOOT = 10000

# Colonne CSV -> libelle (et chiffre de couverture correspondant au manuscrit).
DEFINITIONS = [
    ("episode_overlap", "couverture attribuable (87,9%)"),
    ("detected_any_overlap", "nouveau depart attribuable (57,6%)"),
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


def main(csv: str | Path = CSV, output: str | Path = OUTPUT) -> None:
    df = pd.read_csv(csv).drop_duplicates(["event_id", "variante"])
    df["pos"] = df["scenario"].str.startswith("gradual")
    print(
        f"positifs (graduels): {int(df[df.variante.str.startswith('A')].pos.sum())} "
        f"| negatifs (controles): {int((~df[df.variante.str.startswith('A')].pos).sum())}"
    )

    ci_by_definition: dict[str, tuple[float, float]] = {}
    a = df[df.variante.str.startswith("A")]
    cows = a.cow.unique()
    for col, _ in DEFINITIONS:
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(N_BOOT):
            draw = rng.choice(cows, size=len(cows), replace=True)
            sample = pd.concat([a[a.cow == cow] for cow in draw], ignore_index=True)
            boot.append(_prf(sample, col)[6])
        ci_by_definition[col] = tuple(np.percentile(boot, [2.5, 97.5]))

    rows: list[dict[str, object]] = []
    for col, label in DEFINITIONS:
        print(f"\n=== F1 evenementiel -- detection = {label} ===")
        print(f"  {'variante':40}{'Prec':>7}{'Rappel':>8}{'F1':>7}")
        for v in sorted(df.variante.unique()):
            tp, fp, fn, tn, p, r, f1 = _prf(df[df.variante == v], col)
            print(f"  {v:40}{p:>7.3f}{r:>8.3f}{f1:>7.3f}")
            lo, hi = ci_by_definition[col] if v.startswith("A") else (np.nan, np.nan)
            rows.append(
                {
                    "definition": col,
                    "definition_label": label,
                    "variante": v,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "f1_ci95_low": lo,
                    "f1_ci95_high": hi,
                    "bootstrap_clusters": len(cows) if v.startswith("A") else np.nan,
                    "bootstrap_repetitions": N_BOOT if v.startswith("A") else np.nan,
                    "bootstrap_seed": SEED if v.startswith("A") else np.nan,
                }
            )

    print(f"\n=== HYPO : F1 + IC95 par vache (bootstrap par grappe, {N_BOOT}, graine {SEED}) ===")
    for col, label in DEFINITIONS:
        lo, hi = ci_by_definition[col]
        print(f"  {label:38}: F1={_prf(a, col)[6]:.3f}  IC95 [{lo:.3f} ; {hi:.3f}]")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nArtefact ecrit: {output_path}")


if __name__ == "__main__":
    main()
