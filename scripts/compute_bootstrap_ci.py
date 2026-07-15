"""Intervalles de confiance (bootstrap regroupé par vache) et tailles d'effet.

Recalcule, de façon reproductible, les IC95 des métriques attribuables du module
HYPO et les tailles d'effet de l'ablation rapportées au chapitre des resultats. L'unite de
reechantillonnage est la vache (cluster), conformement a l'unite statistique du
memoire. Aucun resultat n'est saisi a la main.

Usage:
    python scripts/compute_bootstrap_ci.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
HYPO = ROOT / "data/validation/hypo_module"
EVENTS = HYPO / "events_primary.csv"
ABLATION = HYPO / "ablation_primary.csv"
OUT = HYPO / "bootstrap_ci.csv"

SEED = 11
N_BOOT = 10_000
HYPO_VARIANT = "A. Alerte temporelle multivariee"  # tolere l'accent ci-dessous


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _event_value(row: dict[str, str], col: str, binarize: bool) -> float:
    value = float(row[col])
    return 1.0 if (binarize and value >= 1) else value


def hypo_metrics(rng: np.random.Generator) -> list[dict[str, object]]:
    rows = [r for r in _rows(EVENTS) if r["profile"].startswith("gradual")]
    cows = sorted({r["cow"] for r in rows})
    metrics = {
        "couverture_attribuable": ("episode_overlap", False),
        "nouveau_depart": ("novel_start_count", True),
        "iou20": ("detected_iou20", False),
    }
    results: list[dict[str, object]] = []
    for name, (col, binarize) in metrics.items():
        by_cow = {c: [_event_value(r, col, binarize) for r in rows if r["cow"] == c] for c in cows}
        point = float(np.mean([v for c in cows for v in by_cow[c]]))
        boot = np.empty(N_BOOT)
        for i in range(N_BOOT):
            sample = rng.choice(cows, size=len(cows), replace=True)
            boot[i] = np.mean([v for c in sample for v in by_cow[c]])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        results.append(
            {"analyse": "hypo_metrique", "cible": name, "valeur": round(point, 4),
             "ic95_bas": round(float(lo), 4), "ic95_haut": round(float(hi), 4), "effet": ""}
        )
    return results


def ablation_effects(rng: np.random.Generator) -> list[dict[str, object]]:
    rows = _rows(ABLATION)
    variants = sorted({r["variante"] for r in rows})
    hypo_name = next(v for v in variants if v.upper().startswith("A."))
    cows = sorted({r["cow"] for r in rows})

    def iou_by_cow(variant: str) -> np.ndarray:
        return np.array(
            [np.mean([float(r["best_iou"]) for r in rows if r["variante"] == variant and r["cow"] == c])
             for c in cows]
        )

    hypo = iou_by_cow(hypo_name)
    results: list[dict[str, object]] = []
    for variant in [v for v in variants if not v.upper().startswith("A.")]:
        other = iou_by_cow(variant)
        diff = hypo - other
        boot = np.array([np.mean(diff[rng.choice(len(cows), len(cows), replace=True)]) for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        nonzero = diff[diff != 0]
        if len(nonzero):
            ranks = rankdata(np.abs(nonzero), method="average")
            positive = float(ranks[nonzero > 0].sum())
            negative = float(ranks[nonzero < 0].sum())
            rank_biserial = (positive - negative) / (positive + negative)
        else:
            rank_biserial = float("nan")
        p = wilcoxon(hypo, other).pvalue if np.any(diff != 0) else float("nan")
        results.append(
            {"analyse": "ablation_effet", "cible": f"HYPO_vs_{variant.split('.')[0]}",
             "valeur": round(float(np.mean(diff)), 4), "ic95_bas": round(float(lo), 4),
             "ic95_haut": round(float(hi), 4),
             "effet": f"rang_biseriale={rank_biserial:+.2f}; favorables={int((diff > 0).sum())}/{len(cows)}; p={p:.5f}"}
        )
    return results


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = hypo_metrics(rng) + ablation_effects(rng)
    with open(OUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["analyse", "cible", "valeur", "ic95_bas", "ic95_haut", "effet"])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)
    print(f"\nEcrit: {OUT}")


if __name__ == "__main__":
    main()
