#!/usr/bin/env python3
"""Tests statistiques appariés entre les 4 variantes du pipeline (étude d'ablation).

Produit des p-values formelles pour les comparaisons pair-à-pair entre variantes
sur les mêmes 90 cas expérimentaux :

  - Test de McNemar (exact) sur `detection_any` (métrique binaire 0/1)
  - Test de Wilcoxon des rangs signés sur `best_iou` et `fausses_notif_cow_day`
    (métriques continues)

Les 6 paires testées :
  A-B, A-C, A-D, B-C, B-D, C-D
où A=IF+règles, B=Règles seules, C=IF seul, D=LOF+règles.

Sorties :
  - data/ablation_significance/pvalues_mcnemar.csv
  - data/ablation_significance/pvalues_wilcoxon.csv
  - data/ablation_significance/summary.md (résumé lisible pour le mémoire)
  - data/ablation_significance/manifest.json (SHA-256 inputs/outputs)

Exécution :
  python3 scripts/compute_ablation_significance_tests.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "ablation_expanded_results.csv"
OUT_DIR = ROOT / "data" / "ablation_significance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORDER = ["A. Complet", "B. Sans IF", "C. Sans regles", "D. LOF + regles"]
SHORT = {
    "A. Complet": "A (IF+règles)",
    "B. Sans IF": "B (Règles seules)",
    "C. Sans regles": "C (IF seul)",
    "D. LOF + regles": "D (LOF+règles)",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mcnemar_test(y1: np.ndarray, y2: np.ndarray) -> dict:
    """Test de McNemar sur deux vecteurs binaires appariés.

    Retourne statistique + p-value (test exact binomial, recommandé pour n<25
    dans chaque discordance).
    """
    n11 = int(((y1 == 1) & (y2 == 1)).sum())  # les deux corrects
    n10 = int(((y1 == 1) & (y2 == 0)).sum())  # 1 seul correct
    n01 = int(((y1 == 0) & (y2 == 1)).sum())  # 2 seul correct
    n00 = int(((y1 == 0) & (y2 == 0)).sum())  # aucun

    # Table 2x2 : lignes = classifieur 1, colonnes = classifieur 2
    table = [[n11, n10], [n01, n00]]
    # exact=True => binomial exact, recommandé
    result = mcnemar(table, exact=True)

    return {
        "n11_both_correct": n11,
        "n10_only_1": n10,
        "n01_only_2": n01,
        "n00_both_wrong": n00,
        "discordants": n10 + n01,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def wilcoxon_test(x1: np.ndarray, x2: np.ndarray) -> dict:
    """Test de Wilcoxon des rangs signés sur deux vecteurs continus appariés."""
    diffs = x1 - x2
    # Retirer les NaN et les exactes égalités (standard)
    mask = ~np.isnan(diffs) & (diffs != 0)
    n_effective = int(mask.sum())
    if n_effective < 1:
        return {
            "n_pairs": 0,
            "n_nonzero": 0,
            "median_diff": 0.0,
            "statistic": float("nan"),
            "p_value": float("nan"),
        }
    try:
        stat, p = wilcoxon(x1[mask], x2[mask], alternative="two-sided",
                           zero_method="wilcox", correction=False, mode="auto")
    except ValueError:
        stat, p = float("nan"), float("nan")
    return {
        "n_pairs": int((~np.isnan(diffs)).sum()),
        "n_nonzero": n_effective,
        "median_diff": float(np.median(diffs[~np.isnan(diffs)])),
        "statistic": float(stat),
        "p_value": float(p),
    }


def main() -> None:
    df = pd.read_csv(SRC)

    # Pivote sur case_id pour aligner les 90 cas
    wide = df.pivot(index="case_id", columns="variante",
                    values=["detection_any", "best_iou", "fausses_notif_cow_day"])

    # ==================================================================
    # 1. McNemar sur detection_any (binaire)
    # ==================================================================
    mc_rows = []
    for v1, v2 in itertools.combinations(ORDER, 2):
        y1 = wide[("detection_any", v1)].to_numpy()
        y2 = wide[("detection_any", v2)].to_numpy()
        # Cas avec NaN retirés (rare, devrait être 0)
        mask = ~(np.isnan(y1) | np.isnan(y2))
        y1, y2 = y1[mask].astype(int), y2[mask].astype(int)
        out = mcnemar_test(y1, y2)
        out.update({"variante_1": v1, "variante_2": v2,
                    "v1_accuracy": float(y1.mean()), "v2_accuracy": float(y2.mean()),
                    "n_pairs": int(len(y1))})
        mc_rows.append(out)
    mc_df = pd.DataFrame(mc_rows)
    mc_out = OUT_DIR / "pvalues_mcnemar.csv"
    mc_df.to_csv(mc_out, index=False)

    # ==================================================================
    # 2. Wilcoxon sur métriques continues (best_iou, fausses_notif_cow_day)
    # ==================================================================
    wil_rows = []
    for metric in ["best_iou", "fausses_notif_cow_day"]:
        for v1, v2 in itertools.combinations(ORDER, 2):
            x1 = wide[(metric, v1)].to_numpy()
            x2 = wide[(metric, v2)].to_numpy()
            out = wilcoxon_test(x1, x2)
            out.update({"metric": metric, "variante_1": v1, "variante_2": v2,
                        "v1_mean": float(np.nanmean(x1)), "v2_mean": float(np.nanmean(x2))})
            wil_rows.append(out)
    wil_df = pd.DataFrame(wil_rows)
    wil_out = OUT_DIR / "pvalues_wilcoxon.csv"
    wil_df.to_csv(wil_out, index=False)

    # ==================================================================
    # 3. Résumé lisible (Markdown, intégrable mémoire)
    # ==================================================================
    def star(p: float) -> str:
        if np.isnan(p): return "n.a."
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "n.s."

    lines = []
    lines.append("# Tests statistiques appariés — étude d'ablation (N = 90 cas)\n")
    lines.append("Comparaisons pair-à-pair sur les mêmes cas expérimentaux. Significativité :\n")
    lines.append("`*** p<0.001`, `** p<0.01`, `* p<0.05`, `n.s.` non significatif.\n")

    lines.append("\n## 1. McNemar — détection globale (`detection_any`, binaire)\n")
    lines.append("| Paire | Acc. 1 | Acc. 2 | Discordants (1→0 / 0→1) | p-value | Signif. |")
    lines.append("|---|---|---|---|---|---|")
    for r in mc_rows:
        lines.append(
            f"| {SHORT[r['variante_1']]} vs {SHORT[r['variante_2']]} | "
            f"{r['v1_accuracy']:.1%} | {r['v2_accuracy']:.1%} | "
            f"{r['n10_only_1']} / {r['n01_only_2']} | "
            f"{r['p_value']:.4f} | {star(r['p_value'])} |"
        )

    for metric, label in [("best_iou", "Meilleur IoU"),
                           ("fausses_notif_cow_day", "Fausses notif./cow-day")]:
        lines.append(f"\n## 2. Wilcoxon signé — {label} (`{metric}`, continu)\n")
        lines.append("| Paire | Moy. 1 | Moy. 2 | Médiane diff. | p-value | Signif. |")
        lines.append("|---|---|---|---|---|---|")
        for r in wil_rows:
            if r["metric"] != metric: continue
            lines.append(
                f"| {SHORT[r['variante_1']]} vs {SHORT[r['variante_2']]} | "
                f"{r['v1_mean']:.3f} | {r['v2_mean']:.3f} | "
                f"{r['median_diff']:+.3f} | "
                f"{r['p_value']:.4f} | {star(r['p_value'])} |"
            )

    summary_out = OUT_DIR / "summary.md"
    summary_out.write_text("\n".join(lines), encoding="utf-8")

    # ==================================================================
    # 4. Manifest SHA-256 (reproductibilité)
    # ==================================================================
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "input": {
            "ablation_expanded_results.csv": _sha256(SRC),
            "n_cases_per_variant": 90,
            "variants": ORDER,
        },
        "outputs": {
            str(mc_out.relative_to(ROOT)): _sha256(mc_out),
            str(wil_out.relative_to(ROOT)): _sha256(wil_out),
            str(summary_out.relative_to(ROOT)): _sha256(summary_out),
        },
        "tests": {
            "mcnemar": "exact binomial (statsmodels)",
            "wilcoxon": "signed-rank two-sided (scipy)",
        },
    }
    manifest_out = OUT_DIR / "manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"✓ McNemar p-values : {mc_out}")
    print(f"✓ Wilcoxon p-values: {wil_out}")
    print(f"✓ Résumé Markdown  : {summary_out}")
    print(f"✓ Manifest         : {manifest_out}")
    print()
    print("── Résumé ──")
    print(summary_out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
