"""Sensibilite des comparateurs IF et LOF a leur parametre de contamination.

L'analyse OFAT de HYPO fait varier dix parametres du detecteur propose, mais aucun
parametre des comparateurs. Or l'argument de non-circularite du chapitre 6 repose sur
l'echec d'Isolation Forest et de LOF a localiser les fenetres injectees: si cet echec
tenait au seul reglage de ``contamination``, que le chapitre 2 designe comme parametre
critique de ces methodes, l'argument s'affaiblirait.

Ce script rejoue donc l'ablation complete a cinq valeurs de contamination, de 0,02 a
0,15, sur les memes onze vaches et les memes quarante-quatre evenements. HYPO et le
comparateur pedometrique n'utilisent pas Isolation Forest: leurs lignes servent de
temoin et doivent rester constantes.

Usage : ``python scripts/compute_comparator_sensitivity.py``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from validation_hypo.ablation import ablation_summary, run_clean_ablation
from validation_hypo.campaign import final_params

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/hypo_module/comparator_contamination_sensitivity.csv"
CONTAMINATIONS = (0.02, 0.04, 0.06, 0.10, 0.15)
REFERENCE = 0.06


def main() -> None:
    rows: list[dict[str, object]] = []
    for contamination in CONTAMINATIONS:
        params = final_params()
        params["contamination"] = contamination
        summary = ablation_summary(
            run_clean_ablation(params=params, verbose=False)
        ).set_index("variante")
        for variante in summary.index:
            rows.append(
                {
                    "contamination": contamination,
                    "est_reference": int(contamination == REFERENCE),
                    "variante": variante,
                    "detect_any": round(float(summary.loc[variante, "detect_any"]), 6),
                    "iou20": round(float(summary.loc[variante, "iou20"]), 6),
                    "best_iou": round(float(summary.loc[variante, "best_iou"]), 6),
                    "false_notif_cow_day": round(
                        float(summary.loc[variante, "false_notif_cow_day"]), 6
                    ),
                }
            )
        print(f"contamination {contamination:.2f} terminee", flush=True)

    frame = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)

    comparateurs = frame[frame["variante"].str.startswith(("B.", "C.", "D."))]
    print(f"Ecrit: {OUTPUT} ({len(frame)} lignes)")
    print(f"IoU20 maximal des comparateurs sur la plage: {comparateurs['iou20'].max():.6f}")
    print(f"IoU moyen maximal des comparateurs: {comparateurs['best_iou'].max():.6f}")


if __name__ == "__main__":
    main()
