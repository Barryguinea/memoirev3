"""Effet de la tolerance d'appariement des departs sur les metriques attribuables.

Un nouveau depart n'est credite que s'il n'existe pas de depart comparable dans
l'execution propre de la meme vache. La comparaison admet une tolerance temporelle,
fixee a une heure. Ce parametre n'apparait dans aucune configuration publiee et
influence pourtant le taux de nouveau depart: sans tolerance, deux departs distants
de quinze minutes seraient comptes comme differents et le taux monterait.

Ce script rejoue la campagne principale a trois tolerances et ecrit l'effet obtenu.
La couverture attribuable et l'IoU se calculent sur les intervalles attribuables sans
jamais apparier de departs: elles servent de temoin et doivent rester constantes.

Usage : ``python scripts/compute_match_tolerance_sensitivity.py``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from validation_hypo import campaign as campaign_module
from validation_hypo.ablation import ablation_summary, run_clean_ablation

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/hypo_module/match_tolerance_sensitivity.csv"
TOLERANCES = (0.0, 1.0, 2.0)
REFERENCE = campaign_module.DEFAULT_MATCH_TOLERANCE_HOURS


def main() -> None:
    original = campaign_module.DEFAULT_MATCH_TOLERANCE_HOURS
    rows: list[dict[str, object]] = []
    try:
        for tolerance in TOLERANCES:
            campaign_module.DEFAULT_MATCH_TOLERANCE_HOURS = tolerance
            summary = ablation_summary(
                run_clean_ablation(verbose=False)
            ).set_index("variante")
            hypo = next(v for v in summary.index if str(v).startswith("A."))
            rows.append(
                {
                    "match_tolerance_hours": tolerance,
                    "est_reference": int(tolerance == REFERENCE),
                    "variante": hypo,
                    "new_start_rate": round(float(summary.loc[hypo, "detect_any"]), 6),
                    "iou20": round(float(summary.loc[hypo, "iou20"]), 6),
                    "best_iou": round(float(summary.loc[hypo, "best_iou"]), 6),
                }
            )
            print(f"tolerance {tolerance:.1f} h terminee", flush=True)
    finally:
        campaign_module.DEFAULT_MATCH_TOLERANCE_HOURS = original

    frame = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(f"Ecrit: {OUTPUT} ({len(frame)} lignes)")
    print(f"Taux de nouveau depart: {frame['new_start_rate'].min():.4f} a {frame['new_start_rate'].max():.4f}")
    print(f"IoU20 constant sur la plage: {frame['iou20'].nunique() == 1}")


if __name__ == "__main__":
    main()
