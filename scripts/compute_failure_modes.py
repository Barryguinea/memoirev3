"""Calcule la typologie reproductible des limites des campagnes V3.

Les fréquences sont dérivées des sorties événementielles vérifiées. Le script
ne relance pas les détecteurs et ne modifie aucun paramètre expérimental.

Usage:
    python scripts/compute_failure_modes.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "data/revalidation/hypo_module/events_primary.csv"
HYBRID = ROOT / "data/revalidation/v3_refined_full/events_fusion_hierarchical.csv"
OUTPUT = ROOT / "data/revalidation/hypo_module/failure_modes.csv"

PURE_INSTABILITY = {
    "instability_mild",
    "instability_moderate",
    "instability_marked",
}


def _row(family: str, mode: str, numerator: int, denominator: int, definition: str) -> dict:
    return {
        "family": family,
        "mode": mode,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "percentage": round(100.0 * numerator / denominator, 1),
        "definition": definition,
    }


def compute_failure_modes(primary: pd.DataFrame, hybrid: pd.DataFrame) -> pd.DataFrame:
    gradual = primary[primary["profile"].str.startswith("gradual")].copy()
    brief = primary[primary["profile"].eq("isolated_short_variation")].copy()
    nonloc = hybrid[hybrid["scenario"].eq("nonlocomotor_hypoactivity")].copy()
    instability = hybrid[hybrid["scenario"].isin(PURE_INSTABILITY)].copy()

    covered = gradual[gradual["episode_overlap"].eq(1)]
    rows = [
        _row(
            "HYPO",
            "gradual_uncovered",
            gradual["episode_overlap"].eq(0).sum(),
            len(gradual),
            "Aucun intervalle attribuable ne recouvre la fenêtre injectée.",
        ),
        _row(
            "HYPO",
            "gradual_poorly_localized",
            (gradual["episode_overlap"].eq(1) & gradual["detected_iou20"].eq(0)).sum(),
            len(gradual),
            "Recouvrement attribuable présent, mais meilleur IoU inférieur à 0,20.",
        ),
        _row(
            "HYPO",
            "covered_without_novel_start",
            covered["novel_start_count"].eq(0).sum(),
            len(covered),
            "Fenêtre recouverte sans nouveau départ par rapport à l'exécution propre.",
        ),
        _row(
            "Fusion hiérarchique",
            "nonlocomotor_hypoactivity_alerted",
            nonloc["hybrid_detected_any_overlap"].eq(1).sum(),
            len(nonloc),
            "Alerte attribuable sur le confondant d'hypoactivité non locomotrice.",
        ),
        _row(
            "INSTABILITÉ",
            "pure_instability_in_surveillance",
            instability["instability_detected_any_overlap"].eq(1).sum(),
            len(instability),
            "Instabilité pure recouverte par la branche de surveillance.",
        ),
        _row(
            "Fusion hiérarchique",
            "pure_instability_actionable_episode",
            instability["hybrid_episode_overlap"].eq(1).sum(),
            len(instability),
            "Instabilité pure recouverte par un épisode actionnable de la fusion.",
        ),
        _row(
            "HYPO",
            "brief_control_alerted",
            brief["detected_any_overlap"].eq(1).sum(),
            len(brief),
            "Alerte attribuable sur la variation brève isolée.",
        ),
    ]
    return pd.DataFrame(rows)


def main() -> None:
    primary = pd.read_csv(PRIMARY)
    hybrid = pd.read_csv(HYBRID)
    result = compute_failure_modes(primary, hybrid)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"Écrit: {OUTPUT}")


if __name__ == "__main__":
    main()
