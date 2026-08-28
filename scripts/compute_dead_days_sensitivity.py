"""Sensibilite des resultats aux journees sans activite mesuree.

Le corpus comporte cinq journees ou les pas, le Motion Index et les transitions
sont nuls pour toutes les vaches observees, et ou la posture est enregistree
comme couchee sur la totalite des intervalles. Le chapitre 4 les decrit comme un
defaut d'acquisition et les conserve, en indiquant que leur retrait ameliorerait
les resultats.

Ce script produit le chiffre qui soutient cette derniere affirmation : il rejoue
la campagne HYPO sur le corpus prive de ces cinq journees et compare l'ablation
obtenue a celle publiee. Sans lui, les deux valeurs citees au chapitre 4 ne
seraient reproductibles par personne.

    python scripts/compute_dead_days_sensitivity.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/brut.csv"
PUBLIE = ROOT / "data/validation/hypo_module/ablation_summary.csv"
SORTIE = ROOT / "data/validation/derived_metrics/dead_days_sensitivity.json"

JOURNEES_SANS_ACTIVITE = [
    "2023-10-16",
    "2023-10-17",
    "2023-10-19",
    "2023-10-21",
    "2023-10-22",
]


def corpus_filtre(destination: Path) -> int:
    brut = pd.read_csv(RAW)
    jours = pd.to_datetime(brut["Start"]).dt.floor("D").astype(str)
    garde = ~jours.isin(JOURNEES_SANS_ACTIVITE)
    brut[garde].to_csv(destination, index=False)
    return int((~garde).sum())


def ligne_hypo(chemin: Path) -> pd.Series:
    table = pd.read_csv(chemin)
    return table[table["variante"].str.startswith("A.")].iloc[0]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        filtre = tmp / "brut_sans_journees_sans_activite.csv"
        retirees = corpus_filtre(filtre)
        sortie = tmp / "hypo_module"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_hypo_module_validation.py"),
                "--raw-csv", str(filtre),
                "--output-dir", str(sortie),
                "--quiet",
            ],
            check=True,
            cwd=ROOT,
        )
        avec = ligne_hypo(PUBLIE)
        sans = ligne_hypo(sortie / "ablation_summary.csv")

    rapport = {
        "journees_retirees": JOURNEES_SANS_ACTIVITE,
        "intervalles_retires": retirees,
        "avec_journees_sans_activite": {
            "detection_attribuable": round(float(avec["detect_any"]), 4),
            "iou20": round(float(avec["iou20"]), 4),
            "iou_moyen": round(float(avec["best_iou"]), 4),
            "fond_par_vache_jour": round(float(avec["false_notif_cow_day"]), 4),
        },
        "sans_journees_sans_activite": {
            "detection_attribuable": round(float(sans["detect_any"]), 4),
            "iou20": round(float(sans["iou20"]), 4),
            "iou_moyen": round(float(sans["best_iou"]), 4),
            "fond_par_vache_jour": round(float(sans["false_notif_cow_day"]), 4),
        },
    }
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps(rapport, indent=2, ensure_ascii=False) + "\n", encoding="utf8")

    a = rapport["avec_journees_sans_activite"]
    b = rapport["sans_journees_sans_activite"]
    print(f"{retirees} intervalles retires sur {len(JOURNEES_SANS_ACTIVITE)} journees.")
    print(f"  detection attribuable : {a['detection_attribuable']*100:.1f} % -> {b['detection_attribuable']*100:.1f} %")
    print(f"  IoU20                 : {a['iou20']*100:.1f} % -> {b['iou20']*100:.1f} %")
    print(f"Rapport ecrit dans {SORTIE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
