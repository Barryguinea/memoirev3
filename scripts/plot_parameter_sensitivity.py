"""Produit la figure de synthese de l'analyse de sensibilite OFAT.

La figure est derivee du resume tabulaire final. Elle montre l'amplitude
maximale des ecarts observes autour de la configuration de reference, sans
classer les variantes ni proposer une nouvelle configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "hypo_module"
    / "parameter_sensitivity"
    / "parameter_sensitivity_summary.csv"
)
DEFAULT_PDF = PROJECT_ROOT / "memoire" / "figures" / "hypo_parameter_sensitivity_effects.pdf"
DEFAULT_PNG = PROJECT_ROOT / "memoire" / "figures" / "hypo_parameter_sensitivity_effects.png"

PARAMETER_LABELS = {
    "min_families": "Familles concordantes",
    "aggregation_hours": "Agrégation",
    "family_min_change": "Baisse d'activité",
    "persistence_hours": "Persistance",
    "score_threshold": "Seuil du score",
    "posture_min_change": "Changement postural",
    "cusum_drift": "Dérive CUSUM",
    "cusum_threshold": "Seuil CUSUM",
    "cooldown_hours": "Période réfractaire",
    "coverage_min_pct": "Couverture minimale",
}

EVENT_DELTAS = {
    "Nouveau départ": "delta_progressive_new_start_rate_vs_reference",
    "Couverture": "delta_progressive_attributable_coverage_rate_vs_reference",
    "IoU20": "delta_progressive_iou20_rate_vs_reference",
}
BACKGROUND_DELTA = "delta_background_per_cow_day_mean_vs_reference"


def load_summary(path: Path) -> pd.DataFrame:
    """Charge le resume et verifie les colonnes necessaires a la figure."""
    frame = pd.read_csv(path)
    required = {
        "configuration",
        "parameter",
        "direction",
        *EVENT_DELTAS.values(),
        BACKGROUND_DELTA,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Colonnes absentes du resume: {', '.join(missing)}")
    if (frame["configuration"] == "reference").sum() != 1:
        raise ValueError("Le resume doit contenir exactement une configuration de reference.")
    return frame


def build_parameter_effects(summary: pd.DataFrame) -> pd.DataFrame:
    """Agrege les ecarts absolus maximaux par parametre."""
    variants = summary.loc[summary["configuration"] != "reference"].copy()
    unknown = sorted(set(variants["parameter"]) - set(PARAMETER_LABELS))
    if unknown:
        raise ValueError(f"Parametres sans libelle: {', '.join(unknown)}")

    rows: list[dict[str, float | str]] = []
    for parameter, group in variants.groupby("parameter", sort=False):
        row: dict[str, float | str] = {
            "parameter": parameter,
            "label": PARAMETER_LABELS[parameter],
        }
        for label, column in EVENT_DELTAS.items():
            row[label] = float(group[column].abs().max() * 100.0)
        row["Fond"] = float(group[BACKGROUND_DELTA].abs().max())
        rows.append(row)

    effects = pd.DataFrame(rows)
    event_columns = list(EVENT_DELTAS)
    effects["event_max"] = effects[event_columns].max(axis=1)
    return effects.sort_values(
        ["event_max", "Fond", "label"], ascending=[False, False, True]
    ).reset_index(drop=True)


def plot_effects(effects: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    """Ecrit une figure a deux panneaux adaptee a la largeur du memoire."""
    colors = {
        "Nouveau départ": "#375A7F",
        "Couverture": "#4B8764",
        "IoU20": "#BE7832",
        "Fond": "#826E9C",
    }
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, (event_axis, background_axis) = plt.subplots(
        1,
        2,
        figsize=(10.2, 6.2),
        sharey=True,
        gridspec_kw={"width_ratios": [2.35, 1.0], "wspace": 0.08},
    )
    positions = np.arange(len(effects), dtype=float)
    bar_height = 0.22

    for offset, metric in zip((-bar_height, 0.0, bar_height), EVENT_DELTAS, strict=True):
        event_axis.barh(
            positions + offset,
            effects[metric],
            height=bar_height * 0.9,
            color=colors[metric],
            label=metric,
        )

    event_axis.set_yticks(positions)
    event_axis.set_yticklabels(effects["label"])
    event_axis.invert_yaxis()
    event_axis.set_xlabel("Variation maximale absolue (points de %)")
    event_axis.set_title("(a) Réponse aux événements progressifs", loc="left", fontsize=10)
    event_axis.grid(axis="x", linestyle=":", color="#B8B8B8", linewidth=0.7)
    event_axis.set_axisbelow(True)
    event_axis.legend(frameon=False, ncol=3, loc="lower right")

    background_axis.barh(
        positions,
        effects["Fond"],
        height=0.52,
        color=colors["Fond"],
    )
    background_axis.set_xlabel("Variation absolue\ndu fond (/vache-jour)")
    background_axis.set_title("(b) Charge de fond", loc="left", fontsize=10)
    background_axis.grid(axis="x", linestyle=":", color="#B8B8B8", linewidth=0.7)
    background_axis.set_axisbelow(True)
    background_axis.tick_params(axis="y", left=False, labelleft=False)

    for axis in (event_axis, background_axis):
        axis.margins(x=0.04)

    figure.subplots_adjust(left=0.23, right=0.98, top=0.94, bottom=0.12)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight", transparent=False)
    plt.close(figure)
    with Image.open(png_path) as image:
        rgb = image.convert("RGB")
    rgb.save(png_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    effects = build_parameter_effects(load_summary(args.summary))
    plot_effects(effects, args.pdf, args.png)
    print(f"Figure PDF: {args.pdf}")
    print(f"Apercu PNG: {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
