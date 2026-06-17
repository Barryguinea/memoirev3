"""Figure d'admissibilite des 28 vaches a la validation synthetique.

Affiche la duree de serie disponible par vache et distingue les onze vaches
admissibles aux campagnes techniques (duree suffisante et signaux informatifs
dans la periode future). Explique pourquoi les 28 vaches ne sont pas toutes
utilisees en validation. Aucune valeur n'est saisie a la main.

Usage:
    python scripts/plot_cow_admissibility.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BRUT = ROOT / "data/brut.csv"
EVENTS = ROOT / "data/validation/hypo_module/events_primary.csv"
PNG = ROOT / "memoire/figures/cow_admissibility.png"
PDF = ROOT / "memoire/figures/cow_admissibility.pdf"
MIN_DAYS = 14.0

GREEN = "#4b8764"
GREY = "#9c9a92"


def main() -> None:
    df = pd.read_csv(BRUT, parse_dates=["Start"])
    span = (
        df.groupby("Cow")["Start"].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400.0)
    )
    span.index = span.index.astype(str)
    span = span.sort_values(ascending=True)

    admissible = {str(r["cow"]) for r in csv.DictReader(open(EVENTS))}

    colors = [GREEN if c in admissible else GREY for c in span.index]
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    ax.barh(range(len(span)), span.values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(span)))
    ax.set_yticklabels(span.index, fontsize=7)
    ax.axvline(MIN_DAYS, color="black", linestyle="--", linewidth=1.0)
    ax.text(MIN_DAYS + 0.3, 0.5, "seuil 14 jours", rotation=90, va="bottom", fontsize=8)
    ax.set_xlabel("Durée de série disponible (jours)")
    ax.set_ylabel("Vache")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GREEN, ec="black", lw=0.4),
        plt.Rectangle((0, 0), 1, 1, color=GREY, ec="black", lw=0.4),
    ]
    ax.legend(handles, [f"Admissible ({len(admissible)})",
                        f"Non admissible ({len(span) - len(admissible)})"],
              loc="lower right", fontsize=8, frameon=False)
    ax.margins(y=0.01)
    fig.tight_layout()
    fig.savefig(PNG, dpi=200)
    fig.savefig(PDF)
    plt.close(fig)
    with Image.open(PNG) as image:
        rgb = image.convert("RGB")
    rgb.save(PNG)
    n_short = int((span < MIN_DAYS).sum())
    print(f"Vaches: {len(span)} | admissibles: {len(admissible)} | < 14 j: {n_short}")
    print(f"Ecrit: {PNG.name}, {PDF.name}")


if __name__ == "__main__":
    main()
