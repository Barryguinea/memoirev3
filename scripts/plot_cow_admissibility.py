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


# Les axes, les graduations et la grille sont dessines par matplotlib a 0,8
# point, valeur qui ne suit pas la reduction des figures : ramenees a la
# largeur de la page, elles ressortaient plus epaisses qu'avant le
# redimensionnement. Ces reglages leur rendent leur finesse d'origine.
plt.rcParams.update({
    "axes.linewidth": 0.64,
    "xtick.major.width": 0.64,
    "ytick.major.width": 0.64,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "grid.linewidth": 0.5,
})

def main() -> None:
    df = pd.read_csv(BRUT, parse_dates=["Start"])
    span = (
        df.groupby("Cow")["Start"].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400.0)
    )
    span.index = span.index.astype(str)
    span = span.sort_values(ascending=True)

    admissible = {str(r["cow"]) for r in csv.DictReader(open(EVENTS))}

    colors = [GREEN if c in admissible else GREY for c in span.index]
    # 5,6 pouces et non 7,0 : posee a 0,80 de la largeur du texte, la figure
    # ramenait ses etiquettes a 5,2 points. Le rapport est conserve, les
    # epaisseurs suivent le facteur 1,25.
    fig, ax = plt.subplots(figsize=(5.6, 5.12))
    ax.barh(range(len(span)), span.values, color=colors, edgecolor="black", linewidth=0.32)
    ax.set_yticks(range(len(span)))
    ax.set_yticklabels(span.index, fontsize=7)
    ax.axvline(MIN_DAYS, color="black", linestyle="--", linewidth=0.8)
    ax.text(MIN_DAYS + 0.3, 0.5, "seuil 14 jours", rotation=90, va="bottom", fontsize=8)
    ax.set_xlabel("Durée de série disponible (jours)")
    ax.set_ylabel("Vache")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GREEN, ec="black", lw=0.32),
        plt.Rectangle((0, 0), 1, 1, color=GREY, ec="black", lw=0.32),
    ]
    ax.legend(handles, [f"Admissible ({len(admissible)})",
                        f"Non admissible ({len(span) - len(admissible)})"],
              loc="lower right", fontsize=8, frameon=False)
    ax.margins(y=0.01)
    fig.tight_layout()
    fig.savefig(PNG, dpi=300, transparent=False)
    # Meme aplatissement que les autres generateurs : le canal alpha ferait
    # echouer la validation PDF/A-1b.
    with Image.open(PNG) as image:
        rgb = image.convert("RGB")
    rgb.save(PNG, dpi=(300, 300))
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
