"""Génère la figure tornade pour l'analyse de sensibilité locale.

Lit data/sensitivity_local_summary.csv et produit
memoire/overleaf_uqam/figures/sensitivity_tornado.pdf : un graphique en
barres horizontales montrant l'amplitude de Δ-détection (en points de
pourcentage) pour chaque paramètre perturbé ±1 cran.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_PARAM_LABELS = {
    "contamination": r"contamination (0.06 $\rightarrow$ 0.05/0.07)",
    "persist_hours": r"persist\_hours (7 $\rightarrow$ 6/8)",
    "alert_min": r"alert\_min (2 $\rightarrow$ 1/3)",
    "mix_rate_thr": r"mix\_rate\_thr (0.24 $\rightarrow$ 0.20/0.28)",
    "mi_z_high_thr": r"mi\_z\_high\_thr (2.2 $\rightarrow$ 2.0/2.4)",
}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    summary_csv = root / "data" / "sensitivity_local_summary.csv"
    out_pdf = root / "memoire" / "overleaf_uqam" / "figures" / "sensitivity_tornado.pdf"
    if not summary_csv.exists():
        print(f"ERREUR : {summary_csv} introuvable. Lancer d'abord scripts/sensitivity_local.py.")
        return 1

    df = pd.read_csv(summary_csv)
    df = df[df["param"] != "baseline"].copy()

    # Pour chaque paramètre, on a 2 lignes (low et high). On veut la position
    # de chaque perturbation en points de pourcentage de Δ-détection.
    rows = []
    for param in df["param"].unique():
        sub = df[df["param"] == param]
        for _, row in sub.iterrows():
            rows.append(
                {
                    "param": param,
                    "label": row["label"],
                    "delta_det_pp": float(row["delta_detection_any"]) * 100,
                    "delta_iou20_pp": float(row["delta_detection_iou20"]) * 100,
                    "delta_fn": float(row["delta_fausses_notif"]),
                }
            )
    plot_df = pd.DataFrame(rows)

    # Ordonner les paramètres par amplitude max de Δ-détection (impact décroissant)
    amp = (
        plot_df.groupby("param")["delta_det_pp"]
        .apply(lambda s: float(s.abs().max()))
        .sort_values(ascending=True)  # ascending pour matplotlib (bas vers haut)
    )
    param_order = list(amp.index)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    # --- Panel A : Δ détection (any) ---
    ax = axes[0]
    y_positions = []
    labels_y = []
    for i, param in enumerate(param_order):
        sub = plot_df[plot_df["param"] == param]
        y_low = i - 0.18
        y_high = i + 0.18
        # Barres
        for _, row in sub.iterrows():
            sign = "low" if "(-" in row["label"] else "high"
            y = y_low if sign == "low" else y_high
            color = "#d7873b" if sign == "low" else "#2e7e3a"
            ax.barh(y, row["delta_det_pp"], height=0.32, color=color, alpha=0.85,
                    edgecolor="black", linewidth=0.4)
        y_positions.append(i)
        labels_y.append(_PARAM_LABELS.get(param, param))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels_y)
    ax.set_xlabel(r"$\Delta$ détection \textit{any} (points de \%)")
    ax.set_title("(a) Sensibilité de la détection")
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # --- Panel B : Δ fausses notifications ---
    ax2 = axes[1]
    for i, param in enumerate(param_order):
        sub = plot_df[plot_df["param"] == param]
        for _, row in sub.iterrows():
            sign = "low" if "(-" in row["label"] else "high"
            y = i - 0.18 if sign == "low" else i + 0.18
            color = "#d7873b" if sign == "low" else "#2e7e3a"
            ax2.barh(y, row["delta_fn"], height=0.32, color=color, alpha=0.85,
                     edgecolor="black", linewidth=0.4)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel(r"$\Delta$ fausses notifications / cow-day")
    ax2.set_title("(b) Sensibilité du bruit d'alerte")
    ax2.grid(axis="x", linestyle=":", alpha=0.4)

    # Légende couleur
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d7873b", alpha=0.85, edgecolor="black", linewidth=0.4,
              label="perturbation $-1$ cran"),
        Patch(facecolor="#2e7e3a", alpha=0.85, edgecolor="black", linewidth=0.4,
              label="perturbation $+1$ cran"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["text.usetex"] = False  # éviter dep tex
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Figure écrite : {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
