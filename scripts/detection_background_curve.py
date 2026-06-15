# ruff: noqa: E402
"""Courbe descriptive entre detection attribuable et charge de fond.

Le seuil de score HYPO varie tandis que tous les autres parametres, les onze
vaches, les fenetres physiques et la graine restent fixes. Les injections sont
strictement post-baseline. Cette analyse caracterise le compromis technique;
elle ne selectionne pas un nouveau seuil et ne constitue pas une courbe ROC
clinique.

Usage:
    python scripts/detection_background_curve.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.early_warning import EarlyWarningConfig
from revalidation.campaign import run_clean_campaign

GRADUAL = {"gradual_mild", "gradual_moderate", "gradual_marked"}
DEFAULT_THRESHOLDS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20)
REFERENCE_THRESHOLD = EarlyWarningConfig().score_threshold


def _summary(events: pd.DataFrame, threshold: float) -> dict[str, float]:
    if events.empty or not events["event_after_heldout"].astype(bool).all():
        raise RuntimeError("La campagne contient un evenement non post-baseline.")

    gradual = events[events["scenario"].isin(GRADUAL)]
    controls = events[events["scenario"] == "isolated_short_variation"]
    cows = events.groupby("cow", as_index=False).first()
    background = (
        pd.to_numeric(cows["cow_background_notifs"], errors="coerce").sum()
        / pd.to_numeric(cows["cow_monitoring_days"], errors="coerce").sum()
    )
    delays = pd.to_numeric(gradual["onset_delay_hours"], errors="coerce").dropna()
    return {
        "score_threshold": float(threshold),
        "n_cows": int(events["cow"].nunique()),
        "n_gradual_events": int(len(gradual)),
        "background_per_cow_day": float(background),
        "attributable_coverage": float(
            (pd.to_numeric(gradual["episode_overlap"], errors="coerce") >= 1).mean()
        ),
        "new_start_rate": float(
            (pd.to_numeric(gradual["novel_start_count"], errors="coerce") >= 1).mean()
        ),
        "iou20_rate": float(
            (pd.to_numeric(gradual["detected_iou20"], errors="coerce") >= 1).mean()
        ),
        "mean_attributable_iou": float(
            pd.to_numeric(gradual["best_iou"], errors="coerce").mean()
        ),
        "median_onset_delay_hours": float(delays.median()) if len(delays) else float("nan"),
        "brief_control_rejection": float(
            1.0
            - (pd.to_numeric(controls["episode_overlap"], errors="coerce") >= 1).mean()
        ),
    }


def _plot(summary: pd.DataFrame, pdf_out: Path, png_out: Path) -> None:
    ordered = summary.sort_values("score_threshold")
    fig, (ax_metrics, ax_background) = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.3),
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )
    series = [
        ("attributable_coverage", "Couverture attribuable", "#28666e", "o"),
        ("new_start_rate", "Nouveau départ", "#d97706", "s"),
        ("iou20_rate", "IoU >= 0,20", "#9b2226", "^"),
    ]
    for column, label, color, marker in series:
        ax_metrics.plot(
            ordered["score_threshold"],
            100.0 * ordered[column],
            marker=marker,
            linewidth=1.8,
            markersize=5.5,
            color=color,
            label=label,
        )
    ax_background.plot(
        ordered["score_threshold"],
        ordered["background_per_cow_day"],
        color="#4d4d4d",
        marker="D",
        linewidth=1.8,
        markersize=5.5,
    )

    for axis in (ax_metrics, ax_background):
        axis.axvline(REFERENCE_THRESHOLD, color="#333333", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Seuil de score HYPO")
        axis.grid(True, alpha=0.25)
    ax_metrics.set_ylabel("Événements graduels (%)")
    ax_metrics.set_ylim(0, 105)
    ax_metrics.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax_metrics.set_title("Réponse aux injections")
    ax_background.set_ylabel("Notifications par vache-jour")
    ax_background.set_title("Charge de fond")
    ax_background.annotate(
        "référence 0,12",
        (REFERENCE_THRESHOLD, float(ordered.loc[np.isclose(ordered["score_threshold"], REFERENCE_THRESHOLD), "background_per_cow_day"].iloc[0])),
        xytext=(7, 8),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, dpi=220, bbox_inches="tight", transparent=False)
    plt.close(fig)
    with Image.open(png_out) as image:
        rgb = image.convert("RGB")
    rgb.save(png_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Liste separee par des virgules.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/revalidation/hypo_module/detection_background_curve",
    )
    parser.add_argument(
        "--figure-dir",
        default="memoire/figures",
    )
    args = parser.parse_args()

    thresholds = tuple(float(value) for value in args.thresholds.split(","))
    if REFERENCE_THRESHOLD not in thresholds:
        raise ValueError("Le seuil de reference 0,12 doit figurer dans la courbe.")

    output_dir = ROOT / args.output_dir
    figure_dir = ROOT / args.figure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        print(f"Seuil {threshold:.2f}...")
        config = EarlyWarningConfig(score_threshold=threshold)
        events = run_clean_campaign(warning_config=config, verbose=False)
        rows.append(_summary(events, threshold))

    summary = pd.DataFrame(rows).sort_values("score_threshold").reset_index(drop=True)
    csv_out = output_dir / "detection_background_curve.csv"
    pdf_out = figure_dir / "detection_background_curve.pdf"
    png_out = figure_dir / "detection_background_curve.png"
    summary.to_csv(csv_out, index=False, float_format="%.6f")
    _plot(summary, pdf_out, png_out)

    print(summary.to_string(index=False))
    print(f"Ecrit: {csv_out}")
    print(f"Ecrit: {pdf_out}")
    print(f"Ecrit: {png_out}")


if __name__ == "__main__":
    main()
