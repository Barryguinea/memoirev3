"""Génère les figures du manuscrit à partir des artefacts final vérifiés."""

# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from core.io import COW, TIME, load_csv
from core.pipeline import run_pipeline_one_cow
from validation_hypo.campaign import _heldout_start_time, has_informative_heldout_signals
from validation_hybrid.campaign import final_params, inject_profile


RESULTS = ROOT / "data" / "validation" / "hybrid_refined_full"
SLS = ROOT / "data" / "validation" / "mcgill_sls"
FIGURES = ROOT / "memoire" / "figures"


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES / f"{name}.png"
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight", transparent=False)
    plt.close(fig)
    with Image.open(png_path) as image:
        rgb = image.convert("RGB")
    rgb.save(png_path)


def plot_fusion_comparison() -> None:
    data = pd.read_csv(RESULTS / "comparison_summary.csv")
    data = data[data["experiment"].eq("fusion")].copy()
    order = ["hypo_only", "instability_only", "or", "hierarchical", "sequential_24_72h"]
    labels = ["HYPO", "INSTABILITÉ", "OU", "HIÉRARCHIQUE", "SÉQUENTIELLE"]
    data = data.set_index("configuration").loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    x = np.arange(len(data))
    width = 0.25
    axes[0].bar(x - width, 100 * data["actionable_detection"], width, label="Événements actionnables")
    axes[0].bar(x, 100 * data["instability_surveillance_detection"], width, label="Instabilité en surveillance")
    axes[0].bar(x + width, 100 * data["confound_alert_rate"], width, label="Confondants alertés")
    axes[0].set_ylabel("Proportion (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, data["background_per_cow_day"], color="#4b8764")
    axes[1].set_ylabel("Notifications par vache-jour")
    axes[1].set_xticks(x, labels, rotation=22, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].axhline(data.loc["hierarchical", "background_per_cow_day"], color="#af4646", linestyle="--", linewidth=1)
    _save(fig, "fusion_comparison")


def plot_scenarios() -> None:
    data = pd.read_csv(RESULTS / "events_fusion_hierarchical.csv")
    grouped = data.groupby(["target_branch", "scenario"], sort=False).agg(
        actionable=("hybrid_detected_any_overlap", "mean"),
        surveillance=("instability_detected_any_overlap", "mean"),
    ).reset_index()
    order = [
        "hypo_mild", "hypo_moderate", "hypo_marked",
        "instability_mild", "instability_moderate", "instability_marked",
        "instability_then_hypo", "isolated_sensor_spike", "short_exercise",
        "handling_manipulation", "estrus_like_activity", "nonlocomotor_hypoactivity",
    ]
    labels = [
        "Hypo légère", "Hypo modérée", "Hypo marquée",
        "Instabilité légère", "Instabilité modérée", "Instabilité marquée",
        "Séquence", "Pic capteur", "Exercice bref", "Manipulation",
        "Type œstrus", "Hypo non locomotrice",
    ]
    grouped = grouped.set_index("scenario").loc[order]
    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.bar(x - 0.19, 100 * grouped["actionable"], 0.38, label="Alerte actionnable", color="#376078")
    ax.bar(x + 0.19, 100 * grouped["surveillance"], 0.38, label="Surveillance d'instabilité", color="#be7832")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Événements détectés (%)")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, "scenario_results")


def plot_sls() -> None:
    data = pd.read_csv(SLS / "mcgill_cohort_all_variants.csv")
    data = data[data["variant"].eq("hierarchical")].copy()
    sls_col = "sls_mar"
    notif_col = "pre7_hybrid_notifs"
    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    rng = np.random.default_rng(42)
    for sls, group in data.groupby(sls_col):
        jitter = rng.normal(0, 0.035, len(group))
        ax.scatter(np.full(len(group), float(sls)) + jitter, group[notif_col], s=48, alpha=0.85)
    ax.set_xticks(sorted(data[sls_col].dropna().unique()))
    ax.set_xlabel("Score SLS du 12 mars 2019")
    ax.set_ylabel("Notifications dans les 7 jours précédents")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "sls_notifications")


def plot_manual_review() -> None:
    raw = load_csv(str(ROOT / "data" / "brut.csv"))
    raw[COW] = raw[COW].astype(str)
    params = final_params()
    eligible: list[str] = []
    heldout: dict[str, pd.Timestamp] = {}
    for cow in sorted(raw[COW].unique()):
        cow_raw = raw[raw[COW].eq(cow)]
        span = (cow_raw[TIME].max() - cow_raw[TIME].min()).total_seconds() / 86400.0
        if span < 14:
            continue
        start = _heldout_start_time(
            cow_raw,
            interval=str(params["interval"]),
            window_baseline=int(params["window_baseline"]),
            baseline_ratio=float(params["baseline_ratio"]),
            coverage_min_pct=float(params["coverage_min_pct"]),
        )
        if has_informative_heldout_signals(cow_raw, heldout_start=start):
            eligible.append(cow)
            heldout[cow] = start

    cow = "8081" if "8081" in eligible else eligible[0]
    cow_raw = raw[raw[COW].eq(cow)]
    cow_index = eligible.index(cow)
    scenarios = [
        ("hypo_moderate", "Hypoactivité modérée"),
        ("instability_moderate", "Instabilité modérée"),
        ("isolated_sensor_spike", "Pic capteur isolé"),
        ("nonlocomotor_hypoactivity", "Hypoactivité non locomotrice"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 8.4), constrained_layout=True)
    colors = {"Steps_sum": "#4b8764", "Motion Index_sum": "#376078", "Transitions_sum": "#be7832"}
    for ax, (scenario, label) in zip(axes, scenarios):
        injected, event = inject_profile(
            cow_raw,
            cow=cow,
            scenario=scenario,
            interval=str(params["interval"]),
            heldout_start=heldout[cow],
            schedule_index=cow_index,
        )
        pred = run_pipeline_one_cow(injected, cow, **params)
        pred[TIME] = pd.to_datetime(pred[TIME], errors="coerce")
        left = pd.Timestamp(event["start"]) - pd.Timedelta(hours=12)
        right = pd.Timestamp(event["end"]) + pd.Timedelta(hours=12)
        view = pred[pred[TIME].between(left, right)].copy()
        for column, color in colors.items():
            injected_values = pd.to_numeric(view[column], errors="coerce")
            scale = max(
                1.0,
                float(injected_values.max()),
            )
            ax.plot(
                view[TIME],
                injected_values / scale,
                color=color,
                linewidth=1.0,
                label=column.replace("_sum", ""),
            )
        episode = pd.to_numeric(view["hybrid_warning_episode"], errors="coerce").fillna(0).astype(bool)
        if episode.any():
            ax.fill_between(view[TIME], 0, 1.03, where=episode, color="#af4646", alpha=0.16, step="mid")
        ax.axvspan(pd.Timestamp(event["start"]), pd.Timestamp(event["end"]), color="#777777", alpha=0.10)
        in_event = view[TIME].between(event["start"], event["end"])
        detected = bool(view.loc[in_event, "hybrid_warning_episode"].max())
        surveillance = bool(view.loc[in_event, "instability_warning_episode"].max())
        status = "alerte" if detected else ("surveillance" if surveillance else "rejet")
        ax.set_title(f"{label} : {status}", loc="left", fontsize=10)
        ax.set_xlim(left, right)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Signal\nnormalisé")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Temps")
    _save(fig, "manual_review")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    plot_fusion_comparison()
    plot_scenarios()
    plot_sls()
    plot_manual_review()


if __name__ == "__main__":
    main()
