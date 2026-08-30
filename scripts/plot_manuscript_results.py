"""Génère les figures du manuscrit à partir des artefacts final vérifiés."""

# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

# Zones ombrees de la figure d'inspection visuelle. Les trois issues du systeme
# (alerte, surveillance, rejet) doivent se distinguer a l'oeil : le rejet est la
# seule qui reste sans aplat.
ZONE_INJECTEE = "#777777"
ZONE_ALERTE = "#af4646"
ZONE_SURVEILLANCE = "#d9a441"
# Bandeau des sorties du systeme, place sous l'axe des signaux : il occupe une
# plage negative reservee, si bien qu'aucune courbe ne peut y descendre.
YMAX = 1.15
LANE_TOP = -0.05
LANE_BOTTOM = -0.19
YMIN = -0.23


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

    # Figure volontairement compacte : le rapport hauteur/largeur est conserve,
    # donc la place prise dans la page ne bouge pas, mais la reduction a la
    # largeur d'une colonne est moins forte et le texte y reste lisible.
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    x = np.arange(len(data))
    width = 0.25
    # Memes couleurs que la figure des scenarios pour les memes grandeurs :
    # l'alerte actionnable et la surveillance d'instabilite s'y lisent deja en
    # bleu et en ambre. Le rouge des confondants reprend celui du bandeau
    # d'alerte, reserve dans tout le memoire a ce qu'on ne veut pas declencher.
    axes[0].bar(x - width, 100 * data["actionable_detection"], width,
                label="Événements actionnables", color="#376078")
    axes[0].bar(x, 100 * data["instability_surveillance_detection"], width,
                label="Instabilité en surveillance", color="#be7832")
    axes[0].bar(x + width, 100 * data["confound_alert_rate"], width,
                label="Confondants alertés", color="#af4646")
    axes[0].set_ylabel("Proportion (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_title("(a) Réponse par règle de fusion", loc="left", fontsize=10)

    axes[1].bar(x, data["background_per_cow_day"], color="#4b8764")
    axes[1].set_ylabel("Notifications par vache-jour")
    axes[1].set_title("(b) Charge de fond", loc="left", fontsize=10)
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
    # La couleur porte la dichotomie de l'analyse, SLS >= 2 contre SLS < 2, et
    # non le score lui-meme que l'axe des abscisses affiche deja : c'est cette
    # separation qui produit l'AUC, le test exact et le Mann-Whitney.
    for sls, group in data.groupby(sls_col):
        jitter = rng.normal(0, 0.035, len(group))
        positif = float(sls) >= 2
        ax.scatter(np.full(len(group), float(sls)) + jitter, group[notif_col],
                   s=48, alpha=0.85,
                   color="#af4646" if positif else "#376078",
                   label=("SLS $\\geq$ 2" if positif else "SLS < 2") if sls in (0.0, 2.0) else None)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
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
    # Meme raison que pour la figure des regles de fusion : a 13,5 pouces, la
    # reduction a la largeur d'une colonne ramenait les etiquettes sous 4 points.
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 5.6), constrained_layout=True)
    # Le temps couche accompagne les trois canaux d'activite : la posture pese
    # 0,20 dans le score HYPO et porte la fragmentation posturale de la branche
    # INSTABILITE. Sans elle, le panneau de l'instabilite illustrait un
    # mecanisme postural sans montrer la posture.
    # Le temps couche est dessine en premier, donc sous les canaux d'activite :
    # sature a 15 min sur 15, il forme de larges creneaux qui masquaient 14 %
    # de la courbe des transitions lorsqu'il passait au-dessus.
    colors = {
        "Lying Time_sum": "#7a5195",
        "Steps_sum": "#4b8764",
        "Motion Index_sum": "#376078",
        "Transitions_sum": "#be7832",
    }
    # Libelles en francais, comme le corps du memoire. « Motion Index » reste
    # le nom de la variable restituee par le capteur.
    labels_signaux = {
        "Lying Time_sum": "Temps couché",
        "Steps_sum": "Pas",
        "Motion Index_sum": "Motion Index",
        "Transitions_sum": "Transitions",
    }
    # Style de trait en plus de la couleur : imprimees en niveaux de gris, les
    # luminances des quatre courbes ne different que de 12 a 15 points sur 255,
    # soit moins de 6 %. La Figure du compromis detection-charge distingue deja
    # ses series par des marqueurs ; ici, 300 points par courbe les rendraient
    # illisibles, d'ou le trait.
    traits = {
        "Lying Time_sum": "-",
        "Steps_sum": (0, (5, 2)),
        "Motion Index_sum": (0, (6, 2, 1, 2)),
        "Transitions_sum": (0, (1, 1.6)),
    }
    for letter, (ax, (scenario, label)) in zip("abcd", zip(axes, scenarios)):
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
        # Le trace brut reste dessine en transparence sous une moyenne
        # glissante de 1 h : la tendance devient lisible sans perdre les pics,
        # ce qui reste indispensable au panneau du pic capteur isole.
        step = pd.Series(view[TIME]).diff().median()
        smooth_bins = max(1, int(round(pd.Timedelta(hours=1) / step))) if pd.notna(step) else 1
        for column, color in colors.items():
            raw_values = pd.to_numeric(view[column], errors="coerce")
            smoothed = raw_values.rolling(smooth_bins, center=True, min_periods=1).mean()
            # Echelle calee sur le maximum de la courbe lissee, et non sur le
            # maximum brut : un pic isole ne comprime plus tout le reste contre
            # l'axe. Un centile eleve reléverait les canaux d'activite ecrases
            # du panneau (b), mais tronquerait 2 a 4 % des points au sommet.
            scale = max(1.0, float(smoothed.max()))
            ax.plot(view[TIME], raw_values / scale, color=color, linewidth=0.7, alpha=0.18)
            ax.plot(
                view[TIME],
                smoothed / scale,
                color=color,
                linewidth=1.6,
                linestyle=traits[column],
                label=labels_signaux[column],
            )
        episode = pd.to_numeric(view["hybrid_warning_episode"], errors="coerce").fillna(0).astype(bool)
        instability = pd.to_numeric(view["instability_warning_episode"], errors="coerce").fillna(0).astype(bool)
        # Verite terrain et sortie du systeme sur deux canaux visuels distincts,
        # comme le fait deja la figure du chapitre 3. Empiler deux aplats
        # translucides produisait une teinte mixte que la legende n'expliquait
        # pas, et c'etait justement celle de la detection reussie. La fenetre
        # injectee reste un fond gris derriere les courbes; alerte et
        # surveillance descendent dans un bandeau qui leur est reserve, sous
        # l'axe des signaux. Plus aucune superposition n'est donc possible.
        surveillance_only = instability & ~episode
        ax.fill_between(view[TIME], LANE_BOTTOM, LANE_TOP, where=surveillance_only,
                        color=ZONE_SURVEILLANCE, step="mid", lw=0)
        ax.fill_between(view[TIME], LANE_BOTTOM, LANE_TOP, where=episode,
                        color=ZONE_ALERTE, step="mid", lw=0)
        ax.axhline(0.0, color="#999999", linewidth=0.6)
        signal_floor = (0.0 - YMIN) / (YMAX - YMIN)
        ax.axvspan(pd.Timestamp(event["start"]), pd.Timestamp(event["end"]),
                   ymin=signal_floor, ymax=1.0, color=ZONE_INJECTEE, alpha=0.12)
        # Bornes verticales : sans elles, une fenetre injectee d'un seul pas de
        # quinze minutes est invisible sur trois jours d'axe.
        for bound in (event["start"], event["end"]):
            ax.axvline(pd.Timestamp(bound), color="#555555", linestyle=":", linewidth=0.9)
        in_event = view[TIME].between(event["start"], event["end"])
        detected = bool(view.loc[in_event, "hybrid_warning_episode"].max())
        surveillance = bool(view.loc[in_event, "instability_warning_episode"].max())
        status = "alerte" if detected else ("surveillance" if surveillance else "rejet")
        ax.set_title(f"({letter}) {label} : {status}", loc="left", fontsize=10)
        ax.set_xlim(left, right)
        ax.set_ylim(YMIN, YMAX)
        # Graduations sur la seule plage des signaux : le bandeau n'est pas une
        # echelle de valeurs, il ne doit donc pas en porter.
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        ax.set_ylabel("Signal\nnormalisé")
        ax.grid(axis="y", alpha=0.2)
    # Legende placee au-dessus du premier panneau plutot qu'a l'interieur :
    # elle ne recouvre plus les courbes ni les zones ombrees, et s'aligne sur
    # la ligne du titre (titre a gauche, legende a droite).
    axes[0].legend(
        frameon=False,
        ncol=4,
        fontsize=8,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        borderaxespad=0.0,
    )
    # Legende des aplats : sans elle, le lecteur ne pouvait pas savoir ce que
    # signifiait une zone ombree, ni que la superposition de deux aplats
    # translucides produisait une troisieme teinte.
    zone_handles = [
        Patch(facecolor=ZONE_INJECTEE, alpha=0.12, label="Fenêtre injectée (fond)"),
        Patch(facecolor=ZONE_ALERTE, label="Alerte (bandeau)"),
        Patch(facecolor=ZONE_SURVEILLANCE, label="Surveillance seule (bandeau)"),
    ]
    fig.legend(
        handles=zone_handles,
        frameon=False,
        ncol=3,
        fontsize=8,
        loc="lower center",
        # Descendue nettement sous l'etiquette « Temps » : collee a l'axe, la
        # legende se confondait avec le dernier panneau.
        bbox_to_anchor=(0.5, -0.065),
    )
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
