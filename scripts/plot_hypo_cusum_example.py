"""Figure d'illustration HYPO : detection attribuable au niveau de l'evenement.

Genere ``memoire/figures/hypo_cusum_example.{png,pdf}`` a partir des donnees
brutes. Pour une vache, on compare l'execution injectee (degradation graduelle
posee apres la periode de reference) a l'execution propre de la meme vache :
le score comportemental s'eleve pendant la fenetre injectee et l'episode
detecte se superpose a cette fenetre, alors que l'execution propre ne produit
pas le meme episode au meme endroit.

Usage : ``PYTHONPATH=. python scripts/plot_hypo_cusum_example.py``
"""

from __future__ import annotations

import warnings

import matplotlib
import pandas as pd

from core.features import build_interval_features
from core.io import COW, TIME, available_base_cols, load_csv
from revalidation.ablation import _run_variant_a
from revalidation.campaign import (
    _heldout_start_time,
    final_params,
    inject_events_for_cow,
)

COW_ID = "8154"
SCENARIO = "gradual_marked"
SEED = 11


def _run(raw: pd.DataFrame, params: dict, cow: str) -> pd.DataFrame:
    features = build_interval_features(
        raw,
        time_col=TIME,
        interval=str(params["interval"]),
        cols=available_base_cols(raw),
        window_baseline=int(params["window_baseline"]),
    )
    return _run_variant_a(features, cow, params, None)


def main(
    raw_csv: str = "data/brut.csv",
    out_stem: str = "memoire/figures/hypo_cusum_example",
) -> None:
    warnings.filterwarnings("ignore")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = final_params()
    df = load_csv(raw_csv)
    df[COW] = df[COW].astype(str)
    raw = df[df[COW] == COW_ID]

    heldout_start = _heldout_start_time(
        raw,
        interval=str(params["interval"]),
        window_baseline=int(params["window_baseline"]),
        baseline_ratio=float(params["baseline_ratio"]),
        coverage_min_pct=float(params["coverage_min_pct"]),
    )
    clean = _run(raw, params, COW_ID)
    injected_raw, events = inject_events_for_cow(
        raw,
        cow=COW_ID,
        scenario=SCENARIO,
        seed=SEED,
        interval=str(params["interval"]),
        persist_hours=int(params["persist_hours"]),
        baseline_ratio=float(params["baseline_ratio"]),
        window_baseline=int(params["window_baseline"]),
        coverage_min_pct=float(params["coverage_min_pct"]),
        heldout_start=heldout_start,
        schedule_rotation=0,
    )
    if events.empty:
        raise RuntimeError(f"Aucun evenement injectable pour la vache {COW_ID}.")
    injected = _run(injected_raw, params, COW_ID)

    event = events.iloc[0]
    e0, e1 = event["start"], event["end"]
    w0, w1 = e0 - pd.Timedelta("1.25D"), e1 + pd.Timedelta("1.25D")

    def window(frame: pd.DataFrame) -> pd.DataFrame:
        times = pd.to_datetime(frame[TIME])
        mask = (times >= w0) & (times <= w1)
        return frame[mask].assign(_t=times[mask])

    clean_w, inj_w = window(clean), window(injected)

    fig, ax = plt.subplots(
        2, 1, figsize=(8.2, 5.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax[0].axvspan(e0, e1, color="C1", alpha=0.15, label="fenêtre injectée")
    ax[0].plot(clean_w["_t"], clean_w["behavioral_warning_score"], color="0.6", lw=1.1,
               label="exécution propre")
    ax[0].plot(inj_w["_t"], inj_w["behavioral_warning_score"], color="C0", lw=1.3,
               label="exécution injectée")
    ax[0].axhline(0.12, ls="--", color="C3", lw=1.0, label="seuil de candidat (0,12)")
    ax[0].set_ylabel("score comportemental $S_H(t)$")
    ax[0].legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax[1].axvspan(e0, e1, color="C1", alpha=0.15)

    def lane(frame: pd.DataFrame, y0: float, y1: float, color: str) -> None:
        episode = frame["behavioral_warning_episode"].to_numpy()
        ax[1].fill_between(frame["_t"], y0, y1, where=episode == 1, color=color,
                           step="mid", lw=0)

    lane(inj_w, 0.55, 0.95, "C0")
    lane(clean_w, 0.05, 0.45, "0.6")
    ax[1].set_yticks([0.25, 0.75])
    ax[1].set_yticklabels(["propre", "injectée"], fontsize=8)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("épisode\ndétecté", fontsize=8)
    ax[1].set_xlabel("temps")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150 if ext == "png" else None,
                    bbox_inches="tight")
    print(f"Figure écrite : {out_stem}.png / .pdf")


if __name__ == "__main__":
    main()
