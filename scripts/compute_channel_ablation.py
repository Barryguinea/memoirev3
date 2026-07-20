# ruff: noqa: E402
"""Ablation par canal : apport de chaque famille de signaux prise isolement.

Repond a la question << peut-on tester l'impact de chaque variable a part ? >>. On
rejoue exactement le protocole d'ablation attribuable (memes vaches eligibles, memes
injections post-baseline, meme soustraction de l'execution propre, memes metriques),
mais en restreignant le detecteur temporel a une seule famille a la fois :
pas, Motion Index, transitions, posture. La coherence multi-familles etant impossible
a un seul canal, ``min_families`` passe a 1 (comme le comparateur pedometrique deja au
manuscrit). On compare ensuite chaque canal isole a HYPO complet (4 familles).

Controle de coherence : le canal << pas seuls >> doit reproduire la variante E
(comparateur pedometrique) et << 4 familles >> la variante A du tableau d'ablation
attribuable du module HYPO.

Les resultats detailles et leur synthese sont ecrits dans des artefacts CSV.

Usage : ``python scripts/compute_channel_ablation.py``
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.early_warning import EarlyWarningConfig, apply_behavioral_early_warning
from core.features import build_interval_features
from core.io import COW, TIME, available_base_cols, load_csv
from validation_hypo.ablation import _run_if
from validation_hypo.campaign import (
    _evaluate_binary_output,
    _heldout_start_time,
    _monitoring_duration_days,
    final_params,
    has_informative_heldout_signals,
    inject_events_for_cow,
)

RAW = ROOT / "data/brut.csv"
OUTPUT_EVENTS = ROOT / "data/validation/derived_metrics/channel_ablation_events.csv"
OUTPUT_SUMMARY = ROOT / "data/validation/derived_metrics/channel_ablation_summary.csv"
SEED = 11
SCENARIOS = ("gradual_mild", "gradual_moderate", "gradual_marked", "isolated_short_variation")

# Libelle -> familles actives du detecteur temporel.
CHANNELS: dict[str, tuple[str, ...]] = {
    "Pas seuls": ("steps",),
    "Motion Index seul": ("motion",),
    "Transitions seules": ("transitions",),
    "Posture seule": ("posture",),
    "HYPO (4 familles)": ("steps", "motion", "transitions", "posture"),
}


def _config(families: tuple[str, ...]) -> EarlyWarningConfig:
    base = EarlyWarningConfig()
    return replace(
        base,
        active_families=families,
        min_families=base.min_families if len(families) > 1 else 1,
    )


def _behavioral(features: pd.DataFrame, cow: str, params: dict, cfg: EarlyWarningConfig) -> pd.DataFrame:
    out = apply_behavioral_early_warning(_run_if(features, params),
                                         interval=str(params["interval"]), config=cfg)
    out["pred_lameness_episode"] = out["behavioral_warning_episode"]
    out["pred_lameness_start"] = out["behavioral_warning_start"]
    out["notif_lameness"] = out["behavioral_warning_notification"]
    out[COW] = str(cow)
    return out


def main(
    raw: str | Path = RAW,
    output_events: str | Path = OUTPUT_EVENTS,
    output_summary: str | Path = OUTPUT_SUMMARY,
) -> None:
    params = final_params()
    interval = str(params["interval"])
    df_all = load_csv(raw)
    df_all[COW] = df_all[COW].astype(str)
    configs = {name: _config(fam) for name, fam in CHANNELS.items()}

    # Vaches eligibles : memes criteres que run_clean_ablation.
    eligible: list[str] = []
    for cow in sorted(df_all[COW].unique()):
        raw_cow = df_all[df_all[COW] == cow]
        if (raw_cow[TIME].max() - raw_cow[TIME].min()).total_seconds() / 86400.0 < 14:
            continue
        hs = _heldout_start_time(raw_cow, interval=interval,
                                 window_baseline=int(params["window_baseline"]),
                                 baseline_ratio=float(params["baseline_ratio"]),
                                 coverage_min_pct=float(params["coverage_min_pct"]))
        if has_informative_heldout_signals(raw_cow, heldout_start=hs):
            eligible.append(cow)

    rows: list[dict[str, object]] = []
    for ci, cow in enumerate(eligible):
        raw_cow = df_all[df_all[COW] == cow]
        hs = _heldout_start_time(raw_cow, interval=interval,
                                 window_baseline=int(params["window_baseline"]),
                                 baseline_ratio=float(params["baseline_ratio"]),
                                 coverage_min_pct=float(params["coverage_min_pct"]))
        clean_feat = build_interval_features(raw_cow, time_col=TIME, interval=interval,
                                             cols=available_base_cols(raw_cow),
                                             window_baseline=int(params["window_baseline"]))
        clean = {name: _behavioral(clean_feat, cow, params, cfg) for name, cfg in configs.items()}
        bg = {}
        for name, pred in clean.items():
            days = _monitoring_duration_days(pred, heldout_start=hs, interval=interval)
            mask = pd.to_datetime(pred[TIME], errors="coerce") >= pd.Timestamp(hs)
            bg[name] = (float("nan") if days <= 0
                        else float(pd.to_numeric(pred.loc[mask, "notif_lameness"], errors="coerce").fillna(0).sum() / days))

        for scenario in SCENARIOS:
            injected, events = inject_events_for_cow(
                raw_cow, cow=cow, scenario=scenario, seed=SEED, interval=interval,
                persist_hours=int(params["persist_hours"]), baseline_ratio=float(params["baseline_ratio"]),
                window_baseline=int(params["window_baseline"]), coverage_min_pct=float(params["coverage_min_pct"]),
                heldout_start=hs, schedule_rotation=ci % 4)
            if events.empty:
                continue
            feat = build_interval_features(injected, time_col=TIME, interval=interval,
                                           cols=available_base_cols(injected),
                                           window_baseline=int(params["window_baseline"]))
            event = events.iloc[0]
            for name, cfg in configs.items():
                pred = _behavioral(feat, cow, params, cfg)
                m = _evaluate_binary_output(pred, event, episode_col="pred_lameness_episode",
                                            start_col="pred_lameness_start", score_col="behavioral_warning_score",
                                            interval=interval, reference_predictions=clean[name])
                rows.append({"canal": name, "cow": cow, "scenario": scenario,
                             "pos": scenario.startswith("gradual"),
                             "nouv_depart": int(m["detected_any_overlap"]),
                             "couverture": int(m["episode_overlap"]),
                             "iou20": int(m["detected_iou20"]),
                             "best_iou": float(m["best_iou"]), "bg": bg[name]})

    res = pd.DataFrame(rows)
    print(f"{len(eligible)} vaches eligibles, {res.cow.nunique()} avec evenements, "
          f"{res.drop_duplicates(['cow','scenario']).shape[0]} evenements par canal\n")
    print(f"{'Canal':22}{'Nouv.dep':>10}{'Couv.':>8}{'IoU20':>8}{'IoU moy':>9}{'Fond':>8}{'F1 n.d.':>9}")
    summary_rows: list[dict[str, object]] = []
    for name in CHANNELS:
        s = res[res.canal == name]
        tp = int(((s.pos) & (s.nouv_depart == 1)).sum())
        fn = int(((s.pos) & (s.nouv_depart == 0)).sum())
        fp = ((~s.pos) & (s.nouv_depart == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{name:22}{s.nouv_depart.mean()*100:>9.1f}%{s.couverture.mean()*100:>7.1f}%"
              f"{s.iou20.mean()*100:>7.1f}%{s.best_iou.mean():>9.3f}{s.bg.mean():>8.3f}{f1:>9.3f}")
        summary_rows.append(
            {
                "canal": name,
                "n_events": len(s),
                "new_start_rate": s.nouv_depart.mean(),
                "coverage_rate": s.couverture.mean(),
                "iou20_rate": s.iou20.mean(),
                "mean_best_iou": s.best_iou.mean(),
                "background_per_cow_day": s.bg.mean(),
                "precision_new_start": prec,
                "recall_new_start": rec,
                "f1_new_start": f1,
            }
        )

    events_path = Path(output_events)
    summary_path = Path(output_summary)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(events_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\nControle : 'Pas seuls' doit reproduire la variante E (0,533 / 27,3%) "
          "et 'HYPO (4 familles)' la variante A (0,731 / 43,2%) de l'ablation attribuable.")
    print(f"Artefacts ecrits: {events_path} ; {summary_path}")


if __name__ == "__main__":
    main()
