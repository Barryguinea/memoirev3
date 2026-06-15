"""Agrégation et artefacts de sortie pour la validation robuste."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from core.io import COW
from scripts.validation_common import aggregate_distribution as _aggregate_distribution
from scripts.validation_injection_raw import build_streamlit_raw_with_events


def write_partial_outputs(
    *,
    run_dir: Path,
    all_runs: List[pd.DataFrame],
    all_events: List[pd.DataFrame],
    start_meta: Dict[str, object],
    run_counter: int,
) -> None:
    """Écrit des sorties partielles + meta de progression pour reprise/inspection."""
    if len(all_runs):
        pd.concat(all_runs, ignore_index=True).to_csv(run_dir / "robust_runs.partial.csv", index=False)
    if len(all_events):
        pd.concat(all_events, ignore_index=True).to_csv(run_dir / "robust_events_evaluation.partial.csv", index=False)
    partial_meta = dict(start_meta)
    partial_meta["run_counter"] = int(run_counter)
    partial_meta["status"] = "running"
    (run_dir / "run_meta.json").write_text(json.dumps(partial_meta, indent=2), encoding="utf-8")


def build_robust_summary(
    *,
    runs_df: pd.DataFrame,
    group_cols: List[str],
    scenarios: List[str],
    go_thresholds: Dict[str, float],
    go_mode: str,
) -> pd.DataFrame:
    """Construit le résumé robuste agrégé par configuration avec verdict et score de sélection."""
    if not len(runs_df):
        return pd.DataFrame()

    summary_rows = []
    for keys, g in runs_df.groupby(group_cols, dropna=False):
        row: Dict[str, object] = {}
        if isinstance(keys, tuple):
            for i, k in enumerate(group_cols):
                row[k] = keys[i]
        else:
            row[group_cols[0]] = keys
        stats = _aggregate_distribution(
            g,
            [
                "detectable_recall_any",
                "detectable_recall_iou20",
                "detectable_iou_mean",
                "hidden_leak_rate_any",
                "hidden_leak_rate_iou20",
                "false_notif_per_cow_day",
            ],
        ).iloc[0].to_dict()
        row.update(stats)
        row["n_runs"] = int(len(g))
        legacy_pass = int(
            (row.get("detectable_recall_iou20_p10", np.nan) >= 0.25)
            and (row.get("hidden_leak_rate_any_p90", np.nan) <= 0.20)
            and (row.get("false_notif_per_cow_day_p90", np.nan) <= 0.30)
        )
        row["robust_pass_legacy"] = int(legacy_pass)

        for sc in scenarios:
            gs = g[g["scenario"] == str(sc)]
            if len(gs) == 0:
                continue
            sc_stats = _aggregate_distribution(
                gs,
                [
                    "detectable_recall_any",
                    "detectable_recall_iou20",
                    "hidden_leak_rate_any",
                    "false_notif_per_cow_day",
                ],
            ).iloc[0].to_dict()
            for mk, mv in sc_stats.items():
                row[f"{sc}_{mk}"] = mv

        strong_iou20_p10 = row.get("detectable_strong_detectable_recall_iou20_p10", np.nan)
        border_any_mean = row.get("detectable_borderline_detectable_recall_any_mean", np.nan)
        hidden_any_p90 = row.get("non_detectable_short_hidden_leak_rate_any_p90", row.get("hidden_leak_rate_any_p90", np.nan))
        false_notif_p90 = row.get("false_notif_per_cow_day_p90", np.nan)

        scenario_pass = int(
            (strong_iou20_p10 >= go_thresholds["strong_iou20_p10_min"])
            and (border_any_mean >= go_thresholds["borderline_any_mean_min"])
            and (hidden_any_p90 <= go_thresholds["hidden_leak_any_p90_max"])
            and (false_notif_p90 <= go_thresholds["false_notif_p90_max"])
        )
        row["robust_pass_scenario"] = int(scenario_pass)
        row["robust_pass"] = int(scenario_pass if go_mode == "scenario_aware" else legacy_pass)

        go_reason = []
        if not (strong_iou20_p10 >= go_thresholds["strong_iou20_p10_min"]):
            go_reason.append(
                f"strong_iou20_p10 {strong_iou20_p10:.3f} < {go_thresholds['strong_iou20_p10_min']:.3f}"
                if np.isfinite(strong_iou20_p10)
                else "strong_iou20_p10 is NaN"
            )
        if not (border_any_mean >= go_thresholds["borderline_any_mean_min"]):
            go_reason.append(
                f"borderline_any_mean {border_any_mean:.3f} < {go_thresholds['borderline_any_mean_min']:.3f}"
                if np.isfinite(border_any_mean)
                else "borderline_any_mean is NaN"
            )
        if not (hidden_any_p90 <= go_thresholds["hidden_leak_any_p90_max"]):
            go_reason.append(
                f"hidden_leak_any_p90 {hidden_any_p90:.3f} > {go_thresholds['hidden_leak_any_p90_max']:.3f}"
                if np.isfinite(hidden_any_p90)
                else "hidden_leak_any_p90 is NaN"
            )
        if not (false_notif_p90 <= go_thresholds["false_notif_p90_max"]):
            go_reason.append(
                f"false_notif_p90 {false_notif_p90:.3f} > {go_thresholds['false_notif_p90_max']:.3f}"
                if np.isfinite(false_notif_p90)
                else "false_notif_p90 is NaN"
            )
        row["go_reason"] = "OK" if len(go_reason) == 0 else " | ".join(go_reason)

        row["selection_score"] = (
            0.60 * float(row.get("detectable_strong_detectable_recall_iou20_mean", row.get("detectable_recall_iou20_mean", 0.0)))
            + 0.20 * float(row.get("detectable_borderline_detectable_recall_any_mean", row.get("detectable_recall_any_mean", 0.0)))
            + 0.10 * float(row.get("detectable_recall_any_mean", 0.0))
            - 0.25 * float(row.get("hidden_leak_rate_any_mean", 0.0))
            - 0.30 * float(row.get("false_notif_per_cow_day_mean", 0.0))
            - 0.10 * float(row.get("false_notif_per_cow_day_std", 0.0))
        )
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values(
        ["robust_pass", "selection_score", "detectable_recall_iou20_mean"],
        ascending=[False, False, False],
    )


def write_final_outputs(
    *,
    run_dir: Path,
    base_summary: pd.DataFrame,
    base_episodes: pd.DataFrame,
    runs_df: pd.DataFrame,
    events_df: pd.DataFrame,
    robust_summary: pd.DataFrame,
    sample_streamlit_files: List[Dict[str, object]],
    group_cols: List[str],
    raw_df: pd.DataFrame,
    write_best_streamlit_artifacts: bool,
) -> Path:
    """Écrit CSV/artefacts finaux et retourne le chemin du résumé base engine."""
    base_summary.to_csv(run_dir / "baseline_summary_by_cow.csv", index=False)
    base_episodes.to_csv(run_dir / "baseline_detected_episodes.csv", index=False)
    runs_df.to_csv(run_dir / "robust_runs.csv", index=False)
    events_df.to_csv(run_dir / "robust_events_evaluation.csv", index=False)
    robust_summary_base_file = run_dir / "robust_summary_base_engine.csv"
    robust_summary.to_csv(robust_summary_base_file, index=False)
    pd.DataFrame(sample_streamlit_files).to_csv(run_dir / "streamlit_samples_manifest.csv", index=False)

    if write_best_streamlit_artifacts and len(runs_df):
        best = robust_summary.iloc[0]
        f = runs_df.copy()
        for k in group_cols:
            f = f[f[k] == best[k]]
        if len(f):
            pick = f.sort_values(["detectable_recall_iou20", "detectable_recall_any"], ascending=[False, False]).iloc[0]
            chosen_events = events_df.copy()
            for k in group_cols + ["seed", "scenario"]:
                if k in chosen_events.columns:
                    chosen_events = chosen_events[chosen_events[k] == pick[k]]
            if len(chosen_events) == 0:
                chosen_events = events_df[(events_df["seed"] == pick["seed"]) & (events_df["scenario"] == pick["scenario"])].copy()
            chosen_manifest = chosen_events[
                ["event_id", "cow", "start", "end", "duration_bins", "profile", "expected_detected", "design_reason"]
            ].drop_duplicates(subset=["event_id", "cow", "start", "end", "profile", "expected_detected"])
            streamlit_best, applied = build_streamlit_raw_with_events(raw_df, chosen_manifest, seed=int(pick["seed"]))
            streamlit_best.to_csv(run_dir / "streamlit_brut_with_best_injections.csv", index=False)
            applied.to_csv(run_dir / "streamlit_best_injection_windows.csv", index=False)

    for pth in [run_dir / "robust_runs.partial.csv", run_dir / "robust_events_evaluation.partial.csv"]:
        if pth.exists():
            pth.unlink()

    return robust_summary_base_file


def write_completed_meta(
    *,
    run_dir: Path,
    raw_input_path: str | Path,
    raw_source_for_streamlit: str | Path | None,
    params0: Dict[str, object],
    seeds: List[int],
    scenarios: List[str],
    grid: Dict[str, List[object]],
    grid_s: Dict[str, List[object]],
    raw_df: pd.DataFrame,
    verbose: bool,
    progress_every: int,
    checkpoint_every: int,
    detectable_events_per_run: int,
    go_mode: str,
    go_thresholds: Dict[str, float],
    write_best_streamlit_artifacts: bool,
    robust_summary_base_file: Path,
    run_counter: int,
    total_expected: int,
) -> None:
    """Écrit le fichier `run_meta.json` final de la validation robuste."""
    meta = {
        "raw_input_path": str(Path(raw_input_path).resolve()),
        "output_dir": str(run_dir.resolve()),
        "base_params": params0,
        "seeds": [int(s) for s in seeds],
        "scenarios": [str(s) for s in scenarios],
        "grid": {k: [x for x in v] for k, v in grid.items()},
        "effective_grid": {k: [x for x in v] for k, v in grid_s.items()},
        "n_base_rows": int(len(raw_df)),
        "n_base_cows": int(raw_df[COW].astype(str).nunique()),
        "streamlit_source": str(raw_source_for_streamlit) if raw_source_for_streamlit is not None else str(Path(raw_input_path).resolve()),
        "verbose": bool(verbose),
        "progress_every": int(progress_every),
        "checkpoint_every": int(checkpoint_every),
        "detectable_events_per_run": int(detectable_events_per_run),
        "go_mode": str(go_mode),
        "go_thresholds": {k: float(v) for k, v in go_thresholds.items()},
        "write_best_streamlit_artifacts": bool(write_best_streamlit_artifacts),
        "base_summary_file": str(robust_summary_base_file.resolve()),
        "status": "completed",
        "run_counter": int(run_counter),
        "total_expected": int(total_expected),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
