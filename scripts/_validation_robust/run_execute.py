"""Boucle d'exécution du grid-search robuste (injection -> pipeline -> évaluation)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.io import TIME
from core.pipeline import run_pipeline_herd
from scripts._validation_robust.scenarios import _inject_scenario_raw
from scripts.validation_common import count_cow_days, normalize_params
from scripts.validation_eval import event_windows_mask as _event_windows_mask, evaluate_injected_events
from scripts._validation_robust.run_summary import write_partial_outputs


def execute_robust_grid_runs(
    *,
    raw_df: pd.DataFrame,
    base_summary: pd.DataFrame,
    params0: Dict[str, object],
    param_sets: List[Dict[str, object]],
    param_names: List[str],
    seeds: List[int],
    scenarios: List[str],
    selected_detect_cow: str,
    selected_hidden_cow: str,
    detectable_events_per_run: int,
    verbose: bool,
    progress_every: int,
    total_expected: int,
    run_dir: Path,
    checkpoint_every: int,
    start_meta: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, object]], int]:
    """Exécute tous les runs du grid-search et retourne runs/events + manifest Streamlit."""
    all_runs: List[pd.DataFrame] = []
    all_events: List[pd.DataFrame] = []
    sample_streamlit_files: List[Dict[str, object]] = []
    run_counter = 0

    for cfg in param_sets:
        params = dict(params0)
        params.update(cfg)
        params = normalize_params(params)

        for seed in seeds:
            for sc in scenarios:
                run_counter += 1
                if verbose and ((run_counter - 1) % progress_every == 0):
                    print(
                        f"[robust] run {run_counter}/{total_expected} | scenario={sc} | seed={seed} | cfg={cfg}",
                        flush=True,
                    )
                df_inj, ev = _inject_scenario_raw(
                    raw_df,
                    base_summary,
                    interval=str(params["interval"]),
                    persist_hours=int(params["persist_hours"]),
                    seed=int(seed),
                    scenario=str(sc),
                    n_events=(detectable_events_per_run if str(sc).startswith("detectable_") else 1),
                    detect_cow=str(selected_detect_cow),
                    hidden_cow=str(selected_hidden_cow),
                )
                if len(ev) == 0:
                    if verbose:
                        print(f"[robust] skip run {run_counter}: no injectable window for scenario={sc} seed={seed}", flush=True)
                    continue

                _, pred_inj = run_pipeline_herd(df_inj, **params)
                pred_inj[TIME] = pd.to_datetime(pred_inj[TIME], errors="coerce")

                ev_eval = evaluate_injected_events(
                    pred_inj,
                    ev,
                    interval=str(params["interval"]),
                    persist_hours=int(params["persist_hours"]),
                    alert_min=int(params["alert_min"]),
                    mix_rate_thr=float(params["mix_rate_thr"]),
                )
                if len(ev_eval) and len(ev):
                    keep_cols = ["event_id", "design_reason"]
                    extra = ev[keep_cols].drop_duplicates()
                    ev_eval = ev_eval.merge(extra, on="event_id", how="left")
                ev_eval["scenario"] = str(sc)
                ev_eval["seed"] = int(seed)
                for k, v in cfg.items():
                    ev_eval[k] = v

                outside_mask = ~_event_windows_mask(pred_inj, ev)
                notif = pd.to_numeric(pred_inj.get("notif_lameness", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
                false_notif = int(notif[outside_mask].sum())
                false_notif_per_cow_day = float(false_notif / max(1, count_cow_days(pred_inj)))

                det = ev_eval[ev_eval["expected_detected"] == 1]
                hid = ev_eval[ev_eval["expected_detected"] == 0]
                row = {
                    "scenario": str(sc),
                    "seed": int(seed),
                    "false_notif_per_cow_day": float(false_notif_per_cow_day),
                    "detectable_recall_any": float(det["detected_any_overlap"].mean()) if len(det) else np.nan,
                    "detectable_recall_iou20": float(det["detected_iou20"].mean()) if len(det) else np.nan,
                    "detectable_iou_mean": float(pd.to_numeric(det["best_iou"], errors="coerce").mean()) if len(det) else np.nan,
                    "hidden_leak_rate_any": float(hid["detected_any_overlap"].mean()) if len(hid) else np.nan,
                    "hidden_leak_rate_iou20": float(hid["detected_iou20"].mean()) if len(hid) else np.nan,
                    "n_events": int(len(ev_eval)),
                }
                for k, v in cfg.items():
                    row[k] = v
                all_runs.append(pd.DataFrame([row]))
                all_events.append(ev_eval)
                if verbose and (run_counter % progress_every == 0):
                    print(
                        "[robust] result "
                        f"detect_any={row['detectable_recall_any']:.3f} "
                        f"detect_iou20={row['detectable_recall_iou20']:.3f} "
                        f"hidden_leak={row['hidden_leak_rate_any']:.3f} "
                        f"false_notif={row['false_notif_per_cow_day']:.3f}",
                        flush=True,
                    )

                if int(seed) == int(seeds[0]):
                    sc_dir = run_dir / "streamlit_samples"
                    sc_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"streamlit_{sc}_" + "_".join([f"{k}-{cfg[k]}" for k in param_names]) + f"_seed-{seed}.csv"
                    fpath = sc_dir / fname.replace("/", "_")
                    df_inj.to_csv(fpath, index=False)
                    sample_streamlit_files.append({"scenario": sc, "seed": int(seed), "file": str(fpath.resolve()), **cfg})

                if (run_counter % checkpoint_every) == 0:
                    write_partial_outputs(run_dir=run_dir, all_runs=all_runs, all_events=all_events, start_meta=start_meta, run_counter=run_counter)

    runs_df = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    return runs_df, events_df, sample_streamlit_files, run_counter
