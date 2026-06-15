"""Moteur de validation robuste (grid-search) et agrégation des runs."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from core.io import COW, TIME, load_csv
from core.pipeline import run_pipeline_herd
from scripts._validation_robust.run_execute import execute_robust_grid_runs
from scripts._validation_robust.run_summary import build_robust_summary, write_completed_meta, write_final_outputs
from scripts.validation_common import normalize_params
from scripts.validation_selection import (
    _augment_summary_with_injectability,
    _resolve_target_cow,
    extract_detected_episodes,
    pick_cows_for_injection,
)


def run_robust_validation(
    *,
    raw_input_path: str | Path,
    output_root: str | Path,
    base_params: Dict[str, object],
    seeds: List[int],
    scenarios: List[str],
    grid: Dict[str, List[object]],
    raw_source_for_streamlit: str | Path | None = None,
    verbose: bool = False,
    progress_every: int = 1,
    detectable_events_per_run: int = 1,
    go_mode: str = "legacy",
    go_thresholds: Optional[Dict[str, float]] = None,
    checkpoint_every: int = 25,
    write_best_streamlit_artifacts: bool = True,
    injection_mode: str = "auto",
    fixed_detect_cow: Optional[str] = None,
    fixed_hidden_cow: Optional[str] = None,
) -> Dict[str, object]:
    """Exécute une validation robuste complète par grid-search : injecte x seeds x scénarios, agrège et produit GO/NO-GO."""
    params0 = normalize_params(base_params)
    params0["persist_hours"] = max(6, int(params0.get("persist_hours", 6)))
    params0["alert_min"] = max(2, int(params0.get("alert_min", 2)))
    params0["contamination"] = max(0.05, float(params0.get("contamination", 0.06)))
    out_root = Path(output_root)
    run_dir = out_root / f"robust_validation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_csv(str(raw_input_path))
    base_summary, base_pred = run_pipeline_herd(raw_df, **params0)
    base_pred[TIME] = pd.to_datetime(base_pred[TIME], errors="coerce")
    base_episodes = extract_detected_episodes(base_pred)
    base_summary_auto = _augment_summary_with_injectability(base_summary, raw_df)

    injection_mode = str(injection_mode).strip().lower()
    if injection_mode not in {"auto", "fixed"}:
        raise ValueError("injection_mode must be one of: 'auto', 'fixed'")
    if injection_mode == "fixed":
        if not fixed_detect_cow or not fixed_hidden_cow:
            raise ValueError("injection_mode='fixed' requires fixed_detect_cow and fixed_hidden_cow.")
        selected_detect_cow = _resolve_target_cow(base_summary_auto, str(fixed_detect_cow))
        selected_hidden_cow = _resolve_target_cow(base_summary_auto, str(fixed_hidden_cow))
        if selected_hidden_cow == selected_detect_cow:
            s = base_summary_auto.copy()
            s[COW] = s[COW].astype(str)
            alt = [c for c in s[COW].tolist() if c != selected_detect_cow]
            if alt:
                selected_hidden_cow = str(alt[0])
    else:
        selected_detect_cow, selected_hidden_cow = pick_cows_for_injection(base_summary_auto)

    grid_s = {k: list(v) for k, v in grid.items()}
    if "persist_hours" in grid_s:
        grid_s["persist_hours"] = sorted(set(int(x) for x in grid_s["persist_hours"] if int(x) >= 6))
    if "alert_min" in grid_s:
        grid_s["alert_min"] = sorted(set(int(x) for x in grid_s["alert_min"] if int(x) >= 2))
    if "contamination" in grid_s:
        grid_s["contamination"] = sorted(set(float(x) for x in grid_s["contamination"] if float(x) >= 0.05))

    if "persist_hours" in grid_s and len(grid_s["persist_hours"]) == 0:
        grid_s["persist_hours"] = [6]
    if "alert_min" in grid_s and len(grid_s["alert_min"]) == 0:
        grid_s["alert_min"] = [2]
    if "contamination" in grid_s and len(grid_s["contamination"]) == 0:
        grid_s["contamination"] = [0.05]

    param_names = list(grid_s.keys())
    param_sets = [dict(zip(param_names, vals)) for vals in product(*[grid_s[k] for k in param_names])]
    total_expected = int(len(param_sets) * len(seeds) * len(scenarios))
    progress_every = max(1, int(progress_every))
    detectable_events_per_run = max(1, int(detectable_events_per_run))
    checkpoint_every = max(1, int(checkpoint_every))
    default_go_thresholds: Dict[str, float] = {
        "strong_iou20_p10_min": 0.20,
        "borderline_any_mean_min": 0.20,
        "hidden_leak_any_p90_max": 0.20,
        "false_notif_p90_max": 0.30,
    }
    if go_thresholds:
        default_go_thresholds.update({k: float(v) for k, v in go_thresholds.items()})
    go_mode = str(go_mode).strip().lower()
    if go_mode not in {"legacy", "scenario_aware"}:
        raise ValueError("go_mode must be one of: 'legacy', 'scenario_aware'")

    if verbose:
        print(
            f"[robust] start: params={len(param_sets)} x seeds={len(seeds)} x scenarios={len(scenarios)} -> {total_expected} runs",
            flush=True,
        )
        print("[robust] minima enforced: contamination>=0.05, persist_hours>=6, alert_min>=2", flush=True)
        print(f"[robust] effective grid: {grid_s}", flush=True)
        print(
            f"[robust] mode: go_mode={go_mode}, detectable_events_per_run={detectable_events_per_run}, go_thresholds={default_go_thresholds}",
            flush=True,
        )
        print(
            f"[robust] injection: mode={injection_mode}, detect_cow={selected_detect_cow}, hidden_cow={selected_hidden_cow}",
            flush=True,
        )

    start_meta = {
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
        "go_thresholds": {k: float(v) for k, v in default_go_thresholds.items()},
        "status": "running",
        "run_counter": 0,
        "total_expected": int(total_expected),
        "injection_mode": str(injection_mode),
        "fixed_detect_cow_requested": "" if fixed_detect_cow is None else str(fixed_detect_cow),
        "fixed_hidden_cow_requested": "" if fixed_hidden_cow is None else str(fixed_hidden_cow),
        "injection_detect_cow": str(selected_detect_cow),
        "injection_hidden_cow": str(selected_hidden_cow),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(start_meta, indent=2), encoding="utf-8")

    runs_df, events_df, sample_streamlit_files, run_counter = execute_robust_grid_runs(
        raw_df=raw_df,
        base_summary=base_summary,
        params0=params0,
        param_sets=param_sets,
        param_names=param_names,
        seeds=seeds,
        scenarios=scenarios,
        selected_detect_cow=str(selected_detect_cow),
        selected_hidden_cow=str(selected_hidden_cow),
        detectable_events_per_run=detectable_events_per_run,
        verbose=bool(verbose),
        progress_every=progress_every,
        total_expected=total_expected,
        run_dir=run_dir,
        checkpoint_every=checkpoint_every,
        start_meta=start_meta,
    )

    group_cols = list(param_names)
    robust_summary = build_robust_summary(
        runs_df=runs_df,
        group_cols=group_cols,
        scenarios=[str(s) for s in scenarios],
        go_thresholds=default_go_thresholds,
        go_mode=go_mode,
    )

    robust_summary_base_file = write_final_outputs(
        run_dir=run_dir,
        base_summary=base_summary,
        base_episodes=base_episodes,
        runs_df=runs_df,
        events_df=events_df,
        robust_summary=robust_summary,
        sample_streamlit_files=sample_streamlit_files,
        group_cols=group_cols,
        raw_df=raw_df,
        write_best_streamlit_artifacts=bool(write_best_streamlit_artifacts),
    )

    write_completed_meta(
        run_dir=run_dir,
        raw_input_path=raw_input_path,
        raw_source_for_streamlit=raw_source_for_streamlit,
        params0=params0,
        seeds=seeds,
        scenarios=scenarios,
        grid=grid,
        grid_s=grid_s,
        raw_df=raw_df,
        verbose=bool(verbose),
        progress_every=progress_every,
        checkpoint_every=checkpoint_every,
        detectable_events_per_run=detectable_events_per_run,
        go_mode=go_mode,
        go_thresholds=default_go_thresholds,
        write_best_streamlit_artifacts=bool(write_best_streamlit_artifacts),
        robust_summary_base_file=robust_summary_base_file,
        run_counter=run_counter,
        total_expected=total_expected,
    )

    return {
        "run_dir": run_dir,
        "runs": runs_df,
        "events": events_df,
        "summary": robust_summary,
        "baseline_summary": base_summary,
        "baseline_episodes": base_episodes,
    }
