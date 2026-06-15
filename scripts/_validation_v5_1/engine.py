"""Moteur du wrapper robuste V5.1/V5.2 (agrégation, Pareto, exports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from scripts._validation_v5_1.helpers import (
    _archive_base_summary_file,
    _build_summary_v5_1,
    _call_base_run_robust_validation_compat,
    _clean_grid,
    _json_safe,
    _pareto_frontier,
    _pick_best_streamlit,
    _scenario_report,
)

def run_robust_validation_v5_1(
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
    detectable_events_per_run: int = 3,
    checkpoint_every: int = 25,
    go_thresholds_v5_1: Optional[Dict[str, float]] = None,
    reference_cfg: Optional[Dict[str, object]] = None,
    nonregression_tolerances: Optional[Dict[str, float]] = None,
    injection_mode: str = "auto",
    fixed_detect_cow: Optional[str] = None,
    fixed_hidden_cow: Optional[str] = None,
) -> Dict[str, object]:
    """
    V5.1: optimisation borderline sans casser la securite.
    - Reutilise le moteur robuste de base pour les runs
    - Ajoute quality + safety + non-regression adaptative
    - Utilise une reference auto "balanced" si la reference fixee est non representative
    """
    effective_grid = _clean_grid(grid)
    grid_cols = list(effective_grid.keys())

    thresholds = {
        "strong_iou20_p10_min": 0.25,
        "strong_any_mean_min": 0.90,
        "borderline_any_mean_min": 0.20,
        "borderline_any_p10_min": 0.00,
        "hidden_leak_any_p90_max": 0.10,
        "false_notif_p90_max": 0.25,
    }
    if go_thresholds_v5_1:
        thresholds.update({k: float(v) for k, v in go_thresholds_v5_1.items()})

    # Garde-fous de base utilisés pendant l'agrégation des runs dans le moteur robuste.
    base_gate = {
        "strong_iou20_p10_min": float(thresholds["strong_iou20_p10_min"]),
        "borderline_any_mean_min": float(thresholds["borderline_any_mean_min"]),
        "hidden_leak_any_p90_max": float(thresholds["hidden_leak_any_p90_max"]),
        "false_notif_p90_max": float(thresholds["false_notif_p90_max"]),
    }

    base_res = _call_base_run_robust_validation_compat(
        raw_input_path=raw_input_path,
        output_root=output_root,
        base_params=base_params,
        seeds=seeds,
        scenarios=scenarios,
        grid=effective_grid,
        raw_source_for_streamlit=raw_source_for_streamlit,
        verbose=verbose,
        progress_every=progress_every,
        detectable_events_per_run=detectable_events_per_run,
        go_mode="scenario_aware",
        go_thresholds=base_gate,
        checkpoint_every=checkpoint_every,
        # Éviter les doublons "best streamlit" du moteur de base ; v5.1 écrit les siens.
        write_best_streamlit_artifacts=False,
        injection_mode=injection_mode,
        fixed_detect_cow=fixed_detect_cow,
        fixed_hidden_cow=fixed_hidden_cow,
    )

    run_dir = Path(base_res["run_dir"])
    archived_base_summary = _archive_base_summary_file(run_dir)
    runs_df = base_res.get("runs", pd.DataFrame()).copy()
    events_df = base_res.get("events", pd.DataFrame()).copy()

    summary_v5_1, ref_payload = _build_summary_v5_1(
        runs_df,
        grid_cols=grid_cols,
        thresholds=thresholds,
        reference_cfg=reference_cfg,
        nonregression_tolerances=nonregression_tolerances,
    )
    scenario_v5_1 = _scenario_report(events_df)
    frontier_v5_1 = _pareto_frontier(summary_v5_1)

    summary_file = run_dir / "robust_summary_v5_1.csv"
    scen_file = run_dir / "robust_events_by_scenario_v5_1.csv"
    frontier_file = run_dir / "robust_pareto_frontier_v5_1.csv"

    summary_v5_1.to_csv(summary_file, index=False)
    scenario_v5_1.to_csv(scen_file, index=False)
    frontier_v5_1.to_csv(frontier_file, index=False)

    best_files = _pick_best_streamlit(
        run_dir=run_dir,
        summary_v5_1=summary_v5_1,
        runs_df=runs_df,
        events_df=events_df,
        raw_input_path=raw_input_path,
        grid_cols=grid_cols,
    )

    nonreg_default = {
        "delta_false_notif_p90_max": float((nonregression_tolerances or {}).get("delta_false_notif_p90_max", 0.07)),
        "delta_hidden_leak_p90_max": float((nonregression_tolerances or {}).get("delta_hidden_leak_p90_max", 0.02)),
        "ratio_false_notif_max": float((nonregression_tolerances or {}).get("ratio_false_notif_max", 1.45)),
    }

    meta = {
        "raw_input_path": str(Path(raw_input_path).resolve()),
        "run_dir": str(run_dir.resolve()),
        "scenarios": [str(s) for s in scenarios],
        "seeds": [int(s) for s in seeds],
        "effective_grid": {k: [x for x in v] for k, v in effective_grid.items()},
        "detectable_events_per_run": int(max(1, detectable_events_per_run)),
        "go_thresholds_v5_1": {k: float(v) for k, v in thresholds.items()},
        "reference_cfg_requested": {k: v for k, v in (reference_cfg or {}).items()},
        "reference_payload": ref_payload,
        "nonregression_tolerances": nonreg_default,
        "n_runs": int(len(runs_df)),
        "n_events": int(len(events_df)),
        "n_cfg": int(len(summary_v5_1)),
        "n_go_v5_1": int((summary_v5_1["robust_pass_v5_1"] == 1).sum()) if "robust_pass_v5_1" in summary_v5_1.columns else 0,
        "n_frontier": int(len(frontier_v5_1)),
        "base_summary_archived_file": str(archived_base_summary.resolve()) if archived_base_summary else "",
        "injection_mode": str(injection_mode),
        "fixed_detect_cow_requested": "" if fixed_detect_cow is None else str(fixed_detect_cow),
        "fixed_hidden_cow_requested": "" if fixed_hidden_cow is None else str(fixed_hidden_cow),
    }
    meta.update(best_files)
    (run_dir / "run_meta_v5_1.json").write_text(json.dumps(_json_safe(meta), indent=2), encoding="utf-8")

    out = dict(base_res)
    out.update(
        {
            "summary_v5_1": summary_v5_1,
            "summary_v5_1_file": summary_file,
            "scenario_report_v5_1": scenario_v5_1,
            "scenario_report_v5_1_file": scen_file,
            "frontier_v5_1": frontier_v5_1,
            "frontier_v5_1_file": frontier_file,
            "go_thresholds_v5_1": thresholds,
            "reference_payload_v5_1": ref_payload,
            "base_summary_archived_file": archived_base_summary,
        }
    )
    out.update(best_files)
    return out
