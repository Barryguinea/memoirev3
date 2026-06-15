"""Moteur de validation (facade) : API stable pour notebooks, fonctions reparties par modules.

Ce module conserve les noms historiques utilises par les notebooks de validation
(v2, v2.1, v4, v5.1, v5.2) mais délègue désormais l'implémentation à des
sous-modules spécialisés (selection, injection raw/processed, évaluation,
orchestrateurs manual/robust).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from core.features import interval_to_minutes
from core.io import COW, LYING, MI, STANDING, STEPS, TIME, TRANSITIONS, TR_DOWN, TR_UP, load_csv
from core.pipeline import run_pipeline_herd
from scripts import validation_runner_manual as _manual_runner
from scripts import validation_runner_robust as _robust_runner
from scripts.validation_common import (
    aggregate_distribution as _aggregate_distribution,
    count_cow_days,
    has_raw_sensor_cols,
    looks_processed_predictions,
    normalize_params,
    resolve_baseline_ratio,
    to_numeric_cols as _to_numeric_cols,
)
from scripts.validation_eval import (
    event_windows_mask as _event_windows_mask,
    evaluate_injected_events,
    rerun_alert_logic_processed,
)
from scripts.validation_injection_processed import (
    inject_manual_plan_processed,
    inject_two_events_processed,
    pick_rrz_col,
)
from scripts.validation_injection_raw import (
    InjectedEvent,
    _profile_to_raw_multipliers,
    apply_event_on_raw,
    build_streamlit_raw_with_events,
    choose_start_index,
    inject_manual_plan_raw,
    inject_two_events_raw,
)
from scripts.validation_selection import (
    _augment_summary_with_injectability,
    _manual_injection_plan,
    _pick_window_for_cow,
    _resolve_target_cow,
    extract_detected_episodes,
    pick_cows_for_injection,
    summarize_from_predictions,
)

# Re-export du helper scenario (utilise potentiellement dans des notebooks avancés)
_inject_scenario_raw = _robust_runner._inject_scenario_raw


def _sync_manual_runner_hooks() -> None:
    """Synchronise les dependances du runner manuel pour preservier le patching des tests/notebooks.

    Les tests existants patchent des symboles de ``scripts.validation_notebook_utils``.
    Comme l'implementation a ete deplacee dans ``validation_runner_manual``, on recopie
    ici les references courantes (potentiellement patchées) dans le module runner avant appel.
    """
    _manual_runner.load_csv = load_csv
    _manual_runner.run_pipeline_herd = run_pipeline_herd
    _manual_runner.has_raw_sensor_cols = has_raw_sensor_cols
    _manual_runner.looks_processed_predictions = looks_processed_predictions
    _manual_runner.normalize_params = normalize_params
    _manual_runner.evaluate_injected_events = evaluate_injected_events
    _manual_runner.rerun_alert_logic_processed = rerun_alert_logic_processed
    _manual_runner.inject_manual_plan_processed = inject_manual_plan_processed
    _manual_runner.build_streamlit_raw_with_events = build_streamlit_raw_with_events
    _manual_runner.inject_manual_plan_raw = inject_manual_plan_raw
    _manual_runner._augment_summary_with_injectability = _augment_summary_with_injectability
    _manual_runner._manual_injection_plan = _manual_injection_plan
    _manual_runner.extract_detected_episodes = extract_detected_episodes
    _manual_runner.summarize_from_predictions = summarize_from_predictions


def run_manual_validation(
    *,
    input_path: str | Path,
    output_root: str | Path,
    params: Dict[str, object],
    injection_seed: int = 123,
    raw_source_for_streamlit: str | Path | None = None,
    fixed_detect_cow: str = "8081",
    fixed_hidden_cow: str = "8154",
    n_detectable_events: int = 2,
    remove_preexisting_alert_cows_from_streamlit: bool = True,
    input_mode: str = "auto",
    save_dataset_with_injections: bool = False,
    manual_injection_variant: str = "legacy",
) -> Dict[str, object]:
    """Facade de compatibilite pour l'orchestrateur de validation manuelle."""
    _sync_manual_runner_hooks()
    return _manual_runner.run_manual_validation(
        input_path=input_path,
        output_root=output_root,
        params=params,
        injection_seed=injection_seed,
        raw_source_for_streamlit=raw_source_for_streamlit,
        fixed_detect_cow=fixed_detect_cow,
        fixed_hidden_cow=fixed_hidden_cow,
        n_detectable_events=n_detectable_events,
        remove_preexisting_alert_cows_from_streamlit=remove_preexisting_alert_cows_from_streamlit,
        input_mode=input_mode,
        save_dataset_with_injections=save_dataset_with_injections,
        manual_injection_variant=manual_injection_variant,
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
    """Facade de compatibilite pour l'orchestrateur de validation robuste."""
    return _robust_runner.run_robust_validation(
        raw_input_path=raw_input_path,
        output_root=output_root,
        base_params=base_params,
        seeds=seeds,
        scenarios=scenarios,
        grid=grid,
        raw_source_for_streamlit=raw_source_for_streamlit,
        verbose=verbose,
        progress_every=progress_every,
        detectable_events_per_run=detectable_events_per_run,
        go_mode=go_mode,
        go_thresholds=go_thresholds,
        checkpoint_every=checkpoint_every,
        write_best_streamlit_artifacts=write_best_streamlit_artifacts,
        injection_mode=injection_mode,
        fixed_detect_cow=fixed_detect_cow,
        fixed_hidden_cow=fixed_hidden_cow,
    )


__all__ = [
    "InjectedEvent",
    "summarize_from_predictions",
    "extract_detected_episodes",
    "_augment_summary_with_injectability",
    "pick_cows_for_injection",
    "_pick_window_for_cow",
    "_resolve_target_cow",
    "_manual_injection_plan",
    "choose_start_index",
    "_profile_to_raw_multipliers",
    "apply_event_on_raw",
    "pick_rrz_col",
    "inject_two_events_raw",
    "inject_manual_plan_raw",
    "inject_two_events_processed",
    "inject_manual_plan_processed",
    "rerun_alert_logic_processed",
    "_event_windows_mask",
    "evaluate_injected_events",
    "build_streamlit_raw_with_events",
    "run_manual_validation",
    "_inject_scenario_raw",
    "run_robust_validation",
]
