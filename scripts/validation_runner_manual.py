"""Orchestrateur de validation manuelle (implementation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from core.io import COW, TIME, load_csv
from core.pipeline import run_pipeline_herd
from scripts.validation_common import has_raw_sensor_cols, looks_processed_predictions, normalize_params
from scripts.validation_eval import evaluate_injected_events, rerun_alert_logic_processed
from scripts.validation_injection_processed import inject_manual_plan_processed
from scripts.validation_injection_raw import build_streamlit_raw_with_events, inject_manual_plan_raw
from scripts.validation_selection import _augment_summary_with_injectability, _manual_injection_plan, extract_detected_episodes, summarize_from_predictions

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
    """Exécute une validation manuelle complète : injecte le plan, lance le pipeline, évalue et sauvegarde les artefacts."""
    p = normalize_params(params)
    manual_injection_variant = str(manual_injection_variant).strip().lower()
    if manual_injection_variant not in {"legacy", "v2_1"}:
        raise ValueError("manual_injection_variant must be one of: 'legacy', 'v2_1'.")
    default_streamlit_raw = Path(__file__).resolve().parents[1] / "data" / "brut.csv"
    out_root = Path(output_root)
    run_dir = out_root / f"manual_validation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df_input = load_csv(str(input_path))
    raw_mode = has_raw_sensor_cols(df_input)
    processed_mode = looks_processed_predictions(df_input)
    requested_mode = str(input_mode).strip().lower()
    if requested_mode not in {"auto", "raw", "processed"}:
        raise ValueError("input_mode must be one of: 'auto', 'raw', 'processed'.")
    if requested_mode == "raw" and not raw_mode:
        raise ValueError("input_mode='raw' requested, but input does not look like raw sensor data.")
    if requested_mode == "processed" and not processed_mode:
        raise ValueError("input_mode='processed' requested, but input does not look like processed predictions.")
    if requested_mode == "raw":
        processed_mode = False
    elif requested_mode == "processed":
        raw_mode = False

    if not raw_mode and not processed_mode:
        raise ValueError("Input not recognized as raw sensor data or processed prediction output.")

    if raw_mode:
        mode = "raw"
        summary_base, pred_base = run_pipeline_herd(df_input, **p)
        pred_base[TIME] = pd.to_datetime(pred_base[TIME], errors="coerce")
        summary_base = _augment_summary_with_injectability(summary_base, df_input)
        plan = _manual_injection_plan(
            summary_base,
            detect_cow=str(fixed_detect_cow),
            hidden_cow=str(fixed_hidden_cow),
            n_detectable=int(n_detectable_events),
            variant=manual_injection_variant,
        )
        df_inj, events_df = inject_manual_plan_raw(
            df_input,
            plan=plan,
            interval=str(p["interval"]),
            persist_hours=int(p["persist_hours"]),
            seed=int(injection_seed),
            baseline_pred=pred_base,
            cooldown_hours=int(p["cooldown_hours"]),
            baseline_ratio=p.get("baseline_ratio", None),
            manual_injection_variant=manual_injection_variant,
        )
        summary_inj, pred_inj = run_pipeline_herd(df_inj, **p)
        pred_inj[TIME] = pd.to_datetime(pred_inj[TIME], errors="coerce")
        streamlit_df = df_inj.copy()
        streamlit_applied = pd.DataFrame(
            [{"event_id": r.event_id, "cow": r.cow, "start": r.start, "end": r.end, "profile": r.profile, "rows_affected": int(r.duration_bins)} for r in events_df.itertuples(index=False)]
        )
    else:
        mode = "processed"
        pred_base = df_input.copy()
        pred_base[TIME] = pd.to_datetime(pred_base[TIME], errors="coerce")
        pred_base = pred_base.dropna(subset=[TIME]).copy()
        pred_base[COW] = pred_base[COW].astype(str)
        summary_base = summarize_from_predictions(pred_base)
        plan = _manual_injection_plan(
            summary_base,
            detect_cow=str(fixed_detect_cow),
            hidden_cow=str(fixed_hidden_cow),
            n_detectable=int(n_detectable_events),
            variant=manual_injection_variant,
        )
        df_inj, events_df = inject_manual_plan_processed(
            pred_base,
            plan=plan,
            interval=str(p["interval"]),
            persist_hours=int(p["persist_hours"]),
            seed=int(injection_seed),
        )
        pred_inj = rerun_alert_logic_processed(df_inj, p)
        pred_inj[TIME] = pd.to_datetime(pred_inj[TIME], errors="coerce")
        summary_inj = summarize_from_predictions(pred_inj)

        raw_src = raw_source_for_streamlit
        if raw_src is None:
            raw_src = default_streamlit_raw
        raw_df = load_csv(str(raw_src))
        streamlit_df, streamlit_applied = build_streamlit_raw_with_events(raw_df, events_df, seed=int(injection_seed))
        if remove_preexisting_alert_cows_from_streamlit and len(summary_base):
            alert_cows = set(
                summary_base[pd.to_numeric(summary_base.get("lameness_starts", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int) > 0][COW]
                .astype(str)
                .tolist()
            )
            focus_cows = set(events_df["cow"].astype(str).tolist()) if len(events_df) else set()
            if alert_cows:
                keep_mask = (~streamlit_df[COW].astype(str).isin(alert_cows)) | (streamlit_df[COW].astype(str).isin(focus_cows))
                streamlit_df = streamlit_df[keep_mask].copy().reset_index(drop=True)

    baseline_episodes = extract_detected_episodes(pred_base)
    eval_df = evaluate_injected_events(
        pred_inj,
        events_df,
        interval=str(p["interval"]),
        persist_hours=int(p["persist_hours"]),
        alert_min=int(p["alert_min"]),
        mix_rate_thr=float(p["mix_rate_thr"]),
    )

    alert_cols = [c for c in [COW, TIME, "pred_lameness_start", "notif_lameness", "alert_level", "if_anom_k", "anom_rate_k", "coherence_boiterie"] if c in pred_inj.columns]
    if "pred_lameness_start" in pred_inj.columns:
        inj_alerts = pred_inj[pred_inj["pred_lameness_start"].astype(int) == 1].copy()
    else:
        inj_alerts = pred_inj.copy()
    if len(events_df):
        focus_cows = set(events_df["cow"].astype(str).tolist())
        inj_alerts = inj_alerts[inj_alerts[COW].astype(str).isin(focus_cows)].copy()
    inj_alerts = inj_alerts[alert_cols].sort_values(TIME, ascending=False) if len(inj_alerts) else pd.DataFrame(columns=alert_cols)

    summary_base.to_csv(run_dir / "baseline_summary_by_cow.csv", index=False)
    baseline_episodes.to_csv(run_dir / "baseline_detected_episodes.csv", index=False)
    summary_inj.to_csv(run_dir / "injected_summary_by_cow.csv", index=False)
    events_df.to_csv(run_dir / "injected_events_manifest.csv", index=False)
    eval_df.to_csv(run_dir / "injected_events_evaluation.csv", index=False)
    inj_alerts.to_csv(run_dir / "injected_alerts.csv", index=False)
    dataset_with_inj_saved = bool(mode == "processed" or save_dataset_with_injections)
    if dataset_with_inj_saved:
        df_inj.to_csv(run_dir / "dataset_with_injections.csv", index=False)
    streamlit_df.to_csv(run_dir / "streamlit_brut_with_injections.csv", index=False)
    streamlit_applied.to_csv(run_dir / "streamlit_injection_windows.csv", index=False)

    # Comparaison avant/après pour contrôle rapide.
    cmp = summary_base[[COW, "lameness_notifs", "lameness_starts", "if_anomaly_points"]].merge(
        summary_inj[[COW, "lameness_notifs", "lameness_starts", "if_anomaly_points"]],
        on=COW,
        suffixes=("_before", "_after"),
    )
    cmp["delta_notifs"] = pd.to_numeric(cmp["lameness_notifs_after"], errors="coerce").fillna(0).astype(int) - pd.to_numeric(
        cmp["lameness_notifs_before"], errors="coerce"
    ).fillna(0).astype(int)
    cmp["delta_starts"] = pd.to_numeric(cmp["lameness_starts_after"], errors="coerce").fillna(0).astype(int) - pd.to_numeric(
        cmp["lameness_starts_before"], errors="coerce"
    ).fillna(0).astype(int)
    cmp["delta_if_anom"] = pd.to_numeric(cmp["if_anomaly_points_after"], errors="coerce").fillna(0).astype(int) - pd.to_numeric(
        cmp["if_anomaly_points_before"], errors="coerce"
    ).fillna(0).astype(int)
    cmp.to_csv(run_dir / "alerts_before_after_comparison.csv", index=False)

    meta = {
        "input": str(Path(input_path).resolve()),
        "output_dir": str(run_dir.resolve()),
        "mode": mode,
        "params": p,
        "injection_seed": int(injection_seed),
        "n_rows_input": int(len(df_input)),
        "n_rows_injected": int(len(df_inj)),
        "n_cows": int(df_input[COW].astype(str).nunique()),
        "streamlit_source": (
            str(raw_source_for_streamlit)
            if raw_source_for_streamlit is not None
            else ("input" if mode == "raw" else str(default_streamlit_raw.resolve()))
        ),
        "fixed_detect_cow": str(fixed_detect_cow),
        "fixed_hidden_cow": str(fixed_hidden_cow),
        "n_detectable_events": int(n_detectable_events),
        "manual_injection_variant": str(manual_injection_variant),
        "remove_preexisting_alert_cows_from_streamlit": bool(remove_preexisting_alert_cows_from_streamlit),
        "input_mode_requested": requested_mode,
        "save_dataset_with_injections": bool(save_dataset_with_injections),
        "dataset_with_injections_saved": bool(dataset_with_inj_saved),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "run_dir": run_dir,
        "mode": mode,
        "events": events_df,
        "evaluation": eval_df,
        "comparison": cmp,
        "baseline_summary": summary_base,
        "baseline_episodes": baseline_episodes,
        "injected_summary": summary_inj,
    }

