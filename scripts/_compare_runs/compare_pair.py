"""Comparaison robuste pair-à-pair (v4 vs v5.1)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts._compare_runs.helpers import (
    ComparisonResult,
    REQUIRED_V4,
    REQUIRED_V5_BASE,
    SCENARIO_CANDIDATES_V5_1,
    SUMMARY_CANDIDATES_V5_1,
    _build_full_key_cols,
    _collapse_configs_by_keys,
    _ensure_cfg_columns,
    _is_complete,
    _is_complete_with_candidates,
    _merge_meta,
    _read_optional_json,
    _resolve_existing_file,
    _standardize_key_columns,
    _to_float_safe,
)

def compare_runs(v4_run: Path, v5_1_run: Path, output_root: Path) -> ComparisonResult:
    """Compare un run de validation v4 et v5.1, en produisant des CSV config et scénario."""
    v4_summary_path = v4_run / "robust_summary_v4.csv"
    v51_summary_path = _resolve_existing_file(v5_1_run, SUMMARY_CANDIDATES_V5_1)

    if not _is_complete(v4_run, REQUIRED_V4):
        missing = [f for f in REQUIRED_V4 if not (v4_run / f).exists()]
        raise FileNotFoundError(f"V4 run incomplet: {v4_run} | fichiers manquants: {missing}")
    if not _is_complete_with_candidates(v5_1_run, REQUIRED_V5_BASE, SUMMARY_CANDIDATES_V5_1):
        missing = [f for f in REQUIRED_V5_BASE if not (v5_1_run / f).exists()]
        if v51_summary_path is None:
            missing.append("robust_summary_v5_1.csv|robust_summary_v5_2.csv")
        raise FileNotFoundError(f"V5.1 run incomplet: {v5_1_run} | fichiers manquants: {missing}")
    if v51_summary_path is None:
        raise FileNotFoundError(f"Summary v5.1 introuvable dans {v5_1_run}")

    v4 = pd.read_csv(v4_summary_path)
    v51 = pd.read_csv(v51_summary_path)
    meta_v4 = _merge_meta(
        _read_optional_json(v4_run / "run_meta.json"),
        _read_optional_json(v4_run / "run_meta_v4.json"),
    )
    meta_v51 = _merge_meta(
        _read_optional_json(v5_1_run / "run_meta.json"),
        _read_optional_json(v5_1_run / "run_meta_v5_1.json"),
    )

    if "robust_pass_v4" not in v4.columns:
        raise ValueError(f"Colonne introuvable dans v4: robust_pass_v4 ({v4_summary_path})")
    if "robust_pass_v5_1" not in v51.columns:
        raise ValueError(f"Colonne introuvable dans v5.1: robust_pass_v5_1 ({v51_summary_path})")

    keys = _build_full_key_cols(v4, v51, meta_v4, meta_v51)
    if not keys:
        raise ValueError("Impossible de trouver des colonnes de config communes pour fusionner v4 et v5.1.")

    v4_pre = _ensure_cfg_columns(v4, keys, meta_v4)
    v51_pre = _ensure_cfg_columns(v51, keys, meta_v51)
    v4_std = _standardize_key_columns(v4_pre, keys)
    v51_std = _standardize_key_columns(v51_pre, keys)

    missing_cfg_cols_in_v4 = [c for c in keys if c not in v4.columns]
    missing_cfg_cols_in_v51 = [c for c in keys if c not in v51.columns]

    keep_v4 = keys + [c for c in ["robust_pass_v4", "selection_score_v4", "go_reason_v4"] if c in v4_std.columns]
    keep_v51 = keys + [
        c
        for c in [
            "robust_pass_v5_1",
            "selection_score_v5_1",
            "go_reason_v5_1",
            "quality_pass_v5_1",
            "safety_pass_v5_1",
            "nonregression_pass_v5_1",
        ]
        if c in v51_std.columns
    ]

    v4_cmp = _collapse_configs_by_keys(
        v4_std[keep_v4],
        key_cols=keys,
        pass_col="robust_pass_v4",
        score_col="selection_score_v4",
        reason_col="go_reason_v4",
    ).rename(columns={"n_variants": "n_variants_v4"})

    v51_cmp = _collapse_configs_by_keys(
        v51_std[keep_v51],
        key_cols=keys,
        pass_col="robust_pass_v5_1",
        score_col="selection_score_v5_1",
        reason_col="go_reason_v5_1",
        extra_flag_cols=["quality_pass_v5_1", "safety_pass_v5_1", "nonregression_pass_v5_1"],
    ).rename(columns={"n_variants": "n_variants_v5_1"})

    cfg_cmp = v4_cmp.merge(v51_cmp, on=keys, how="outer", indicator=True)
    if "robust_pass_v4" in cfg_cmp.columns and "robust_pass_v5_1" in cfg_cmp.columns:
        cfg_cmp["same_pass"] = (
            cfg_cmp["robust_pass_v4"].fillna(-1).astype(int) == cfg_cmp["robust_pass_v5_1"].fillna(-2).astype(int)
        ).astype(int)
    else:
        cfg_cmp["same_pass"] = 0

    v4_events_path = _resolve_existing_file(v4_run, ["robust_events_by_scenario_v4.csv"])
    v51_events_path = _resolve_existing_file(v5_1_run, SCENARIO_CANDIDATES_V5_1)
    scenario_cmp = pd.DataFrame()
    if v4_events_path is not None and v51_events_path is not None:
        ev4 = pd.read_csv(v4_events_path)
        ev51 = pd.read_csv(v51_events_path)
        scenario_cmp = ev4.merge(ev51, on="scenario", how="outer", suffixes=("_v4", "_v5_1"))
        for c in ["detected_any_overlap_mean", "detected_iou20_mean", "best_iou_mean"]:
            c4 = f"{c}_v4"
            c51 = f"{c}_v5_1"
            if c4 in scenario_cmp.columns and c51 in scenario_cmp.columns:
                scenario_cmp[f"delta_{c}_v5_1_minus_v4"] = scenario_cmp[c51] - scenario_cmp[c4]

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"v4_vs_v5_1_comparison_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_cmp_path = out_dir / "config_comparison_v4_vs_v5_1.csv"
    cfg_cmp.to_csv(cfg_cmp_path, index=False)

    scenario_cmp_path = out_dir / "scenario_comparison_v4_vs_v5_1.csv"
    if len(scenario_cmp):
        scenario_cmp.to_csv(scenario_cmp_path, index=False)

    n_go_v4 = int((pd.to_numeric(v4["robust_pass_v4"], errors="coerce").fillna(0) == 1).sum())
    n_go_v51 = int((pd.to_numeric(v51["robust_pass_v5_1"], errors="coerce").fillna(0) == 1).sum())
    overlap = cfg_cmp[cfg_cmp["_merge"] == "both"].copy() if "_merge" in cfg_cmp.columns else cfg_cmp.copy()
    same_pass = int(overlap["same_pass"].sum()) if "same_pass" in overlap.columns else 0
    overlap_count = int(len(overlap))

    summary = {
        "v4_run_dir": str(v4_run.resolve()),
        "v5_1_run_dir": str(v5_1_run.resolve()),
        "output_dir": str(out_dir.resolve()),
        "n_cfg_v4": int(len(v4)),
        "n_cfg_v5_1": int(len(v51)),
        "n_cfg_v4_collapsed_on_full_keys": int(len(v4_cmp)),
        "n_cfg_v5_1_collapsed_on_full_keys": int(len(v51_cmp)),
        "n_go_v4": n_go_v4,
        "n_go_v5_1": n_go_v51,
        "n_overlap_cfg": overlap_count,
        "same_pass_count": same_pass,
        "same_pass_ratio": float(same_pass / max(1, overlap_count)),
        "full_config_keys": keys,
        "missing_cfg_cols_in_v4_summary": missing_cfg_cols_in_v4,
        "missing_cfg_cols_in_v5_1_summary": missing_cfg_cols_in_v51,
        "files": {
            "config_comparison": str(cfg_cmp_path.resolve()),
            "scenario_comparison": str(scenario_cmp_path.resolve()) if scenario_cmp_path.exists() else "",
        },
        "run_meta_v4": meta_v4,
        "run_meta_v5_1": meta_v51,
    }
    summary_path = out_dir / "comparison_summary_v4_vs_v5_1.json"
    summary_path.write_text(json.dumps(_to_float_safe(summary), indent=2), encoding="utf-8")

    return ComparisonResult(output_dir=out_dir, summary=summary)
