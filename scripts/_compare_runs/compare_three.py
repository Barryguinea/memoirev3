"""Comparaison robuste trois voies (v4 vs v5.1 vs v5.2)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from scripts._compare_runs.helpers import (
    ComparisonResult,
    REQUIRED_V4,
    REQUIRED_V5_BASE,
    SCENARIO_CANDIDATES_V5_1,
    SCENARIO_CANDIDATES_V5_2,
    SUMMARY_CANDIDATES_V5_1,
    SUMMARY_CANDIDATES_V5_2,
    _build_full_key_cols_many,
    _ensure_cfg_columns,
    _first_existing_column,
    _is_complete,
    _is_complete_with_candidates,
    _load_scenario_report,
    _merge_meta,
    _prepare_version_frame,
    _read_optional_json,
    _resolve_existing_file,
    _standardize_key_columns,
    _to_float_safe,
)

def compare_runs_three(v4_run: Path, v5_1_run: Path, v5_2_run: Path, output_root: Path) -> ComparisonResult:
    """Compare des runs v4, v5.1 et v5.2 avec accords pair-à-pair et à trois voies."""
    v4_summary_path = v4_run / "robust_summary_v4.csv"
    v51_summary_path = _resolve_existing_file(v5_1_run, SUMMARY_CANDIDATES_V5_1)
    v52_summary_path = _resolve_existing_file(v5_2_run, SUMMARY_CANDIDATES_V5_2)

    if not _is_complete(v4_run, REQUIRED_V4):
        missing = [f for f in REQUIRED_V4 if not (v4_run / f).exists()]
        raise FileNotFoundError(f"V4 run incomplet: {v4_run} | fichiers manquants: {missing}")
    if not _is_complete_with_candidates(v5_1_run, REQUIRED_V5_BASE, SUMMARY_CANDIDATES_V5_1):
        missing = [f for f in REQUIRED_V5_BASE if not (v5_1_run / f).exists()]
        if v51_summary_path is None:
            missing.append("robust_summary_v5_1.csv|robust_summary_v5_2.csv")
        raise FileNotFoundError(f"V5.1 run incomplet: {v5_1_run} | fichiers manquants: {missing}")
    if not _is_complete_with_candidates(v5_2_run, REQUIRED_V5_BASE, SUMMARY_CANDIDATES_V5_2):
        missing = [f for f in REQUIRED_V5_BASE if not (v5_2_run / f).exists()]
        if v52_summary_path is None:
            missing.append("robust_summary_v5_2.csv|robust_summary_v5_1.csv")
        raise FileNotFoundError(f"V5.2 run incomplet: {v5_2_run} | fichiers manquants: {missing}")
    if v51_summary_path is None or v52_summary_path is None:
        raise FileNotFoundError("Impossible de localiser les summaries v5.1/v5.2.")

    v4 = pd.read_csv(v4_summary_path)
    v51 = pd.read_csv(v51_summary_path)
    v52 = pd.read_csv(v52_summary_path)

    meta_v4 = _merge_meta(_read_optional_json(v4_run / "run_meta.json"), _read_optional_json(v4_run / "run_meta_v4.json"))
    meta_v51 = _merge_meta(_read_optional_json(v5_1_run / "run_meta.json"), _read_optional_json(v5_1_run / "run_meta_v5_1.json"))
    meta_v52 = _merge_meta(_read_optional_json(v5_2_run / "run_meta.json"), _read_optional_json(v5_2_run / "run_meta_v5_2.json"))

    if "robust_pass_v4" not in v4.columns:
        raise ValueError(f"Colonne introuvable dans v4: robust_pass_v4 ({v4_summary_path})")

    pass_col_v51 = _first_existing_column(v51, ["robust_pass_v5_1", "robust_pass_v5_2"])
    pass_col_v52 = _first_existing_column(v52, ["robust_pass_v5_2", "robust_pass_v5_1"])
    if pass_col_v51 is None:
        raise ValueError(f"Colonne pass introuvable dans v5.1 ({v51_summary_path})")
    if pass_col_v52 is None:
        raise ValueError(f"Colonne pass introuvable dans v5.2 ({v52_summary_path})")

    keys = _build_full_key_cols_many([v4, v51, v52], [meta_v4, meta_v51, meta_v52])
    if not keys:
        raise ValueError("Impossible de trouver des colonnes de config communes pour fusionner v4/v5.1/v5.2.")

    v4_std = _standardize_key_columns(_ensure_cfg_columns(v4, keys, meta_v4), keys)
    v51_std = _standardize_key_columns(_ensure_cfg_columns(v51, keys, meta_v51), keys)
    v52_std = _standardize_key_columns(_ensure_cfg_columns(v52, keys, meta_v52), keys)

    v4_cmp = _prepare_version_frame(
        v4_std,
        key_cols=keys,
        pass_col="robust_pass_v4",
        score_col=_first_existing_column(v4_std, ["selection_score_v4"]),
        reason_col=_first_existing_column(v4_std, ["go_reason_v4"]),
        extra_flag_cols=[],
        version_tag="v4",
    )
    v51_cmp = _prepare_version_frame(
        v51_std,
        key_cols=keys,
        pass_col=pass_col_v51,
        score_col=_first_existing_column(v51_std, ["selection_score_v5_1", "selection_score_v5_2"]),
        reason_col=_first_existing_column(v51_std, ["go_reason_v5_1", "go_reason_v5_2"]),
        extra_flag_cols=[
            c
            for c in [
                "quality_pass_v5_1",
                "safety_pass_v5_1",
                "nonregression_pass_v5_1",
                "quality_pass_v5_2",
                "safety_pass_v5_2",
                "nonregression_pass_v5_2",
            ]
            if c in v51_std.columns
        ],
        version_tag="v5_1",
    )
    v52_cmp = _prepare_version_frame(
        v52_std,
        key_cols=keys,
        pass_col=pass_col_v52,
        score_col=_first_existing_column(v52_std, ["selection_score_v5_2", "selection_score_v5_1"]),
        reason_col=_first_existing_column(v52_std, ["go_reason_v5_2", "go_reason_v5_1"]),
        extra_flag_cols=[
            c
            for c in [
                "quality_pass_v5_2",
                "safety_pass_v5_2",
                "nonregression_pass_v5_2",
                "quality_pass_v5_1",
                "safety_pass_v5_1",
                "nonregression_pass_v5_1",
            ]
            if c in v52_std.columns
        ],
        version_tag="v5_2",
    )

    cfg_cmp = v4_cmp.merge(v51_cmp, on=keys, how="outer").merge(v52_cmp, on=keys, how="outer")
    for a, b, name in [
        ("robust_pass_v4", "robust_pass_v5_1", "v4_vs_v5_1"),
        ("robust_pass_v4", "robust_pass_v5_2", "v4_vs_v5_2"),
        ("robust_pass_v5_1", "robust_pass_v5_2", "v5_1_vs_v5_2"),
    ]:
        col = f"same_pass_{name}"
        cfg_cmp[col] = np.nan
        if a in cfg_cmp.columns and b in cfg_cmp.columns:
            both = cfg_cmp[a].notna() & cfg_cmp[b].notna()
            cfg_cmp.loc[both, col] = (
                cfg_cmp.loc[both, a].astype(int) == cfg_cmp.loc[both, b].astype(int)
            ).astype(int)

    if all(c in cfg_cmp.columns for c in ["robust_pass_v4", "robust_pass_v5_1", "robust_pass_v5_2"]):
        both3 = cfg_cmp["robust_pass_v4"].notna() & cfg_cmp["robust_pass_v5_1"].notna() & cfg_cmp["robust_pass_v5_2"].notna()
        cfg_cmp["all_three_agree"] = np.nan
        cfg_cmp.loc[both3, "all_three_agree"] = (
            (cfg_cmp.loc[both3, "robust_pass_v4"].astype(int) == cfg_cmp.loc[both3, "robust_pass_v5_1"].astype(int))
            & (cfg_cmp.loc[both3, "robust_pass_v4"].astype(int) == cfg_cmp.loc[both3, "robust_pass_v5_2"].astype(int))
        ).astype(int)

    scen_v4 = _load_scenario_report(v4_run, ["robust_events_by_scenario_v4.csv"], "v4")
    scen_v51 = _load_scenario_report(v5_1_run, SCENARIO_CANDIDATES_V5_1, "v5_1")
    scen_v52 = _load_scenario_report(v5_2_run, SCENARIO_CANDIDATES_V5_2, "v5_2")
    scenario_cmp = pd.DataFrame()
    if len(scen_v4) or len(scen_v51) or len(scen_v52):
        scenario_cmp = scen_v4.copy()
        if len(scen_v51):
            scenario_cmp = scenario_cmp.merge(scen_v51, on="scenario", how="outer")
        if len(scen_v52):
            scenario_cmp = scenario_cmp.merge(scen_v52, on="scenario", how="outer")
        for metric in ["detected_any_overlap_mean", "detected_iou20_mean", "best_iou_mean"]:
            for hi, lo in [("v5_1", "v4"), ("v5_2", "v4"), ("v5_2", "v5_1")]:
                c_hi = f"{metric}_{hi}"
                c_lo = f"{metric}_{lo}"
                if c_hi in scenario_cmp.columns and c_lo in scenario_cmp.columns:
                    scenario_cmp[f"delta_{metric}_{hi}_minus_{lo}"] = scenario_cmp[c_hi] - scenario_cmp[c_lo]

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"v4_vs_v5_1_vs_v5_2_comparison_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_cmp_path = out_dir / "config_comparison_v4_vs_v5_1_vs_v5_2.csv"
    cfg_cmp.to_csv(cfg_cmp_path, index=False)

    scenario_cmp_path = out_dir / "scenario_comparison_v4_vs_v5_1_vs_v5_2.csv"
    if len(scenario_cmp):
        scenario_cmp.to_csv(scenario_cmp_path, index=False)

    n_go_v4 = int((pd.to_numeric(v4["robust_pass_v4"], errors="coerce").fillna(0) == 1).sum())
    n_go_v51 = int((pd.to_numeric(v51[pass_col_v51], errors="coerce").fillna(0) == 1).sum())
    n_go_v52 = int((pd.to_numeric(v52[pass_col_v52], errors="coerce").fillna(0) == 1).sum())

    def _pair_stats(a: str, b: str, same_col: str) -> Tuple[int, int, float]:
        """Calcule recouvrement, nombre d'accords et ratio d'accord pour une paire de versions."""
        if not all(c in cfg_cmp.columns for c in [a, b, same_col]):
            return 0, 0, 0.0
        both = cfg_cmp[a].notna() & cfg_cmp[b].notna()
        overlap = int(both.sum())
        same = int((cfg_cmp.loc[both, same_col] == 1).sum())
        ratio = float(same / max(1, overlap))
        return overlap, same, ratio

    overlap_41, same_41, ratio_41 = _pair_stats("robust_pass_v4", "robust_pass_v5_1", "same_pass_v4_vs_v5_1")
    overlap_42, same_42, ratio_42 = _pair_stats("robust_pass_v4", "robust_pass_v5_2", "same_pass_v4_vs_v5_2")
    overlap_12, same_12, ratio_12 = _pair_stats("robust_pass_v5_1", "robust_pass_v5_2", "same_pass_v5_1_vs_v5_2")
    overlap_all = 0
    agree_all = 0
    if "all_three_agree" in cfg_cmp.columns and all(c in cfg_cmp.columns for c in ["robust_pass_v4", "robust_pass_v5_1", "robust_pass_v5_2"]):
        both3 = cfg_cmp["robust_pass_v4"].notna() & cfg_cmp["robust_pass_v5_1"].notna() & cfg_cmp["robust_pass_v5_2"].notna()
        overlap_all = int(both3.sum())
        agree_all = int((cfg_cmp.loc[both3, "all_three_agree"] == 1).sum())

    summary = {
        "v4_run_dir": str(v4_run.resolve()),
        "v5_1_run_dir": str(v5_1_run.resolve()),
        "v5_2_run_dir": str(v5_2_run.resolve()),
        "summary_file_v4": str(v4_summary_path.resolve()),
        "summary_file_v5_1": str(v51_summary_path.resolve()),
        "summary_file_v5_2": str(v52_summary_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "n_cfg_v4": int(len(v4)),
        "n_cfg_v5_1": int(len(v51)),
        "n_cfg_v5_2": int(len(v52)),
        "n_cfg_v4_collapsed_on_full_keys": int(len(v4_cmp)),
        "n_cfg_v5_1_collapsed_on_full_keys": int(len(v51_cmp)),
        "n_cfg_v5_2_collapsed_on_full_keys": int(len(v52_cmp)),
        "n_go_v4": n_go_v4,
        "n_go_v5_1": n_go_v51,
        "n_go_v5_2": n_go_v52,
        "overlap_v4_vs_v5_1": overlap_41,
        "overlap_v4_vs_v5_2": overlap_42,
        "overlap_v5_1_vs_v5_2": overlap_12,
        "same_pass_count_v4_vs_v5_1": same_41,
        "same_pass_count_v4_vs_v5_2": same_42,
        "same_pass_count_v5_1_vs_v5_2": same_12,
        "same_pass_ratio_v4_vs_v5_1": ratio_41,
        "same_pass_ratio_v4_vs_v5_2": ratio_42,
        "same_pass_ratio_v5_1_vs_v5_2": ratio_12,
        "overlap_all_three": overlap_all,
        "all_three_agree_count": agree_all,
        "all_three_agree_ratio": float(agree_all / max(1, overlap_all)),
        "full_config_keys": keys,
        "missing_cfg_cols_in_v4_summary": [c for c in keys if c not in v4.columns],
        "missing_cfg_cols_in_v5_1_summary": [c for c in keys if c not in v51.columns],
        "missing_cfg_cols_in_v5_2_summary": [c for c in keys if c not in v52.columns],
        "files": {
            "config_comparison": str(cfg_cmp_path.resolve()),
            "scenario_comparison": str(scenario_cmp_path.resolve()) if scenario_cmp_path.exists() else "",
        },
        "run_meta_v4": meta_v4,
        "run_meta_v5_1": meta_v51,
        "run_meta_v5_2": meta_v52,
    }
    summary_path = out_dir / "comparison_summary_v4_vs_v5_1_vs_v5_2.json"
    summary_path.write_text(json.dumps(_to_float_safe(summary), indent=2), encoding="utf-8")
    return ComparisonResult(output_dir=out_dir, summary=summary)
