"""Exports scénario, Pareto et artefacts Streamlit pour le wrapper V5.1/V5.2."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from core.io import load_csv
from scripts._validation_v5_1.base import base_utils

def _scenario_report(events_df: pd.DataFrame) -> pd.DataFrame:
    """Construit un résumé par scénario des comptages d'événements et métriques de détection."""
    if len(events_df) == 0:
        return pd.DataFrame(
            columns=[
                "scenario",
                "n_events",
                "n_expected_detected",
                "detected_any_overlap_mean",
                "detected_iou20_mean",
                "best_iou_mean",
            ]
        )

    rows = []
    for sc, g in events_df.groupby("scenario", dropna=False):
        rows.append(
            {
                "scenario": str(sc),
                "n_events": int(len(g)),
                "n_expected_detected": int(pd.to_numeric(g.get("expected_detected", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
                "detected_any_overlap_mean": float(pd.to_numeric(g.get("detected_any_overlap", pd.Series(dtype=float)), errors="coerce").mean()),
                "detected_iou20_mean": float(pd.to_numeric(g.get("detected_iou20", pd.Series(dtype=float)), errors="coerce").mean()),
                "best_iou_mean": float(pd.to_numeric(g.get("best_iou", pd.Series(dtype=float)), errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario"]).reset_index(drop=True)


def _is_dominated(a: pd.Series, b: pd.Series) -> bool:
    """Retourne ``True`` si la configuration *a* est dominée au sens de Pareto par *b*."""
    cond_all = (
        (b["border_detect_any_mean"] >= a["border_detect_any_mean"])
        and (b["false_notif_p90"] <= a["false_notif_p90"])
        and (b["hidden_leak_any_p90"] <= a["hidden_leak_any_p90"])
    )
    cond_one_strict = (
        (b["border_detect_any_mean"] > a["border_detect_any_mean"])
        or (b["false_notif_p90"] < a["false_notif_p90"])
        or (b["hidden_leak_any_p90"] < a["hidden_leak_any_p90"])
    )
    return bool(cond_all and cond_one_strict)


def _pareto_frontier(summary_v5_1: pd.DataFrame) -> pd.DataFrame:
    """Extrait les configurations Pareto-optimales du résumé v5.1."""
    if len(summary_v5_1) == 0:
        return summary_v5_1.copy()
    pts = summary_v5_1.copy().reset_index(drop=True)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        for j in range(len(pts)):
            if i == j or not keep[j]:
                continue
            if _is_dominated(pts.iloc[i], pts.iloc[j]):
                keep[i] = False
                break
    out = pts[keep].copy()
    out = out.sort_values(["robust_pass_v5_1", "border_detect_any_mean", "false_notif_p90"], ascending=[False, False, True]).reset_index(drop=True)
    return out


def _pick_best_streamlit(
    *,
    run_dir: Path,
    summary_v5_1: pd.DataFrame,
    runs_df: pd.DataFrame,
    events_df: pd.DataFrame,
    raw_input_path: str | Path,
    grid_cols: List[str],
) -> Dict[str, str]:
    """Sélectionne le run au meilleur score et exporte ses artefacts d'injection Streamlit."""
    if len(summary_v5_1) == 0 or len(runs_df) == 0 or len(events_df) == 0:
        return {}

    best_cfg = summary_v5_1.sort_values(["robust_pass_v5_1", "selection_score_v5_1"], ascending=[False, False]).iloc[0]

    sel = runs_df.copy()
    for c in grid_cols:
        if c not in sel.columns or c not in best_cfg.index:
            continue
        if pd.api.types.is_numeric_dtype(sel[c]):
            sel = sel[np.isclose(pd.to_numeric(sel[c], errors="coerce"), float(best_cfg[c]), rtol=1e-9, atol=1e-9)]
        else:
            sel = sel[sel[c].astype(str) == str(best_cfg[c])]

    if len(sel) == 0:
        return {}

    pick = sel.sort_values(["detectable_recall_iou20", "detectable_recall_any"], ascending=[False, False]).iloc[0]

    chosen = events_df.copy()
    for c in grid_cols + ["seed", "scenario"]:
        if c not in chosen.columns or c not in pick.index:
            continue
        if pd.api.types.is_numeric_dtype(chosen[c]):
            chosen = chosen[np.isclose(pd.to_numeric(chosen[c], errors="coerce"), float(pick[c]), rtol=1e-9, atol=1e-9)]
        else:
            chosen = chosen[chosen[c].astype(str) == str(pick[c])]

    if len(chosen) == 0:
        return {}

    keep = ["event_id", "cow", "start", "end", "duration_bins", "profile", "expected_detected", "design_reason"]
    keep = [c for c in keep if c in chosen.columns]
    manifest = chosen[keep].drop_duplicates().copy()

    raw_df = load_csv(str(raw_input_path))
    streamlit_df, windows_df = base_utils.build_streamlit_raw_with_events(raw_df, manifest, seed=int(pick["seed"]))

    brut_path = run_dir / "streamlit_brut_with_best_injections_v5_1.csv"
    win_path = run_dir / "streamlit_best_injection_windows_v5_1.csv"
    streamlit_df.to_csv(brut_path, index=False)
    windows_df.to_csv(win_path, index=False)

    return {
        "streamlit_brut_with_best_injections_v5_1": str(brut_path.resolve()),
        "streamlit_best_injection_windows_v5_1": str(win_path.resolve()),
    }
