"""Logique de reference et calcul du resume V5.1/V5.2."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts._validation_v5_1.base import _scenario, _to_stats
from scripts._validation_v5_1.reference import _choose_reference_row, _is_reference_balanced, _match_cfg_row

def _build_summary_v5_1(
    runs_df: pd.DataFrame,
    *,
    grid_cols: List[str],
    thresholds: Dict[str, float],
    reference_cfg: Optional[Dict[str, object]] = None,
    nonregression_tolerances: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Agrège les stats par configuration, calcule les indicateurs qualité/sécurité/non-régression et classe les configurations."""
    if len(runs_df) == 0:
        return pd.DataFrame(), {"ref_source": "none"}

    rows: List[Dict[str, object]] = []
    for keys, g in runs_df.groupby(grid_cols, dropna=False):
        row: Dict[str, object] = {}
        if isinstance(keys, tuple):
            for i, k in enumerate(grid_cols):
                row[k] = keys[i]
        else:
            row[grid_cols[0]] = keys

        strong = _scenario(g, "detectable_strong")
        border = _scenario(g, "detectable_borderline")
        hidden = _scenario(g, "non_detectable_short")

        s_iou = _to_stats(strong.get("detectable_recall_iou20", pd.Series(dtype=float)))
        s_any = _to_stats(strong.get("detectable_recall_any", pd.Series(dtype=float)))
        b_any = _to_stats(border.get("detectable_recall_any", pd.Series(dtype=float)))
        h_any = _to_stats(hidden.get("hidden_leak_rate_any", pd.Series(dtype=float)))
        f_notif = _to_stats(g.get("false_notif_per_cow_day", pd.Series(dtype=float)))

        row["strong_detect_iou20_mean"] = s_iou["mean"]
        row["strong_detect_iou20_p10"] = s_iou["p10"]
        row["strong_detect_any_mean"] = s_any["mean"]
        row["strong_detect_any_p10"] = s_any["p10"]

        row["border_detect_any_mean"] = b_any["mean"]
        row["border_detect_any_p10"] = b_any["p10"]

        row["hidden_leak_any_mean"] = h_any["mean"]
        row["hidden_leak_any_p90"] = h_any["p90"]

        row["false_notif_mean"] = f_notif["mean"]
        row["false_notif_std"] = f_notif["std"]
        row["false_notif_p90"] = f_notif["p90"]

        row["n_runs"] = int(len(g))
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out, {"ref_source": "none"}

    ref_row, ref_info = _choose_reference_row(out, grid_cols=grid_cols, thresholds=thresholds, reference_cfg=reference_cfg)

    ref_false = float(ref_row.get("false_notif_p90", np.nan))
    ref_hidden = float(ref_row.get("hidden_leak_any_p90", np.nan))
    ref_border = float(ref_row.get("border_detect_any_mean", np.nan))

    out["ref_false_notif_p90"] = ref_false
    out["ref_hidden_leak_any_p90"] = ref_hidden
    out["ref_border_detect_any_mean"] = ref_border
    out["ref_source"] = str(ref_info.get("ref_source", "unknown"))

    out["delta_border_any_mean_vs_ref"] = out["border_detect_any_mean"] - ref_border
    out["delta_false_notif_p90_vs_ref"] = out["false_notif_p90"] - ref_false
    out["delta_hidden_leak_p90_vs_ref"] = out["hidden_leak_any_p90"] - ref_hidden

    # Ratio de non-régression (protège contre une dégradation relative)
    if np.isfinite(ref_false) and ref_false > 0:
        out["ratio_false_notif_vs_ref"] = out["false_notif_p90"] / ref_false
    else:
        out["ratio_false_notif_vs_ref"] = np.nan

    tol = {
        # Non-régression par défaut : limite la dérive tout en autorisant une
        # variance réaliste entre runs robustes sur données réelles de troupeau.
        "delta_false_notif_p90_max": 0.07,
        "delta_hidden_leak_p90_max": 0.02,
        "ratio_false_notif_max": 1.45,
    }
    if nonregression_tolerances:
        tol.update({k: float(v) for k, v in nonregression_tolerances.items()})

    if np.isfinite(ref_false) and ref_false > 0.03:
        ratio_ok = out["ratio_false_notif_vs_ref"] <= float(tol["ratio_false_notif_max"])
    else:
        ratio_ok = pd.Series(True, index=out.index)

    out["nonregression_pass_v5_1"] = (
        (out["delta_false_notif_p90_vs_ref"] <= float(tol["delta_false_notif_p90_max"]))
        & (out["delta_hidden_leak_p90_vs_ref"] <= float(tol["delta_hidden_leak_p90_max"]))
        & ratio_ok
    ).astype(int)

    out["safety_pass_v5_1"] = (
        (out["hidden_leak_any_p90"] <= float(thresholds["hidden_leak_any_p90_max"]))
        & (out["false_notif_p90"] <= float(thresholds["false_notif_p90_max"]))
    ).astype(int)

    out["quality_pass_v5_1"] = (
        (out["strong_detect_iou20_p10"] >= float(thresholds["strong_iou20_p10_min"]))
        & (out["strong_detect_any_mean"] >= float(thresholds["strong_any_mean_min"]))
        & (out["border_detect_any_mean"] >= float(thresholds["borderline_any_mean_min"]))
        & (out["border_detect_any_p10"] >= float(thresholds["borderline_any_p10_min"]))
    ).astype(int)

    out["robust_pass_v5_1"] = (
        (out["quality_pass_v5_1"] == 1)
        & (out["safety_pass_v5_1"] == 1)
        & (out["nonregression_pass_v5_1"] == 1)
    ).astype(int)

    reasons: List[str] = []
    for _, r in out.iterrows():
        fail: List[str] = []
        if r["strong_detect_iou20_p10"] < float(thresholds["strong_iou20_p10_min"]):
            fail.append(f"strong_iou20_p10 {r['strong_detect_iou20_p10']:.3f} < {thresholds['strong_iou20_p10_min']:.3f}")
        if r["strong_detect_any_mean"] < float(thresholds["strong_any_mean_min"]):
            fail.append(f"strong_any_mean {r['strong_detect_any_mean']:.3f} < {thresholds['strong_any_mean_min']:.3f}")
        if r["border_detect_any_mean"] < float(thresholds["borderline_any_mean_min"]):
            fail.append(f"borderline_any_mean {r['border_detect_any_mean']:.3f} < {thresholds['borderline_any_mean_min']:.3f}")
        if r["border_detect_any_p10"] < float(thresholds["borderline_any_p10_min"]):
            fail.append(f"borderline_any_p10 {r['border_detect_any_p10']:.3f} < {thresholds['borderline_any_p10_min']:.3f}")
        if r["hidden_leak_any_p90"] > float(thresholds["hidden_leak_any_p90_max"]):
            fail.append(f"hidden_any_p90 {r['hidden_leak_any_p90']:.3f} > {thresholds['hidden_leak_any_p90_max']:.3f}")
        if r["false_notif_p90"] > float(thresholds["false_notif_p90_max"]):
            fail.append(f"false_notif_p90 {r['false_notif_p90']:.3f} > {thresholds['false_notif_p90_max']:.3f}")
        if r["delta_false_notif_p90_vs_ref"] > float(tol["delta_false_notif_p90_max"]):
            fail.append(
                f"delta_false_notif_vs_ref {r['delta_false_notif_p90_vs_ref']:.3f} > {tol['delta_false_notif_p90_max']:.3f}"
            )
        if r["delta_hidden_leak_p90_vs_ref"] > float(tol["delta_hidden_leak_p90_max"]):
            fail.append(
                f"delta_hidden_leak_vs_ref {r['delta_hidden_leak_p90_vs_ref']:.3f} > {tol['delta_hidden_leak_p90_max']:.3f}"
            )
        if np.isfinite(r.get("ratio_false_notif_vs_ref", np.nan)) and np.isfinite(ref_false) and ref_false > 0.03:
            if r["ratio_false_notif_vs_ref"] > float(tol["ratio_false_notif_max"]):
                fail.append(f"ratio_false_notif_vs_ref {r['ratio_false_notif_vs_ref']:.3f} > {tol['ratio_false_notif_max']:.3f}")
        reasons.append("OK" if len(fail) == 0 else " | ".join(fail))
    out["go_reason_v5_1"] = reasons

    out["selection_score_v5_1"] = (
        0.35 * out["strong_detect_iou20_mean"].fillna(0)
        + 0.20 * out["strong_detect_any_mean"].fillna(0)
        + 0.35 * out["border_detect_any_mean"].fillna(0)
        + 0.10 * out["delta_border_any_mean_vs_ref"].fillna(0)
        - 0.30 * out["false_notif_mean"].fillna(0)
        - 0.25 * out["hidden_leak_any_mean"].fillna(0)
        - 0.10 * out["false_notif_std"].fillna(0)
    )

    out = out.sort_values(["robust_pass_v5_1", "selection_score_v5_1"], ascending=[False, False]).reset_index(drop=True)

    ref_payload = {
        "ref_source": str(ref_info.get("ref_source", "unknown")),
        "ref_rejected": str(ref_info.get("ref_rejected", "")),
        "ref_values": {
            k: (
                float(ref_row[k])
                if isinstance(ref_row.get(k, None), (int, float, np.floating, np.integer))
                else ref_row.get(k, None)
            )
            for k in [
                *grid_cols,
                "strong_detect_iou20_p10",
                "strong_detect_any_mean",
                "border_detect_any_mean",
                "hidden_leak_any_p90",
                "false_notif_p90",
            ]
            if k in ref_row.index
        },
        "nonreg_tolerances": {k: float(v) for k, v in tol.items()},
    }
    return out, ref_payload


__all__ = [
    "_match_cfg_row",
    "_is_reference_balanced",
    "_choose_reference_row",
    "_build_summary_v5_1",
]
