"""Helpers de résumé/scoring pour le wrapper robuste V4."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def _series_stats(x: Iterable[object]) -> Dict[str, float]:
    """Calcule mean, std, p10 et p90 descriptifs à partir d'un itérable de valeurs numériques."""
    arr = pd.to_numeric(pd.Series(list(x)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": np.nan, "std": np.nan, "p10": np.nan, "p90": np.nan}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def _filter_cfg_grid(grid: Dict[str, List[object]]) -> Dict[str, List[object]]:
    """Nettoie la grille de configuration en imposant des minima pour contamination, persist_hours et alert_min."""
    g = {k: list(v) for k, v in grid.items()}
    if "contamination" in g:
        g["contamination"] = sorted(set(float(x) for x in g["contamination"] if float(x) >= 0.05)) or [0.05]
    if "persist_hours" in g:
        g["persist_hours"] = sorted(set(int(x) for x in g["persist_hours"] if int(x) >= 6)) or [6]
    if "alert_min" in g:
        g["alert_min"] = sorted(set(int(x) for x in g["alert_min"] if int(x) >= 2)) or [2]
    return g


def _scenario_slice(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Retourne le sous-ensemble des lignes correspondant au scénario donné."""
    if "scenario" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["scenario"].astype(str) == str(scenario)].copy()


def _build_v4_summary(
    runs_df: pd.DataFrame,
    *,
    grid: Dict[str, List[object]],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    """Agrège les métriques par configuration en un résumé V4 avec verdicts pass/fail et scores de sélection."""
    if len(runs_df) == 0:
        return pd.DataFrame()

    group_cols = [k for k in grid.keys() if k in runs_df.columns]
    if not group_cols:
        candidates = ["contamination", "persist_hours", "alert_min", "mix_mode", "mix_rate_thr"]
        group_cols = [k for k in candidates if k in runs_df.columns]

    rows: List[Dict[str, object]] = []
    for keys, g in runs_df.groupby(group_cols, dropna=False):
        row: Dict[str, object] = {}
        if isinstance(keys, tuple):
            for i, k in enumerate(group_cols):
                row[k] = keys[i]
        else:
            row[group_cols[0]] = keys

        strong = _scenario_slice(g, "detectable_strong")
        borderline = _scenario_slice(g, "detectable_borderline")
        hidden = _scenario_slice(g, "non_detectable_short")

        strong_iou20 = _series_stats(strong.get("detectable_recall_iou20", pd.Series(dtype=float)))
        strong_any = _series_stats(strong.get("detectable_recall_any", pd.Series(dtype=float)))
        border_any = _series_stats(borderline.get("detectable_recall_any", pd.Series(dtype=float)))
        hidden_any = _series_stats(hidden.get("hidden_leak_rate_any", pd.Series(dtype=float)))
        false_notif = _series_stats(g.get("false_notif_per_cow_day", pd.Series(dtype=float)))

        row["strong_detect_iou20_mean"] = strong_iou20["mean"]
        row["strong_detect_iou20_p10"] = strong_iou20["p10"]
        row["strong_detect_any_mean"] = strong_any["mean"]
        row["strong_detect_any_p10"] = strong_any["p10"]

        row["border_detect_any_mean"] = border_any["mean"]
        row["border_detect_any_p10"] = border_any["p10"]

        row["hidden_leak_any_mean"] = hidden_any["mean"]
        row["hidden_leak_any_p90"] = hidden_any["p90"]

        row["false_notif_mean"] = false_notif["mean"]
        row["false_notif_std"] = false_notif["std"]
        row["false_notif_p90"] = false_notif["p90"]

        row["n_runs"] = int(len(g))

        fail = []
        if not (row["strong_detect_iou20_p10"] >= thresholds["strong_iou20_p10_min"]):
            if np.isfinite(row["strong_detect_iou20_p10"]):
                fail.append(
                    f"strong_iou20_p10 {row['strong_detect_iou20_p10']:.3f} < {thresholds['strong_iou20_p10_min']:.3f}"
                )
            else:
                fail.append("strong_iou20_p10 is NaN")

        if not (row["strong_detect_any_mean"] >= thresholds["strong_any_mean_min"]):
            if np.isfinite(row["strong_detect_any_mean"]):
                fail.append(
                    f"strong_any_mean {row['strong_detect_any_mean']:.3f} < {thresholds['strong_any_mean_min']:.3f}"
                )
            else:
                fail.append("strong_any_mean is NaN")

        if not (row["border_detect_any_mean"] >= thresholds["borderline_any_mean_min"]):
            if np.isfinite(row["border_detect_any_mean"]):
                fail.append(
                    f"borderline_any_mean {row['border_detect_any_mean']:.3f} < {thresholds['borderline_any_mean_min']:.3f}"
                )
            else:
                fail.append("borderline_any_mean is NaN")

        if not (row["border_detect_any_p10"] >= thresholds["borderline_any_p10_min"]):
            if np.isfinite(row["border_detect_any_p10"]):
                fail.append(
                    f"borderline_any_p10 {row['border_detect_any_p10']:.3f} < {thresholds['borderline_any_p10_min']:.3f}"
                )
            else:
                fail.append("borderline_any_p10 is NaN")

        if not (row["hidden_leak_any_p90"] <= thresholds["hidden_leak_any_p90_max"]):
            if np.isfinite(row["hidden_leak_any_p90"]):
                fail.append(
                    f"hidden_any_p90 {row['hidden_leak_any_p90']:.3f} > {thresholds['hidden_leak_any_p90_max']:.3f}"
                )
            else:
                fail.append("hidden_any_p90 is NaN")

        if not (row["false_notif_p90"] <= thresholds["false_notif_p90_max"]):
            if np.isfinite(row["false_notif_p90"]):
                fail.append(f"false_notif_p90 {row['false_notif_p90']:.3f} > {thresholds['false_notif_p90_max']:.3f}")
            else:
                fail.append("false_notif_p90 is NaN")

        row["robust_pass_v4"] = int(len(fail) == 0)
        row["go_reason_v4"] = "OK" if len(fail) == 0 else " | ".join(fail)
        row["selection_score_v4"] = (
            0.50 * float(np.nan_to_num(row["strong_detect_iou20_mean"], nan=0.0))
            + 0.20 * float(np.nan_to_num(row["strong_detect_any_mean"], nan=0.0))
            + 0.20 * float(np.nan_to_num(row["border_detect_any_mean"], nan=0.0))
            - 0.25 * float(np.nan_to_num(row["hidden_leak_any_mean"], nan=0.0))
            - 0.30 * float(np.nan_to_num(row["false_notif_mean"], nan=0.0))
            - 0.10 * float(np.nan_to_num(row["false_notif_std"], nan=0.0))
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["robust_pass_v4", "selection_score_v4"], ascending=[False, False]).reset_index(drop=True)
    return out


def _events_scenario_report(events_df: pd.DataFrame) -> pd.DataFrame:
    """Construit un rapport par scénario avec comptages d'événements et métriques moyennes de détection."""
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


__all__ = [
    "_series_stats",
    "_filter_cfg_grid",
    "_scenario_slice",
    "_build_v4_summary",
    "_events_scenario_report",
]
