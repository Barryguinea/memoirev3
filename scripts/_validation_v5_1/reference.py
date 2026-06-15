"""Sélection de la configuration de référence V5.1/V5.2."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def _match_cfg_row(df: pd.DataFrame, cfg: Dict[str, object], cols: List[str]) -> pd.Series:
    """Retourne un masque booléen des lignes de ``df`` correspondant à ``cfg`` sur les colonnes spécifiées."""
    if len(df) == 0:
        return pd.Series(False, index=df.index)
    m = pd.Series(True, index=df.index)
    for c in cols:
        if c not in df.columns or c not in cfg:
            continue
        val = cfg[c]
        if pd.api.types.is_numeric_dtype(df[c]):
            m = m & np.isclose(pd.to_numeric(df[c], errors="coerce"), float(val), rtol=1e-9, atol=1e-9)
        else:
            m = m & (df[c].astype(str) == str(val))
    return m


def _is_reference_balanced(row: pd.Series, thresholds: Dict[str, float]) -> bool:
    """Vérifie si une ligne de résumé satisfait les conditions qualité/sécurité d'une référence équilibrée."""
    if row is None or len(row) == 0:
        return False
    min_ref_strong_any = max(0.85, float(thresholds["strong_any_mean_min"]) - 0.10)
    min_ref_border = max(0.15, float(thresholds["borderline_any_mean_min"]) - 0.05)
    return bool(
        (float(row.get("strong_detect_iou20_p10", np.nan)) >= float(thresholds["strong_iou20_p10_min"]))
        and (float(row.get("strong_detect_any_mean", np.nan)) >= min_ref_strong_any)
        and (float(row.get("border_detect_any_mean", np.nan)) >= min_ref_border)
        and (float(row.get("hidden_leak_any_p90", np.nan)) <= float(thresholds["hidden_leak_any_p90_max"]))
        and (float(row.get("false_notif_p90", np.nan)) <= float(thresholds["false_notif_p90_max"]))
    )


def _choose_reference_row(
    summary_cfg: pd.DataFrame,
    *,
    grid_cols: List[str],
    thresholds: Dict[str, float],
    reference_cfg: Optional[Dict[str, object]] = None,
) -> Tuple[pd.Series, Dict[str, object]]:
    """Sélectionne la meilleure référence via un repli priorisé (utilisateur, équilibrée, qualité+sûreté, conservatrice)."""
    user_rejected_reason = ""

    # 1) Essayer la référence fournie par l'utilisateur si elle existe et est équilibrée.
    if reference_cfg:
        m = _match_cfg_row(summary_cfg, reference_cfg, grid_cols)
        if m.any():
            r = summary_cfg[m].iloc[0]
            if _is_reference_balanced(r, thresholds):
                return r, {"ref_source": "user_cfg", "ref_rejected": ""}
            user_rejected_reason = "user reference does not satisfy balanced reference conditions"
        else:
            user_rejected_reason = "user reference config not found in evaluated grid"

    # 2) Référence automatique équilibrée (préférée).
    balanced = summary_cfg[summary_cfg.apply(lambda r: _is_reference_balanced(r, thresholds), axis=1)].copy()
    if len(balanced):
        r = balanced.sort_values(
            ["false_notif_p90", "hidden_leak_any_p90", "border_detect_any_mean", "strong_detect_any_mean"],
            ascending=[True, True, False, False],
        ).iloc[0]
        return r, {"ref_source": "auto_balanced", "ref_rejected": user_rejected_reason}

    # 3) Repli automatique "qualité + sécurité".
    min_ref_strong_any = max(0.80, float(thresholds["strong_any_mean_min"]) - 0.15)
    qs = summary_cfg[
        (summary_cfg["strong_detect_iou20_p10"] >= float(thresholds["strong_iou20_p10_min"]))
        & (summary_cfg["strong_detect_any_mean"] >= min_ref_strong_any)
        & (summary_cfg["hidden_leak_any_p90"] <= float(thresholds["hidden_leak_any_p90_max"]))
        & (summary_cfg["false_notif_p90"] <= float(thresholds["false_notif_p90_max"]))
    ].copy()
    if len(qs):
        r = qs.sort_values(
            ["false_notif_p90", "hidden_leak_any_p90", "border_detect_any_mean", "strong_detect_any_mean"],
            ascending=[True, True, False, False],
        ).iloc[0]
        return r, {"ref_source": "auto_quality_safe", "ref_rejected": user_rejected_reason}

    # 4) Repli conservateur.
    cand = summary_cfg.copy()
    sort_cols = [c for c in ["mix_rate_thr", "persist_hours", "contamination"] if c in cand.columns]
    if len(sort_cols):
        asc = [False if c in {"mix_rate_thr", "persist_hours"} else True for c in sort_cols]
        r = cand.sort_values(sort_cols, ascending=asc).iloc[0]
    else:
        r = cand.iloc[0]
    return r, {"ref_source": "auto_conservative", "ref_rejected": user_rejected_reason}
