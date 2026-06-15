"""Helpers internes de cohérence multi-familles pour la logique d'alertes."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def _feature_family(col: str) -> str:
    """Mappe un nom de colonne vers sa famille capteur."""
    c = str(col).lower()
    if "motion index" in c or "steps" in c:
        return "activity"
    if "lying time" in c or "standing time" in c:
        return "rest"
    if "transitions" in c or "lying bout" in c:
        return "transitions"
    return "other"


def _pick_rrz_candidates(df: pd.DataFrame, pref_rrz_cols: List[str]) -> List[str]:
    """Sélectionne les colonnes de rolling robust z-score pour la cohérence."""
    rz_cols = [c for c in df.columns if str(c).endswith("_rrz")]
    cand = [name for name in pref_rrz_cols if name in df.columns]
    if not cand and rz_cols:
        cand = rz_cols[:8]
    else:
        for c in rz_cols:
            if c not in cand:
                cand.append(c)
    return cand


def _family_presence(
    df: pd.DataFrame,
    cols: List[str],
    z_low: float,
    z_high: float,
    *,
    window_k: int,
    rate_thr: float,
) -> pd.DataFrame:
    """Calcule les hits et persistances par famille sur une fenêtre glissante."""
    fams = ["activity", "rest", "transitions", "other"]
    hit_any = {f: np.zeros(len(df), dtype=int) for f in fams}

    for c in cols:
        if c not in df.columns:
            continue
        fam = _feature_family(c)
        v = pd.to_numeric(df[c], errors="coerce").fillna(0.0).values
        hit = ((v <= z_low) | (v >= z_high)).astype(int)
        hit_any[fam] = np.maximum(hit_any[fam], hit)

    hit_df = pd.DataFrame({f"hit_{f}": hit_any[f] for f in fams}, index=df.index)

    persist = {}
    for f in fams:
        r = hit_df[f"hit_{f}"].rolling(window_k, min_periods=1).mean()
        persist[f] = (r >= float(rate_thr)).astype(int)

    out = pd.DataFrame(
        {
            "fam_activity": persist["activity"],
            "fam_rest": persist["rest"],
            "fam_transitions": persist["transitions"],
            "fam_other": persist["other"],
            "hit_activity": hit_df["hit_activity"],
            "hit_rest": hit_df["hit_rest"],
            "hit_transitions": hit_df["hit_transitions"],
            "hit_other": hit_df["hit_other"],
        },
        index=df.index,
    )
    return out


def _mi_spike_series(df: pd.DataFrame, mi_name: str, mi_z_high_thr: float) -> Tuple[np.ndarray, str]:
    """Retourne une série binaire de pics MI (dépassements du seuil z)."""
    for c in [
        f"{mi_name}_sum_log_rrz",
        f"{mi_name}_sum_rrz",
        f"{mi_name}_mean_log_rrz",
        f"{mi_name}_mean_rrz",
    ]:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0.0).values
            return (v >= float(mi_z_high_thr)).astype(int), c
    return np.zeros(len(df), dtype=int), ""

