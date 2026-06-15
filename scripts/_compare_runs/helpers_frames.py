"""Helpers DataFrame/agrégation pour la comparaison inter-versions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts._compare_runs.helpers_fs import _resolve_existing_file


def _collapse_configs_by_keys(
    df: pd.DataFrame,
    *,
    key_cols: List[str],
    pass_col: str,
    score_col: str,
    reason_col: str,
    extra_flag_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Regroupe les configurations dupliquées par clés en agrégeant pass/score/reason."""
    if len(df) == 0:
        return df.copy()

    extra_flag_cols = extra_flag_cols or []
    if not key_cols:
        out = df.copy()
        out["n_variants"] = 1
        return out

    rows: List[Dict[str, object]] = []
    for keys, g in df.groupby(key_cols, dropna=False):
        row: Dict[str, object] = {}
        if isinstance(keys, tuple):
            for i, k in enumerate(key_cols):
                row[k] = keys[i]
        else:
            row[key_cols[0]] = keys

        row["n_variants"] = int(len(g))

        if pass_col in g.columns:
            p = pd.to_numeric(g[pass_col], errors="coerce").fillna(0).astype(int)
            row[pass_col] = int((p == 1).any())
        if score_col in g.columns:
            s = pd.to_numeric(g[score_col], errors="coerce")
            row[score_col] = float(s.max()) if s.notna().any() else np.nan

        for c in extra_flag_cols:
            if c in g.columns:
                cc = pd.to_numeric(g[c], errors="coerce").fillna(0).astype(int)
                row[c] = int((cc == 1).any())

        if reason_col in g.columns:
            if row.get(pass_col, 0) == 1:
                row[reason_col] = "OK"
            else:
                r = g[reason_col].dropna().astype(str)
                uniq = []
                for x in r.tolist():
                    if x not in uniq:
                        uniq.append(x)
                row[reason_col] = " | ".join(uniq[:2]) if uniq else ""

        rows.append(row)

    return pd.DataFrame(rows)


def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Première colonne existante parmi *candidates*.

    Note : logique similaire à core.io._first_existing(), dupliquée ici
    volontairement pour que le script de comparaison reste autonome
    (pas de dépendance vers core/ qui pourrait évoluer entre versions).
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _prepare_version_frame(
    df: pd.DataFrame,
    *,
    key_cols: List[str],
    pass_col: str,
    score_col: Optional[str],
    reason_col: Optional[str],
    extra_flag_cols: Optional[List[str]],
    version_tag: str,
) -> pd.DataFrame:
    """Agrége et renomme le DataFrame de résumé d'une version pour la fusion inter-versions."""
    keep = list(key_cols)
    if pass_col:
        keep.append(pass_col)
    if score_col:
        keep.append(score_col)
    if reason_col:
        keep.append(reason_col)
    for c in extra_flag_cols or []:
        if c:
            keep.append(c)
    keep = [c for c in keep if c in df.columns]
    if pass_col not in keep:
        raise ValueError(f"Colonne pass introuvable pour {version_tag}: {pass_col}")

    cmp_df = _collapse_configs_by_keys(
        df[keep],
        key_cols=key_cols,
        pass_col=pass_col,
        score_col=(score_col or "__selection_missing__"),
        reason_col=(reason_col or "__go_reason_missing__"),
        extra_flag_cols=[c for c in (extra_flag_cols or []) if c in keep],
    )
    rename_map: Dict[str, str] = {
        "n_variants": f"n_variants_{version_tag}",
        pass_col: f"robust_pass_{version_tag}",
    }
    if score_col and score_col in cmp_df.columns:
        rename_map[score_col] = f"selection_score_{version_tag}"
    if reason_col and reason_col in cmp_df.columns:
        rename_map[reason_col] = f"go_reason_{version_tag}"
    for flag in extra_flag_cols or []:
        if flag in cmp_df.columns:
            base_flag = flag
            for prefix in ["quality_pass_", "safety_pass_", "nonregression_pass_"]:
                if base_flag.startswith(prefix):
                    base_flag = prefix[:-1]
                    break
            rename_map[flag] = f"{base_flag}_{version_tag}"
    return cmp_df.rename(columns=rename_map)


def _load_scenario_report(run_dir: Path, candidates: List[str], version_tag: str) -> pd.DataFrame:
    """Charge un CSV de rapport par scénario et renomme les métriques avec le tag de version."""
    p = _resolve_existing_file(run_dir, candidates)
    if p is None:
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "scenario" not in df.columns:
        return pd.DataFrame()
    keep = [
        "scenario",
        "n_events",
        "n_expected_detected",
        "detected_any_overlap_mean",
        "detected_iou20_mean",
        "best_iou_mean",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    rename = {c: f"{c}_{version_tag}" for c in out.columns if c != "scenario"}
    return out.rename(columns=rename)

