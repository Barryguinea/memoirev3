"""Helpers de base du wrapper robuste V5.1/V5.2 (compat, grid, JSON, utilitaires)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from scripts import validation_notebook_utils as base_utils

def _call_base_run_robust_validation_compat(**kwargs):
    """
    Appel rétrocompatible pour les notebooks pouvant conserver en mémoire
    un module de base plus ancien (sans les nouveaux kwargs).
    """
    try:
        sig = inspect.signature(base_utils.run_robust_validation)
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not accepts_var_kw:
            allowed = set(sig.parameters.keys())
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        # Repli : appeler avec les kwargs fournis.
        pass
    return base_utils.run_robust_validation(**kwargs)


def _archive_base_summary_file(run_dir: Path) -> Path | None:
    """
    Évite les résumés robustes ambigus dans les sorties du wrapper.
    - Le moteur de base récent écrit `robust_summary_base_engine.csv`
    - Les anciens runs peuvent encore contenir `robust_summary.csv`
    - La sortie officielle v5.1 est `robust_summary_v5_1.csv`
    On ne conserve que `robust_summary_base_engine.csv` comme artefact base.
    """
    archived = run_dir / "robust_summary_base_engine.csv"
    legacy = run_dir / "robust_summary.csv"

    if archived.exists():
        if legacy.exists():
            legacy.unlink()
        return archived

    if not legacy.exists():
        return None

    legacy.replace(archived)
    return archived


def _json_safe(value: object) -> object:
    """Convertit récursivement les types numpy en primitifs Python sérialisables en JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _to_stats(x: Iterable[object]) -> Dict[str, float]:
    """Calcule mean, std, p10 et p90 à partir d'un itérable numérique en ignorant les valeurs non finies."""
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


def _scenario(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Filtre un DataFrame pour les lignes correspondant au nom de scénario donné."""
    if "scenario" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["scenario"].astype(str) == str(name)].copy()


def _clean_grid(grid: Dict[str, List[object]]) -> Dict[str, List[object]]:
    """Nettoie et trie les valeurs de grille en appliquant des seuils minimum par paramètre."""
    g = {k: list(v) for k, v in grid.items()}
    if "contamination" in g:
        g["contamination"] = sorted(set(float(x) for x in g["contamination"] if float(x) >= 0.05)) or [0.05]
    if "persist_hours" in g:
        g["persist_hours"] = sorted(set(int(x) for x in g["persist_hours"] if int(x) >= 6)) or [6]
    if "alert_min" in g:
        g["alert_min"] = sorted(set(int(x) for x in g["alert_min"] if int(x) >= 2)) or [2]
    return g

