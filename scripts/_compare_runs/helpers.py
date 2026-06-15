"""Helpers et constantes pour la comparaison inter-versions de runs robustes.

Ce module reste le point d'import de compatibilité pour les autres modules
du package. Les sous-blocs volumineux (filesystem/discovery et agrégations
DataFrame) sont déplacés dans des sous-modules internes dédiés.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts._compare_runs.helpers_frames import (
    _collapse_configs_by_keys,
    _first_existing_column,
    _load_scenario_report,
    _prepare_version_frame,
)
from scripts._compare_runs.helpers_fs import (
    _discover_latest_run,
    _is_complete,
    _is_complete_with_candidates,
    _read_optional_json,
    _resolve_existing_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_ROOT = ROOT / "data" / "validation_robuste"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "validation_robuste"


REQUIRED_V4 = ["robust_summary_v4.csv", "robust_runs.csv", "robust_events_evaluation.csv"]
REQUIRED_V5_BASE = ["robust_runs.csv", "robust_events_evaluation.csv"]
# Candidats ordonnés par préférence : on cherche d'abord le fichier natif
# de la version, puis on accepte le fichier de l'autre version en repli
# (le moteur v5.1 et v5.2 produisent des fichiers au format identique).
SUMMARY_CANDIDATES_V5_1 = ["robust_summary_v5_1.csv", "robust_summary_v5_2.csv"]
SUMMARY_CANDIDATES_V5_2 = ["robust_summary_v5_2.csv", "robust_summary_v5_1.csv"]
SCENARIO_CANDIDATES_V5_1 = ["robust_events_by_scenario_v5_1.csv", "robust_events_by_scenario_v5_2.csv"]
SCENARIO_CANDIDATES_V5_2 = ["robust_events_by_scenario_v5_2.csv", "robust_events_by_scenario_v5_1.csv"]
PREFERRED_CFG_KEYS = ["contamination", "persist_hours", "alert_min", "mix_mode", "mix_rate_thr", "mi_z_high_thr"]


def _to_float_safe(value: object) -> object:
    """Convertit récursivement les scalaires numpy en types Python natifs pour la sérialisation JSON."""
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _to_float_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_float_safe(v) for v in value]
    return value


def _standardize_key_columns(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """Normalise les colonnes clés en numérique (arrondi) ou chaîne pour une fusion cohérente."""
    out = df.copy()
    for c in key_cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        if s.notna().mean() > 0.95:
            out[c] = s.round(6)
        else:
            out[c] = out[c].astype(str)
    return out


def _meta_dict(meta: Dict[str, object], key: str) -> Dict[str, object]:
    """Extrait en sécurité un sous-dictionnaire de métadonnées, sinon retourne un dict vide."""
    v = meta.get(key, {})
    return v if isinstance(v, dict) else {}


def _merge_meta(base_meta: Dict[str, object], wrapper_meta: Dict[str, object]) -> Dict[str, object]:
    """Fusionne en profondeur les métadonnées du wrapper dans celles de base."""
    out = dict(base_meta)
    for k, v in wrapper_meta.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            merged = dict(out[k])
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    return out


def _infer_fixed_cfg_value(meta: Dict[str, object], col: str) -> object:
    """Infère une valeur de configuration fixe depuis ``base_params`` ou une grille à un seul élément."""
    base_params = _meta_dict(meta, "base_params")
    if col in base_params:
        return base_params[col]
    effective_grid = _meta_dict(meta, "effective_grid")
    if col in effective_grid and isinstance(effective_grid[col], list) and len(effective_grid[col]) == 1:
        return effective_grid[col][0]
    return np.nan


def _build_full_key_cols(v4: pd.DataFrame, v51: pd.DataFrame, meta_v4: Dict[str, object], meta_v51: Dict[str, object]) -> List[str]:
    """Construit la liste ordonnée des colonnes clés de configuration pour une fusion v4/v5.1."""
    keys: List[str] = []
    grid_v4 = _meta_dict(meta_v4, "effective_grid")
    grid_v51 = _meta_dict(meta_v51, "effective_grid")
    base_v4 = _meta_dict(meta_v4, "base_params")
    base_v51 = _meta_dict(meta_v51, "base_params")

    for c in PREFERRED_CFG_KEYS:
        if (
            (c in v4.columns)
            or (c in v51.columns)
            or (c in grid_v4)
            or (c in grid_v51)
            or (c in base_v4)
            or (c in base_v51)
        ):
            keys.append(c)

    extra = sorted((set(grid_v4.keys()) | set(grid_v51.keys())) - set(keys))
    keys.extend(extra)
    return keys


def _build_full_key_cols_many(dfs: List[pd.DataFrame], metas: List[Dict[str, object]]) -> List[str]:
    """Construit la liste ordonnée des colonnes clés de configuration sur plusieurs versions."""
    keys: List[str] = []
    grids = [_meta_dict(m, "effective_grid") for m in metas]
    bases = [_meta_dict(m, "base_params") for m in metas]

    for c in PREFERRED_CFG_KEYS:
        if any(c in d.columns for d in dfs) or any(c in g for g in grids) or any(c in b for b in bases):
            keys.append(c)

    extra = sorted(set().union(*(set(g.keys()) for g in grids)) - set(keys))
    keys.extend(extra)
    return keys


def _ensure_cfg_columns(df: pd.DataFrame, key_cols: List[str], meta: Dict[str, object]) -> pd.DataFrame:
    """Ajoute les colonnes de configuration manquantes au DataFrame via les métadonnées inférées."""
    out = df.copy()
    for c in key_cols:
        if c not in out.columns:
            out[c] = _infer_fixed_cfg_value(meta, c)
        else:
            inferred = _infer_fixed_cfg_value(meta, c)
            if np.isfinite(pd.to_numeric(pd.Series([inferred]), errors="coerce")).all():
                s = pd.to_numeric(out[c], errors="coerce")
                if s.isna().all() and pd.notna(inferred):
                    out[c] = inferred
    return out
@dataclass
class ComparisonResult:
    """Conteneur pour le dossier de sortie et le dict résumé d'un run de comparaison."""
    output_dir: Path
    summary: Dict[str, object]
