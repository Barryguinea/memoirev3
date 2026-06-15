"""Helpers communs pour la validation (parametres, segments, IoU, statistiques simples)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from core.io import COW, LYING, MI, STANDING, STEPS, TIME, TRANSITIONS, TR_DOWN, TR_UP


def to_numeric_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Convertit en float les colonnes numeriques listees (in-place, NaN->0)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)


def has_raw_sensor_cols(df: pd.DataFrame) -> bool:
    """Retourne ``True`` si *df* contient au moins une colonne capteur brute canonique."""
    return any(c in df.columns for c in [STEPS, MI, LYING, STANDING, TRANSITIONS, TR_UP, TR_DOWN])


def looks_processed_predictions(df: pd.DataFrame) -> bool:
    """Retourne ``True`` si *df* ressemble a une sortie pipeline (Cow, T, if_anomaly_point)."""
    needed = {COW, TIME, "if_anomaly_point"}
    return needed.issubset(set(df.columns))


def resolve_baseline_ratio(v):
    """Convertit une valeur ``baseline_ratio`` en float ou ``None`` (gere la chaine ``'None'``)."""
    if str(v).strip().lower() == "none":
        return None
    return float(v)


def normalize_params(params: Dict[str, object]) -> Dict[str, object]:
    """Normalise le dictionnaire de parametres du pipeline et applique les minima."""
    p = dict(params)
    p["baseline_ratio"] = resolve_baseline_ratio(p.get("baseline_ratio", 0.6))
    p["interval"] = str(p.get("interval", "15T"))
    p["window_baseline"] = int(p.get("window_baseline", 24))
    # Validation minima enforced globally (manual + robust).
    p["contamination"] = max(0.05, float(p.get("contamination", 0.06)))
    p["random_state"] = int(p.get("random_state", 42))
    p["persist_hours"] = max(6, int(p.get("persist_hours", 6)))
    p["alert_min"] = max(2, int(p.get("alert_min", 2)))
    p["mix_mode"] = str(p.get("mix_mode", "MIX"))
    p["mix_rate_thr"] = float(p.get("mix_rate_thr", 0.24))
    p["z_low_thr"] = float(p.get("z_low_thr", -2.0))
    p["z_high_thr"] = float(p.get("z_high_thr", 2.0))
    p["cooldown_hours"] = int(p.get("cooldown_hours", 12))
    p["mi_z_high_thr"] = float(p.get("mi_z_high_thr", 2.2))
    p["coverage_min_pct"] = float(p.get("coverage_min_pct", 25.0))
    max_cows = p.get("max_cows", None)
    if max_cows is not None:
        max_cows = int(max_cows)
        p["max_cows"] = None if max_cows <= 0 else max_cows
    return p


def segments_from_binary(df: pd.DataFrame, col: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Extrait des segments temporels contigus (start, end) ou *col* == 1."""
    x = df.sort_values(TIME).copy()
    if col not in x.columns:
        return []
    v = pd.to_numeric(x[col], errors="coerce").fillna(0).astype(int)
    starts = x.loc[(v == 1) & (v.shift(1).fillna(0) == 0), TIME].tolist()
    ends = x.loc[(v == 1) & (v.shift(-1).fillna(0) == 0), TIME].tolist()
    return list(zip(starts, ends))


def overlap_seconds(a: Tuple[pd.Timestamp, pd.Timestamp], b: Tuple[pd.Timestamp, pd.Timestamp], bin_seconds: int) -> float:
    """Retourne le recouvrement temporel en secondes entre deux intervalles (start, end)."""
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    if e < s:
        return 0.0
    return float((e - s).total_seconds() + bin_seconds)


def iou(a: Tuple[pd.Timestamp, pd.Timestamp], b: Tuple[pd.Timestamp, pd.Timestamp], bin_seconds: int) -> float:
    """Intersection over Union (IoU) entre deux intervalles temporels."""
    inter = overlap_seconds(a, b, bin_seconds)
    if inter <= 0:
        return 0.0
    la = float((a[1] - a[0]).total_seconds() + bin_seconds)
    lb = float((b[1] - b[0]).total_seconds() + bin_seconds)
    return float(inter / max(1e-9, la + lb - inter))


def count_cow_days(df: pd.DataFrame) -> int:
    """Retourne le nombre de couples distincts (vache, jour) dans *df* (min 1)."""
    if len(df) == 0:
        return 1
    x = df.copy()
    x["Day"] = pd.to_datetime(x[TIME], errors="coerce").dt.floor("D")
    return int(max(1, x[[COW, "Day"]].drop_duplicates().shape[0]))


def aggregate_distribution(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Calcule mean/std/p10/p90 pour une liste de colonnes de metriques."""
    out = {}
    for m in metric_cols:
        x = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            out[f"{m}_mean"] = np.nan
            out[f"{m}_std"] = np.nan
            out[f"{m}_p10"] = np.nan
            out[f"{m}_p90"] = np.nan
            continue
        out[f"{m}_mean"] = float(np.mean(x))
        out[f"{m}_std"] = float(np.std(x, ddof=0))
        out[f"{m}_p10"] = float(np.quantile(x, 0.10))
        out[f"{m}_p90"] = float(np.quantile(x, 0.90))
    return pd.DataFrame([out])
