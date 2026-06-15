"""Construction de features pour des données capteurs agrégées par intervalle.
Construit des z-scores robustes (MAD), des rolling robust z-scores,
des différences premières, des moyennes mobiles sur 7 bins et des
features temporelles cycliques à partir des colonnes resamplées.
"""

import numpy as np
import pandas as pd
from typing import List

# Constantes historiques conservées (stables)
MAD_FACTOR = 1.4826
EPS = 1e-9
MIN_PERIODS_ROLLING = lambda w: max(5, w // 4)


def interval_to_minutes(interval: str) -> int:
    """Convertit un intervalle (ex. ``15T``, ``1H``) en minutes."""
    s = str(interval).strip().upper()
    if s.endswith("T") or s.endswith("MIN"):
        return int(s.rstrip("TMIN"))
    if s.endswith("H"):
        return int(s[:-1]) * 60
    raise ValueError(f"Intervalle non supporté: {interval}")


def _pandas_freq(interval: str) -> str:
    """Convertit un intervalle UI en fréquence pandas moderne (ex: '15T' -> '15min')."""
    s = str(interval).strip().upper()
    if s.endswith("T"):
        return s[:-1] + "min"
    return interval


def _safe_iqr(x: np.ndarray) -> float:
    """Calcule l'IQR, ou ``NaN`` s'il y a moins de 5 valeurs finies."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    q75, q25 = np.nanpercentile(x, [75, 25])
    return float(q75 - q25)


def robust_z(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Z-score robuste global (MAD) avec replis IQR puis écart-type."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    denom = MAD_FACTOR * mad

    if (not np.isfinite(denom)) or (denom < eps):
        iqr = _safe_iqr(x)
        if np.isfinite(iqr) and iqr > eps:
            denom = iqr / 1.349
        else:
            s = np.nanstd(x)
            denom = s if (np.isfinite(s) and s > eps) else 1.0

    z = (x - med) / denom
    z[~np.isfinite(z)] = 0.0
    return z


def rolling_robust_z(series: pd.Series, window_size: int, eps: float = EPS) -> pd.Series:
    """Rolling robust z-score (MAD) avec replis IQR puis écart-type."""
    series = pd.to_numeric(series, errors="coerce").astype(float)
    min_periods = MIN_PERIODS_ROLLING(window_size)

    med = series.rolling(window_size, min_periods=min_periods).median()
    mad = (series - med).abs().rolling(window_size, min_periods=min_periods).median()
    denom = MAD_FACTOR * mad

    z = (series - med) / denom
    z = z.replace([np.inf, -np.inf], np.nan)

    bad = (~np.isfinite(denom)) | (denom < eps)
    if bad.any():
        q25 = series.rolling(window_size, min_periods=min_periods).quantile(0.25)
        q75 = series.rolling(window_size, min_periods=min_periods).quantile(0.75)
        iqr = (q75 - q25)
        denom_iqr = (iqr / 1.349).replace([np.inf, -np.inf], np.nan)

        z2 = (series - med) / denom_iqr
        z2 = z2.replace([np.inf, -np.inf], np.nan)

        std = series.rolling(window_size, min_periods=min_periods).std()
        denom_std = std.replace([np.inf, -np.inf], np.nan)
        z3 = (series - med) / denom_std
        z3 = z3.replace([np.inf, -np.inf], np.nan)

        z = z.where(~bad, other=z2)
        z = z.where(np.isfinite(z), other=z3)

    z = z.where(np.isfinite(z), other=np.nan).fillna(0.0)
    return z


def build_interval_features(
    df_cow: pd.DataFrame,
    time_col: str,
    interval: str,
    cols: List[str],
    window_baseline: int,
    mi_name: str = "Motion Index",
) -> pd.DataFrame:
    """
    Reprend exactement la logique de l'implémentation historique de référence :
    - resample interval (sum+mean)
    - log(MI) sur sum et mean
    - coverage_pct (n_raw_samples / expected_raw_per_bin)
    - rz / rrz / d1 / d1_per_hour / mm7 / vs_mm7
    - features temporelles sin/cos
    """
    x = df_cow.set_index(time_col).copy()
    x = x.sort_index()

    # garde seulement cols existantes
    cols = [c for c in cols if c in x.columns]
    if not cols:
        out = x.reset_index().rename(columns={time_col: "T"})
        out["coverage_pct"] = 0.0
        out["Day"] = pd.to_datetime(out["T"]).dt.floor("D")
        return out

    freq = _pandas_freq(interval)
    agg = x[cols].resample(freq).agg(["sum", "mean"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    out = agg.reset_index().rename(columns={time_col: "T"})

    # log MI
    if f"{mi_name}_sum" in out.columns:
        s = pd.to_numeric(out[f"{mi_name}_sum"], errors="coerce").fillna(0.0).clip(lower=0.0)
        out[f"{mi_name}_sum_log"] = np.log1p(s)
    if f"{mi_name}_mean" in out.columns:
        s = pd.to_numeric(out[f"{mi_name}_mean"], errors="coerce").fillna(0.0).clip(lower=0.0)
        out[f"{mi_name}_mean_log"] = np.log1p(s)

    base_cols = [c for c in out.columns if c != "T"]
    mins = interval_to_minutes(interval)

    # Aligner n_raw sur les bins agrégés
    n_raw_s = x.resample(freq).size().astype(float)
    n_raw_s = n_raw_s.reindex(agg.index).fillna(0.0)
    out["n_raw_samples"] = n_raw_s.values

    # expected_raw_per_bin (robuste, par vache)
    nz = out["n_raw_samples"].replace(0, np.nan).values
    p50 = float(np.nanpercentile(nz, 50)) if np.isfinite(np.nanpercentile(nz, 50)) else 0.0
    p80 = float(np.nanpercentile(nz, 80)) if np.isfinite(np.nanpercentile(nz, 80)) else 0.0

    exp = p50 if p50 > 0 else p80
    if exp <= 0:
        mx = np.nanmax(nz)
        exp = float(mx) if np.isfinite(mx) else 1.0
    if p80 > 0:
        exp = min(exp, p80)

    exp = max(1.0, exp)
    out["expected_raw_per_bin"] = exp
    # La couverture est plafonnée à 100 %: un bin ne peut pas être « plus que
    # complet ». Comme exp est la mediane des echantillons par bin, les bins
    # au-dessus de la mediane depassaient sinon 100 % (artefact d'affichage).
    out["coverage_pct"] = np.minimum(out["n_raw_samples"] / exp, 1.0) * 100.0

    # features robustes
    for col in base_cols:
        series = pd.to_numeric(out[col], errors="coerce").astype(float)

        out[f"{col}_rz"] = robust_z(series.values)
        out[f"{col}_rrz"] = rolling_robust_z(series, window_baseline)

        out[f"{col}_d1"] = series.diff()
        out[f"{col}_d1_per_hour"] = out[f"{col}_d1"] / (mins / 60.0)

        out[f"{col}_mm7"] = series.rolling(7, min_periods=3).mean()
        out[f"{col}_vs_mm7"] = series - out[f"{col}_mm7"]

    # temps
    out["T"] = pd.to_datetime(out["T"], errors="coerce")
    out["dow"] = out["T"].dt.dayofweek
    out["hour"] = out["T"].dt.hour

    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["sin_dow"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["cos_dow"] = np.cos(2 * np.pi * out["dow"] / 7)

    out["Day"] = out["T"].dt.floor("D")
    return out
