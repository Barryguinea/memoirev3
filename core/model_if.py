"""Détection d'anomalies Isolation Forest par animal.

Entraîne un modèle IF par vache sur les features d'intervalle
(``rrz``, ``d1_per_hour``, ``vs_mm7``, variables temporelles cycliques).
Sorties: ``if_pred``, ``if_score``, ``if_anomaly_point``.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------
# LIMITATIONS CONNUES :
# 1. IF entraine par animal : chaque vache a son propre modele. Avec peu de
#    donnees par animal le modele peut etre instable. Un modele troupeau avec
#    normalisation intra-animal serait plus robuste mais necessite une
#    architecture differente.
# 2. Pas de modelisation temporelle : l'IF traite chaque intervalle de facon
#    independante. Il ne detecte pas les tendances progressives ni
#    l'autocorrelation. Un LSTM ou un modele auto-regressif pourrait capturer
#    les patterns temporels (evolution future).
# -----------------------------------------------------------------------

# Paramètres historiques conservés (compatibilité)
N_ESTIMATORS = 800
MAX_FEATURES = 0.8
QUANTILE_RANGE = (10, 90)
EPS = 1e-9


def _default_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Sélection de features cohérente avec l'implémentation historique:
    - *_rrz, *_d1_per_hour, *_vs_mm7
    - sin/cos time
    """
    cols = []

    for c in df.columns:
        s = str(c)
        if s.endswith("_rrz") or s.endswith("_d1_per_hour") or s.endswith("_vs_mm7"):
            cols.append(c)

    for c in ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]:
        if c in df.columns:
            cols.append(c)

    # unique, stable
    cols = list(dict.fromkeys(cols))
    return cols


def run_if_core(
    interval_df: pd.DataFrame,
    *,
    time_col: str = "T",
    contamination: float = 0.06,
    random_state: int = 42,
    feature_cols: Optional[List[str]] = None,
    baseline_ratio: Optional[float] = None,  # None => fit sur tout (mode historique)
    coverage_min_pct: float = 25.0,
    sensor_warmup_bins: int = 3,
) -> pd.DataFrame:
    """
    Isolation Forest sur df d'intervalle.
    Sorties standard:
    - if_pred : +1 normal, -1 anomal
    - if_score : decision_function (plus bas => plus anormal)
    - if_anomaly_point : 0/1
    - if_anom_k / anom_rate_k seront calculés ensuite dans alerts.py
    """
    df = interval_df.sort_values(time_col).copy().reset_index(drop=True)

    if feature_cols is None:
        feature_cols = _default_feature_cols(df)

    if not feature_cols:
        # pas de features => aucun scoring possible
        df["if_pred"] = 1
        df["if_score"] = np.nan
        df["if_anomaly_point"] = 0
        df["if_feat_count"] = 0
        df["if_feat_cols"] = ""
        df["dataset_split"] = "all"
        df["if_train_candidate"] = 0
        df["if_train_point"] = 0
        return df

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Mask de fiabilite: ne pas apprendre sur la phase capteur inactif / couverture faible.
    if "coverage_pct" in df.columns:
        valid_cov = (pd.to_numeric(df["coverage_pct"], errors="coerce").fillna(0.0) >= float(coverage_min_pct))
    else:
        valid_cov = pd.Series(np.ones(len(df), dtype=bool), index=df.index)

    warmup_k = max(1, int(sensor_warmup_bins))
    active_mask = valid_cov.copy()
    if warmup_k > 1:
        streak = valid_cov.astype(int).rolling(warmup_k, min_periods=warmup_k).sum()
        active_candidates = streak[streak >= warmup_k]
        if len(active_candidates) > 0:
            first_active_idx = int(active_candidates.index[0])
            active_mask = pd.Series(df.index >= first_active_idx, index=df.index)

    train_candidate_mask = (valid_cov & active_mask)
    candidate_idx = np.flatnonzero(train_candidate_mask.values)
    if candidate_idx.size == 0:
        candidate_idx = np.flatnonzero(valid_cov.values)
    if candidate_idx.size == 0:
        candidate_idx = np.arange(len(df))
    effective_candidate_mask = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    if candidate_idx.size > 0:
        effective_candidate_mask.iloc[candidate_idx] = 1

    # split baseline optionnel applique uniquement sur les points candidats
    if baseline_ratio is None:
        train_idx = candidate_idx
        df["dataset_split"] = "excluded"
        df.loc[candidate_idx, "dataset_split"] = "all"
    else:
        br = float(baseline_ratio)
        br = max(0.05, min(0.95, br))
        n_cand = len(candidate_idx)
        n_train = max(30, int(round(n_cand * br)))
        n_train = min(n_train, n_cand)
        train_idx = candidate_idx[:n_train]

        df["dataset_split"] = "excluded"
        df.loc[candidate_idx, "dataset_split"] = "futur"
        df.loc[train_idx, "dataset_split"] = "baseline"

    df["if_train_candidate"] = effective_candidate_mask.astype(int)
    df["if_train_point"] = 0
    df.loc[train_idx, "if_train_point"] = 1

    if len(train_idx) == 0:
        df["if_pred"] = 1
        df["if_score"] = np.nan
        df["if_anomaly_point"] = 0
        df["if_feat_count"] = int(len(feature_cols))
        df["if_feat_cols"] = ", ".join(feature_cols[:40]) + (" ..." if len(feature_cols) > 40 else "")
        return df

    X_train = X.iloc[train_idx]

    pipe = Pipeline([
        ("scaler", RobustScaler(quantile_range=QUANTILE_RANGE)),
        ("if", IsolationForest(
            n_estimators=int(N_ESTIMATORS),
            max_features=float(MAX_FEATURES),
            bootstrap=True,
            contamination=float(contamination),
            random_state=int(random_state),
            n_jobs=1
        ))
    ])

    pipe.fit(X_train)

    df["if_pred"] = pipe.predict(X)
    df["if_score"] = pipe.decision_function(X)

    # convention unique pour le reste du projet
    df["if_anomaly_point"] = (df["if_pred"] == -1).astype(int)

    # Les points non fiables ne doivent pas porter de signal IF.
    invalid = ~valid_cov.values
    if invalid.any():
        df.loc[invalid, "if_pred"] = 1
        df.loc[invalid, "if_score"] = np.nan
        df.loc[invalid, "if_anomaly_point"] = 0

    df["if_feat_count"] = int(len(feature_cols))
    df["if_feat_cols"] = ", ".join(feature_cols[:40]) + (" ..." if len(feature_cols) > 40 else "")

    return df
