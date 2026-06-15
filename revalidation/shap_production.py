"""SHAP aligné sur les features et les points d'entraînement de production.

Différences avec le script comparatif ``scripts/compute_shap_feature_importance.py`` :

  | Réglage          | Script comparatif | Configuration alignée |
  |------------------|--------------|------------------|
  | capteurs/features| 7 / 52       | 5 / 40           |
  | window_baseline  | 48 (12 h)    | 24 (6 h)         |
  | entraînement     | toute série  | 60 % premiers    |
  | bootstrap        | absent       | True             |

On entraîne l'Isolation Forest comme ``core.model_if.run_if_core`` (RobustScaler
10–90, n_estimators=800, max_features=0.8, bootstrap=True, fit sur les 60 %
premiers bins fiables), puis on calcule les valeurs de Shapley dessus.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from core import config as C
from core.features import build_interval_features
from core.io import COW, TIME, available_base_cols, load_csv
from core.model_if import MAX_FEATURES, N_ESTIMATORS, QUANTILE_RANGE, _default_feature_cols
from revalidation.training import production_split_indices


def shap_importance_production(
    raw_csv: str = "data/brut.csv",
    *,
    cow: str = "8081",
    sample_size: int = 1000,
    seed: int = 42,
) -> Dict[str, object]:
    """Importance SHAP de l'IF avec les paramètres exacts de production.

    Retourne un dict avec le DataFrame d'importance trié et le décompte de features.
    """
    import shap  # import tardif (dépendance lourde)

    df = load_csv(raw_csv)
    df[COW] = df[COW].astype(str)
    df_cow = df[df[COW] == str(cow)].copy()

    # --- PRODUCTION : available_base_cols (5 capteurs) + fenêtre 24 bins ---
    base_cols = available_base_cols(df_cow)                       # 5 capteurs
    it = build_interval_features(
        df_cow, time_col=TIME, interval=C.DEFAULT_INTERVAL,
        cols=base_cols, window_baseline=C.DEFAULT_WINDOW_BASELINE,  # 24 = 6 h
    )
    feat_cols = _default_feature_cols(it)                         # -> 40 features
    X = it[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(drop=True)

    # Sélection exacte: couverture, warm-up capteur, puis 60 % des candidats.
    n = len(X)
    train_idx, _, candidate_idx = production_split_indices(
        it,
        baseline_ratio=C.DEFAULT_BASELINE_RATIO,
        coverage_min_pct=C.DEFAULT_COVERAGE_MIN_PCT,
        sensor_warmup_bins=C.DEFAULT_SENSOR_WARMUP_BINS,
    )
    X_train = X.iloc[train_idx]

    scaler = RobustScaler(quantile_range=QUANTILE_RANGE)
    Xs_train = scaler.fit_transform(X_train)
    Xs_all = scaler.transform(X)
    iforest = IsolationForest(
        n_estimators=int(N_ESTIMATORS),
        max_features=float(MAX_FEATURES),
        bootstrap=True,                       # <-- comme production (manquait dans l'original)
        contamination=float(C.DEFAULT_CONTAMINATION),
        random_state=int(C.DEFAULT_RANDOM_STATE),
        n_jobs=1,
    )
    iforest.fit(Xs_train)

    # --- SHAP sur un échantillon ---
    rng = np.random.default_rng(seed)
    if n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X_sample = Xs_all[idx]
    else:
        X_sample = Xs_all

    explainer = shap.TreeExplainer(iforest)
    sv = explainer.shap_values(X_sample, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[0]
    mean_abs = np.abs(sv).mean(axis=0)

    imp = (
        pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp["rank"] = imp.index + 1
    imp["cumulative_share"] = imp["mean_abs_shap"].cumsum() / imp["mean_abs_shap"].sum()

    return {
        "cow": str(cow),
        "n_features": len(feat_cols),
        "n_train_bins": int(len(train_idx)),
        "n_candidate_bins": int(len(candidate_idx)),
        "train_first_index": int(train_idx[0]),
        "train_last_index": int(train_idx[-1]),
        "n_total_bins": int(n),
        "params": {
            "n_estimators": N_ESTIMATORS,
            "max_features": MAX_FEATURES,
            "bootstrap": True,
            "window_baseline": C.DEFAULT_WINDOW_BASELINE,
            "baseline_ratio": C.DEFAULT_BASELINE_RATIO,
            "contamination": C.DEFAULT_CONTAMINATION,
            "coverage_min_pct": C.DEFAULT_COVERAGE_MIN_PCT,
            "sensor_warmup_bins": C.DEFAULT_SENSOR_WARMUP_BINS,
            "random_state": C.DEFAULT_RANDOM_STATE,
        },
        "importance": imp,
    }
