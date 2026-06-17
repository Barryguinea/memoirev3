"""Analyses au niveau événement avec inférence regroupée par vache.

Corrige la pseudo-réplication (claim 2) : aucune agrégation de lignes répétées
sur des centaines de configurations. Les séparations train/test et les
intervalles de confiance se font PAR GROUPE (vache), de sorte qu'un même animal
(et donc ses événements) n'apparaisse jamais des deux côtés.

Corrige aussi le claim 4 : on peut faire la ROC sur le VRAI score brut IF
(``if_score_min`` = decision_function la plus anormale), pas seulement sur le
compteur seuillé ``if_anom_k_max`` (les deux sont disponibles, étiquetés).
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------------------
# 1. Synthèse de détection (niveau événement, sans pseudo-réplication)
# ---------------------------------------------------------------------------
def detection_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Couverture et nouveaux départs d'alerte par scénario synthétique."""
    unique = df.drop_duplicates(subset=["event_id"])
    g = unique.groupby("scenario")
    out = g.agg(
        n_events=("event_id", "nunique"),
        new_warning_start_mean=("detected_any_overlap", "mean"),
        warning_coverage_mean=("episode_overlap", "mean"),
        detect_iou20_mean=("detected_iou20", "mean"),
        best_iou_mean=("best_iou", "mean"),
    ).round(3)
    return out.reset_index()


def background_notification_rate(df: pd.DataFrame) -> Dict[str, float]:
    """Alertes de fond par vache-jour sur les séries réelles non étiquetées."""
    # une exécution = (cow, seed, scenario) ; on déduplique pour ne pas compter 2x
    per_run = df.drop_duplicates(subset=["cow", "seed", "scenario"])[
        ["cow", "seed", "scenario", "cow_false_notifs", "cow_days"]
    ].copy()
    per_cow = per_run.groupby(["cow", "seed"], as_index=False).agg(
        cow_false_notifs=("cow_false_notifs", "mean"), cow_days=("cow_days", "first")
    )
    per_cow["fn_per_cow_day"] = per_cow["cow_false_notifs"] / per_cow["cow_days"].clip(lower=1e-9)
    return {
        "background_per_cow_day_mean": round(float(per_cow["fn_per_cow_day"].mean()), 4),
        "background_per_cow_day_p90": round(float(per_cow["fn_per_cow_day"].quantile(0.90)), 4),
        "n_cow_runs": int(len(per_cow)),
    }


def false_notification_rate(df: pd.DataFrame) -> Dict[str, float]:
    """Alias de compatibilite; ces alertes ne sont pas des faux positifs cliniques."""
    return background_notification_rate(df)


def leak_rate(df: pd.DataFrame) -> Dict[str, float]:
    """Taux de fuite : proportion d'événements non-détectables tout de même détectés."""
    nd = df[df["expected_detected"] == 0]
    if len(nd) == 0:
        return {"leak_any_mean": float("nan"), "n_non_detectable": 0}
    return {
        "leak_any_mean": round(float(nd["detected_any_overlap"].mean()), 4),
        "n_non_detectable": int(nd["event_id"].nunique()),
    }


# ---------------------------------------------------------------------------
# 2. ROC au niveau événement, IC bootstrap PAR GROUPE (vache)
# ---------------------------------------------------------------------------
def _metric_grouped_bootstrap(
    score: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    point = float(metric(y, score))
    uniq_groups = np.unique(groups)
    boots = []
    for _ in range(n_boot):
        # rééchantillonnage AU NIVEAU GROUPE (vache), pas au niveau ligne
        sampled = rng.choice(uniq_groups, size=len(uniq_groups), replace=True)
        mask = np.concatenate([np.flatnonzero(groups == gg) for gg in sampled])
        yb, sb = y[mask], score[mask]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(metric(yb, sb))
    lo, hi = (np.nanpercentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return {"auc": round(point, 4), "ic95_lo": round(float(lo), 4), "ic95_hi": round(float(hi), 4), "n_boot": len(boots)}


def roc_event_level(
    df: pd.DataFrame,
    *,
    pos_scenario: str,
    neg_scenario: str,
    score_col: str = "if_anom_k_max",
    higher_is_positive: bool = True,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """AUC ROC ``pos_scenario`` vs ``neg_scenario`` au niveau événement.

    ``score_col`` : 'if_anom_k_max' (compteur seuillé, comparable à l'ancien) ou
    'if_score_min' (VRAI score brut IF ; plus bas = plus anormal -> mis à
    ``higher_is_positive=False``).
    IC95 par bootstrap rééchantillonné PAR VACHE (pas de pseudo-réplication).
    """
    sub = df[df["scenario"].isin([pos_scenario, neg_scenario])].dropna(subset=[score_col]).copy()
    # un événement physique = une observation
    sub = sub.drop_duplicates(subset=["event_id"])
    y = (sub["scenario"] == pos_scenario).astype(int).to_numpy()
    score = sub[score_col].to_numpy(dtype=float)
    if not higher_is_positive:
        score = -score
    groups = sub["cow"].astype(str).to_numpy()
    res = _metric_grouped_bootstrap(
        score,
        y,
        groups,
        metric=roc_auc_score,
        n_boot=n_boot,
        seed=seed,
    )
    res.update({"score_col": score_col, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                "n_groups": int(len(np.unique(groups)))})
    return res


def pr_event_level(
    df: pd.DataFrame,
    *,
    pos_scenario: str,
    neg_scenario: str,
    score_col: str = "if_anom_k_max",
    higher_is_positive: bool = True,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Average precision avec IC bootstrap par vache."""
    sub = df[df["scenario"].isin([pos_scenario, neg_scenario])].dropna(subset=[score_col]).copy()
    sub = sub.drop_duplicates(subset=["event_id"])
    y = (sub["scenario"] == pos_scenario).astype(int).to_numpy()
    score = sub[score_col].to_numpy(dtype=float)
    if not higher_is_positive:
        score = -score
    groups = sub["cow"].astype(str).to_numpy()
    res = _metric_grouped_bootstrap(
        score,
        y,
        groups,
        metric=average_precision_score,
        n_boot=n_boot,
        seed=seed,
    )
    res["average_precision"] = res.pop("auc")
    res.update(
        {
            "score_col": score_col,
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "n_groups": int(len(np.unique(groups))),
        }
    )
    return res


# ---------------------------------------------------------------------------
# 3. Calibration isotonique, prédictions hors pli PAR VACHE
# ---------------------------------------------------------------------------
def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_event_level(
    df: pd.DataFrame,
    *,
    score_col: str = "if_anom_k_max",
    higher_is_positive: bool = True,
    n_splits: int = 5,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, object]:
    """Calibration isotone marked-vs-moderate, validation croisée PAR VACHE.

    Garantit qu'aucun événement d'une vache présente à l'entraînement n'est
    évalué en test -> Brier/ECE honnêtes, IC réalistes (corrige claim 2).
    """
    det = df[df["scenario"].isin(["gradual_marked", "gradual_moderate"])].dropna(subset=[score_col]).copy()
    det = det.drop_duplicates(subset=["event_id"])
    det["y"] = (det["scenario"] == "gradual_marked").astype(int)
    score = det[score_col].to_numpy(dtype=float)
    if not higher_is_positive:
        score = -score
    y = det["y"].to_numpy()
    groups = det["cow"].astype(str).to_numpy()

    unique_groups = np.unique(groups)
    folds = min(max(2, int(n_splits)), len(unique_groups))
    if folds < 2:
        raise ValueError("Au moins deux vaches sont nécessaires pour la calibration groupée.")
    splitter = GroupKFold(n_splits=folds)
    p_oof = np.full(len(det), np.nan, dtype=float)
    for tr, te in splitter.split(score, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(score[tr], y[tr])
        p_oof[te] = iso.predict(score[te])

    valid = np.isfinite(p_oof)
    y_eval = y[valid]
    p_eval = p_oof[valid]
    eval_groups = groups[valid]
    brier = float(brier_score_loss(y_eval, p_eval))
    ece = _ece(y_eval, p_eval)

    # IC bootstrap par vache sur le test
    rng = np.random.default_rng(seed)
    uniq = np.unique(eval_groups)
    bb, be = [], []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.concatenate([np.flatnonzero(eval_groups == gg) for gg in sampled])
        yy, pp = y_eval[mask], p_eval[mask]
        if len(np.unique(yy)) > 1:
            bb.append(brier_score_loss(yy, pp))
            be.append(_ece(yy, pp))
    blo, bhi = (np.nanpercentile(bb, [2.5, 97.5]) if bb else (np.nan, np.nan))
    elo, ehi = (np.nanpercentile(be, [2.5, 97.5]) if be else (np.nan, np.nan))

    return {
        "score_col": score_col,
        "method": f"GroupKFold-{folds}",
        "n_oof_events": int(len(y_eval)),
        "n_cows": int(len(uniq)),
        "brier": round(brier, 4), "brier_ic95": [round(float(blo), 4), round(float(bhi), 4)],
        "ece": round(ece, 4), "ece_ic95": [round(float(elo), 4), round(float(ehi), 4)],
    }
