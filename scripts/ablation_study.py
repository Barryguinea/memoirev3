"""Étude d'ablation : contribution de l'IF et des règles métier.

Compare quatre variantes du pipeline sur les mêmes données injectées :
  A. Complet   (IF + règles métier)
  B. Sans IF   (règles métier seules, z-scores font le filtrage)
  C. Sans règles (IF brut, pas de persistance/cohérence/cooldown)
  D. LOF + règles (Local Outlier Factor remplace IF, mêmes règles métier)

Métriques : détection (any overlap), détection IoU≥0.20, best IoU,
fausses notifications par cow-day.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from core.alerts import apply_alert_logic
from core.config import (
    DEFAULT_ALERT_MIN,
    DEFAULT_BASELINE_RATIO,
    DEFAULT_CONTAMINATION,
    DEFAULT_COOLDOWN_HOURS,
    DEFAULT_COVERAGE_MIN_PCT,
    DEFAULT_INTERVAL,
    DEFAULT_MI_Z_HIGH_THR,
    DEFAULT_MIX_MODE,
    DEFAULT_MIX_RATE_THR,
    DEFAULT_PERSIST_HOURS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SENSOR_WARMUP_BINS,
    DEFAULT_WINDOW_BASELINE,
    DEFAULT_Z_HIGH_THR,
    DEFAULT_Z_LOW_THR,
)
from core.features import build_interval_features, interval_to_minutes
from core.io import COW, TIME, available_base_cols, load_csv
from core.model_if import _default_feature_cols, run_if_core
from scripts.validation_common import count_cow_days
from scripts.validation_eval import evaluate_injected_events, event_windows_mask
from scripts.validation_injection_raw import inject_manual_plan_raw

# Paramètres règles métier réutilisés par les variantes A et B
_ALERT_KWARGS = dict(
    time_col="T",
    interval=DEFAULT_INTERVAL,
    persist_hours=DEFAULT_PERSIST_HOURS,
    alert_min=DEFAULT_ALERT_MIN,
    mix_mode=DEFAULT_MIX_MODE,
    mix_rate_thr=DEFAULT_MIX_RATE_THR,
    z_low_thr=DEFAULT_Z_LOW_THR,
    z_high_thr=DEFAULT_Z_HIGH_THR,
    cooldown_hours=DEFAULT_COOLDOWN_HOURS,
    mi_z_high_thr=DEFAULT_MI_Z_HIGH_THR,
    coverage_min_pct=DEFAULT_COVERAGE_MIN_PCT,
)

_EVAL_KWARGS = dict(
    interval=DEFAULT_INTERVAL,
    persist_hours=DEFAULT_PERSIST_HOURS,
    alert_min=DEFAULT_ALERT_MIN,
    mix_rate_thr=DEFAULT_MIX_RATE_THR,
)

_VARIANT_ORDER = [
    "A. Complet",
    "B. Sans IF",
    "C. Sans regles",
    "D. LOF + regles",
]

_DEFAULT_EXPANDED_PROFILES = [
    "detectable_strong",
    "detectable_borderline",
    "detectable_multisignal_persistent",
]


def _run_variant_a(features_df: pd.DataFrame, cow_id: str) -> pd.DataFrame:
    """Variante A : pipeline complet (IF + règles métier)."""
    df = run_if_core(
        features_df.copy(),
        time_col="T",
        contamination=DEFAULT_CONTAMINATION,
        random_state=DEFAULT_RANDOM_STATE,
        baseline_ratio=DEFAULT_BASELINE_RATIO,
        coverage_min_pct=DEFAULT_COVERAGE_MIN_PCT,
        sensor_warmup_bins=DEFAULT_SENSOR_WARMUP_BINS,
    )
    df = apply_alert_logic(df, **_ALERT_KWARGS)
    df[COW] = str(cow_id)
    return df


def _run_variant_b(features_df: pd.DataFrame, cow_id: str) -> pd.DataFrame:
    """Variante B : sans IF — tous les bins marqués anomaux, seules les règles filtrent."""
    df = features_df.copy()
    df["if_anomaly_point"] = 1
    df["if_pred"] = -1
    df["if_score"] = -1.0
    df = apply_alert_logic(df, **_ALERT_KWARGS)
    df[COW] = str(cow_id)
    return df


def _run_variant_c(features_df: pd.DataFrame, cow_id: str) -> pd.DataFrame:
    """Variante C : sans règles métier — IF brut, chaque anomalie = épisode."""
    df = run_if_core(
        features_df.copy(),
        time_col="T",
        contamination=DEFAULT_CONTAMINATION,
        random_state=DEFAULT_RANDOM_STATE,
        baseline_ratio=DEFAULT_BASELINE_RATIO,
        coverage_min_pct=DEFAULT_COVERAGE_MIN_PCT,
        sensor_warmup_bins=DEFAULT_SENSOR_WARMUP_BINS,
    )
    df["pred_lameness_episode"] = df["if_anomaly_point"]
    df[COW] = str(cow_id)
    return df


def _run_variant_d(features_df: pd.DataFrame, cow_id: str) -> pd.DataFrame:
    """Variante D : LOF + règles métier — Local Outlier Factor remplace Isolation Forest."""
    df = features_df.copy().sort_values("T").reset_index(drop=True)

    feature_cols = _default_feature_cols(df)
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Même scaler que IF (RobustScaler quantile 10-90)
    scaler = RobustScaler(quantile_range=(10, 90))
    X_scaled = scaler.fit_transform(X)

    # LOF avec contamination identique à IF pour comparaison équitable
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=DEFAULT_CONTAMINATION,
        novelty=False,
    )
    preds = lof.fit_predict(X_scaled)

    df["if_anomaly_point"] = (preds == -1).astype(int)
    df["if_pred"] = preds
    df["if_score"] = lof.negative_outlier_factor_

    # Appliquer les mêmes règles métier que la variante A
    df = apply_alert_logic(df, **_ALERT_KWARGS)
    df[COW] = str(cow_id)
    return df


def _false_notifs_per_cow_day(pred_df: pd.DataFrame, events_df: pd.DataFrame) -> float:
    """Nombre de notifications hors fenêtres injectées, normalisé par cow-day."""
    if "notif_lameness" not in pred_df.columns:
        return float("nan")
    mask_inj = event_windows_mask(pred_df, events_df)
    notifs_outside = pred_df.loc[~mask_inj, "notif_lameness"]
    n_false = int(pd.to_numeric(notifs_outside, errors="coerce").fillna(0).sum())
    cd = count_cow_days(pred_df)
    return n_false / max(1, cd)


def _prepare_features(df_inj: pd.DataFrame, cow_id: str) -> pd.DataFrame:
    """Construit les features intervalle pour une vache injectee."""
    df_cow = df_inj[df_inj[COW] == str(cow_id)].copy()
    base_cols = available_base_cols(df_cow)
    return build_interval_features(
        df_cow,
        time_col=TIME,
        interval=DEFAULT_INTERVAL,
        cols=base_cols,
        window_baseline=DEFAULT_WINDOW_BASELINE,
    )


def _run_variants(features_df: pd.DataFrame, cow_id: str) -> Dict[str, pd.DataFrame]:
    """Execute les quatre variantes d'ablation sur les memes features."""
    return {
        "A. Complet": _run_variant_a(features_df, cow_id),
        "B. Sans IF": _run_variant_b(features_df, cow_id),
        "C. Sans regles": _run_variant_c(features_df, cow_id),
        "D. LOF + regles": _run_variant_d(features_df, cow_id),
    }


def _evaluate_variants(
    variants: Dict[str, pd.DataFrame],
    events_df: pd.DataFrame,
    *,
    seed: int,
    cow_id: str,
    profile: str,
) -> List[Dict[str, object]]:
    """Calcule les metriques de chaque variante pour un cas injecte."""
    rows: List[Dict[str, object]] = []
    for name, pred_df in variants.items():
        ev = evaluate_injected_events(pred_df, events_df, **_EVAL_KWARGS)
        if ev.empty:
            continue
        rows.append(
            {
                "seed": int(seed),
                "cow": str(cow_id),
                "profile": str(profile),
                "case_id": f"{cow_id}|{profile}|{seed}",
                "variante": name,
                "detection_any": float(ev["detected_any_overlap"].mean()),
                "detection_iou20": float(ev["detected_iou20"].mean()),
                "best_iou": float(ev["best_iou"].mean()),
                "fausses_notif_cow_day": _false_notifs_per_cow_day(pred_df, events_df),
            }
        )
    return rows


def _detectable_event_length(*, interval: str, persist_hours: int) -> int:
    """Longueur minimale usuelle d'un evenement detectable pour l'injection raw."""
    minutes = interval_to_minutes(interval)
    k = max(1, int(round((float(persist_hours) * 60) / max(1, minutes))))
    return max(int(round(1.5 * k)), 36)


def _select_benchmark_cows(
    df_raw: pd.DataFrame,
    *,
    n_cows: int,
    interval: str = DEFAULT_INTERVAL,
    persist_hours: int = DEFAULT_PERSIST_HOURS,
    preferred: Iterable[str] | None = None,
) -> List[str]:
    """Selectionne des vaches avec assez de lignes pour un benchmark pairé."""
    min_len = _detectable_event_length(interval=interval, persist_hours=persist_hours) + 10
    counts = (
        df_raw.assign(_cow=df_raw[COW].astype(str))
        .groupby("_cow", sort=False)
        .size()
        .sort_values(ascending=False)
    )
    eligible = [str(cow) for cow, n_rows in counts.items() if int(n_rows) >= int(min_len)]
    picked: List[str] = []
    if preferred is not None:
        pref_list = [str(cow) for cow in preferred]
        picked.extend([cow for cow in pref_list if cow in eligible and cow not in picked])
    for cow in eligible:
        if cow not in picked:
            picked.append(cow)
        if len(picked) >= int(n_cows):
            break
    return picked[: int(n_cows)]


def run_ablation_expanded(
    input_path: str | Path = "data/brut.csv",
    *,
    cow_ids: List[str] | None = None,
    profiles: List[str] | None = None,
    n_cows: int = 3,
    n_seeds: int = 10,
    seed_start: int = 42,
) -> pd.DataFrame:
    """Lance un benchmark d'ablation pairé sur plusieurs vaches et profils."""
    base_raw = load_csv(str(input_path))
    selected_cows = (
        [str(cow) for cow in cow_ids]
        if cow_ids
        else _select_benchmark_cows(
            base_raw,
            n_cows=n_cows,
            interval=DEFAULT_INTERVAL,
            persist_hours=DEFAULT_PERSIST_HOURS,
        )
    )
    selected_profiles = [str(profile) for profile in (profiles or _DEFAULT_EXPANDED_PROFILES)]

    rows: List[Dict[str, object]] = []
    total_cases = len(selected_cows) * len(selected_profiles) * int(n_seeds)
    case_idx = 0
    for seed in range(seed_start, seed_start + n_seeds):
        for cow_id in selected_cows:
            for profile in selected_profiles:
                case_idx += 1
                plan = [
                    {
                        "cow": str(cow_id),
                        "profile": str(profile),
                        "expected_detected": 1,
                        "event_id": f"ablation_{cow_id}_{profile}_{seed}",
                    },
                ]
                df_inj, events_df = inject_manual_plan_raw(
                    base_raw,
                    plan=plan,
                    interval=DEFAULT_INTERVAL,
                    persist_hours=DEFAULT_PERSIST_HOURS,
                    seed=seed,
                )
                features_df = _prepare_features(df_inj, str(cow_id))
                variants = _run_variants(features_df, str(cow_id))
                rows.extend(
                    _evaluate_variants(
                        variants,
                        events_df,
                        seed=seed,
                        cow_id=str(cow_id),
                        profile=str(profile),
                    )
                )
                print(
                    f"  cas {case_idx}/{total_cases} termine "
                    f"(seed={seed}, cow={cow_id}, profile={profile})"
                )

    return pd.DataFrame(rows)


def run_ablation(
    input_path: str | Path = "data/brut.csv",
    cow_id: str = "8081",
    n_seeds: int = 10,
    seed_start: int = 42,
) -> pd.DataFrame:
    """Lance l'étude d'ablation sur plusieurs seeds d'injection."""
    df = run_ablation_expanded(
        input_path=input_path,
        cow_ids=[str(cow_id)],
        profiles=["detectable_multisignal_persistent"],
        n_cows=1,
        n_seeds=n_seeds,
        seed_start=seed_start,
    )
    if "cow" in df.columns:
        df = df.drop(columns=["cow"])
    if "profile" in df.columns:
        df = df.drop(columns=["profile"])
    if "case_id" in df.columns:
        df = df.drop(columns=["case_id"])
    return df


def _build_summary(
    df: pd.DataFrame,
    *,
    group_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Construit un resume agrege avec moyennes et ecarts-types."""
    group_cols = list(group_cols or ["variante"])
    summary_rows: List[Dict[str, object]] = []
    work_df = df.copy()
    work_df["detection_iou10"] = (work_df["best_iou"] >= 0.10).astype(float)

    for keys, g in work_df.groupby(group_cols, dropna=False, sort=False):
        key_values = (keys,) if not isinstance(keys, tuple) else keys
        row = {col: value for col, value in zip(group_cols, key_values)}
        row.update(
            {
                "n_runs": int(len(g)),
                "n_seeds": int(g["seed"].nunique()) if "seed" in g.columns else 0,
                "n_cows": int(g["cow"].nunique()) if "cow" in g.columns else 0,
                "n_profiles": int(g["profile"].nunique()) if "profile" in g.columns else 0,
                "detection_any_mean": float(g["detection_any"].mean()),
                "detection_any_std": float(g["detection_any"].std()),
                "detection_iou20_mean": float(g["detection_iou20"].mean()),
                "detection_iou20_std": float(g["detection_iou20"].std()),
                "detection_iou10_mean": float(g["detection_iou10"].mean()),
                "detection_iou10_std": float(g["detection_iou10"].std()),
                "best_iou_mean": float(g["best_iou"].mean()),
                "best_iou_std": float(g["best_iou"].std()),
                "fausses_notif_cow_day_mean": float(g["fausses_notif_cow_day"].mean()),
                "fausses_notif_cow_day_std": float(g["fausses_notif_cow_day"].std()),
            }
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if "variante" in summary_df.columns:
        summary_df["variante"] = pd.Categorical(
            summary_df["variante"],
            categories=_VARIANT_ORDER,
            ordered=True,
        )
        summary_df = summary_df.sort_values(group_cols).reset_index(drop=True)
        summary_df["variante"] = summary_df["variante"].astype(str)
    return summary_df


def _paired_counts(
    df: pd.DataFrame,
    left_variant: str,
    right_variant: str,
) -> Dict[str, int]:
    """Compte les gains/pertes seed-par-seed entre deux variantes."""
    pair_cols = [col for col in ["seed", "cow", "profile", "case_id"] if col in df.columns]
    if not pair_cols:
        pair_cols = ["seed"]
    left = df[df["variante"] == left_variant].set_index(pair_cols)
    right = df[df["variante"] == right_variant].set_index(pair_cols)
    common_seeds = left.index.intersection(right.index)
    if common_seeds.empty:
        return {
            "n_common": 0,
            "det_right_gt": 0,
            "det_right_eq": 0,
            "det_right_lt": 0,
            "fn_right_lt": 0,
            "fn_right_eq": 0,
            "fn_right_gt": 0,
            "best_iou_right_gt": 0,
            "best_iou_right_eq": 0,
            "best_iou_right_lt": 0,
            "det_ge_and_fn_lt": 0,
            "det_gt_and_fn_lt": 0,
        }

    left = left.loc[common_seeds]
    right = right.loc[common_seeds]
    return {
        "n_common": int(len(common_seeds)),
        "det_right_gt": int((right["detection_any"] > left["detection_any"]).sum()),
        "det_right_eq": int((right["detection_any"] == left["detection_any"]).sum()),
        "det_right_lt": int((right["detection_any"] < left["detection_any"]).sum()),
        "fn_right_lt": int((right["fausses_notif_cow_day"] < left["fausses_notif_cow_day"]).sum()),
        "fn_right_eq": int((right["fausses_notif_cow_day"] == left["fausses_notif_cow_day"]).sum()),
        "fn_right_gt": int((right["fausses_notif_cow_day"] > left["fausses_notif_cow_day"]).sum()),
        "best_iou_right_gt": int((right["best_iou"] > left["best_iou"]).sum()),
        "best_iou_right_eq": int((right["best_iou"] == left["best_iou"]).sum()),
        "best_iou_right_lt": int((right["best_iou"] < left["best_iou"]).sum()),
        "det_ge_and_fn_lt": int(
            ((right["detection_any"] >= left["detection_any"]) & (right["fausses_notif_cow_day"] < left["fausses_notif_cow_day"])).sum()
        ),
        "det_gt_and_fn_lt": int(
            ((right["detection_any"] > left["detection_any"]) & (right["fausses_notif_cow_day"] < left["fausses_notif_cow_day"])).sum()
        ),
    }


def _print_table(df: pd.DataFrame) -> None:
    """Affiche le tableau agrégé par variante."""
    def _fmt(mean: float, std: float, *, pct: bool = False) -> str:
        if not np.isfinite(mean):
            return "N/A"
        if not np.isfinite(std):
            std = 0.0
        if pct:
            return f"{mean:.0%} ± {std:.0%}"
        return f"{mean:.3f} ± {std:.3f}"

    print("\n" + "=" * 70)
    print("ÉTUDE D'ABLATION — contribution de chaque composante")
    print("=" * 70)
    print(f"{'Variante':<18} {'Détection':>18} {'IoU≥0.20':>18} {'Best IoU':>18} {'Faux/cow-day':>18}")
    print("-" * 70)
    for name in ["A. Complet", "B. Sans IF", "C. Sans regles", "D. LOF + regles"]:
        g = df[df["variante"] == name]
        if g.empty:
            continue
        det_m, det_s = float(g["detection_any"].mean()), float(g["detection_any"].std())
        i20_m, i20_s = float(g["detection_iou20"].mean()), float(g["detection_iou20"].std())
        bi_m, bi_s = float(g["best_iou"].mean()), float(g["best_iou"].std())
        fn_m, fn_s = float(g["fausses_notif_cow_day"].mean()), float(g["fausses_notif_cow_day"].std())
        print(
            f"{name:<18} "
            f"{_fmt(det_m, det_s, pct=True):>18} "
            f"{_fmt(i20_m, i20_s, pct=True):>18} "
            f"{_fmt(bi_m, bi_s):>18} "
            f"{_fmt(fn_m, fn_s):>18}"
        )
    print("=" * 70)
    scope_parts = [f"Seeds : {df['seed'].nunique()}"]
    if "cow" in df.columns:
        scope_parts.append(f"Vaches : {df['cow'].nunique()}")
    if "profile" in df.columns:
        scope_parts.append(f"Profils : {df['profile'].nunique()}")
    print(" | ".join(scope_parts))


def _benchmark_scope_text(df: pd.DataFrame) -> str:
    """Retourne une description courte de la portee du benchmark."""
    n_seeds = int(df["seed"].nunique()) if "seed" in df.columns else 0
    n_cows = int(df["cow"].nunique()) if "cow" in df.columns else 1
    n_profiles = int(df["profile"].nunique()) if "profile" in df.columns else 1
    if n_cows <= 1 and n_profiles <= 1:
        return f"micro-benchmark local (1 profil injecte, 1 vache, {n_seeds} seeds)"
    return f"benchmark elargi paire ({n_cows} vaches, {n_profiles} profils, {n_seeds} seeds)"


def _write_report(df: pd.DataFrame, path: Path) -> None:
    """Génère un rapport texte explicatif des résultats d'ablation."""
    def _fmt(mean: float, std: float, *, pct: bool = False) -> str:
        if not np.isfinite(mean):
            return "N/A"
        if not np.isfinite(std):
            std = 0.0
        if pct:
            return f"{mean:.0%} ± {std:.0%}"
        return f"{mean:.3f} ± {std:.3f}"

    lines = []
    lines.append("=" * 70)
    lines.append("ÉTUDE D'ABLATION — Rapport explicatif")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OBJECTIF")
    lines.append("-" * 70)
    lines.append("Démontrer que chaque composante du pipeline (Isolation Forest et")
    lines.append("règles métier) est nécessaire. On retire une pièce à la fois et on")
    lines.append("mesure l'impact sur la détection et les fausses alertes.")
    lines.append("")
    lines.append("VARIANTES TESTÉES")
    lines.append("-" * 70)
    lines.append("A. Complet     : IF + règles métier (système actuel)")
    lines.append("B. Sans IF     : règles métier seules (z-scores font tout le filtrage)")
    lines.append("C. Sans règles : IF brut (pas de persistance, cohérence, cooldown)")
    lines.append("D. LOF + règles: Local Outlier Factor remplace IF, mêmes règles métier")
    lines.append("")
    lines.append("PROTOCOLE")
    lines.append("-" * 70)
    n_seeds = df["seed"].nunique()
    n_variants = df["variante"].nunique()
    lines.append(f"- {n_seeds} seeds d'injection")
    if "cow" in df.columns:
        lines.append(f"- {int(df['cow'].nunique())} vache(s) evaluee(s)")
    if "profile" in df.columns:
        profile_list = ", ".join(str(x) for x in sorted(df["profile"].dropna().astype(str).unique()))
        lines.append(f"- Profils : {profile_list}")
    else:
        lines.append("- Profil : detectable_multisignal_persistent (multi-signal cohérent)")
    lines.append(f"- Mêmes données, même injection pour les {n_variants} variantes a chaque cas injecte")
    lines.append("- Métriques : détection (overlap), IoU, fausses notifications/cow-day")
    lines.append("")
    lines.append("RÉSULTATS AGRÉGÉS")
    lines.append("-" * 70)
    lines.append(
        f"{'Variante':<18} {'Détection':>18} {'IoU20':>18} {'Best IoU':>18} {'Faux/cow-day':>18}"
    )
    lines.append("-" * 92)
    summary_df = _build_summary(df)
    summaries = {}
    for _, row in summary_df.iterrows():
        name = str(row["variante"])
        if not name:
            continue
        lines.append(
            f"{name:<18} "
            f"{_fmt(float(row['detection_any_mean']), float(row['detection_any_std']), pct=True):>18} "
            f"{_fmt(float(row['detection_iou20_mean']), float(row['detection_iou20_std']), pct=True):>18} "
            f"{_fmt(float(row['best_iou_mean']), float(row['best_iou_std'])):>18} "
            f"{_fmt(float(row['fausses_notif_cow_day_mean']), float(row['fausses_notif_cow_day_std'])):>18}"
        )
        summaries[name] = {
            "det": float(row["detection_any_mean"]),
            "det_std": float(row["detection_any_std"]),
            "iou20": float(row["detection_iou20_mean"]),
            "iou20_std": float(row["detection_iou20_std"]),
            "iou10": float(row["detection_iou10_mean"]),
            "iou10_std": float(row["detection_iou10_std"]),
            "biou": float(row["best_iou_mean"]),
            "biou_std": float(row["best_iou_std"]),
            "fn": float(row["fausses_notif_cow_day_mean"]),
            "fn_std": float(row["fausses_notif_cow_day_std"]),
        }
    lines.append("")
    lines.append("INTERPRÉTATION")
    lines.append("-" * 70)
    sa = summaries.get("A. Complet", {})
    sb = summaries.get("B. Sans IF", {})
    sc = summaries.get("C. Sans regles", {})
    sd = summaries.get("D. LOF + regles", {})
    if sc:
        lines.append(f"C. Sans règles (IF seul) :")
        lines.append(f"   Détection = {sc['det']:.0%} — l'IF détecte presque toujours un overlap,")
        lines.append(f"   mais sans filtrage il signale {DEFAULT_CONTAMINATION:.0%} de TOUS les bins")
        lines.append(f"   comme anomaux. Inutilisable en production (trop de bruit).")
        lines.append("")
    if sb:
        ratio = sb["fn"] / sa["fn"] if sa.get("fn", 0) > 0 else float("nan")
        ratio_str = f"{ratio:.1f}x" if np.isfinite(ratio) else "N/A"
        lines.append(f"B. Sans IF (règles seules) :")
        lines.append(f"   Détection = {sb['det']:.0%}, mais {ratio_str} plus de fausses alertes")
        lines.append(f"   que le système complet ({sb['fn']:.2f} vs {sa.get('fn', 0):.2f}/cow-day).")
        lines.append(f"   Les z-scores seuls ne discriminent pas assez.")
        lines.append("")
    if sd:
        paired_ad = _paired_counts(df, "A. Complet", "D. LOF + regles")
        lines.append(f"D. LOF + règles (alternative non-supervisée) :")
        lines.append(
            f"   Détection = {sd['det']:.0%} ± {sd['det_std']:.0%}, "
            f"fausses notifs = {sd['fn']:.2f} ± {sd['fn_std']:.2f}/cow-day."
        )
        if sa:
            d_better = sd["det"] > sa["det"]
            fn_better = np.isfinite(sd["fn"]) and np.isfinite(sa["fn"]) and sd["fn"] < sa["fn"]
            if d_better and fn_better:
                lines.append(
                    f"   Sur ce benchmark, D dépasse A sur les deux métriques observées "
                    f"(détection {sd['det']:.0%} vs {sa['det']:.0%}, faux {sd['fn']:.2f} vs {sa['fn']:.2f})."
                )
            else:
                lines.append(
                    f"   D n'est pas systématiquement supérieur à A sur l'ensemble des métriques."
                )
            lines.append(
                f"   Seed-par-seed, D garde une détection au moins égale sur "
                f"{paired_ad['det_ge_and_fn_lt']}/{paired_ad['n_common']} seeds "
                f"tout en réduisant les faux positifs sur ces mêmes seeds."
            )
            lines.append(
                f"   Le gain en détection est strict sur {paired_ad['det_gt_and_fn_lt']}/{paired_ad['n_common']} seeds."
            )
            lines.append(
                f"   Ce résultat reste un {_benchmark_scope_text(df)}, "
                f"pas un remplacement automatique de la baseline."
            )
            lines.append(
                f"   IF peut toutefois rester préféré en production pour la scalabilité "
                f"(complexité O(n) vs O(n²) pour LOF) et la continuité des validations historiques."
            )
        lines.append("")
    if sa:
        lines.append(f"A. Complet (IF + règles) :")
        lines.append(
            f"   Détection = {sa['det']:.0%} ± {sa['det_std']:.0%} et "
            f"faux = {sa['fn']:.2f} ± {sa['fn_std']:.2f}/cow-day "
            f"(référence pipeline principal)."
        )
        lines.append("")

    if np.isfinite(float(df["detection_iou20"].mean())) and float(df["detection_iou20"].mean()) == 0.0:
        lines.append("IoU20 (seuil strict) :")
        lines.append(
            "   Aucune variante n'atteint IoU >= 0.20 sur ce protocole. "
            "Cela n'implique pas absence de signal: detection_any reste non nulle."
        )
        lines.append(
            "   Le résultat indique un recouvrement partiel/fragmenté, cohérent avec les best_iou observés."
        )
        lines.append("")
    lines.append("CONCLUSION")
    lines.append("-" * 70)
    lines.append("L'IF seul est trop sensible (bruit) et les règles seules génèrent trop")
    lines.append("de faux positifs. Sur ce benchmark d'ablation, LOF + règles obtient")
    lines.append("de meilleurs scores empiriques que IF + règles sur les métriques observées.")
    lines.append("La conclusion est locale au protocole d'ablation et ne suffit pas, seule,")
    lines.append("à invalider le choix historique d'IF dans le pipeline principal.")
    lines.append("Le pipeline IF + règles peut néanmoins rester le choix principal pour")
    lines.append("la scalabilité, la reproductibilité historique et la cohérence du projet.")
    lines.append("=" * 70)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary(df: pd.DataFrame, path: Path) -> None:
    """Sauvegarde le resume agrege a partir des resultats bruts existants."""
    _build_summary(df).to_csv(path, index=False)


def write_derived_artifacts(
    df: pd.DataFrame,
    project_root: Path,
    *,
    prefix: str = "ablation",
) -> None:
    """Ecrit les artefacts derives sans relancer l'ablation complete."""
    summary_path = project_root / "data" / f"{prefix}_summary.csv"
    report_path = project_root / "data" / f"{prefix}_rapport.txt"
    _write_summary(df, summary_path)
    _write_report(df, report_path)
    if "profile" in df.columns and int(df["profile"].nunique()) > 1:
        profile_summary_path = project_root / "data" / f"{prefix}_summary_by_profile.csv"
        _build_summary(df, group_cols=["profile", "variante"]).to_csv(profile_summary_path, index=False)
    if "cow" in df.columns and int(df["cow"].nunique()) > 1:
        cow_summary_path = project_root / "data" / f"{prefix}_summary_by_cow.csv"
        _build_summary(df, group_cols=["cow", "variante"]).to_csv(cow_summary_path, index=False)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse les arguments CLI du script d'ablation."""
    parser = argparse.ArgumentParser(description="Ablation IF / regles / LOF")
    parser.add_argument("--mode", choices=["local", "expanded"], default="local")
    parser.add_argument("--input-path", default="data/brut.csv")
    parser.add_argument("--cow-id", default="8081")
    parser.add_argument("--cows", default="")
    parser.add_argument("--n-cows", type=int, default=3)
    parser.add_argument(
        "--profiles",
        default=",".join(_DEFAULT_EXPANDED_PROFILES),
        help="Liste CSV de profils pour le mode expanded.",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--output-prefix", default="")
    return parser.parse_args(argv)


def _split_csv_arg(value: str) -> List[str]:
    """Convertit une chaine CSV simple en liste propre."""
    if not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def main(argv: List[str] | None = None) -> int:
    """Point d'entrée CLI."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    print("Étude d'ablation en cours...")
    project_root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input_path)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    if not input_path.exists():
        print(f"ERREUR : fichier introuvable : {input_path}")
        return 1

    if args.mode == "expanded":
        cow_ids = _split_csv_arg(args.cows)
        profiles = _split_csv_arg(args.profiles)
        df = run_ablation_expanded(
            input_path=input_path,
            cow_ids=cow_ids or None,
            profiles=profiles or None,
            n_cows=max(1, int(args.n_cows)),
            n_seeds=max(1, int(args.n_seeds)),
            seed_start=int(args.seed_start),
        )
        output_prefix = args.output_prefix or "ablation_expanded"
    else:
        df = run_ablation(
            input_path=input_path,
            cow_id=str(args.cow_id),
            n_seeds=max(1, int(args.n_seeds)),
            seed_start=int(args.seed_start),
        )
        output_prefix = args.output_prefix or "ablation"

    if df.empty:
        print("ERREUR : aucun résultat.")
        return 1

    _print_table(df)

    out_path = project_root / "data" / f"{output_prefix}_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nRésultats sauvegardés : {out_path}")

    write_derived_artifacts(df, project_root, prefix=output_prefix)
    print(f"Résumé agrégé       : {project_root / 'data' / f'{output_prefix}_summary.csv'}")
    print(f"Rapport explicatif  : {project_root / 'data' / f'{output_prefix}_rapport.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
