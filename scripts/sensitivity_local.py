"""Analyse de sensibilité locale autour de la configuration v5.2 GO retenue.

Pour chaque paramètre clé du pipeline (contamination, persist_hours, alert_min,
mix_rate_thr, mi_z_high_thr), on perturbe la valeur retenue d'un cran vers le
bas et un cran vers le haut, puis on relance le pipeline complet (IF + règles
métier) sur le même benchmark d'ablation (3 vaches × 3 profils × n_seeds).

Objectif : démontrer que la configuration retenue n'est pas un point isolé
fragile dans l'espace des 336 configurations, mais une zone stable dont les
voisins immédiats produisent des métriques comparables.

Sortie : data/sensitivity_local_results.csv (cas par cas) et
data/sensitivity_local_summary.csv (delta vs baseline par paramètre).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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
from core.features import build_interval_features
from core.io import COW, TIME, available_base_cols, load_csv
from core.model_if import run_if_core
from scripts.ablation_study import _select_benchmark_cows
from scripts.validation_common import count_cow_days
from scripts.validation_eval import evaluate_injected_events, event_windows_mask
from scripts.validation_injection_raw import inject_manual_plan_raw

# Configuration baseline retenue (v5.2 GO, alignée sur core/config.py)
_BASELINE = {
    "contamination": float(DEFAULT_CONTAMINATION),
    "persist_hours": int(DEFAULT_PERSIST_HOURS),
    "alert_min": int(DEFAULT_ALERT_MIN),
    "mix_rate_thr": float(DEFAULT_MIX_RATE_THR),
    "mi_z_high_thr": float(DEFAULT_MI_Z_HIGH_THR),
}

# Perturbations ±1 cran autour de la baseline.
# Choix justifiés : un cran représente la granularité opérationnelle naturelle
# (entiers pour persist_hours/alert_min, ±0.01-0.04 pour les seuils continus).
_PERTURBATIONS: Dict[str, Tuple[float, float]] = {
    "contamination":  (0.05, 0.07),   # baseline 0.06
    "persist_hours":  (6,    8),       # baseline 7
    "alert_min":      (1,    3),       # baseline 2
    "mix_rate_thr":   (0.20, 0.28),    # baseline 0.24
    "mi_z_high_thr":  (2.0,  2.4),     # baseline 2.2
}

_PROFILES = [
    "detectable_strong",
    "detectable_borderline",
    "detectable_multisignal_persistent",
]


def _build_alert_kwargs(config: Dict[str, float]) -> dict:
    """Reconstruit les kwargs d'apply_alert_logic à partir d'une config dict."""
    return dict(
        time_col="T",
        interval=DEFAULT_INTERVAL,
        persist_hours=int(config["persist_hours"]),
        alert_min=int(config["alert_min"]),
        mix_mode=DEFAULT_MIX_MODE,
        mix_rate_thr=float(config["mix_rate_thr"]),
        z_low_thr=DEFAULT_Z_LOW_THR,
        z_high_thr=DEFAULT_Z_HIGH_THR,
        cooldown_hours=DEFAULT_COOLDOWN_HOURS,
        mi_z_high_thr=float(config["mi_z_high_thr"]),
        coverage_min_pct=DEFAULT_COVERAGE_MIN_PCT,
    )


def _run_pipeline(features_df: pd.DataFrame, cow_id: str, config: Dict[str, float]) -> pd.DataFrame:
    """Exécute le pipeline complet (IF + règles) avec la config donnée."""
    df = run_if_core(
        features_df.copy(),
        time_col="T",
        contamination=float(config["contamination"]),
        random_state=DEFAULT_RANDOM_STATE,
        baseline_ratio=DEFAULT_BASELINE_RATIO,
        coverage_min_pct=DEFAULT_COVERAGE_MIN_PCT,
        sensor_warmup_bins=DEFAULT_SENSOR_WARMUP_BINS,
    )
    df = apply_alert_logic(df, **_build_alert_kwargs(config))
    df[COW] = str(cow_id)
    return df


def _false_notifs_per_cow_day(pred_df: pd.DataFrame, events_df: pd.DataFrame) -> float:
    """Notifications hors fenêtres injectées, normalisées par cow-day."""
    if "notif_lameness" not in pred_df.columns:
        return float("nan")
    mask_inj = event_windows_mask(pred_df, events_df)
    notifs_outside = pred_df.loc[~mask_inj, "notif_lameness"]
    n_false = int(pd.to_numeric(notifs_outside, errors="coerce").fillna(0).sum())
    cd = count_cow_days(pred_df)
    return n_false / max(1, cd)


def _prepare_features(df_inj: pd.DataFrame, cow_id: str) -> pd.DataFrame:
    """Construit les features intervalle pour une vache injectée."""
    df_cow = df_inj[df_inj[COW] == str(cow_id)].copy()
    base_cols = available_base_cols(df_cow)
    return build_interval_features(
        df_cow,
        time_col=TIME,
        interval=DEFAULT_INTERVAL,
        cols=base_cols,
        window_baseline=DEFAULT_WINDOW_BASELINE,
    )


def _config_label(name: str, value: float, baseline: float) -> str:
    """Étiquette lisible pour une config (ex. 'persist_hours=6 (-1)')."""
    if name == "baseline":
        return "baseline"
    diff = float(value) - float(baseline)
    sign = "+" if diff > 0 else ""
    if name in ("persist_hours", "alert_min"):
        return f"{name}={int(value)} ({sign}{int(diff)})"
    return f"{name}={value:g} ({sign}{diff:.2g})"


def _enumerate_configs() -> List[Tuple[str, Dict[str, float], str]]:
    """Renvoie la liste (param_name, config_dict, label) à tester.

    L'élément 0 est la baseline; les suivants perturbent un seul paramètre.
    """
    configs: List[Tuple[str, Dict[str, float], str]] = []
    configs.append(("baseline", dict(_BASELINE), _config_label("baseline", 0.0, 0.0)))
    for name, (low, high) in _PERTURBATIONS.items():
        for val in (low, high):
            cfg = dict(_BASELINE)
            cfg[name] = val
            configs.append((name, cfg, _config_label(name, val, _BASELINE[name])))
    return configs


def run_sensitivity(
    input_path: str | Path = "data/brut.csv",
    *,
    n_cows: int = 3,
    n_seeds: int = 5,
    seed_start: int = 42,
) -> pd.DataFrame:
    """Lance l'analyse de sensibilité locale et renvoie les résultats bruts."""
    base_raw = load_csv(str(input_path))
    selected_cows = _select_benchmark_cows(
        base_raw,
        n_cows=n_cows,
        interval=DEFAULT_INTERVAL,
        persist_hours=DEFAULT_PERSIST_HOURS,
    )
    configs = _enumerate_configs()
    total_cases = len(configs) * len(selected_cows) * len(_PROFILES) * int(n_seeds)
    case_idx = 0
    rows: List[Dict[str, object]] = []

    print(
        f"Analyse de sensibilité : {len(configs)} configs "
        f"({len(_PERTURBATIONS)} paramètres × 2 perturbations + baseline) "
        f"× {len(selected_cows)} vaches × {len(_PROFILES)} profils × {n_seeds} seeds "
        f"= {total_cases} cas."
    )

    for param_name, config, label in configs:
        for seed in range(seed_start, seed_start + n_seeds):
            for cow_id in selected_cows:
                for profile in _PROFILES:
                    case_idx += 1
                    plan = [
                        {
                            "cow": str(cow_id),
                            "profile": str(profile),
                            "expected_detected": 1,
                            "event_id": f"sensit_{cow_id}_{profile}_{seed}",
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
                    pred_df = _run_pipeline(features_df, str(cow_id), config)
                    ev = evaluate_injected_events(
                        pred_df,
                        events_df,
                        interval=DEFAULT_INTERVAL,
                        persist_hours=int(config["persist_hours"]),
                        alert_min=int(config["alert_min"]),
                        mix_rate_thr=float(config["mix_rate_thr"]),
                    )
                    if ev.empty:
                        continue
                    rows.append(
                        {
                            "param": param_name,
                            "label": label,
                            "contamination": float(config["contamination"]),
                            "persist_hours": int(config["persist_hours"]),
                            "alert_min": int(config["alert_min"]),
                            "mix_rate_thr": float(config["mix_rate_thr"]),
                            "mi_z_high_thr": float(config["mi_z_high_thr"]),
                            "seed": int(seed),
                            "cow": str(cow_id),
                            "profile": str(profile),
                            "detection_any": float(ev["detected_any_overlap"].mean()),
                            "detection_iou20": float(ev["detected_iou20"].mean()),
                            "best_iou": float(ev["best_iou"].mean()),
                            "fausses_notif_cow_day": _false_notifs_per_cow_day(pred_df, events_df),
                        }
                    )
                    if case_idx % 30 == 0 or case_idx == total_cases:
                        print(f"  cas {case_idx}/{total_cases}")

    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les résultats par configuration (label) et calcule les delta vs baseline."""
    if df.empty:
        return df
    grouped = (
        df.groupby(["param", "label"], sort=False)
        .agg(
            n_cases=("detection_any", "size"),
            detection_any_mean=("detection_any", "mean"),
            detection_iou20_mean=("detection_iou20", "mean"),
            best_iou_mean=("best_iou", "mean"),
            fausses_notif_mean=("fausses_notif_cow_day", "mean"),
        )
        .reset_index()
    )

    # Baseline values
    baseline_row = grouped[grouped["param"] == "baseline"].iloc[0]
    base_det = float(baseline_row["detection_any_mean"])
    base_iou = float(baseline_row["detection_iou20_mean"])
    base_biou = float(baseline_row["best_iou_mean"])
    base_fn = float(baseline_row["fausses_notif_mean"])

    grouped["delta_detection_any"] = grouped["detection_any_mean"] - base_det
    grouped["delta_detection_iou20"] = grouped["detection_iou20_mean"] - base_iou
    grouped["delta_best_iou"] = grouped["best_iou_mean"] - base_biou
    grouped["delta_fausses_notif"] = grouped["fausses_notif_mean"] - base_fn
    return grouped


def write_tornado_data(summary_df: pd.DataFrame, path: Path) -> None:
    """Calcule l'amplitude max de Δ-détection par paramètre pour la figure tornade."""
    rows: List[Dict[str, object]] = []
    for param in [p for p in _PERTURBATIONS]:
        sub = summary_df[summary_df["param"] == param]
        if sub.empty:
            continue
        deltas = sub["delta_detection_any"].abs()
        rows.append(
            {
                "param": param,
                "delta_min": float(sub["delta_detection_any"].min()),
                "delta_max": float(sub["delta_detection_any"].max()),
                "amplitude_abs": float(deltas.max()),
            }
        )
    pd.DataFrame(rows).sort_values("amplitude_abs", ascending=False).to_csv(path, index=False)


def _print_table(summary_df: pd.DataFrame) -> None:
    """Affiche un résumé console."""
    print("\n" + "=" * 90)
    print("ANALYSE DE SENSIBILITÉ LOCALE — Δ vs baseline (config v5.2 GO retenue)")
    print("=" * 90)
    print(f"{'Configuration':<30} {'Détec.':>8} {'IoU20':>8} {'Best IoU':>10} {'Faux/cd':>10}")
    print("-" * 90)
    base = summary_df[summary_df["param"] == "baseline"].iloc[0]
    print(
        f"{'baseline':<30} "
        f"{base['detection_any_mean']:>8.3f} "
        f"{base['detection_iou20_mean']:>8.3f} "
        f"{base['best_iou_mean']:>10.3f} "
        f"{base['fausses_notif_mean']:>10.3f}"
    )
    print("-" * 90)
    for _, row in summary_df[summary_df["param"] != "baseline"].iterrows():
        print(
            f"{row['label']:<30} "
            f"{row['delta_detection_any']:+8.3f} "
            f"{row['delta_detection_iou20']:+8.3f} "
            f"{row['delta_best_iou']:+10.3f} "
            f"{row['delta_fausses_notif']:+10.3f}"
        )
    print("=" * 90)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse de sensibilité locale")
    parser.add_argument("--input-path", default="data/brut.csv")
    parser.add_argument("--n-cows", type=int, default=3)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=42)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    project_root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input_path)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if not input_path.exists():
        print(f"ERREUR : fichier introuvable : {input_path}")
        return 1

    df = run_sensitivity(
        input_path=input_path,
        n_cows=int(args.n_cows),
        n_seeds=int(args.n_seeds),
        seed_start=int(args.seed_start),
    )
    if df.empty:
        print("ERREUR : aucun résultat.")
        return 1

    out_dir = project_root / "data"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "sensitivity_local_results.csv", index=False)

    summary = build_summary(df)
    summary.to_csv(out_dir / "sensitivity_local_summary.csv", index=False)
    write_tornado_data(summary, out_dir / "sensitivity_local_tornado.csv")

    _print_table(summary)
    print(f"\nRésultats : {out_dir / 'sensitivity_local_results.csv'}")
    print(f"Résumé   : {out_dir / 'sensitivity_local_summary.csv'}")
    print(f"Tornade  : {out_dir / 'sensitivity_local_tornado.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
