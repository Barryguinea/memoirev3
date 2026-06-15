#!/usr/bin/env python3
"""Baseline « Steps-only » (Alsaaod et al., 2012) comparée au pipeline complet.

Compare notre système (variante A = IF + règles métier) à une baseline
classique de la littérature pédométrique :

  E. Steps-only (Alsaaod) : alerte déclenchée lorsque le z-score robuste
     roulant du compte de pas franchit le seuil `-k_steps_z` (défaut 1.5,
     équivalent à une réduction de ~25-30 % du comptage de pas par
     rapport au baseline mobile 7 jours, le critère de Alsaaod 2012),
     suivi des mêmes règles métier de persistance / cooldown que
     la variante A pour une comparaison équitable (l'unique différence
     porte sur le détecteur d'anomalie : Isolation Forest multivariate
     vs seuil uni-variate sur les pas).

Les 90 cas synthétiques de l'étude d'ablation étendue sont re-exécutés
(3 vaches × 3 profils × 10 seeds) avec les mêmes données injectées,
permettant une comparaison appariée par case_id.

Tests statistiques :
  - McNemar exact binomial sur `detection_any`
  - Wilcoxon des rangs signés sur `best_iou` et `fausses_notif_cow_day`

Sorties :
  - data/alsaaod_baseline/results.csv             (E, 90 cas)
  - data/alsaaod_baseline/pair_comparison.csv     (A vs E sur 90 cas alignés)
  - data/alsaaod_baseline/pvalues_mcnemar.csv
  - data/alsaaod_baseline/pvalues_wilcoxon.csv
  - data/alsaaod_baseline/summary.md
  - data/alsaaod_baseline/manifest.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.alerts import apply_alert_logic  # noqa: E402
from core.config import (  # noqa: E402
    DEFAULT_ALERT_MIN, DEFAULT_COOLDOWN_HOURS, DEFAULT_COVERAGE_MIN_PCT,
    DEFAULT_INTERVAL, DEFAULT_MI_Z_HIGH_THR, DEFAULT_MIX_MODE,
    DEFAULT_MIX_RATE_THR, DEFAULT_PERSIST_HOURS, DEFAULT_Z_HIGH_THR,
    DEFAULT_Z_LOW_THR,
)
from core.io import COW, load_csv  # noqa: E402
from scripts.ablation_study import (  # noqa: E402
    _DEFAULT_EXPANDED_PROFILES, _false_notifs_per_cow_day, _prepare_features,
    _select_benchmark_cows,
)
from scripts.validation_eval import evaluate_injected_events  # noqa: E402
from scripts.validation_injection_raw import inject_manual_plan_raw  # noqa: E402


OUT_DIR = ROOT / "data" / "alsaaod_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ABL_CSV = ROOT / "data" / "ablation_expanded_results.csv"

K_STEPS_Z = 1.5        # Seuil z-score (équivalent ~25% réduction pas)
N_COWS = 3
N_SEEDS = 10
SEED_START = 42

ALERT_KWARGS = dict(
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
EVAL_KWARGS = dict(
    interval=DEFAULT_INTERVAL,
    persist_hours=DEFAULT_PERSIST_HOURS,
    alert_min=DEFAULT_ALERT_MIN,
    mix_rate_thr=DEFAULT_MIX_RATE_THR,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_variant_e_steps_only(features_df: pd.DataFrame, cow_id: str,
                                k_steps_z: float = K_STEPS_Z) -> pd.DataFrame:
    """Variante E : seuil sur Steps_sum_rrz (z-score roulant des pas).

    Équivalent 15-min de la méthode pédométrique Alsaaod (2012) :
    alerte quand le comptage de pas agrégé chute significativement
    par rapport au baseline roulant. Suivi des mêmes règles métier
    (persistance, cooldown, ...) que la variante A pour isoler la
    contribution du détecteur d'anomalie.
    """
    df = features_df.copy()
    # `Steps_sum_rrz` est calculé par build_interval_features (z-score
    # robuste roulant sur une fenêtre de 7 jours).
    if "Steps_sum_rrz" not in df.columns:
        raise RuntimeError("Feature Steps_sum_rrz manquante — pipeline incomplet.")
    # Anomalie ssi les pas chutent de `k` écart-types (seuil -k_steps_z).
    z = df["Steps_sum_rrz"].astype(float)
    df["if_anomaly_point"] = (z < -float(k_steps_z)).astype(int)
    df["if_pred"] = np.where(df["if_anomaly_point"] == 1, -1, 1)
    df["if_score"] = -z  # Plus grand = plus anormal (cohérent avec IF)
    df = apply_alert_logic(df, **ALERT_KWARGS)
    df[COW] = str(cow_id)
    return df


def run_alsaaod_baseline(base_raw: pd.DataFrame, *, n_cows: int = N_COWS,
                          n_seeds: int = N_SEEDS, seed_start: int = SEED_START,
                          k_steps_z: float = K_STEPS_Z) -> pd.DataFrame:
    """Exécute la variante E sur les 90 cas."""
    selected_cows = _select_benchmark_cows(
        base_raw, n_cows=n_cows,
        interval=DEFAULT_INTERVAL, persist_hours=DEFAULT_PERSIST_HOURS,
    )
    profiles = list(_DEFAULT_EXPANDED_PROFILES)
    print(f"Vaches sélectionnées : {selected_cows}")
    print(f"Profils : {profiles}")
    print(f"Seeds   : {seed_start}..{seed_start + n_seeds - 1}")

    rows = []
    total = len(selected_cows) * len(profiles) * n_seeds
    idx = 0
    for seed in range(seed_start, seed_start + n_seeds):
        for cow_id in selected_cows:
            for profile in profiles:
                idx += 1
                plan = [{
                    "cow": str(cow_id), "profile": profile,
                    "expected_detected": 1,
                    "event_id": f"ablation_{cow_id}_{profile}_{seed}",
                }]
                df_inj, events_df = inject_manual_plan_raw(
                    base_raw, plan=plan,
                    interval=DEFAULT_INTERVAL,
                    persist_hours=DEFAULT_PERSIST_HOURS,
                    seed=seed,
                )
                features_df = _prepare_features(df_inj, str(cow_id))
                pred_df = _run_variant_e_steps_only(features_df, str(cow_id),
                                                     k_steps_z=k_steps_z)
                ev = evaluate_injected_events(pred_df, events_df, **EVAL_KWARGS)
                if ev.empty:
                    continue
                rows.append({
                    "seed": int(seed), "cow": str(cow_id), "profile": profile,
                    "case_id": f"{cow_id}|{profile}|{seed}",
                    "variante": f"E. Steps seul (Alsaaod)",
                    "detection_any": float(ev["detected_any_overlap"].mean()),
                    "detection_iou20": float(ev["detected_iou20"].mean()),
                    "best_iou": float(ev["best_iou"].mean()),
                    "fausses_notif_cow_day": _false_notifs_per_cow_day(pred_df, events_df),
                })
                print(f"  {idx:3d}/{total}  seed={seed} cow={cow_id} profile={profile} "
                      f"det={rows[-1]['detection_any']:.0f} iou={rows[-1]['best_iou']:.3f} "
                      f"fn={rows[-1]['fausses_notif_cow_day']:.3f}")
    return pd.DataFrame(rows)


def mcnemar_test(y1: np.ndarray, y2: np.ndarray) -> dict:
    n11 = int(((y1 == 1) & (y2 == 1)).sum())
    n10 = int(((y1 == 1) & (y2 == 0)).sum())
    n01 = int(((y1 == 0) & (y2 == 1)).sum())
    n00 = int(((y1 == 0) & (y2 == 0)).sum())
    result = mcnemar([[n11, n10], [n01, n00]], exact=True)
    return {"n11": n11, "n10": n10, "n01": n01, "n00": n00,
            "discordants": n10 + n01,
            "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def wilcoxon_test(x1: np.ndarray, x2: np.ndarray) -> dict:
    diffs = x1 - x2
    mask = ~np.isnan(diffs) & (diffs != 0)
    n_eff = int(mask.sum())
    if n_eff < 1:
        return {"n_pairs": 0, "n_nonzero": 0, "median_diff": 0.0,
                "statistic": float("nan"), "p_value": float("nan")}
    try:
        stat, p = wilcoxon(x1[mask], x2[mask], alternative="two-sided",
                           zero_method="wilcox", correction=False, mode="auto")
    except ValueError:
        stat, p = float("nan"), float("nan")
    return {"n_pairs": int((~np.isnan(diffs)).sum()),
            "n_nonzero": n_eff,
            "median_diff": float(np.median(diffs[~np.isnan(diffs)])),
            "statistic": float(stat), "p_value": float(p)}


def main() -> None:
    print(f"Source : {ABL_CSV.relative_to(ROOT)}")
    if not ABL_CSV.exists():
        sys.exit(f"❌ {ABL_CSV} manquant — exécuter scripts/ablation_study.py d'abord.")

    # 1) Rejoindre les résultats de l'ablation A
    ablation_df = pd.read_csv(ABL_CSV)
    ablation_A = ablation_df[ablation_df["variante"] == "A. Complet"].copy()
    print(f"Variante A : {len(ablation_A)} cas")

    # 2) Exécuter variante E (Steps-only Alsaaod)
    print("\n=== Variante E : Steps-only (Alsaaod) ===")
    base_raw = load_csv(str(ROOT / "data" / "brut.csv"))
    results_E = run_alsaaod_baseline(base_raw)
    results_csv = OUT_DIR / "results.csv"
    results_E.to_csv(results_csv, index=False)
    print(f"\n✓ Variante E : {len(results_E)} cas → {results_csv.relative_to(ROOT)}")

    # 3) Fusion A + E pour comparaison appariée
    merged = ablation_A[["case_id", "detection_any", "best_iou",
                          "fausses_notif_cow_day"]].merge(
        results_E[["case_id", "detection_any", "best_iou",
                    "fausses_notif_cow_day"]],
        on="case_id", suffixes=("_A", "_E"),
    )
    pair_csv = OUT_DIR / "pair_comparison.csv"
    merged.to_csv(pair_csv, index=False)
    print(f"✓ Paires A vs E : {len(merged)} → {pair_csv.relative_to(ROOT)}")

    # 4) Tests statistiques appariés
    mc_out = mcnemar_test(merged["detection_any_A"].astype(int).to_numpy(),
                           merged["detection_any_E"].astype(int).to_numpy())
    mc_out["variante_1"] = "A. Complet"
    mc_out["variante_2"] = "E. Steps seul (Alsaaod)"
    mc_out["A_accuracy"] = float(merged["detection_any_A"].mean())
    mc_out["E_accuracy"] = float(merged["detection_any_E"].mean())
    mc_out["n_pairs"] = int(len(merged))
    pd.DataFrame([mc_out]).to_csv(OUT_DIR / "pvalues_mcnemar.csv", index=False)

    wil_rows = []
    for metric in ["best_iou", "fausses_notif_cow_day"]:
        row = wilcoxon_test(merged[f"{metric}_A"].to_numpy(),
                             merged[f"{metric}_E"].to_numpy())
        row["metric"] = metric
        row["variante_1"] = "A. Complet"
        row["variante_2"] = "E. Steps seul (Alsaaod)"
        row["A_mean"] = float(merged[f"{metric}_A"].mean())
        row["E_mean"] = float(merged[f"{metric}_E"].mean())
        wil_rows.append(row)
    pd.DataFrame(wil_rows).to_csv(OUT_DIR / "pvalues_wilcoxon.csv", index=False)

    # 5) Résumé Markdown
    def star(p):
        if np.isnan(p): return "n.a."
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "n.s."

    lines = [
        "# Baseline Alsaaod Steps-only vs pipeline complet\n",
        f"N = 90 cas appariés (3 vaches × 3 profils × 10 seeds).  ",
        f"Seuil Steps z-score : **-{K_STEPS_Z}** (équivalent ~25-30 % "
        f"de réduction des pas).  ",
        f"Mêmes règles métier de persistance que la variante A pour une "
        f"comparaison équitable.",
        "",
        "## Moyennes (90 cas)",
        "",
        "| Métrique | A. Complet | E. Steps seul (Alsaaod) | Δ (A − E) |",
        "|---|---|---|---|",
        f"| `detection_any`            | {mc_out['A_accuracy']:.1%} | "
        f"{mc_out['E_accuracy']:.1%} | {mc_out['A_accuracy']-mc_out['E_accuracy']:+.1%} |",
    ]
    for r in wil_rows:
        lines.append(
            f"| `{r['metric']}` | {r['A_mean']:.3f} | {r['E_mean']:.3f} | "
            f"{r['A_mean']-r['E_mean']:+.3f} |")

    lines.append("")
    lines.append("## Tests appariés (90 cas)")
    lines.append("")
    lines.append("| Test | Métrique | Statistique | p-value | Signif. |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| McNemar | detection_any | disc={mc_out['discordants']} "
        f"(A seul={mc_out['n10']}, E seul={mc_out['n01']}) | "
        f"{mc_out['p_value']:.4f} | {star(mc_out['p_value'])} |"
    )
    for r in wil_rows:
        lines.append(
            f"| Wilcoxon | {r['metric']} | W={r['statistic']:.1f} "
            f"(n_nz={r['n_nonzero']}) | {r['p_value']:.4f} | "
            f"{star(r['p_value'])} |"
        )

    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    # 6) Manifest
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "input": {str(ABL_CSV.relative_to(ROOT)): _sha256(ABL_CSV)},
        "outputs": {str(f.relative_to(ROOT)): _sha256(f)
                     for f in OUT_DIR.glob("*") if f.is_file()},
        "params": {
            "k_steps_z": K_STEPS_Z, "n_cows": N_COWS, "n_seeds": N_SEEDS,
            "seed_start": SEED_START, "alert_rules": "identical to variante A",
            "baseline_reference": "Alsaaod et al. (2012) Electronic Detection of Lameness",
        },
        "metrics": {
            "A_accuracy": mc_out["A_accuracy"], "E_accuracy": mc_out["E_accuracy"],
            "mcnemar_p": mc_out["p_value"],
            "wilcoxon_iou_p": wil_rows[0]["p_value"],
            "wilcoxon_fn_p": wil_rows[1]["p_value"],
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                             encoding="utf-8")

    print()
    print((OUT_DIR / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
