#!/usr/bin/env python3
"""Compute paired IF/LOF comparison summaries for the memoire.

This script reuses existing experimental CSV artifacts and produces:
  - data/if_lof_stats/if_lof_paired_ablation_expanded.csv
  - data/if_lof_stats/if_lof_paired_v4.csv
  - data/if_lof_stats/if_lof_paired_summary.md

The confidence intervals are paired percentile bootstrap intervals with a fixed
random seed for deterministic regeneration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "if_lof_stats"
BOOT_SEED = 42
N_BOOT = 20_000


def paired_bootstrap_mean_diff(values: np.ndarray, *, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")

    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sample = rng.choice(values, size=len(values), replace=True)
        boots[i] = sample.mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def compute_expanded() -> pd.DataFrame:
    rng = np.random.default_rng(BOOT_SEED)
    df = pd.read_csv(DATA / "ablation_expanded_results.csv")
    wide = df.pivot(
        index="case_id",
        columns="variante",
        values=[
            "detection_any",
            "detection_iou20",
            "best_iou",
            "fausses_notif_cow_day",
        ],
    )

    lof_col = "D. LOF + regles"
    rows: list[dict[str, object]] = []
    labels = {
        "detection_any": "Detection any",
        "detection_iou20": "IoU20",
        "best_iou": "Best IoU",
        "fausses_notif_cow_day": "Fausses notif/cow-day",
    }
    for metric, metric_label in labels.items():
        if_mean = wide[(metric, "A. Complet")].mean()
        lof_mean = wide[(metric, lof_col)].mean()
        diffs = wide[(metric, lof_col)] - wide[(metric, "A. Complet")]
        diff, lo, hi = paired_bootstrap_mean_diff(diffs.to_numpy(), rng=rng)
        rows.append(
            {
                "dataset": "ablation_expanded",
                "comparison": "D - A",
                "metric": metric,
                "metric_label": metric_label,
                "scenario": "all_profiles",
                "n_pairs": int(diffs.notna().sum()),
                "if_mean": float(if_mean),
                "lof_mean": float(lof_mean),
                "diff_lof_minus_if": diff,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    return pd.DataFrame(rows)


def compute_v4() -> pd.DataFrame:
    rng = np.random.default_rng(BOOT_SEED)
    if_df = pd.read_csv(
        DATA / "validation_robuste" / "robust_validation_20260221_162027" / "robust_runs.csv"
    )
    lof_df = pd.read_csv(DATA / "exploration_lof" / "lof_v4_runs.csv")

    key = ["contamination", "persist_hours", "alert_min", "mix_mode", "mix_rate_thr", "seed", "scenario"]
    lof_small = lof_df.rename(
        columns={
            "detection_any": "detectable_recall_any",
            "detection_iou20": "detectable_recall_iou20",
            "best_iou": "detectable_iou_mean",
            "false_notif_cd": "false_notif_per_cow_day",
        }
    )

    merged = if_df.merge(
        lof_small[key + ["detectable_recall_any", "detectable_recall_iou20", "detectable_iou_mean", "false_notif_per_cow_day"]],
        on=key,
        suffixes=("_if", "_lof"),
    )

    rows: list[dict[str, object]] = []
    metric_labels = {
        "detectable_recall_any": "Detection any",
        "detectable_recall_iou20": "IoU20",
        "detectable_iou_mean": "Best IoU",
        "false_notif_per_cow_day": "Fausses notif/cow-day",
    }
    for scenario in ["detectable_strong", "detectable_borderline"]:
        subset = merged[merged["scenario"] == scenario].copy()
        for metric, metric_label in metric_labels.items():
            if_mean = subset[f"{metric}_if"].mean()
            lof_mean = subset[f"{metric}_lof"].mean()
            diffs = subset[f"{metric}_lof"] - subset[f"{metric}_if"]
            diff, lo, hi = paired_bootstrap_mean_diff(diffs.to_numpy(), rng=rng)
            rows.append(
                {
                    "dataset": "v4_paired",
                    "comparison": "LOF - IF",
                    "metric": metric,
                    "metric_label": metric_label,
                    "scenario": scenario,
                    "n_pairs": int(diffs.notna().sum()),
                    "if_mean": float(if_mean),
                    "lof_mean": float(lof_mean),
                    "diff_lof_minus_if": diff,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    return pd.DataFrame(rows)


def fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}\\%"


def fmt_delta(value: float, low: float, high: float, *, pct: bool) -> str:
    if pct:
        return f"{100 * value:+.1f} pts [{100 * low:+.1f}; {100 * high:+.1f}]"
    return f"{value:+.3f} [{low:+.3f}; {high:+.3f}]"


def build_summary(expanded: pd.DataFrame, v4: pd.DataFrame) -> str:
    lines = [
        "# Comparaison appariée IF vs LOF",
        "",
        "Bootstrap apparié percentile à 95 %, 20 000 rééchantillonnages, seed 42.",
        "",
        "## Ablation étendue (D = LOF + règles, A = IF + règles)",
    ]
    for _, row in expanded.iterrows():
        pct = row["metric"] in {"detection_any", "detection_iou20", "fausses_notif_cow_day"}
        if row["metric"] == "best_iou":
            left = f"IF={row['if_mean']:.3f}, LOF={row['lof_mean']:.3f}"
        else:
            left = f"IF={fmt_pct(row['if_mean'])}, LOF={fmt_pct(row['lof_mean'])}"
        lines.append(
            f"- {row['metric_label']}: {left}; Δ(LOF-IF)={fmt_delta(row['diff_lof_minus_if'], row['ci95_low'], row['ci95_high'], pct=pct)}"
        )
    lines += ["", "## Validation exploratoire v4"]
    for scenario, title in [
        ("detectable_strong", "Cas forts"),
        ("detectable_borderline", "Cas borderline"),
    ]:
        lines.append(f"### {title}")
        sub = v4[v4["scenario"] == scenario]
        for _, row in sub.iterrows():
            pct = row["metric"] in {"detectable_recall_any", "detectable_recall_iou20", "false_notif_per_cow_day"}
            if row["metric"] == "detectable_iou_mean":
                left = f"IF={row['if_mean']:.3f}, LOF={row['lof_mean']:.3f}"
            else:
                left = f"IF={fmt_pct(row['if_mean'])}, LOF={fmt_pct(row['lof_mean'])}"
            lines.append(
                f"- {row['metric_label']}: {left}; Δ(LOF-IF)={fmt_delta(row['diff_lof_minus_if'], row['ci95_low'], row['ci95_high'], pct=pct)}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expanded = compute_expanded()
    v4 = compute_v4()

    expanded.to_csv(OUT / "if_lof_paired_ablation_expanded.csv", index=False)
    v4.to_csv(OUT / "if_lof_paired_v4.csv", index=False)
    (OUT / "if_lof_paired_summary.md").write_text(build_summary(expanded, v4), encoding="utf-8")


if __name__ == "__main__":
    main()
