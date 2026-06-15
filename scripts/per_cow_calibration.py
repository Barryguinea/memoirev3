# ruff: noqa: E402
"""Test exploratoire d'une calibration du seuil par vache.

L'hypothese testee est qu'une normalisation par la variabilite du score de
baseline pourrait homogeneiser la charge de fond entre individus. Le facteur
est estime UNIQUEMENT sur la baseline (aucune fuite), puis applique au seuil de
score de la vache. Le resultat peut etre positif, nul ou negatif; le script ne
selectionne pas de nouvelle configuration.

Usage:
    python scripts/per_cow_calibration.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.early_warning import EarlyWarningConfig
from core.io import COW, load_csv
from core.pipeline import run_pipeline_one_cow
from revalidation.campaign import final_params, run_clean_campaign

OUT_DIR = ROOT / "data/revalidation/hypo_module"
SUMMARY_OUT = OUT_DIR / "per_cow_calibration.csv"
DETAIL_OUT = OUT_DIR / "per_cow_calibration_by_cow.csv"
GRAD = {"gradual_mild", "gradual_moderate", "gradual_marked"}
BASE_SCORE_THR = EarlyWarningConfig().score_threshold
CLAMP = (0.6, 1.8)


def _per_cow_background(df: pd.DataFrame) -> dict[str, float]:
    rates: dict[str, float] = {}
    for cow, grp in df.groupby("cow"):
        row = grp.iloc[0]
        days = float(row["cow_monitoring_days"])
        notifs = float(row["cow_background_notifs"])
        rates[str(cow)] = notifs / days if days > 0 else float("nan")
    return rates


def _detection_rates(df: pd.DataFrame) -> dict[str, float]:
    grad = df[df["scenario"].isin(GRAD)]
    cov = (pd.to_numeric(grad["episode_overlap"], errors="coerce") >= 1).mean()
    new = (pd.to_numeric(grad["novel_start_count"], errors="coerce") >= 1).mean()
    iou = (pd.to_numeric(grad["detected_iou20"], errors="coerce") >= 1).mean()
    return {"couverture": float(cov), "nouveau_depart": float(new), "iou20": float(iou)}


def _cv(values: list[float]) -> float:
    arr = np.array([v for v in values if np.isfinite(v)])
    return float(arr.std() / arr.mean()) if arr.mean() else float("nan")


def _baseline_score_std(raw_cow: pd.DataFrame, cow: str, params: dict) -> float:
    pred = run_pipeline_one_cow(raw_cow, cow, **params, warning_config=None)
    base = pred[pred["dataset_split"].astype(str) == "baseline"]
    score = pd.to_numeric(base["behavioral_warning_score"], errors="coerce").dropna()
    return float(score.std()) if len(score) > 1 else float("nan")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = final_params()

    # 1) Reference (seuil global identique pour toutes les vaches).
    ref = run_clean_campaign(warning_config=None, verbose=False)
    cows = sorted(ref["cow"].astype(str).unique())
    ref_bg = _per_cow_background(ref)
    ref_det = _detection_rates(ref)

    # 2) Facteur de calibration par vache, estime sur la baseline.
    df_all = load_csv("data/brut.csv")
    df_all[COW] = df_all[COW].astype(str)
    stds = {c: _baseline_score_std(df_all[df_all[COW] == c], c, params) for c in cows}
    med = float(np.nanmedian(list(stds.values())))
    factors = {c: float(np.clip((stds[c] / med) if med else 1.0, *CLAMP)) for c in cows}

    # 3) Re-run calibre, une vache a la fois (seuil de score mis a l'echelle).
    calib_rows = []
    for c in cows:
        cfg = EarlyWarningConfig(score_threshold=BASE_SCORE_THR * factors[c])
        out = run_clean_campaign(cows=[c], warning_config=cfg, verbose=False)
        calib_rows.append(out)
    calib = pd.concat(calib_rows, ignore_index=True)
    cal_bg = _per_cow_background(calib)
    cal_det = _detection_rates(calib)

    # 4) Synthese comparative.
    ref_rates = [ref_bg[c] for c in cows]
    cal_rates = [cal_bg[c] for c in cows]
    summary = [
        ("Fond moyen / vache-jour", np.nanmean(ref_rates), np.nanmean(cal_rates)),
        ("Fond etendue min", np.nanmin(ref_rates), np.nanmin(cal_rates)),
        ("Fond etendue max", np.nanmax(ref_rates), np.nanmax(cal_rates)),
        ("Fond CV inter-vache", _cv(ref_rates), _cv(cal_rates)),
        ("Couverture (graduels)", ref_det["couverture"], cal_det["couverture"]),
        ("Nouveau depart (graduels)", ref_det["nouveau_depart"], cal_det["nouveau_depart"]),
        ("IoU20 (graduels)", ref_det["iou20"], cal_det["iou20"]),
    ]

    with open(SUMMARY_OUT, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["indicateur", "reference", "calibre", "difference"])
        for name, r, cval in summary:
            w.writerow(
                [name, round(float(r), 4), round(float(cval), 4), round(float(cval - r), 4)]
            )

    detail_rows = []
    for cow in cows:
        detail_rows.append(
            {
                "cow": cow,
                "baseline_score_std": round(float(stds[cow]), 6),
                "calibration_factor": round(float(factors[cow]), 6),
                "reference_score_threshold": round(float(BASE_SCORE_THR), 6),
                "calibrated_score_threshold": round(float(BASE_SCORE_THR * factors[cow]), 6),
                "reference_background_per_cow_day": round(float(ref_bg[cow]), 6),
                "calibrated_background_per_cow_day": round(float(cal_bg[cow]), 6),
            }
        )
    pd.DataFrame(detail_rows).to_csv(DETAIL_OUT, index=False)

    print(f"{'indicateur':30s} {'reference':>12s} {'calibre':>12s}")
    for name, r, cval in summary:
        print(f"{name:30s} {r:12.4f} {cval:12.4f}")
    print(f"\nFacteurs par vache (min/med/max): "
          f"{min(factors.values()):.2f} / {np.median(list(factors.values())):.2f} / {max(factors.values()):.2f}")
    print(f"Ecrit: {SUMMARY_OUT}")
    print(f"Ecrit: {DETAIL_OUT}")


if __name__ == "__main__":
    main()
