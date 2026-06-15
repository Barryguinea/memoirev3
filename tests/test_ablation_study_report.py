from pathlib import Path

import pandas as pd

from core.io import COW, TIME
from scripts.ablation_study import _build_summary, _paired_counts, _select_benchmark_cows, _write_report


def _sample_ablation_df() -> pd.DataFrame:
    rows = []
    for seed in [42, 43]:
        rows.extend(
            [
                {
                    "seed": seed,
                    "variante": "A. Complet",
                    "detection_any": 0.5 if seed == 42 else 1.0,
                    "detection_iou20": 0.0,
                    "best_iou": 0.03 if seed == 42 else 0.04,
                    "fausses_notif_cow_day": 0.13,
                },
                {
                    "seed": seed,
                    "variante": "B. Sans IF",
                    "detection_any": 1.0,
                    "detection_iou20": 0.0,
                    "best_iou": 0.05,
                    "fausses_notif_cow_day": 0.43,
                },
                {
                    "seed": seed,
                    "variante": "C. Sans regles",
                    "detection_any": 1.0,
                    "detection_iou20": 0.0,
                    "best_iou": 0.07,
                    "fausses_notif_cow_day": float("nan"),
                },
                {
                    "seed": seed,
                    "variante": "D. LOF + regles",
                    "detection_any": 1.0,
                    "detection_iou20": 0.0,
                    "best_iou": 0.06,
                    "fausses_notif_cow_day": 0.05,
                },
            ]
        )
    return pd.DataFrame(rows)


def _sample_expanded_ablation_df() -> pd.DataFrame:
    rows = []
    for seed in [42, 43]:
        for cow in ["8081", "9001"]:
            for profile in ["detectable_strong", "detectable_borderline"]:
                rows.extend(
                    [
                        {
                            "seed": seed,
                            "cow": cow,
                            "profile": profile,
                            "case_id": f"{cow}|{profile}|{seed}",
                            "variante": "A. Complet",
                            "detection_any": 0.0 if (seed == 42 and cow == "8081") else 1.0,
                            "detection_iou20": 0.0,
                            "best_iou": 0.02,
                            "fausses_notif_cow_day": 0.12,
                        },
                        {
                            "seed": seed,
                            "cow": cow,
                            "profile": profile,
                            "case_id": f"{cow}|{profile}|{seed}",
                            "variante": "D. LOF + regles",
                            "detection_any": 1.0,
                            "detection_iou20": 0.0,
                            "best_iou": 0.05,
                            "fausses_notif_cow_day": 0.04,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def test_write_report_mentions_four_variants_and_iou20_note(tmp_path: Path) -> None:
    report_path = tmp_path / "ablation_report.txt"
    _write_report(_sample_ablation_df(), report_path)
    text = report_path.read_text(encoding="utf-8")

    assert "4 variantes" in text
    assert "IoU20 (seuil strict)" in text
    assert "Aucune variante n'atteint IoU >= 0.20" in text


def test_write_report_aligns_conclusion_when_lof_beats_if(tmp_path: Path) -> None:
    report_path = tmp_path / "ablation_report.txt"
    _write_report(_sample_ablation_df(), report_path)
    text = report_path.read_text(encoding="utf-8")

    assert "D dépasse A sur les deux métriques observées" in text
    assert "Seed-par-seed" in text
    assert "micro-benchmark local" in text
    assert "LOF comme alternative non-supervisée ne surpasse pas IF." not in text


def test_build_summary_includes_mean_and_std_columns() -> None:
    summary = _build_summary(_sample_ablation_df())

    expected_columns = {
        "variante",
        "n_runs",
        "n_seeds",
        "detection_any_mean",
        "detection_any_std",
        "detection_iou20_mean",
        "detection_iou20_std",
        "detection_iou10_mean",
        "detection_iou10_std",
        "best_iou_mean",
        "best_iou_std",
        "fausses_notif_cow_day_mean",
        "fausses_notif_cow_day_std",
    }

    assert expected_columns.issubset(summary.columns)
    a_row = summary.loc[summary["variante"] == "A. Complet"].iloc[0]
    assert a_row["n_seeds"] == 2
    assert a_row["n_runs"] == 2
    assert a_row["detection_iou10_mean"] == 0.0
    assert a_row["detection_any_std"] > 0.0


def test_build_summary_tracks_cows_and_profiles_for_expanded_benchmark() -> None:
    summary = _build_summary(_sample_expanded_ablation_df())

    a_row = summary.loc[summary["variante"] == "A. Complet"].iloc[0]
    assert a_row["n_runs"] == 8
    assert a_row["n_cows"] == 2
    assert a_row["n_profiles"] == 2


def test_paired_counts_uses_composite_keys_for_expanded_benchmark() -> None:
    counts = _paired_counts(_sample_expanded_ablation_df(), "A. Complet", "D. LOF + regles")

    assert counts["n_common"] == 8
    assert counts["det_ge_and_fn_lt"] == 8
    assert counts["det_gt_and_fn_lt"] == 2


def test_select_benchmark_cows_keeps_only_eligible_rows() -> None:
    df = pd.DataFrame(
        {
            COW: ["8081"] * 60 + ["9001"] * 54 + ["short"] * 20,
            TIME: pd.date_range("2026-01-01", periods=134, freq="15min"),
        }
    )

    selected = _select_benchmark_cows(df, n_cows=3)

    assert selected == ["8081", "9001"]
