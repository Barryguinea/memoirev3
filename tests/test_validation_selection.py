import pandas as pd

from core.io import COW, TIME
from scripts.validation_selection import (
    _manual_injection_plan,
    extract_detected_episodes,
    pick_cows_for_injection,
    summarize_from_predictions,
)


def test_summarize_from_predictions_and_extract_detected_episodes():
    df = pd.DataFrame(
        [
            {COW: "8081", TIME: pd.Timestamp("2024-01-01 00:00"), "if_anomaly_point": 1, "pred_lameness_episode": 0, "pred_lameness_start": 0, "notif_lameness": 0, "coverage_pct": 80},
            {COW: "8081", TIME: pd.Timestamp("2024-01-01 01:00"), "if_anomaly_point": 1, "pred_lameness_episode": 1, "pred_lameness_start": 1, "notif_lameness": 1, "coverage_pct": 90, "if_anom_k": 3, "anom_rate_k": 0.5, "coherence_boiterie": 1.0},
            {COW: "8081", TIME: pd.Timestamp("2024-01-01 02:00"), "if_anomaly_point": 0, "pred_lameness_episode": 1, "pred_lameness_start": 0, "notif_lameness": 0, "coverage_pct": 85, "if_anom_k": 2, "anom_rate_k": 0.4, "coherence_boiterie": 0.8},
            {COW: "8154", TIME: pd.Timestamp("2024-01-01 00:00"), "if_anomaly_point": 0, "pred_lameness_episode": 0, "pred_lameness_start": 0, "notif_lameness": 0, "coverage_pct": 95},
        ]
    )

    summary = summarize_from_predictions(df)
    assert summary.iloc[0][COW] == "8081"
    assert int(summary.iloc[0]["lameness_notifs"]) == 1

    episodes = extract_detected_episodes(df)
    assert len(episodes) == 1
    assert episodes.iloc[0]["cow"] == "8081"
    assert int(episodes.iloc[0]["duration_bins"]) == 2


def test_pick_cows_for_injection_and_manual_plan_keep_distinct_roles():
    summary_df = pd.DataFrame(
        [
            {COW: "8081", "lameness_starts": 0, "lameness_notifs": 0, "if_anomaly_points": 3, "injectability_score": 0.9, "injectability_ok": 1, "series_length_score": 0.9, "long_series_ok": 1},
            {COW: "8154", "lameness_starts": 0, "lameness_notifs": 0, "if_anomaly_points": 1, "injectability_score": 0.8, "injectability_ok": 1, "series_length_score": 0.8, "long_series_ok": 1},
        ]
    )

    cow_detect, cow_hidden = pick_cows_for_injection(summary_df)
    assert cow_detect in {"8081", "8154"}
    assert cow_hidden in {"8081", "8154"}
    assert cow_detect != cow_hidden

    plan = _manual_injection_plan(
        summary_df,
        detect_cow=cow_detect,
        hidden_cow=cow_hidden,
        n_detectable=2,
        variant="v2_1",
    )
    assert len(plan) == 3
    assert [int(x["expected_detected"]) for x in plan] == [1, 1, 0]
    assert plan[-1]["cow"] != plan[0]["cow"]
