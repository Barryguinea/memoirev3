import numpy as np
import pandas as pd

from core.hybrid_warning import HybridFusionConfig, apply_hybrid_warning
from core.io import COW, TIME, load_csv
from validation_hybrid.campaign import final_params, inject_profile
from core.pipeline import run_pipeline_one_cow


def test_pipeline_exposes_three_separate_outputs():
    raw = load_csv("data/brut.csv")
    raw[COW] = raw[COW].astype(str)
    cow = raw[raw[COW] == "8081"]
    out = run_pipeline_one_cow(cow, "8081", **final_params())
    required = {
        "behavioral_warning_episode",
        "instability_warning_episode",
        "hybrid_warning_episode",
        "hybrid_warning_type",
        "hybrid_warning_notification",
    }
    assert required.issubset(out.columns)
    assert set(out["hybrid_warning_type"].unique()).issubset(
        {"AUCUN", "HYPO", "INSTABILITE", "MIXTE", "SEQUENCE"}
    )
    assert out["hybrid_warning_fusion_mode"].eq("HIERARCHICAL").all()


def test_fusion_or_preserves_branch_identity():
    times = pd.date_range("2026-01-01", periods=8, freq="15min")
    frame = pd.DataFrame(
        {
            "T": times,
            "dataset_split": ["futur"] * 8,
            "coverage_pct": [100.0] * 8,
            "Steps_sum": [10.0] * 8,
            "Motion Index_sum": [10.0] * 8,
            "Transitions_sum": [4.0] * 8,
            "behavioral_warning_episode": [0, 1, 1, 0, 0, 1, 1, 0],
            "behavioral_warning_score": [0, .4, .4, 0, 0, .4, .4, 0],
        }
    )
    # La fonction recalcule la branche instabilité; on vérifie au minimum que
    # l'épisode HYPO existant est conservé par la fusion OR.
    out = apply_hybrid_warning(
        frame,
        interval="15T",
        fusion_config=HybridFusionConfig(mode="OR", cooldown_hours=1),
    )
    assert (out.loc[[1, 2, 5, 6], "hybrid_warning_episode"] == 1).all()
    assert (out.loc[[1, 2, 5, 6], "hybrid_warning_type"].isin(["HYPO", "MIXTE"])).all()


def test_hierarchical_fusion_keeps_instability_as_surveillance_only():
    times = pd.date_range("2026-01-01", periods=40, freq="15min")
    frame = pd.DataFrame(
        {
            "T": times,
            "dataset_split": ["futur"] * 40,
            "coverage_pct": [100.0] * 40,
            "Steps_sum": [10.0] * 40,
            "Motion Index_sum": [10.0] * 40,
            "Transitions_sum": [4.0] * 40,
            "Lying Time_sum": [8.0] * 40,
            "Standing Time_sum": [7.0] * 40,
            "behavioral_warning_episode": [0] * 40,
            "behavioral_warning_start": [0] * 40,
            "behavioral_warning_score": [0.0] * 40,
        }
    )
    out = apply_hybrid_warning(
        frame,
        interval="15T",
        fusion_config=HybridFusionConfig(mode="HIERARCHICAL"),
    )
    assert out["hybrid_warning_notification"].sum() == 0
    assert (out["hybrid_warning_priority"] <= 1).all()


def test_profiles_are_post_baseline_and_preserve_posture_total():
    raw = load_csv("data/brut.csv")
    raw[COW] = raw[COW].astype(str)
    cow = raw[raw[COW] == "8081"].copy()
    heldout = pd.Timestamp("2023-10-16")
    injected, event = inject_profile(
        cow,
        cow="8081",
        scenario="hypo_moderate",
        interval="15T",
        heldout_start=heldout,
        schedule_index=0,
    )
    assert bool(event["event_after_heldout"])
    mask = (pd.to_datetime(injected[TIME]) >= event["start"]) & (
        pd.to_datetime(injected[TIME]) <= event["end"]
    )
    original = cow.sort_values(TIME).reset_index(drop=True)
    assert np.allclose(
        injected.loc[mask, "Lying Time"] + injected.loc[mask, "Standing Time"],
        original.loc[mask, "Lying Time"] + original.loc[mask, "Standing Time"],
    )
