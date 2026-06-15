import pandas as pd

from ui.clinical_ui import build_episode_why_table, load_clinical_kb


def test_build_episode_why_table_adds_reason_and_clinical_guidance():
    df = pd.DataFrame(
        [
            {
                "T": pd.Timestamp("2024-01-01 02:00:00"),
                "pred_lameness_start": 1,
                "alert_level": "probable",
                "if_score": -0.5,
                "if_anom_k": 3,
                "K": 6,
                "anom_rate_k": 0.5,
                "fam_activity": 1,
                "mi_spike_k": 2,
                "coherence_boiterie": 1,
                "in_cooldown": 0,
                "is_critique": 0,
            }
        ]
    )
    kb = {
        "levels": {
            "probable": {
                "summary": "Intervention rapide.",
                "actions_0_24h": ["Examiner le pied"],
            }
        }
    }

    out = build_episode_why_table(df, kb=kb)
    assert list(out.columns)[:3] == ["T", "alert_level", "if_score"]
    assert "reason" in out.columns
    assert "consigne_clinique" in out.columns
    assert "Intervention rapide." in str(out.iloc[0]["consigne_clinique"])


def test_load_clinical_kb_returns_payload_with_levels():
    kb = load_clinical_kb()
    assert isinstance(kb, dict)
    assert "levels" in kb
    assert isinstance(kb["levels"], dict)
