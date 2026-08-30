import pandas as pd

from ui.presentation import (
    build_filterable_table,
    build_reason_row,
    clinical_actions,
    clinical_payload,
    clinical_short,
    compact_feature_title,
    filter_detection_table,
    level_priority,
    normalize_alert_level,
    plot_options_for_mode,
    plain_feature_name,
    pretty_feature_name,
)


def test_pretty_feature_name_formats_rrz_labels():
    assert pretty_feature_name("Motion Index_sum_log_rrz") == "Activité - Motion Index (log) - écart robuste glissant"
    assert pretty_feature_name("Steps_mean") == "Activité - Pas - moyenne"


def test_compact_feature_title_shortens_and_rewrites_suffixes():
    out = compact_feature_title("Motion Index_sum_log_rrz", max_len=24)
    assert "(rrz)" in out
    assert len(out) <= 24


def test_plot_options_for_mode_returns_expected_defaults():
    opts_diag, defaults_diag = plot_options_for_mode("Variables centrées (rrz)")
    assert "Motion Index_sum_log_rrz" in opts_diag
    assert defaults_diag == ["Motion Index_sum_log_rrz", "Steps_sum_rrz", "Lying Time_sum_rrz"]

    opts_std, defaults_std = plot_options_for_mode("Standard")
    assert "Motion Index_sum" in opts_std
    assert defaults_std == ["Motion Index_sum", "Steps_sum", "Lying Time_sum"]


def test_alert_level_helpers_normalize_and_rank():
    assert normalize_alert_level("critical") == "critique"
    assert normalize_alert_level("SUSPECT") == "suspect"
    assert level_priority("critique") > level_priority("probable") > level_priority("normal")


def test_clinical_helpers_provide_fallbacks_and_short_summary():
    kb = {
        "levels": {
            "probable": {
                "summary": "Intervention rapide.",
                "actions_0_24h": ["Examiner le pied"],
            }
        }
    }
    payload = clinical_payload("probable", kb)
    assert payload["summary"] == "Intervention rapide."
    assert clinical_actions(payload) == ["Examiner le pied"]
    assert clinical_short("probable", kb) == "Intervention rapide. Action: Examiner le pied"

    fallback = clinical_payload("inconnu", kb)
    assert "summary" in fallback


def test_build_reason_row_and_filterable_table_behaviors():
    row = pd.Series(
        {
            "if_anom_k": 3,
            "K": 6,
            "anom_rate_k": 0.5,
            "fam_activity": 1,
            "mi_spike_k": 2,
            "coherence_boiterie": 1,
            "in_cooldown": 0,
            "is_critique": 1,
        }
    )
    reason = build_reason_row(row)
    assert "Persistance IF: 3/6 anomalies" in reason
    assert "Taux anomalies=50.0%" in reason
    assert "Niveau critique" in reason

    df = pd.DataFrame(
        [
            {
                "T": pd.Timestamp("2024-01-01 01:00:00"),
                "hybrid_warning_episode": 0,
                "behavioral_warning_candidate": 1,
                "if_score": -0.2,
                "pred_lameness_episode": 1,
                "extra": 1,
            },
            {
                "T": pd.Timestamp("2024-01-01 03:00:00"),
                "hybrid_warning_episode": 1,
                "behavioral_warning_candidate": 0,
                "if_score": -0.5,
                "pred_lameness_episode": 0,
                "extra": 2,
            },
        ]
    )
    out = build_filterable_table(df)
    assert list(out.columns) == [
        "T",
        "hybrid_warning_episode",
        "behavioral_warning_candidate",
        "if_score",
    ]
    assert out.iloc[0]["T"] == pd.Timestamp("2024-01-01 03:00:00")
    assert "pred_lameness_episode" not in out


def test_filter_detection_table_uses_current_pipeline_outputs() -> None:
    frame = pd.DataFrame(
        {
            "T": pd.date_range("2024-01-01", periods=4, freq="15min"),
            "hybrid_warning_episode": [0, 1, 1, 0],
            "hybrid_warning_notification": [0, 1, 0, 0],
            "behavioral_warning_candidate": [1, 1, 0, 0],
            "if_anomaly_point": [0, 0, 1, 1],
        }
    )

    assert len(filter_detection_table(frame, only_episode=True)) == 2
    assert len(filter_detection_table(frame, only_notification=True)) == 1
    assert len(filter_detection_table(frame, only_hypo_candidate=True)) == 2
    assert len(filter_detection_table(frame, only_if=True)) == 2
    combined = filter_detection_table(
        frame,
        only_episode=True,
        only_notification=True,
        only_hypo_candidate=True,
    )
    assert list(combined.index) == [1]


def test_plain_feature_name_donne_le_nom_du_manuscrit():
    """Le manuscrit ecrit « Motion Index », pas la forme a trois segments.

    ``pretty_feature_name`` sert a nommer les traces d'un graphique, ou la
    categorie et l'agregation aident a distinguer des variantes voisines. Dans
    une cle de lecture destinee a etre lue, elle produisait « Activite - Motion
    Index - somme », que le lecteur ne rattache a rien.
    """
    assert plain_feature_name("Motion Index_sum") == "Motion Index"
    assert plain_feature_name("Steps_sum_rrz") == "Pas"
    assert plain_feature_name("Lying Time_sum") == "Temps couché"
    assert plain_feature_name(None) == "Variable"
