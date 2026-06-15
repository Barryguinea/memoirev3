import pandas as pd

from ui.plots import build_multi_panel_figure, build_small_fig


def _sample_plot_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "T": pd.date_range("2024-01-01", periods=4, freq="h"),
            "Motion Index_sum": [1.0, 2.0, 1.5, 2.5],
            "if_anomaly_point": [0, 1, 0, 0],
            "pred_lameness_episode": [0, 1, 1, 0],
            "notif_lameness": [0, 0, 1, 0],
        }
    )


def test_build_multi_panel_figure_adds_expected_traces_and_layout():
    df = _sample_plot_df()
    fig = build_multi_panel_figure(df, ["Motion Index_sum"], title="Test")

    assert fig.layout.title.text == "Test"
    assert fig.layout.height == 300
    assert len(fig.data) == 4  # line + anomaly + lameness + notification


def test_build_small_fig_returns_compact_figure_and_handles_missing_col():
    df = _sample_plot_df()
    fig = build_small_fig(df, "Motion Index_sum")
    assert fig.layout.height == 180
    assert len(fig.data) == 2  # line + lameness markers

    missing = build_small_fig(df, "does_not_exist")
    assert len(missing.data) == 0
