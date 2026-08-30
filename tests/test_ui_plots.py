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
    assert fig.layout.height == 240
    assert len(fig.data) == 2  # line + lameness markers

    missing = build_small_fig(df, "does_not_exist")
    assert len(missing.data) == 0


def test_build_small_fig_porte_des_graduations_sur_les_deux_axes():
    """Une vignette sans graduation ne dit ni la periode ni l'amplitude.

    Les axes etaient masques : la vue servait au survol dans l'application, mais
    reproduite dans un document elle ne se lisait plus.
    """
    fig = build_small_fig(_sample_plot_df(), "Motion Index_sum")
    assert fig.layout.xaxis.showticklabels
    assert fig.layout.yaxis.showticklabels


def test_build_small_fig_accepte_un_maximum_commun():
    """Sans maximum impose, chaque vignette se cale sur le sien.

    Deux vaches d'amplitudes tres differentes paraissaient alors comparables.
    """
    libre = build_small_fig(_sample_plot_df(), "Motion Index_sum")
    assert libre.layout.yaxis.range is None

    borne = build_small_fig(_sample_plot_df(), "Motion Index_sum", y_max=1234.0)
    assert borne.layout.yaxis.range == (0, 1234.0)


def test_build_small_fig_ne_montre_la_legende_que_sur_demande():
    """Une legende par vignette repeterait six fois la meme cle."""
    muette = build_small_fig(_sample_plot_df(), "Motion Index_sum")
    assert all(not trace.showlegend for trace in muette.data)

    parlante = build_small_fig(_sample_plot_df(), "Motion Index_sum", show_legend=True)
    assert all(trace.showlegend for trace in parlante.data)
    assert {trace.name for trace in parlante.data} == {"Signal", "Épisode d'alerte"}
