from pathlib import Path


_NB_ROOT = Path(__file__).resolve().parents[1] / "notebooks"

NOTEBOOKS = [
    _NB_ROOT / "revalidation_complete.ipynb",
]


def test_notebooks_do_not_depend_on_core_default_constants():
    for nb in NOTEBOOKS:
        text = nb.read_text(encoding="utf-8")
        assert "DEFAULT_" not in text, f"{nb.name} still references global defaults"


def test_notebooks_keep_pinned_params_markers():
    # Markers communs a tous les notebooks
    common_markers = [
        "Parametres figes pour reproductibilite",
        "interval='15T'",
        "window_baseline=24",
        "contamination=0.06",
        "alert_min=2",
        "mix_mode='MIX'",
        "mix_rate_thr=0.24",
        "mi_z_high_thr=2.2",
        "cooldown_hours=12",
        "coverage_min_pct=25.0",
    ]
    for nb in NOTEBOOKS:
        text = nb.read_text(encoding="utf-8")
        for marker in common_markers:
            assert marker in text, f"{nb.name} missing marker: {marker}"
        assert "persist_hours=7" in text, f"{nb.name} missing pinned persist_hours"
        assert "baseline_ratio=0.60" in text, f"{nb.name} missing baseline ratio"
        assert "random_state=42" in text, f"{nb.name} missing random state"
        assert "PRIMARY_SEEDS = [11]" in text, f"{nb.name} must use one primary seed"
