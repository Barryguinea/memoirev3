"""Tableau de bord Streamlit d'alerte comportementale bidirectionnelle.

Fournit quatre onglets : analyse individuelle, classement du troupeau par severite,
comparaison journaliere inter-animaux (heatmap) et export CSV.  Tous les parametres
du pipeline sont exposes dans la barre laterale et l'analyse s'execute a la volee
via le pipeline principal.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import hashlib
import uuid
from pathlib import Path
from typing import Any

from core.io import load_csv, COW
from core.pipeline import run_pipeline_one_cow
from ui.panels import (
    render_tab_daily_comparison,
    render_tab_export,
    render_tab_herd,
    render_tab_individual,
)
from ui.presentation import (
    plot_options_for_mode,
)
from core.config import (
    DEFAULT_INTERVAL,
    DEFAULT_WINDOW_BASELINE,
    DEFAULT_CONTAMINATION,
    DEFAULT_BASELINE_RATIO,
    DEFAULT_RANDOM_STATE,
    DEFAULT_PERSIST_HOURS,
    DEFAULT_ALERT_MIN,
    DEFAULT_MIX_MODE,
    DEFAULT_MIX_RATE_THR,
    DEFAULT_Z_LOW_THR,
    DEFAULT_Z_HIGH_THR,
    DEFAULT_COOLDOWN_HOURS,
    DEFAULT_MI_Z_HIGH_THR,
    DEFAULT_COVERAGE_MIN_PCT,
)

# ============================================================
# CONFIGURATION ET STYLE
# ============================================================
st.set_page_config(page_title="Alerte comportementale bidirectionnelle", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "brut.csv"

if "_upload_widget_nonce" not in st.session_state:
    st.session_state["_upload_widget_nonce"] = 0

with st.sidebar:
    if st.button("Recalculer (vider le cache)"):
        st.cache_data.clear()
        st.session_state["_upload_widget_nonce"] += 1
        st.rerun()

# Clés uniques pour Plotly
if "RUN_ID" not in st.session_state:
    st.session_state["RUN_ID"] = str(uuid.uuid4())[:8]
if "_PLOT_SEQ" not in st.session_state:
    st.session_state["_PLOT_SEQ"] = 0

def _mk_key(*parts: Any) -> str:
    base = "|".join(map(str, parts)) + "|" + st.session_state["RUN_ID"]
    return "k_" + hashlib.md5(base.encode("utf-8")).hexdigest()[:16]

def st_plotly(fig: go.Figure, *key_parts: Any, **kwargs: Any) -> Any:
    key = kwargs.pop("key", None)
    if key is None:
        st.session_state["_PLOT_SEQ"] += 1
        key = _mk_key("plot", st.session_state["_PLOT_SEQ"], *key_parts)
    return st.plotly_chart(fig, key=key, **kwargs)

# ============================================================
# MISE EN CACHE
# ============================================================
@st.cache_data(show_spinner=False)
def load_and_process_cached(path: str) -> pd.DataFrame:
    """Wrapper avec cache autour de :func:`core.io.load_csv`."""
    return load_csv(path)

@st.cache_data(show_spinner=False)
def compute_cow_cached(
    df_all: pd.DataFrame,
    cow_id: str,
    interval: str,
    window_baseline: int,
    contamination: float,
    baseline_ratio,
    random_state: int,
    persist_hours: int,
    alert_min: int,
    mix_mode: str,
    mix_rate_thr: float,
    z_low_thr: float,
    z_high_thr: float,
    cooldown_hours: int,
    mi_z_high_thr: float,
    coverage_min_pct: float,
) -> pd.DataFrame:
    """Execution pipeline avec cache pour une vache (cle = toutes les valeurs de parametres)."""
    return run_pipeline_one_cow(
        df_all, cow_id,
        interval=interval,
        window_baseline=window_baseline,
        contamination=contamination,
        baseline_ratio=baseline_ratio,
        random_state=random_state,
        persist_hours=persist_hours,
        alert_min=alert_min,
        mix_mode=mix_mode,
        mix_rate_thr=mix_rate_thr,
        z_low_thr=z_low_thr,
        z_high_thr=z_high_thr,
        cooldown_hours=cooldown_hours,
        mi_z_high_thr=mi_z_high_thr,
        coverage_min_pct=coverage_min_pct,
    )

# ============================================================
# BARRE LATERALE
# ============================================================
st.title("Alerte comportementale bidirectionnelle")
st.caption("Prototype exploratoire de priorisation terrain; aucune sortie ne constitue un diagnostic de boiterie.")

with st.sidebar:
    st.header("Données")
    upload_widget_key = f"upload_csv_{st.session_state['_upload_widget_nonce']}"
    uploaded_file = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"],
        help="Le fichier importé remplace le corpus intégré pour cette session.",
        key=upload_widget_key,
    )

    use_builtin = st.toggle(
        "Utiliser le corpus intégré si aucun fichier n'est importé",
        value=True,
        help="Charge data/brut.csv afin de pouvoir explorer immédiatement l'interface.",
    )
    if uploaded_file is not None:
        st.success(f"Fichier chargé : {uploaded_file.name}")
    elif use_builtin:
        st.caption("Source active : corpus IceQube 2023 intégré")

    st.header("Intervalle / Variables")
    interval_options = ["15T", "30T", "1H", "2H", "4H"]
    default_interval_idx = interval_options.index(DEFAULT_INTERVAL) if DEFAULT_INTERVAL in interval_options else 0
    interval = st.selectbox("Intervalle de rééchantillonnage", interval_options, index=default_interval_idx)
    window_baseline = st.slider("Fenêtre du z-score robuste glissant", 12, 240, int(DEFAULT_WINDOW_BASELINE), 6)

    st.header("Comparateur Isolation Forest")
    contamination = st.slider("Contamination", 0.001, 0.1, float(DEFAULT_CONTAMINATION), 0.001)
    baseline_options = [None, 0.5, 0.6, 0.7, 0.8]
    default_baseline_idx = baseline_options.index(DEFAULT_BASELINE_RATIO) if DEFAULT_BASELINE_RATIO in baseline_options else 0
    baseline_ratio = st.selectbox("Baseline ratio (fit IF)", baseline_options, index=default_baseline_idx)
    random_state = st.number_input("Random state", value=int(DEFAULT_RANDOM_STATE), step=1)

    st.header("Agrégation du comparateur")
    persist_hours = st.slider("Fenêtre de persistance (heures)", 2, 24, int(DEFAULT_PERSIST_HOURS), 1)
    alert_min = st.slider("Anomalies min. dans la fenêtre", 1, 10, int(DEFAULT_ALERT_MIN), 1)
    mix_mode_options = ["MIX", "IF-ONLY"]
    default_mix_idx = mix_mode_options.index(DEFAULT_MIX_MODE) if DEFAULT_MIX_MODE in mix_mode_options else 0
    mix_mode = st.selectbox("Mode de détection", mix_mode_options, index=default_mix_idx)
    mix_rate_thr = st.slider("Taux min. anomalies (mix_rate_thr)", 0.05, 1.0, float(DEFAULT_MIX_RATE_THR), 0.01)
    z_low_thr = st.slider("Seuil z-score bas", -6.0, -0.5, float(DEFAULT_Z_LOW_THR), 0.1)
    z_high_thr = st.slider("Seuil z-score haut", 0.5, 6.0, float(DEFAULT_Z_HIGH_THR), 0.1)
    cooldown_hours = st.slider("Cooldown notifications (heures)", 1, 48, int(DEFAULT_COOLDOWN_HOURS), 1)
    mi_z_high_thr = st.slider("Seuil spike Motion Index", 1.0, 6.0, float(DEFAULT_MI_Z_HIGH_THR), 0.1)

    st.header("Qualité des données")
    coverage_min_pct = st.slider("Couverture minimale (%)", 0.0, 100.0, float(DEFAULT_COVERAGE_MIN_PCT), 1.0)

    st.header("Visualisation")
    plot_mode = st.radio(
        "Mode graphiques",
        ["Brut (recommandé)", "Variables centrées (rrz)"],
        index=0,
        horizontal=True,
    )
    plot_options, plot_defaults = plot_options_for_mode(plot_mode)
    plot_pref = st.multiselect(
        "Variables à afficher",
        plot_options,
        default=plot_defaults
    )
    if "rrz" in plot_mode.lower():
        st.caption("Les variables rrz sont centrées; des valeurs négatives sont normales.")

    st.caption(
        "Les branches HYPO et INSTABILITÉ ont des persistances distinctes. "
        "La fusion hiérarchique conserve l'instabilité isolée comme surveillance "
        "et applique un cooldown commun aux notifications actionnables."
    )

# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================
if uploaded_file is None and not use_builtin:
    st.info("Importez un fichier CSV ou activez le corpus intégré pour commencer l'analyse.")
    st.markdown("""
    ### Démarrage rapide

    1. Glissez-déposez votre fichier CSV dans la barre latérale
    2. L'analyse démarre automatiquement

    #### Format CSV attendu

    Colonnes minimales requises :
    - `Cow` (ou ID, Animal) : Identifiant animal
    - `T` (ou Time, Timestamp) : Date/heure
    - `Motion Index` (ou Steps) : Données d'activité

    Colonnes fortement recommandees pour une alerte multivariee plus robuste :
    - `Lying Time`, `Standing Time`, `Transitions`
    """)
    st.stop()

with st.spinner("Chargement et normalisation des données..."):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            from core.io import normalize_columns
            df = normalize_columns(df)
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
            source_label = f"fichier importé ({uploaded_file.name})"
        else:
            if not DEFAULT_DATA_FILE.is_file():
                raise FileNotFoundError(f"Corpus intégré introuvable : {DEFAULT_DATA_FILE}")
            df = load_and_process_cached(str(DEFAULT_DATA_FILE))
            file_hash = hashlib.md5(DEFAULT_DATA_FILE.read_bytes()).hexdigest()[:8]
            source_label = "corpus intégré data/brut.csv"
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        st.info("Vérifiez le format du fichier CSV et la présence du corpus intégré.")
        st.stop()

cows = sorted(df[COW].astype(str).unique().tolist())
st.success(f"Données chargées : **{len(df):,}** lignes | **{len(cows)}** animaux | Hash `{file_hash}`")
st.caption(f"Source analysée : {source_label}")

# ============================================================
# CLES DE CACHE
# ============================================================
def _build_cache_key_full() -> str:
    """Cle de cache pour le troupeau complet (sans max_cows)."""
    return (
        f"{file_hash}_{interval}_{window_baseline}_{contamination}_{baseline_ratio}_"
        f"{random_state}_{persist_hours}_{alert_min}_{mix_mode}_{mix_rate_thr}_"
        f"{z_low_thr}_{z_high_thr}_{cooldown_hours}_{mi_z_high_thr}_{coverage_min_pct}_"
        f"None"
    )

# ============================================================
# ONGLETS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Analyse individuelle",
    "Classement troupeau",
    "Comparaison journalière",
    "Export"
])

pipeline_kwargs = {
    "interval": interval,
    "window_baseline": window_baseline,
    "contamination": contamination,
    "baseline_ratio": baseline_ratio,
    "random_state": random_state,
    "persist_hours": persist_hours,
    "alert_min": alert_min,
    "mix_mode": mix_mode,
    "mix_rate_thr": mix_rate_thr,
    "z_low_thr": z_low_thr,
    "z_high_thr": z_high_thr,
    "cooldown_hours": cooldown_hours,
    "mi_z_high_thr": mi_z_high_thr,
    "coverage_min_pct": coverage_min_pct,
}

with tab1:
    render_tab_individual(
        df=df,
        cows=cows,
        interval=interval,
        plot_mode=plot_mode,
        plot_pref=plot_pref,
        file_hash=file_hash,
        compute_cow_cached=compute_cow_cached,
        st_plotly=st_plotly,
        mk_key=_mk_key,
        pipeline_kwargs=pipeline_kwargs,
    )

with tab2:
    render_tab_herd(
        df=df,
        interval=interval,
        plot_pref=plot_pref,
        file_hash=file_hash,
        st_plotly=st_plotly,
        mk_key=_mk_key,
        build_cache_key_full=_build_cache_key_full,
        pipeline_kwargs=pipeline_kwargs,
    )

with tab3:
    render_tab_daily_comparison(
        df=df,
        file_hash=file_hash,
        st_plotly=st_plotly,
        build_cache_key_full=_build_cache_key_full,
        pipeline_kwargs=pipeline_kwargs,
    )

with tab4:
    render_tab_export(
        df=df,
        interval=interval,
        file_hash=file_hash,
        mk_key=_mk_key,
        build_cache_key_full=_build_cache_key_full,
        pipeline_kwargs=pipeline_kwargs,
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(f"Prototype MemoireV3 | Hash : `{file_hash}`")
