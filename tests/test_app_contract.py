from pathlib import Path


APP_FILE = Path(__file__).resolve().parents[1] / "app.py"


def _read_app() -> str:
    return APP_FILE.read_text(encoding="utf-8")


def test_builtin_corpus_is_explicit_and_upload_keeps_priority():
    text = _read_app()
    assert 'DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "brut.csv"' in text
    assert "Utiliser le corpus intégré si aucun fichier n'est importé" in text
    assert "if uploaded_file is not None:" in text
    assert "elif use_builtin:" in text


def test_duplicate_upload_warning_removed_and_no_html_injection():
    text = _read_app()
    assert 'st.error("Meme fichier recharge: resultats possiblement inchanges.")' not in text
    assert "unsafe_allow_html=True" not in text
