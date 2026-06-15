import io
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


APP_FILE = Path(__file__).resolve().parents[1] / "app.py"


def _fake_upload_csv() -> io.BytesIO:
    rows = []
    ts = pd.date_range("2024-01-01", periods=48, freq="15min")
    for cow in ["8081", "8154"]:
        for i, t in enumerate(ts):
            rows.append(
                {
                    "Cow": cow,
                    "T": t,
                    "Steps": float(10 + (i % 5)),
                    "Motion Index": float(20 + (i % 7)),
                    "Lying Time": float(5 + (i % 3)),
                    "Standing Time": float(5 + (i % 4)),
                    "Transitions": float(1 + (i % 2)),
                }
            )
    payload = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

    class FakeUpload(io.BytesIO):
        def __init__(self, data: bytes, name: str):
            super().__init__(data)
            self.name = name

    return FakeUpload(payload, "sample.csv")


def test_app_smoke_without_upload_uses_builtin_corpus_and_no_exception():
    at = AppTest.from_file(APP_FILE)
    at.run(timeout=40)

    assert len(at.exception) == 0
    assert len(at.get("file_uploader")) >= 1
    assert len(at.get("tab")) == 4

    success_msgs = [getattr(x, "value", "") for x in at.success]
    assert any("Données chargées :" in str(msg) for msg in success_msgs)


def test_app_smoke_with_monkeypatched_upload_reaches_tabs_and_no_exception(monkeypatch):
    # AppTest expose le file_uploader comme UnknownElement dans ce contexte.
    # On monkeypatch streamlit.file_uploader pour tester la branche post-upload.
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: _fake_upload_csv())

    at = AppTest.from_file(APP_FILE)
    at.run(timeout=40)

    assert len(at.exception) == 0
    assert len(at.get("tab")) == 4

    success_msgs = [getattr(x, "value", "") for x in at.success]
    assert any("Fichier chargé : sample.csv" in str(msg) for msg in success_msgs)
    assert any("Données chargées :" in str(msg) for msg in success_msgs)

    infos = [getattr(x, "value", "") for x in at.info]
    assert not any("Importez un fichier CSV" in str(msg) for msg in infos)


def test_app_smoke_sidebar_recompute_button_reruns_without_exception():
    at = AppTest.from_file(APP_FILE)
    at.run(timeout=40)

    nonce_before = at.session_state["_upload_widget_nonce"]
    assert len(at.button) >= 1
    assert getattr(at.button[0], "label", "") == "Recalculer (vider le cache)"

    at.button[0].click()
    at.run(timeout=40)

    assert len(at.exception) == 0
    assert at.session_state["_upload_widget_nonce"] == nonce_before + 1
    assert len(at.get("tab")) == 4
