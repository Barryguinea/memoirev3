"""Configuration partagee de la suite de tests.

Le corpus brut est confidentiel et n'est pas versionne. Les tests qui en
dependent portent la marque ``corpus`` : lorsque le fichier est absent, ils
sont ignores avec un motif explicite plutot que de tomber en echec. Un lecteur
du depot peut ainsi lancer ``pytest -q`` et distinguer sans ambiguite un test
defaillant d'un test simplement prive de donnees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parents[1] / "data" / "brut.csv"
MOTIF = "corpus confidentiel data/brut.csv absent du depot"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "corpus: test exigeant le corpus brut confidentiel"
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if CORPUS.exists():
        return
    ignorer = pytest.mark.skip(reason=MOTIF)
    for item in items:
        if "corpus" in item.keywords:
            item.add_marker(ignorer)
