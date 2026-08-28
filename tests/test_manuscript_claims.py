"""Affirmations que le manuscrit porte sur le depot lui-meme.

Le registre d'audit compare les resultats empiriques a leurs artefacts, mais le
manuscrit avance aussi des faits sur le code : la taille de la suite de tests,
par exemple. Ces affirmations vieillissent silencieusement, puisque aucun
artefact ne les porte. Elles ont deja ete fausses deux fois.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
CH3 = ROOT / "memoire/ch3_systeme.tex"


def _tests_declares() -> int:
    """Nombre de fonctions de test du depot.

    Compte les definitions plutot que la collecte pytest : appeler pytest depuis
    un test le ferait s'executer lui-meme. Les deux totaux coincident tant que
    la suite n'utilise pas de parametrage, ce que le second test verifie.
    """
    total = 0
    for chemin in sorted(TESTS.glob("test_*.py")):
        total += len(re.findall(r"^\s*def (test_\w+)", chemin.read_text(encoding="utf8"), re.M))
    return total


def test_le_manuscrit_annonce_le_bon_nombre_de_tests() -> None:
    annonce = re.search(
        r"La suite automatisée contient (\d+) tests", CH3.read_text(encoding="utf8")
    )
    assert annonce is not None, "la phrase du chapitre 3 a change de forme"
    assert int(annonce.group(1)) == _tests_declares()


def test_aucun_test_parametre_ne_fausse_le_compte() -> None:
    """Garde-fou sur la methode de comptage.

    Un ``@pytest.mark.parametrize`` ferait diverger le nombre de fonctions et le
    nombre de cas collectes, et le test precedent deviendrait trompeur.
    """
    decorateur = re.compile(r"@pytest\.mark\.parametrize")
    ici = Path(__file__).name
    for chemin in sorted(TESTS.glob("test_*.py")):
        if chemin.name == ici:
            # Ce fichier nomme le decorateur dans sa propre documentation.
            continue
        contenu = chemin.read_text(encoding="utf8")
        assert not decorateur.search(contenu), (
            f"{chemin.name} parametre ses cas : le comptage par fonction devient faux"
        )
