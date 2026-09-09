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
README = ROOT / "README.md"

# Tests de la politique de reference fixe, posterieurs au gel du manuscrit.
# Le compteur du chapitre 3 decrit la suite scellee ; celui du README decrit la
# suite courante. Les deux sont verifies separement.
TESTS_HORS_MANUSCRIT = {
    "test_hypo_stress_campaign.py::test_all_variants_keep_clean_reference_after_injection",
    "test_hypo_stress_campaign.py::test_fixed_reference_rejects_changed_training_points",
    "test_hypo_stress_campaign.py::test_stress_campaign_passes_clean_reference_to_every_injected_run",
    "test_hypo_stress_campaign.py::test_stress_campaign_recomputes_the_reference_by_default",
    "test_hypo_stress_campaign.py::test_fixed_reference_output_cannot_overwrite_sealed_artifacts",
}


def _tests_declares() -> set[str]:
    """Identifiants des fonctions de test du depot.

    Compte les definitions plutot que la collecte pytest : appeler pytest depuis
    un test le ferait s'executer lui-meme. Les deux totaux coincident tant que
    la suite n'utilise pas de parametrage, ce que le second test verifie.
    """
    identifiers = set()
    for chemin in sorted(TESTS.glob("test_*.py")):
        names = re.findall(r"^\s*def (test_\w+)", chemin.read_text(encoding="utf8"), re.M)
        identifiers.update(f"{chemin.name}::{name}" for name in names)
    return identifiers


def test_le_manuscrit_annonce_le_bon_nombre_de_tests() -> None:
    annonce = re.search(
        r"La suite automatisée contient (\d+) tests", CH3.read_text(encoding="utf8")
    )
    assert annonce is not None, "la phrase du chapitre 3 a change de forme"
    tests = _tests_declares()
    assert TESTS_HORS_MANUSCRIT <= tests, "un test declare hors manuscrit est absent"
    assert int(annonce.group(1)) == len(tests - TESTS_HORS_MANUSCRIT)
    current = re.search(r"tests/\s+# (\d+) tests", README.read_text(encoding="utf8"))
    assert current is not None
    assert int(current.group(1)) == len(tests)


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
