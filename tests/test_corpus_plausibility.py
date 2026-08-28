"""Plausibilite biologique du corpus d'entree.

La chaine de verification controlait les sorties : 398 valeurs confrontees aux
artefacts, 76 fichiers scelles, campagnes rejouables au bit pres. Rien ne
regardait si les donnees d'entree etaient plausibles, alors qu'elles portent
tout le reste.

Le corpus contient cinq journees ou les pas, le Motion Index et les transitions
sont nuls pour toutes les vaches observees, et ou la posture est enregistree
comme couchee sur la totalite des intervalles. Une vache laitiere se leve
plusieurs fois par jour : ces journees relevent de l'acquisition, non du
comportement. Elles sont conservees, car les retirer ameliorerait les resultats
publies, mais elles sont documentees au chapitre 4 et verrouillees ici. Un
sixieme jour de ce type, ou un corpus modifie, doit faire echouer ces tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.io import COW, load_csv

# Journees sans aucune activite mesuree, reperees le 2026-08-26.
JOURNEES_SANS_ACTIVITE = {
    "2023-10-16",
    "2023-10-17",
    "2023-10-19",
    "2023-10-21",
    "2023-10-22",
}
FAMILLES_ACTIVITE = ["Steps", "Motion Index", "Transitions"]


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    df = load_csv("data/brut.csv")
    colonne = "Start" if "Start" in df.columns else "T"
    df = df.copy()
    df["jour"] = pd.to_datetime(df[colonne]).dt.floor("D").dt.strftime("%Y-%m-%d")
    return df


def _journees_sans_activite(corpus: pd.DataFrame) -> set[str]:
    par_jour = corpus.groupby("jour")[FAMILLES_ACTIVITE].sum()
    return set(par_jour[par_jour.sum(axis=1) == 0].index)


def test_les_journees_sans_activite_sont_celles_documentees(corpus) -> None:
    """Ni plus ni moins que les cinq journees decrites au chapitre 4."""
    assert _journees_sans_activite(corpus) == JOURNEES_SANS_ACTIVITE


def test_ces_journees_touchent_toutes_les_vaches_observees(corpus) -> None:
    """Une panne d'acquisition, pas un comportement individuel.

    Si une seule vache restait active un de ces jours, l'explication
    instrumentale tomberait et le chapitre 4 devrait etre revu.
    """
    for jour in JOURNEES_SANS_ACTIVITE:
        du_jour = corpus[corpus["jour"] == jour]
        actives = du_jour.groupby(COW)[FAMILLES_ACTIVITE].sum().sum(axis=1)
        assert (actives == 0).all(), f"{jour}: une vache conserve de l'activite"


def test_la_posture_y_est_couchee_sur_tous_les_intervalles(corpus) -> None:
    """Signature d'un capteur immobile plutot que d'une vache au repos.

    ``load_csv`` normalise les durees posturales en minutes : un intervalle de
    15 minutes entierement couche vaut 15,0 et 0,0 debout.
    """
    zone = corpus[corpus["jour"].isin(JOURNEES_SANS_ACTIVITE)]
    debout = pd.to_numeric(zone["Standing Time"], errors="coerce")
    couche = pd.to_numeric(zone["Lying Time"], errors="coerce")
    assert (debout == 0.0).all()
    assert (couche == 15.0).all()


def test_leur_part_du_corpus_reste_celle_annoncee(corpus) -> None:
    """Le chapitre 4 annonce 15,6 % des couples vache-jour."""
    par_couple = corpus.groupby([COW, "jour"])[FAMILLES_ACTIVITE].sum()
    nulles = (par_couple.sum(axis=1) == 0).sum()
    part = nulles / len(par_couple) * 100.0
    assert nulles == 63
    assert part == pytest.approx(15.6, abs=0.05)


def test_aucune_autre_journee_n_est_majoritairement_inactive(corpus) -> None:
    """Garde-fou sur les journees de reprise partielle.

    Les 18 et 20 octobre restent tres en dessous du niveau habituel sans etre
    nuls. Ce test signale l'apparition d'une journee encore plus degradee, qui
    demanderait le meme examen que les cinq autres.

    L'activite est rapportee par vache observee : le 11 novembre n'a qu'une seule
    vache, si bien qu'un total de troupeau ferait passer cette journee pour une
    anomalie alors qu'elle est simplement peu peuplee.
    """
    par_jour = corpus.groupby("jour").apply(
        lambda g: g[FAMILLES_ACTIVITE].sum().sum() / g[COW].nunique(),
        include_groups=False,
    )
    ordinaires = par_jour[~par_jour.index.isin(JOURNEES_SANS_ACTIVITE)]
    assert (ordinaires > 0).all()
    assert ordinaires.min() > 0.05 * ordinaires.median()
