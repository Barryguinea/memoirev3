"""Invariants des deux regles temporelles de la fusion.

La periode refractaire et la fenetre de sequence gouvernent directement deux
chiffres du manuscrit : la charge de fond par vache-jour et la proportion de
sequences instabilite-hypoactivite recuperees. Les deux mecanismes etaient
verifiables par lecture du code, mais aucun test ne les exercait : neutraliser
le cooldown ou supprimer la borne inferieure de la fenetre laissait la suite
entierement verte.
"""

import numpy as np
import pandas as pd

from core.hybrid_warning import _cooldown_notifications, _recent_prior_event


def _starts(indices: list[int], length: int) -> pd.Series:
    valeurs = np.zeros(length, dtype=int)
    for i in indices:
        valeurs[i] = 1
    return pd.Series(valeurs)


# --- periode refractaire ---------------------------------------------------


def test_cooldown_supprime_un_second_depart_trop_proche() -> None:
    """Deux departs separes par moins que le cooldown ne notifient qu'une fois."""
    notifications = _cooldown_notifications(_starts([0, 3], 12), cooldown_bins=6)
    assert notifications.tolist() == [1] + [0] * 11


def test_cooldown_laisse_passer_un_depart_au_dela_de_la_fenetre() -> None:
    """Au-dela du cooldown, le depart suivant produit bien une notification."""
    notifications = _cooldown_notifications(_starts([0, 7], 12), cooldown_bins=6)
    assert notifications[0] == 1
    assert notifications[7] == 1
    assert notifications.sum() == 2


def test_cooldown_compte_depuis_la_derniere_notification_emise() -> None:
    """Un depart etouffe ne prolonge pas la periode refractaire.

    Sans cette propriete, une suite de departs rapproches repousserait
    indefiniment la notification suivante.
    """
    notifications = _cooldown_notifications(_starts([0, 2, 4, 7], 12), cooldown_bins=6)
    assert notifications[0] == 1
    assert notifications[2] == 0
    assert notifications[4] == 0
    assert notifications[7] == 1


def test_cooldown_nul_notifie_chaque_depart() -> None:
    notifications = _cooldown_notifications(_starts([0, 1, 2], 6), cooldown_bins=0)
    assert notifications.sum() == 3


# --- fenetre de sequence ---------------------------------------------------


def test_sequence_rejette_un_evenement_anterieur_trop_recent() -> None:
    """Sous la borne inferieure, l'anteriorite ne fait pas une sequence.

    C'est cette borne qui distingue une sequence d'une simple convergence
    simultanee des deux branches.
    """
    presence = _recent_prior_event(_starts([0], 10), min_bins=4, max_bins=8)
    assert not presence.iloc[1:4].any()


def test_sequence_accepte_un_evenement_dans_la_fenetre() -> None:
    presence = _recent_prior_event(_starts([0], 12), min_bins=4, max_bins=8)
    assert presence.iloc[4:9].all()


def test_sequence_oublie_un_evenement_trop_ancien() -> None:
    presence = _recent_prior_event(_starts([0], 14), min_bins=4, max_bins=8)
    assert not presence.iloc[9:].any()


def test_sequence_ne_se_declenche_pas_sur_sa_propre_occurrence() -> None:
    """L'intervalle du depart lui-meme ne compte jamais comme anteriorite."""
    presence = _recent_prior_event(_starts([0, 5], 12), min_bins=0, max_bins=8)
    assert not presence.iloc[0]
