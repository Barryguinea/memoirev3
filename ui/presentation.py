"""Fonctions pures de presentation pour le dashboard (labels, tri, tables)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def pretty_feature_name(col: Optional[str]) -> str:
    """Retourne un libelle lisible en francais pour un nom de colonne feature."""
    if not col:
        return "Variable"

    raw = str(col)
    name = raw.lower()

    def _base_label(n: str) -> str:
        if "motion index" in n:
            return "Activite — Motion Index"
        if "steps" in n:
            return "Activite — Pas"
        if "lying time" in n:
            return "Repos — Temps couche"
        if "standing time" in n:
            return "Repos — Temps debout"
        if "transitions" in n:
            return "Transitions"
        return raw

    base = _base_label(name)

    if "_log" in name:
        base = f"{base} (log)"

    if name.endswith("_rrz"):
        return f"{base} — ecart robuste glissant"
    if name.endswith("_rz"):
        return f"{base} — ecart robuste global"
    if name.endswith("_sum"):
        return f"{base} — somme"
    if name.endswith("_mean"):
        return f"{base} — moyenne"

    return base


def compact_feature_title(col: Optional[str], max_len: int = 38) -> str:
    """Libelle feature raccourci pour les titres de sous-graphiques (tronque a *max_len*)."""
    label = pretty_feature_name(col)
    replacements = [
        ("Activite — ", ""),
        ("Repos — ", ""),
        ("Transitions — ", "Transitions "),
        (" — ecart robuste glissant", " (rrz)"),
        (" — ecart robuste global", " (rz)"),
        (" — somme", " (sum)"),
        (" — moyenne", " (mean)"),
    ]
    for old, new in replacements:
        label = label.replace(old, new)
    label = " ".join(label.split())
    if len(label) > max_len:
        label = label[: max_len - 1].rstrip() + "…"
    return label


def plot_options_for_mode(mode: str) -> Tuple[List[str], List[str]]:
    """Retourne (colonnes_disponibles, selection_par_defaut) pour le mode graphique choisi."""
    mode_norm = str(mode).strip().lower()
    if "rrz" in mode_norm or "diagnostic" in mode_norm:
        options = [
            "Motion Index_sum_log_rrz",
            "Motion Index_sum_rrz",
            "Steps_sum_rrz",
            "Lying Time_sum_rrz",
            "Standing Time_sum_rrz",
            "Transitions_sum_rrz",
            "Motion Index_mean_log_rrz",
            "Motion Index_mean_rrz",
            "Steps_mean_rrz",
            "Lying Time_mean_rrz",
            "Standing Time_mean_rrz",
            "Transitions_mean_rrz",
        ]
        defaults = ["Motion Index_sum_log_rrz", "Steps_sum_rrz", "Lying Time_sum_rrz"]
        return options, defaults

    options = [
        "Motion Index_sum",
        "Steps_sum",
        "Lying Time_sum",
        "Standing Time_sum",
        "Transitions_sum",
        "Motion Index_mean",
        "Steps_mean",
        "Lying Time_mean",
        "Standing Time_mean",
        "Transitions_mean",
    ]
    defaults = ["Motion Index_sum", "Steps_sum", "Lying Time_sum"]
    return options, defaults


def build_reason_row(row: pd.Series) -> str:
    """Construit une phrase expliquant pourquoi un intervalle est detecte/alerte."""
    reasons: List[str] = []

    if "if_anom_k" in row and "K" in row and pd.notna(row["if_anom_k"]) and pd.notna(row["K"]):
        try:
            k = int(row["K"])
            if k > 0:
                reasons.append(f"Persistance IF: {int(row['if_anom_k'])}/{k} anomalies")
        except Exception:
            pass

    if "anom_rate_k" in row and pd.notna(row["anom_rate_k"]):
        try:
            reasons.append(f"Taux anomalies={float(row['anom_rate_k']) * 100:.1f}%")
        except Exception:
            pass

    fams: List[str] = []
    for col, label in [
        ("fam_activity", "activite"),
        ("fam_rest", "repos"),
        ("fam_transitions", "transitions"),
    ]:
        if col in row and int(row.get(col, 0)) == 1:
            fams.append(label)
    if fams:
        reasons.append("Familles: " + ", ".join(fams))

    if "mi_spike_k" in row and pd.notna(row["mi_spike_k"]):
        try:
            if int(row["mi_spike_k"]) > 0:
                reasons.append(f"MI spike persistant={int(row['mi_spike_k'])}")
        except Exception:
            pass

    if "coherence_boiterie" in row:
        try:
            if int(row.get("coherence_boiterie", 0)) == 1:
                reasons.append("Cohérence multivariée=OK")
        except Exception:
            pass

    if "in_cooldown" in row:
        try:
            if int(row.get("in_cooldown", 0)) == 1:
                reasons.append("En cooldown")
        except Exception:
            pass

    if "is_critique" in row:
        try:
            if int(row.get("is_critique", 0)) == 1:
                reasons.append("Niveau critique")
        except Exception:
            pass

    if not reasons:
        return "Signal faible / contexte normal"
    return " | ".join(reasons)


def build_filterable_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sous-ensemble des colonnes de detection, tri recent-en-premier, pour filtrage interactif."""
    x = df.copy().sort_values("T", ascending=False)
    cols = [
        c
        for c in [
            "T",
            "Day",
            "if_score",
            "if_score_q05",
            "if_score_pct",
            "is_critique",
            "if_anomaly_point",
            "if_anom_k",
            "anom_rate_k",
            "K",
            "mi_spike",
            "mi_spike_k",
            "n_families",
            "flag_coherent_episode",
            "coherence_boiterie",
            "pred_problem_episode",
            "pred_lameness_episode",
            "pred_lameness_start",
            "notif_lameness",
            "in_cooldown",
            "alert_level",
            "lame_confidence",
        ]
        if c in x.columns
    ]
    return x[cols]


def normalize_alert_level(level: Any) -> str:
    """Normalise un niveau d'alerte heterogene vers {normal,suspect,probable,critique}."""
    x = str(level).strip().lower()
    if x in {"critique", "critical"}:
        return "critique"
    if x in {"probable"}:
        return "probable"
    if x in {"suspect"}:
        return "suspect"
    return "normal"


def level_priority(level: str) -> int:
    """Priorite de tri croissante des niveaux d'alerte."""
    order = {"normal": 0, "suspect": 1, "probable": 2, "critique": 3}
    return order.get(normalize_alert_level(level), 0)


def clinical_payload(level: str, kb: Dict[str, Any]) -> Dict[str, Any]:
    """Retourne la fiche de consignes pour un niveau d'alerte a partir de la KB."""
    lvl = normalize_alert_level(level)
    levels = kb.get("levels", {}) if isinstance(kb, dict) else {}
    if lvl in levels:
        return levels[lvl]
    if lvl == "normal":
        return {
            "summary": "Pas d'indice significatif; poursuivre la surveillance de routine.",
            "actions_0_24h": [
                "Continuer le suivi quotidien et l'hygiene des sols/litiere.",
                "Recontroler si changement de comportement ou de locomotion.",
            ],
        }
    return {
        "summary": "Niveau non documente.",
        "actions_0_24h": ["Verifier le dossier clinique et reevaluer le cas."],
    }


def clinical_actions(payload: Dict[str, Any]) -> List[str]:
    """Aplatit les listes d'actions (0-2h, 0-24h, 24-48h, etc.) en liste ordonnee."""
    actions: List[str] = []
    for key in ["actions_0_2h", "actions_0_24h", "actions_24h", "actions_24_48h"]:
        vals = payload.get(key, [])
        if isinstance(vals, list):
            actions.extend([str(v) for v in vals if str(v).strip()])
    return actions


def clinical_short(level: Any, kb: Dict[str, Any]) -> str:
    """Resume court de consigne clinique pour un niveau d'alerte."""
    payload = clinical_payload(str(level), kb)
    summary = str(payload.get("summary", "")).strip()
    actions = clinical_actions(payload)
    if actions:
        return f"{summary} Action: {actions[0]}"
    return summary or "Surveillance clinique selon protocole."
