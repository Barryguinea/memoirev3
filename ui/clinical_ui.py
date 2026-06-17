"""Helpers UI pour episodes detectes et consignes cliniques (tables + rendu Streamlit)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from ui.presentation import (
    build_reason_row as _build_reason_row,
    clinical_actions as _clinical_actions,
    clinical_payload as _clinical_payload,
    clinical_short as _clinical_short,
    level_priority as _level_priority,
    normalize_alert_level as _normalize_alert_level,
)


# Libelles d'affichage neutres (cadrage surveillance/verification, non diagnostique).
# Utilises par defaut si un niveau ne fournit pas son propre "label" (ex: KB de secours).
_NEUTRAL_LEVEL_LABELS = {
    "suspect": "À surveiller",
    "probable": "À vérifier",
    "critique": "Vérification prioritaire",
    "normal": "Normal",
}


def _slug_source_id(name: str, idx: int) -> str:
    """Construit un identifiant source stable-ish si absent du JSON."""
    base = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return base or f"source_{idx}"


def _normalize_clinical_kb(kb: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise la KB clinique vers un schema v1.1 compatible (retrocompatible v1.0)."""
    out = dict(kb) if isinstance(kb, dict) else {}

    meta = out.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta = dict(meta)
    meta.setdefault("title", "Base de consignes cliniques (boiterie)")
    meta.setdefault("version", "1.1.0")
    meta.setdefault("date", "")
    meta.setdefault("language", "fr")
    meta.setdefault("disclaimer", "Aide a l'interpretation. Ne remplace pas l'avis veterinaire.")
    meta.setdefault("owner", "Equipe projet")
    meta.setdefault("reviewed_by", "A renseigner")
    meta.setdefault("review_date", meta.get("date", ""))
    meta.setdefault("next_review_date", "")
    meta.setdefault("source_policy", "Sources de reference veterinaires / institutionnelles priorisees")
    out["meta"] = meta

    # Sources: ajoute id + metadonnees minimales si absents.
    raw_sources = out.get("sources", [])
    srcs: List[Dict[str, Any]] = []
    for i, s in enumerate(raw_sources if isinstance(raw_sources, list) else [], start=1):
        if isinstance(s, dict):
            src = dict(s)
            src.setdefault("id", _slug_source_id(str(src.get("name", "")), i))
            src.setdefault("publisher", "")
            src.setdefault("type", "reference")
            src.setdefault("language", "fr")
            src.setdefault("evidence_level", "reference")
            src.setdefault("reliability", "medium")
            src.setdefault("review_date", meta.get("review_date", ""))
            src.setdefault("note", "")
            srcs.append(src)
        else:
            srcs.append(
                {
                    "id": f"source_{i}",
                    "name": str(s),
                    "url": "",
                    "publisher": "",
                    "type": "reference",
                    "language": "fr",
                    "evidence_level": "reference",
                    "reliability": "medium",
                    "review_date": meta.get("review_date", ""),
                    "note": "",
                }
            )
    out["sources"] = srcs
    source_ids = {str(s.get("id")) for s in srcs if str(s.get("id", "")).strip()}

    default_refs = {
        "suspect": [
            "merck_veterinary_manual_overview_of_lameness_in_cattle",
            "ahdb_mobility_scoring_for_dairy_cows",
        ],
        "probable": [
            "merck_veterinary_manual_lameness_originating_in_the_hoof_in_cattle",
            "dairynz_treating_lameness",
            "ahdb_mobility_score_score_sheet_48h_pour_score_2",
        ],
        "critique": [
            "merck_veterinary_manual_overview_of_lameness_in_cattle",
            "merck_veterinary_manual_lameness_originating_in_the_hoof_in_cattle",
        ],
    }

    levels = out.get("levels", {})
    if not isinstance(levels, dict):
        levels = {}
    normalized_levels: Dict[str, Dict[str, Any]] = {}
    for level_key, payload in levels.items():
        lvl = str(level_key).strip().lower()
        p = dict(payload) if isinstance(payload, dict) else {"summary": str(payload)}
        p.setdefault("label", _NEUTRAL_LEVEL_LABELS.get(lvl, lvl.capitalize()))
        p.setdefault("priority", _level_priority(lvl))
        p.setdefault("confidence", "high" if lvl == "critique" else "medium")
        p.setdefault("last_review_date", meta.get("review_date", ""))
        p.setdefault("call_vet_if", [])
        p.setdefault("do_not", [])
        refs = p.get("references", [])
        if not isinstance(refs, list):
            refs = []
        refs = [str(r) for r in refs if str(r).strip()]
        if not refs:
            refs = [r for r in default_refs.get(lvl, []) if r in source_ids]
        p["references"] = [r for r in refs if r in source_ids]
        normalized_levels[lvl] = p
    out["levels"] = normalized_levels

    # herd_prevention: conserve list[str] si present, accepte aussi list[dict]
    herd_prevention = out.get("herd_prevention", [])
    if not isinstance(herd_prevention, list):
        out["herd_prevention"] = []

    return out


def _render_bullets(title: str, values: Any) -> None:
    if not isinstance(values, list):
        return
    vals = [str(v).strip() for v in values if str(v).strip()]
    if not vals:
        return
    st.markdown(f"**{title}**")
    for v in vals:
        st.markdown(f"- {v}")


def _render_level_sources(payload: Dict[str, Any], kb: Dict[str, Any]) -> None:
    refs = payload.get("references", [])
    if not isinstance(refs, list) or not refs:
        return
    src_by_id = {
        str(s.get("id")): s
        for s in kb.get("sources", [])
        if isinstance(s, dict) and str(s.get("id", "")).strip()
    }
    resolved = [src_by_id[r] for r in refs if r in src_by_id]
    if not resolved:
        return

    with st.expander("Sources (niveau)", expanded=False):
        for s in resolved:
            name = str(s.get("name", "")).strip() or str(s.get("id", "source"))
            url = str(s.get("url", "")).strip()
            reliability = str(s.get("reliability", "")).strip()
            review_date = str(s.get("review_date", "")).strip()
            note = str(s.get("note", "")).strip()

            line = f"- [{name}]({url})" if url else f"- {name}"
            extras = [x for x in [f"fiabilite: {reliability}" if reliability else "", f"revue: {review_date}" if review_date else ""] if x]
            if extras:
                line += " (" + ", ".join(extras) + ")"
            st.markdown(line)
            if note:
                st.caption(note)


def build_episode_why_table(df: pd.DataFrame, kb: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Construit une table d'épisodes détectés avec raisons lisibles et consignes cliniques."""
    x = df.copy()
    hybrid = "hybrid_warning_start" in x.columns
    behavioral = "behavioral_warning_start" in x.columns
    if hybrid:
        ep = x[x["hybrid_warning_start"].astype(int) == 1].copy()
    elif behavioral:
        ep = x[x["behavioral_warning_start"].astype(int) == 1].copy()
    elif "pred_lameness_start" in x.columns:
        ep = x[x["pred_lameness_start"].astype(int) == 1].copy()
    elif "pred_lameness_episode" in x.columns:
        ep = x[x["pred_lameness_episode"].astype(int) == 1].copy()
    else:
        ep = pd.DataFrame(columns=x.columns)

    if ep.empty:
        return ep

    if kb is None:
        kb = load_clinical_kb()

    ep = ep.sort_values("T", ascending=False)
    if hybrid:
        ep["alert_level"] = "suspect"
        ep["reason"] = ep.apply(
            lambda row: (
                f"Alerte {row.get('hybrid_warning_type', 'INDETERMINE')}: "
                f"score HYPO={float(row.get('behavioral_warning_score', 0)):.2f}, "
                f"score INSTABILITÉ={float(row.get('instability_warning_score', 0)):.2f}. "
                "Vérification terrain requise."
            ),
            axis=1,
        )
    elif behavioral:
        ep["alert_level"] = "suspect"
        ep["reason"] = ep.apply(
            lambda row: (
                f"Dégradation persistante: score={float(row.get('behavioral_warning_score', 0)):.2f}, "
                f"familles concordantes={int(row.get('behavioral_warning_families', 0))}. "
                "Vérification terrain requise."
            ),
            axis=1,
        )
    else:
        ep["reason"] = ep.apply(_build_reason_row, axis=1)
    if "alert_level" in ep.columns:
        ep["consigne_clinique"] = ep["alert_level"].apply(lambda x: _clinical_short(x, kb))
    else:
        ep["consigne_clinique"] = _clinical_short("normal", kb)

    cols = [
        c
        for c in [
            "T",
            "alert_level",
            "hybrid_warning_type",
            "hybrid_warning_score",
            "behavioral_warning_score",
            "behavioral_warning_cusum",
            "behavioral_warning_families",
            "instability_warning_score",
            "instability_warning_families",
            "if_score",
            "if_score_q05",
            "if_score_pct",
            "mi_spike_k",
            "is_critique",
            "reason",
            "consigne_clinique",
        ]
        if c in ep.columns
    ]
    return ep[cols]


@st.cache_data(show_spinner=False)
def load_clinical_kb() -> Dict[str, Any]:
    """Charge la base de consignes cliniques (JSON) avec fallback intégré."""
    candidates = [Path(__file__).resolve().parents[1] / "data" / "consignes_boiterie_kb.json"]
    env_kb = os.environ.get("LAMENESS_KB_PATH", "").strip()
    if env_kb:
        candidates.append(Path(env_kb))
    candidates.append(Path.home() / "Desktop" / "consignes_boiterie_kb.json")
    for p in candidates:
        if p.is_file():
            try:
                return _normalize_clinical_kb(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
    return _normalize_clinical_kb(
        {
        "meta": {
            "title": "Consignes cliniques boiterie",
            "version": "fallback",
            "disclaimer": "Informations generales. Ne remplace pas un avis veterinaire.",
        },
        "levels": {
            "suspect": {
                "summary": "Surveillance active (<24h).",
                "actions_0_24h": [
                    "Observer locomotion et appui.",
                    "Verifier hygiene du pied et environnement.",
                    "Consigner le cas et suivre l'evolution.",
                ],
            },
            "probable": {
                "summary": "Intervention rapide (<24-48h).",
                "actions_0_2h": [
                    "Mettre au calme, litiere seche, limiter marche.",
                    "Preparer examen du pied par personnel forme.",
                ],
                "actions_24_48h": [
                    "Parage therapeutique selon protocole.",
                    "Gestion de la douleur selon protocole veterinaire.",
                ],
            },
            "critique": {
                "summary": "Priorite immediate + vet si signes d'urgence.",
                "actions_0_2h": [
                    "Isoler et limiter la marche.",
                    "Contacter le veterinaire si red flags.",
                ],
                "red_flags": [
                    "Incapacite a se lever/boire/manger.",
                    "Fievre, abattement ou douleur extreme.",
                ],
            },
        },
    }
    )


def render_clinical_guidance(levels: List[str], kb: Dict[str, Any]) -> None:
    """Affiche les consignes cliniques depliables pour chaque niveau d'alerte présent."""
    kb = _normalize_clinical_kb(kb)
    uniq = sorted(
        {_normalize_alert_level(level) for level in levels},
        key=_level_priority,
        reverse=True,
    )
    if not uniq:
        uniq = ["normal"]

    st.markdown("#### Consignes de vérification terrain par niveau")
    meta = kb.get("meta", {}) if isinstance(kb, dict) else {}
    version = str(meta.get("version", "")).strip()
    review_date = str(meta.get("review_date", "")).strip()
    next_review_date = str(meta.get("next_review_date", "")).strip()
    reviewed_by = str(meta.get("reviewed_by", "")).strip()
    source_policy = str(meta.get("source_policy", "")).strip()

    if version:
        st.caption(f"KB v{version}")

    detail_lines = [
        f"Revue: {review_date}" if review_date else "",
        f"Prochaine revue: {next_review_date}" if next_review_date else "",
        f"Validation: {reviewed_by}" if reviewed_by else "",
        f"Politique sources: {source_policy}" if source_policy else "",
    ]
    detail_lines = [line for line in detail_lines if line]
    if detail_lines:
        with st.expander("Infos KB", expanded=False):
            for line in detail_lines:
                st.caption(line)

    for lvl in uniq:
        payload = _clinical_payload(lvl, kb)
        summary = str(payload.get("summary", "")).strip() or "Consignes disponibles."
        label = str(payload.get("label", lvl)).strip() or lvl
        priority = payload.get("priority", _level_priority(lvl))
        expanded = lvl in {"critique", "probable"}
        with st.expander(f"{label.upper()} (P{priority}) : {summary}", expanded=expanded):
            # Rendu detaille par bloc d'actions pour conserver la temporalite.
            _render_bullets("Actions 0-2h", payload.get("actions_0_2h", []))
            _render_bullets("Actions 0-24h", payload.get("actions_0_24h", []))
            _render_bullets("Actions 24h", payload.get("actions_24h", []))
            _render_bullets("Actions 24-48h", payload.get("actions_24_48h", []))

            # Fallback si une fiche utilise des cles non standard mais passe via clinical_actions
            actions = _clinical_actions(payload)
            known_action_values = set()
            for k in ["actions_0_2h", "actions_0_24h", "actions_24h", "actions_24_48h"]:
                vals = payload.get(k, [])
                if isinstance(vals, list):
                    known_action_values.update(str(v).strip() for v in vals if str(v).strip())
            extra_actions = [a for a in actions if str(a).strip() and str(a).strip() not in known_action_values]
            if extra_actions:
                _render_bullets("Autres actions", extra_actions)

            red_flags = payload.get("red_flags", [])
            if isinstance(red_flags, list) and len(red_flags) > 0:
                st.markdown("**Signes d'urgence (contacter veterinaire)**")
                for rf in red_flags:
                    st.markdown(f"- {rf}")

            _render_bullets("Appeler le veterinaire si", payload.get("call_vet_if", []))
            _render_bullets("A eviter", payload.get("do_not", []))

            recheck = payload.get("recheck")
            if recheck:
                st.markdown(f"**Recontrole:** {recheck}")

            _render_level_sources(payload, kb)

    herd_prev = kb.get("herd_prevention", [])
    if isinstance(herd_prev, list) and herd_prev:
        with st.expander("Prevention troupeau (rappels)", expanded=False):
            for item in herd_prev:
                if isinstance(item, dict):
                    text = str(item.get("text", "")).strip()
                    refs = item.get("references", [])
                    st.markdown(f"- {text}" if text else "-")
                    if isinstance(refs, list) and refs:
                        st.caption(f"Refs: {', '.join(str(r) for r in refs)}")
                else:
                    text = str(item).strip()
                    if text:
                        st.markdown(f"- {text}")

    all_sources = kb.get("sources", [])
    if isinstance(all_sources, list) and all_sources:
        with st.expander("Sources de reference (KB)", expanded=False):
            for s in all_sources:
                if not isinstance(s, dict):
                    st.markdown(f"- {s}")
                    continue
                name = str(s.get("name", "")).strip() or str(s.get("id", "source"))
                url = str(s.get("url", "")).strip()
                reliability = str(s.get("reliability", "")).strip()
                review_date = str(s.get("review_date", "")).strip()
                line = f"- [{name}]({url})" if url else f"- {name}"
                extras = [x for x in [reliability, review_date] if x]
                if extras:
                    line += " (" + ", ".join(extras) + ")"
                st.markdown(line)

    disclaimer = kb.get("meta", {}).get("disclaimer", "")
    if disclaimer:
        st.caption(f"Avertissement: {disclaimer}")
