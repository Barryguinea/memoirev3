"""Chargement des données, normalisation des colonnes et noms canoniques.
Lit les CSV capteurs bruts, mappe les noms de colonnes hétérogènes vers
un schéma canonique unique (Cow, T, Steps, Motion Index, Lying Time,
Standing Time, Transitions), puis convertit les durées.
"""

import pandas as pd
from typing import Dict, List, Optional

# Noms canoniques INTERNES (un seul standard pour tout le projet)
COW = "Cow"
TIME = "T"

STEPS = "Steps"
MI = "Motion Index"

LYING = "Lying Time"
STANDING = "Standing Time"

TR_UP = "Transitions Up"
TR_DOWN = "Transitions Down"
TRANSITIONS = "Transitions"


# Synonymes -> canonique
COLUMN_SYNONYMS: Dict[str, List[str]] = {
    # Identifiants
    COW: ["Cow", "cow", "ID", "Id", "Animal", "animal", "Cow ID", "cow_id", "cowid"],
    TIME: ["T", "t", "Time", "time", "Start", "start", "Timestamp", "DateTime", "Datetime", "date", "Date"],

    # Activité
    STEPS: ["Steps", "steps", "Step", "step", "Steps Count", "Step Count"],
    MI: ["Motion Index", "MotionIndex", "MI", "mi", "Motion index", "Motion_Index"],

    # Repos
    LYING: ["Lying Time", "LyingTime", "Lying Time (s)", "Lying time", "lying time", "Lying (s)"],
    STANDING: ["Standing Time", "StandingTime", "Standing Time (s)", "Standing time", "standing time", "Standing (s)"],

    # Transitions
    TR_UP: ["Transitions Up", "TransitionsUp", "Transitions_up", "Trans Up", "Up", "Up Transitions"],
    TR_DOWN: ["Transitions Down", "TransitionsDown", "Transitions_down", "Trans Down", "Down", "Down Transitions"],
    TRANSITIONS: ["Transitions", "Transition", "transitions", "Transitions total", "Total Transitions"],
}


def _duration_to_minutes(s: pd.Series) -> pd.Series:
    """
    Convertit une colonne au format durée (H:MM:SS ou HH:MM:SS) en minutes (float).
    Si la valeur est déjà numérique, elle est retournée telle quelle.
    """
    numeric = pd.to_numeric(s, errors="coerce")
    # Si tout est déjà numérique, retourner directement
    if numeric.notna().sum() == s.notna().sum():
        return numeric

    def _parse_one(val):
        if pd.isna(val):
            return float("nan")
        v = str(val).strip()
        # Tenter d'abord comme nombre pur
        try:
            return float(v)
        except ValueError:
            pass
        # Format H:MM:SS ou HH:MM:SS
        parts = v.split(":")
        if len(parts) == 3:
            try:
                h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
                return h * 60.0 + m + sec / 60.0
            except ValueError:
                pass
        # Format MM:SS
        if len(parts) == 2:
            try:
                m, sec = int(parts[0]), int(parts[1])
                return m + sec / 60.0
            except ValueError:
                pass
        return float("nan")

    return s.apply(_parse_one)


def _first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    """Retourne le premier candidat présent dans *cols*, sinon ``None``."""
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def _rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes vers les noms canoniques via ``COLUMN_SYNONYMS``."""
    df = df.copy()
    rename = {}
    for canon, candidates in COLUMN_SYNONYMS.items():
        hit = _first_existing(df.columns.tolist(), candidates)
        if hit and hit != canon:
            rename[hit] = canon
    if rename:
        df = df.rename(columns=rename)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantit:
    - colonnes canoniques Cow/T + cast numeric sur variables
    - parse datetime robuste
    - calcule Transitions si seulement Up/Down existent
    - tri (Cow, T)
    """
    df = _rename_to_canonical(df)

    # checks
    if COW not in df.columns:
        raise ValueError(f"[io] Colonne vache introuvable. Colonnes: {list(df.columns)[:40]}")

    if TIME not in df.columns:
        raise ValueError(f"[io] Colonne temps introuvable. Colonnes: {list(df.columns)[:40]}")

    # parse datetime robuste
    df[TIME] = pd.to_datetime(df[TIME], errors="coerce", utc=False)
    df = df.dropna(subset=[TIME])

    # cast Cow en str (stable)
    df[COW] = df[COW].astype(str)

    # Colonnes durée (H:MM:SS) → minutes avant le cast numérique
    for c in [LYING, STANDING]:
        if c in df.columns:
            df[c] = _duration_to_minutes(df[c])

    # numeric casts (si présents)
    numeric_cols = [STEPS, MI, TR_UP, TR_DOWN, TRANSITIONS]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # construire TRANSITIONS si absent mais Up/Down présents
    if TRANSITIONS not in df.columns:
        if (TR_UP in df.columns) and (TR_DOWN in df.columns):
            df[TRANSITIONS] = df[TR_UP].fillna(0) + df[TR_DOWN].fillna(0)

    # si transitions existe mais NaN -> 0 (souvent)
    if TRANSITIONS in df.columns:
        df[TRANSITIONS] = df[TRANSITIONS].fillna(0)

    # option: clamp négatifs (capteurs => pas de négatifs)
    for c in [STEPS, MI, LYING, STANDING, TR_UP, TR_DOWN, TRANSITIONS]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = 0

    # tri
    df = df.sort_values([COW, TIME]).reset_index(drop=True)
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Lit un CSV et retourne un DataFrame normalisé (colonnes canoniques, trié)."""
    df = pd.read_csv(path)
    return normalize_columns(df)


def available_base_cols(df: pd.DataFrame) -> List[str]:
    """
    Colonnes “utiles” pour build_interval_features
    """
    cols = []
    for c in [STEPS, MI, LYING, STANDING, TRANSITIONS]:
        if c in df.columns:
            cols.append(c)
    return cols
