"""Réapplication d'événements injectés pour artefacts/visualisation Streamlit."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from core.io import COW, LYING, MI, STANDING, STEPS, TIME, TRANSITIONS, TR_DOWN, TR_UP
from scripts.validation_common import to_numeric_cols as _to_numeric_cols

from scripts._validation_injection_raw.profiles import apply_event_on_raw


def build_streamlit_raw_with_events(
    raw_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Réapplique les événements injectés sur les données brutes pour visualisation Streamlit."""
    df = raw_df.sort_values([COW, TIME]).copy().reset_index(drop=True)
    _to_numeric_cols(df, [STEPS, MI, LYING, STANDING, TRANSITIONS, TR_UP, TR_DOWN])
    rng = np.random.default_rng(seed)
    applied = []

    if events_df is None or len(events_df) == 0:
        return df, pd.DataFrame(columns=["event_id", "cow", "start", "end", "profile", "rows_affected"])

    for r in events_df.itertuples(index=False):
        cow = str(r.cow)
        s = pd.Timestamp(r.start)
        e = pd.Timestamp(r.end)
        mask = (
            (df[COW].astype(str) == cow)
            & (pd.to_datetime(df[TIME], errors="coerce") >= s)
            & (pd.to_datetime(df[TIME], errors="coerce") <= e)
        )
        idx = np.flatnonzero(mask.values)
        if len(idx) > 0:
            apply_event_on_raw(df, idx, str(r.profile), rng)
        applied.append(
            {
                "event_id": r.event_id,
                "cow": cow,
                "start": s,
                "end": e,
                "profile": r.profile,
                "rows_affected": int(len(idx)),
            }
        )

    df["injection_flag"] = 0
    for r in events_df.itertuples(index=False):
        mask = (df[COW].astype(str) == str(r.cow)) & (
            pd.to_datetime(df[TIME], errors="coerce") >= pd.Timestamp(r.start)
        ) & (pd.to_datetime(df[TIME], errors="coerce") <= pd.Timestamp(r.end))
        df.loc[mask, "injection_flag"] = 1
    return df, pd.DataFrame(applied)

