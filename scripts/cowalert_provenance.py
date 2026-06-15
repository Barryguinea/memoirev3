"""Traçabilite des donnees : corpus IceQube contre l'export CowAlert.

Verifie que le corpus IceQube utilise dans ce memoire correspond exactement a
l'export du systeme commercial CowAlert (IceRobotics) pour les vaches dont un
export est disponible. Aligne les deux sources sur (vache, debut d'intervalle)
et compare les variables d'activite. Aucune valeur n'est saisie a la main.

Le fichier source CowAlert (data/cowalert_export.csv) contient des donnees
capteurs confidentielles et n'est pas diffuse; seules les correlations agregees
sont publiees.

Usage:
    python scripts/cowalert_provenance.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ICEQUBE = ROOT / "data/brut.csv"
COWALERT = ROOT / "data/cowalert_export.csv"
OUT = ROOT / "data/revalidation/hypo_module/cowalert_provenance.csv"
COLUMNS = ["Steps", "Motion Index", "Transitions"]
KEYS = ["Cow", "Start"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Cow"] = df["Cow"].astype(str)
    df["Start"] = pd.to_datetime(df["Start"], errors="coerce")
    if df[KEYS].isna().any().any():
        raise ValueError("Identifiant ou date manquant dans une source de provenance.")
    if df.duplicated(KEYS).any():
        raise ValueError("Doublons (Cow, Start) détectés: appariement un-à-un impossible.")
    return df


def main() -> None:
    ice = _normalise(pd.read_csv(ICEQUBE))
    cow = _normalise(pd.read_csv(COWALERT))
    merged = ice.merge(
        cow,
        on=KEYS,
        suffixes=("_IQ", "_CA"),
        how="inner",
        validate="one_to_one",
    )

    n_pairs = len(merged)
    n_cows = merged["Cow"].nunique()
    if n_pairs == 0:
        raise ValueError("Aucun intervalle commun entre le corpus et l'export CowAlert.")
    source_rows = len(cow)
    match_pct = 100.0 * n_pairs / source_rows
    rows = [{
        "variable": "n_observations_appariees",
        "n": n_pairs,
        "correlation": "",
        "pct_identique": "",
        "n_vaches": n_cows,
        "n_source": source_rows,
        "appariement_pct": round(match_pct, 1),
    }]
    for col in COLUMNS:
        pair = merged[[f"{col}_IQ", f"{col}_CA"]].dropna()
        a, b = pair.iloc[:, 0].to_numpy(float), pair.iloc[:, 1].to_numpy(float)
        corr = float(np.corrcoef(a, b)[0, 1]) if len(pair) > 2 else float("nan")
        identical = float((a == b).mean() * 100.0) if len(pair) else float("nan")
        rows.append({
            "variable": col,
            "n": len(pair),
            "correlation": round(corr, 4),
            "pct_identique": round(identical, 1),
            "n_vaches": n_cows,
            "n_source": source_rows,
            "appariement_pct": round(match_pct, 1),
        })

    with open(OUT, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variable",
                "n",
                "correlation",
                "pct_identique",
                "n_vaches",
                "n_source",
                "appariement_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Vaches communes: {n_cows} | observations appariees: "
        f"{n_pairs}/{source_rows} ({match_pct:.1f}%)"
    )
    for r in rows[1:]:
        print(f"  {r['variable']:14s}: r={r['correlation']} | identiques={r['pct_identique']}% (n={r['n']})")
    print(f"Ecrit: {OUT}")


if __name__ == "__main__":
    main()
