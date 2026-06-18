"""Temps de correlation effectif (tau) et taille d'echantillon effective (n_eff).

Quantifie la pseudo-replication du corpus IceQube: les mesures aux 15 minutes ne sont
pas independantes. Pour chaque vache et chaque signal (pas, Motion Index, transitions,
posture couchee), on estime sur une grille reguliere de 15 minutes:
  - rho(k) : autocorrelation au decalage k (paires completes, gaps ignores) ;
  - tau_e : premier decalage ou rho(k) < 1/e (~0,368), temps de premiere decorrelation ;
  - tau_int : temps d'autocorrelation integre 1 + 2*somme(rho) sur la sequence positive
              initiale (troncature de Geyer), qui donne l'inflation de variance ;
  - n_eff = n_obs / tau_int : nombre d'observations reellement independantes.

Agrege ensuite par mediane entre vaches et par somme des n_eff. Aucune donnee brute
n'est ecrite: seules les statistiques derivees sont affichees.

Usage : ``python scripts/compute_correlation_time.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.io import load_csv, COW, TIME, STEPS, MI, TRANSITIONS, LYING

CSV = "data/brut.csv"
STEP_MIN = 15            # cadence nominale du capteur (minutes)
MAX_LAG = 192           # 48 h de decalages explores
WINDOW = 48             # 12 h de totaux glissants = niveau de decision HYPO
SIGNALS = [STEPS, MI, TRANSITIONS, LYING]
INV_E = float(np.exp(-1.0))  # ~0,3679


def _pairwise_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """ACF par paires completes (NaN = trou), normalisee par la variance globale."""
    mask = ~np.isnan(x)
    mu = np.nanmean(x)
    var = np.nanmean((x[mask] - mu) ** 2)
    rho = np.full(max_lag + 1, np.nan)
    rho[0] = 1.0
    if var <= 0:
        return rho
    xc = x - mu
    for k in range(1, max_lag + 1):
        a, b = xc[:-k], xc[k:]
        both = ~np.isnan(a) & ~np.isnan(b)
        if both.sum() >= 30:
            rho[k] = np.mean(a[both] * b[both]) / var
    return rho


def _tau_e(rho: np.ndarray) -> float:
    """Premier decalage (en pas) ou l'ACF retombe sous 1/e."""
    for k in range(1, len(rho)):
        if not np.isnan(rho[k]) and rho[k] < INV_E:
            return float(k)
    return float(len(rho) - 1)


def _tau_int(rho: np.ndarray) -> float:
    """Temps d'autocorrelation integre, troncature a la sequence positive initiale."""
    s = 0.0
    for k in range(1, len(rho)):
        if np.isnan(rho[k]) or rho[k] <= 0:
            break
        s += rho[k]
    return max(1.0, 1.0 + 2.0 * s)


def main(csv: str = CSV) -> None:
    df = load_csv(csv)
    print(f"corpus : {len(df)} lignes, {df[COW].nunique()} vaches, cadence {STEP_MIN} min\n")

    rows = []
    for sig in SIGNALS:
        for cow, g in df.groupby(COW):
            g = g.drop_duplicates(TIME).set_index(TIME).sort_index()
            s = g[sig].asfreq(f"{STEP_MIN}min")     # grille reguliere, trous = NaN
            n_obs = int(s.notna().sum())
            if n_obs < 96:                           # < 1 jour exploitable : ignorer
                continue
            rho = _pairwise_acf(s.to_numpy(dtype=float), MAX_LAG)
            te, ti = _tau_e(rho), _tau_int(rho)
            rows.append({"signal": sig, "cow": cow, "n_obs": n_obs,
                         "tau_e": te, "tau_int": ti, "n_eff": n_obs / ti})

    res = pd.DataFrame(rows)
    print(f"{'signal':16}{'tau_e (h)':>12}{'tau_int (h)':>14}{'n_obs':>9}{'n_eff':>9}{'ratio':>8}")
    for sig in SIGNALS:
        sub = res[res.signal == sig]
        te_h = sub.tau_e.median() * STEP_MIN / 60
        ti_h = sub.tau_int.median() * STEP_MIN / 60
        n_obs, n_eff = int(sub.n_obs.sum()), sub.n_eff.sum()
        print(f"{sig:16}{te_h:>12.2f}{ti_h:>14.2f}{n_obs:>9}{n_eff:>9.0f}{n_obs / n_eff:>8.1f}")

    n_obs_tot = int(res.groupby("cow").n_obs.first().sum())
    print(f"\nLecture (signal brut 15 min) : l'ACF retombe sous 1/e en ~"
          f"{res[res.signal == MI].tau_e.median() * STEP_MIN / 60:.1f} h ; "
          f"temps integre median ~{res[res.signal == MI].tau_int.median() * STEP_MIN / 60:.1f} h.")
    print(f"Les {n_obs_tot} mesures valent deja moins de ~{res[res.signal == MI].n_eff.sum():.0f} "
          f"observations independantes (Motion Index).")

    # --- Niveau de decision HYPO : totaux glissants de 12 h ---
    wrows = []
    for sig in [STEPS, MI]:
        for cow, g in df.groupby(COW):
            g = g.drop_duplicates(TIME).set_index(TIME).sort_index()
            s = g[sig].asfreq(f"{STEP_MIN}min").rolling(WINDOW, min_periods=WINDOW).sum()
            n_obs = int(s.notna().sum())
            if n_obs < 96:
                continue
            rho = _pairwise_acf(s.to_numpy(dtype=float), MAX_LAG)
            ti = _tau_int(rho)
            wrows.append({"signal": sig, "cow": cow, "n_obs": n_obs,
                          "tau_int": ti, "n_eff": n_obs / ti})
    wres = pd.DataFrame(wrows)
    print(f"\nNiveau de decision (totaux glissants {WINDOW * STEP_MIN // 60} h) :")
    print(f"{'signal':16}{'tau_int (h)':>14}{'n_eff/vache':>14}{'n_eff total':>14}")
    for sig in [STEPS, MI]:
        sub = wres[wres.signal == sig]
        ti_h = sub.tau_int.median() * STEP_MIN / 60
        print(f"{sig:16}{ti_h:>14.1f}{sub.n_eff.median():>14.0f}{sub.n_eff.sum():>14.0f}")
    print(f"\nA ce niveau, le temps de correlation atteint ~"
          f"{wres[wres.signal == MI].tau_int.median() * STEP_MIN / 60:.0f} h (ordre de la fenetre) : "
          f"chaque vache n'apporte que ~{wres[wres.signal == MI].n_eff.median():.0f} points de decision "
          f"independants, d'ou le choix de la vache (n=11) comme unite statistique.")


if __name__ == "__main__":
    main()
