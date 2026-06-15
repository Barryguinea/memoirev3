#!/usr/bin/env python3
"""Diagramme de fiabilité (calibration) du score `if_anom_k_max`.

Le score brut de l'Isolation Forest n'est pas, en soi, une probabilité
calibrée. Cette analyse évalue dans quelle mesure ce score peut être
transformé en probabilité de sévérité clinique (strong vs borderline)
via une calibration isotone, et visualise la composition des strates
par décile de score.

Deux panneaux :

  (a) Reliability diagram — calibration isotone strong vs borderline
      - On entraîne un `IsotonicRegression` sur 50% des événements
        détectables (split aléatoire, seed=42), calibration évaluée sur les 50%
        restants (hold-out).
      - On bin la probabilité prédite en 10 déciles et on compare la
        fréquence observée (% strong) à la probabilité moyenne prédite.
      - Une calibration parfaite suit la diagonale y=x.
      - Métriques: Brier score, Expected Calibration Error (ECE),
        Maximum Calibration Error (MCE).

  (b) Composition par décile de score
      - Les 18,816 événements (toutes strates) binés en 10 déciles de
        score `if_anom_k_max`.
      - Barre empilée: % strong / borderline / non-détectable par bin.
      - Visualise la monotonicité et l'utilité pratique du score pour
        un triage 3 niveaux.

Sorties :
  - data/calibration/metrics.csv               (Brier, ECE, MCE, seuils F1)
  - data/calibration/reliability_data.csv      (points du reliability diagram)
  - data/calibration/composition_by_decile.csv (table composition)
  - data/calibration/score_calibration.{pdf,png}
  - data/calibration/summary.md
  - data/calibration/manifest.json

Déterministe : seed=42. Reproductible via manifest SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "validation_robuste"
OUT_DIR = ROOT / "data" / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_COL = "if_anom_k_max"
N_BINS_RELIAB = 10
N_BINS_COMP = 10
SEED = 42


def _latest_robust_run() -> Path:
    cand = sorted([p for p in SRC_DIR.glob("robust_validation_*")
                    if (p / "robust_events_evaluation.csv").exists()])
    if not cand:
        raise FileNotFoundError("Aucun robust_events_evaluation.csv")
    return cand[-1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_ece_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
                    ) -> tuple[float, float, pd.DataFrame]:
    """Expected + Maximum Calibration Error, sur bins équi-largeur [0,1].

    Retourne (ECE, MCE, reliability_table).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows = []
    ece, mce = 0.0, 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            rows.append({"bin": b + 1, "bin_low": bins[b], "bin_high": bins[b + 1],
                         "n": 0, "mean_predicted": np.nan, "frac_positive": np.nan,
                         "gap": np.nan})
            continue
        mean_pred = float(y_prob[m].mean())
        frac_pos = float(y_true[m].mean())
        gap = abs(frac_pos - mean_pred)
        ece += (m.sum() / n) * gap
        mce = max(mce, gap)
        rows.append({"bin": b + 1, "bin_low": float(bins[b]), "bin_high": float(bins[b + 1]),
                     "n": int(m.sum()), "mean_predicted": mean_pred,
                     "frac_positive": frac_pos, "gap": gap})
    return float(ece), float(mce), pd.DataFrame(rows)


def composition_by_decile(df: pd.DataFrame, score_col: str, profile_col: str,
                          n_bins: int = 10) -> pd.DataFrame:
    """Composition (strong/borderline/non-detectable) par décile de score."""
    # On utilise qcut uniquement si les valeurs sont suffisamment variées.
    # Sinon (score IF entier, beaucoup de zéros), on bin par seuils équi-largeur.
    try:
        labels, edges = pd.qcut(df[score_col], q=n_bins, retbins=True, duplicates="drop")
    except ValueError:
        edges = np.linspace(df[score_col].min(), df[score_col].max(), n_bins + 1)
        labels = pd.cut(df[score_col], bins=edges, include_lowest=True)
    df = df.assign(_bin=labels)
    comp = (df.groupby("_bin", observed=True)[profile_col]
              .value_counts(normalize=False)
              .unstack(fill_value=0))
    # Proportion
    props = comp.div(comp.sum(axis=1), axis=0)
    # Bin center pour axis
    bin_ranges = [f"[{iv.left:.1f}, {iv.right:.1f}]" for iv in comp.index]
    props.insert(0, "bin_range", bin_ranges)
    props.insert(1, "n_events", comp.sum(axis=1).values)
    props.insert(2, "score_bin_center",
                 [(iv.left + iv.right) / 2 for iv in comp.index])
    return props.reset_index(drop=True)


def main() -> None:
    run_dir = _latest_robust_run()
    src_csv = run_dir / "robust_events_evaluation.csv"
    print(f"Source : {src_csv.relative_to(ROOT)}")
    df = pd.read_csv(src_csv)
    print(f"Événements : {len(df):,}")
    print(f"Profils   : {dict(df['profile'].value_counts())}")

    # ==================================================================
    # Panneau (a) — Reliability diagram (calibration isotone)
    # Tâche binaire : strong (y=1) vs borderline (y=0), non-det exclus.
    # ==================================================================
    det = df[df["profile"].isin(["detectable_strong", "detectable_borderline"])].copy()
    det["y"] = (det["profile"] == "detectable_strong").astype(int)
    score = det[SCORE_COL].to_numpy(dtype=float)
    y = det["y"].to_numpy()

    # Split train/test 50/50 stratifié
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.5, stratify=y, random_state=SEED,
    )
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(score[idx_train], y[idx_train])
    prob_test = iso.predict(score[idx_test])
    y_test = y[idx_test]

    brier = float(brier_score_loss(y_test, prob_test))
    ece, mce, reliab = compute_ece_mce(y_test, prob_test, n_bins=N_BINS_RELIAB)

    # Score brut (non calibré) mapé sur [0,1] pour comparaison
    prob_raw = (score[idx_test] - score.min()) / (score.max() - score.min() + 1e-9)
    brier_raw = float(brier_score_loss(y_test, prob_raw))
    ece_raw, mce_raw, reliab_raw = compute_ece_mce(y_test, prob_raw, n_bins=N_BINS_RELIAB)

    print(f"[calibration isotone] Brier = {brier:.4f}, ECE = {ece:.4f}, MCE = {mce:.4f}")
    print(f"[score brut]          Brier = {brier_raw:.4f}, ECE = {ece_raw:.4f}, MCE = {mce_raw:.4f}")

    # Seuil F1-optimal (sur probabilités calibrées)
    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = []
    for t in thresholds:
        pred = (prob_test >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0)
        f1s.append(f1)
    f1s = np.array(f1s)
    t_f1 = float(thresholds[int(f1s.argmax())])
    f1_max = float(f1s.max())
    print(f"[calibration isotone] Seuil F1-optimal = {t_f1:.2f} (F1 = {f1_max:.3f})")

    # ==================================================================
    # Panneau (b) — Composition par décile de score (tout le corpus)
    # ==================================================================
    comp = composition_by_decile(df, SCORE_COL, "profile", n_bins=N_BINS_COMP)
    print()
    print("Composition par bin de score :")
    print(comp.to_string(index=False))

    # ==================================================================
    # Écriture CSV
    # ==================================================================
    reliab["calibration"] = "isotonic"
    reliab_raw["calibration"] = "raw_minmax"
    reliab_all = pd.concat([reliab, reliab_raw], ignore_index=True)
    reliab_csv = OUT_DIR / "reliability_data.csv"
    reliab_all.to_csv(reliab_csv, index=False)

    comp_csv = OUT_DIR / "composition_by_decile.csv"
    comp.to_csv(comp_csv, index=False)

    metrics_df = pd.DataFrame([
        {"metric": "brier_isotonic", "value": brier},
        {"metric": "brier_raw_minmax", "value": brier_raw},
        {"metric": "ece_isotonic", "value": ece},
        {"metric": "ece_raw_minmax", "value": ece_raw},
        {"metric": "mce_isotonic", "value": mce},
        {"metric": "mce_raw_minmax", "value": mce_raw},
        {"metric": "f1_optimal_threshold_prob", "value": t_f1},
        {"metric": "f1_optimal_value", "value": f1_max},
        {"metric": "n_test", "value": int(len(y_test))},
        {"metric": "n_train", "value": int(len(idx_train))},
    ])
    metrics_csv = OUT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # ==================================================================
    # Figure (2 panneaux)
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    # --- (a) Reliability diagram ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Calibration parfaite")
    # Calibration isotone
    r_iso = reliab.dropna(subset=["mean_predicted"])
    ax.plot(r_iso["mean_predicted"], r_iso["frac_positive"],
            "o-", color="#1f4e79", lw=2, markersize=7,
            label=f"Isotonique (ECE = {ece:.3f}, Brier = {brier:.3f})")
    # Score brut (min-max scaled)
    r_raw = reliab_raw.dropna(subset=["mean_predicted"])
    ax.plot(r_raw["mean_predicted"], r_raw["frac_positive"],
            "s--", color="#b05b1f", lw=1.3, markersize=6, alpha=0.7,
            label=f"Score brut (ECE = {ece_raw:.3f}, Brier = {brier_raw:.3f})")
    # Tailles de bin en histogramme secondaire
    ax2 = ax.twinx()
    centers = (r_iso["mean_predicted"].values)
    widths = np.diff(np.concatenate([[0], centers, [1]]))[:-1] * 0.6
    ax2.bar(centers, r_iso["n"].values, width=widths,
            alpha=0.12, color="#1f4e79", edgecolor="none", zorder=0)
    ax2.set_ylabel("Effectif du bin (isotone)", fontsize=9, color="#1f4e79")
    ax2.tick_params(axis="y", labelsize=8, colors="#1f4e79")

    ax.set_xlabel("Probabilité prédite de profil $\\mathit{strong}$", fontsize=10)
    ax.set_ylabel("Fréquence observée (% $\\mathit{strong}$)", fontsize=10)
    ax.set_title("(a) Diagramme de fiabilité (hold-out, n = {:,})".format(len(y_test)),
                 fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")

    # --- (b) Composition par décile ---
    ax = axes[1]
    colors = {"detectable_strong": "#b22222",
              "detectable_borderline": "#e6a200",
              "non_detectable_short_single_signal": "#5a7f9b"}
    labels_fr = {"detectable_strong": "Strong",
                 "detectable_borderline": "Borderline",
                 "non_detectable_short_single_signal": "Non-détectable"}
    x = np.arange(len(comp))
    bottom = np.zeros(len(comp))
    for prof in ["non_detectable_short_single_signal", "detectable_borderline",
                  "detectable_strong"]:
        if prof not in comp.columns:
            continue
        vals = comp[prof].values
        ax.bar(x, vals, bottom=bottom, color=colors[prof],
               label=labels_fr[prof], edgecolor="white", linewidth=0.5)
        bottom += vals
    # annotations effectif
    for i, n_bin in enumerate(comp["n_events"].values):
        ax.text(i, 1.02, f"n={int(n_bin):,}", ha="center", fontsize=7, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(comp["bin_range"].values, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Décile du score if_anom_k_max", fontsize=10, family="monospace")
    ax.set_ylabel("Composition (proportion)", fontsize=10)
    ax.set_title(f"(b) Composition des strates par décile (n = {len(df):,})",
                 fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.15, axis="y")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

    fig.suptitle("Calibration du score $\\mathtt{if\\_anom\\_k\\_max}$", fontsize=12,
                 fontweight="bold")

    pdf_out = OUT_DIR / "score_calibration.pdf"
    png_out = OUT_DIR / "score_calibration.png"
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Summary Markdown
    # ==================================================================
    lines = [
        "# Calibration du score `if_anom_k_max`\n",
        f"Source : `{src_csv.relative_to(ROOT)}`  ",
        f"Événements : {len(df):,} (strong {(df['profile']=='detectable_strong').sum():,} / "
        f"borderline {(df['profile']=='detectable_borderline').sum():,} / "
        f"non-det {(df['profile']=='non_detectable_short_single_signal').sum():,})",
        "",
        "## Métriques (tâche binaire strong vs borderline, hold-out 50%)",
        "",
        "| Mesure | Score brut (min-max) | Calibration isotone |",
        "|---|---|---|",
        f"| Brier score  | {brier_raw:.4f} | {brier:.4f} |",
        f"| ECE (10 bins)| {ece_raw:.4f} | {ece:.4f} |",
        f"| MCE (10 bins)| {mce_raw:.4f} | {mce:.4f} |",
        "",
        f"**Seuil F1-optimal (sur prob. isotone)** : {t_f1:.2f} (F1 = {f1_max:.3f})",
        "",
        "## Composition des strates par décile de score",
        "",
        comp.round(3).to_markdown(index=False),
    ]
    summary_out = OUT_DIR / "summary.md"
    summary_out.write_text("\n".join(lines), encoding="utf-8")

    # ==================================================================
    # Manifest SHA-256
    # ==================================================================
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "source": {str(src_csv.relative_to(ROOT)): _sha256(src_csv)},
        "outputs": {
            str(metrics_csv.relative_to(ROOT)): _sha256(metrics_csv),
            str(reliab_csv.relative_to(ROOT)): _sha256(reliab_csv),
            str(comp_csv.relative_to(ROOT)): _sha256(comp_csv),
            str(pdf_out.relative_to(ROOT)): _sha256(pdf_out),
            str(png_out.relative_to(ROOT)): _sha256(png_out),
            str(summary_out.relative_to(ROOT)): _sha256(summary_out),
        },
        "params": {"score_col": SCORE_COL, "n_bins_reliability": N_BINS_RELIAB,
                   "n_bins_composition": N_BINS_COMP, "seed": SEED,
                   "test_size": 0.5, "calibration": "IsotonicRegression"},
        "metrics": {
            "brier_isotonic": brier, "brier_raw": brier_raw,
            "ece_isotonic": ece, "ece_raw": ece_raw,
            "mce_isotonic": mce, "mce_raw": mce_raw,
            "f1_optimal_threshold_prob": t_f1, "f1_optimal_value": f1_max,
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                             encoding="utf-8")

    print()
    print(summary_out.read_text(encoding="utf-8"))
    print()
    print(f"✓ Figure : {pdf_out}")


if __name__ == "__main__":
    main()
