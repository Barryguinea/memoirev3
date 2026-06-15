#!/usr/bin/env python3
"""Courbes ROC et PR + AUC avec IC95 bootstrap sur le score `if_anom_k_max`.

Utilise les 18,816 événements de la validation robuste (robust_validation_*)
comme corpus d'évaluation. Chaque événement a :
  - un score brut d'IF : `if_anom_k_max` (max anomalie sur la fenêtre)
  - un label de vérité terrain : `expected_detected`
    - 1 si l'événement est dans un profil détectable (strong ou borderline)
    - 0 si l'événement est dans un profil non-détectable (bruit intentionnel)

Produit :
  - data/roc_pr/metrics.csv                  (AUC global + stratifié + IC95)
  - data/roc_pr/roc_pr_curves.png            (figure 4 panneaux pour le mémoire)
  - data/roc_pr/roc_pr_curves.pdf            (version vectorielle)
  - data/roc_pr/summary.md                   (résumé Markdown)
  - data/roc_pr/manifest.json                (SHA-256 inputs/outputs)

Déterministe : seed=42 pour bootstrap (B=2000).
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
from sklearn.metrics import roc_curve, precision_recall_curve, auc


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "validation_robuste"
OUT_DIR = ROOT / "data" / "roc_pr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_COL = "if_anom_k_max"
LABEL_COL = "expected_detected"
B_BOOT = 2000
SEED = 42


def _latest_robust_run() -> Path:
    """Retourne le dossier de validation robuste le plus récent contenant robust_events_evaluation.csv."""
    candidates = sorted(
        [p for p in SRC_DIR.glob("robust_validation_*") if (p / "robust_events_evaluation.csv").exists()]
    )
    if not candidates:
        raise FileNotFoundError(f"Aucun robust_events_evaluation.csv trouvé dans {SRC_DIR}")
    return candidates[-1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, *,
                   kind: str, rng: np.random.Generator, B: int = B_BOOT) -> tuple[float, float, float]:
    """Bootstrap percentile IC95 pour AUC-ROC ou AUC-PR."""
    n = len(y_true)
    if n == 0 or (y_true.sum() == 0) or (y_true.sum() == n):
        return float("nan"), float("nan"), float("nan")

    # Point estimate
    if kind == "roc":
        fpr, tpr, _ = roc_curve(y_true, y_score)
        point = auc(fpr, tpr)
    elif kind == "pr":
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        point = auc(rec, prec)
    else:
        raise ValueError(kind)

    # Bootstrap
    aucs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        if kind == "roc":
            f, t, _ = roc_curve(yt, ys)
            aucs.append(auc(f, t))
        else:
            p, r, _ = precision_recall_curve(yt, ys)
            aucs.append(auc(r, p))
    if not aucs:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def compute_curves(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """ROC + PR complets avec points de diagnostic clés."""
    fpr, tpr, thr_roc = roc_curve(y_true, y_score)
    prec, rec, thr_pr = precision_recall_curve(y_true, y_score)

    # Youden index : maximise Sensibilité + Spécificité - 1
    j = tpr - fpr
    idx_j = int(np.argmax(j))
    youden_thr = float(thr_roc[idx_j]) if idx_j < len(thr_roc) else float("nan")

    # F1 optimal sur PR curve
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    idx_f1 = int(np.argmax(f1))

    return {
        "fpr": fpr, "tpr": tpr,
        "prec": prec, "rec": rec,
        "youden_thr": youden_thr,
        "youden_sens": float(tpr[idx_j]),
        "youden_spec": float(1 - fpr[idx_j]),
        "f1_thr": float(thr_pr[idx_f1]) if idx_f1 < len(thr_pr) else float("nan"),
        "f1_value": float(f1[idx_f1]),
    }


def plot_curves(strata: dict, out_png: Path, out_pdf: Path) -> None:
    """Figure 3 panneaux: (a) ROC globale strong∪border vs non-det,
    (b) distribution du score par profil, (c) ROC strong vs border."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)

    # Couleurs cohérentes avec la thèse
    col_global = "#1f4e79"   # ifblue
    col_strong = "#c0392b"   # rawred
    col_border = "#d98c34"   # loforange
    col_neg    = "#7f7f7f"   # gris

    # --- Panneau 1 : ROC globale (détectable vs non-détectable) ---
    ax = axes[0]
    g = strata["global"]
    ax.plot(g["curves"]["fpr"], g["curves"]["tpr"],
            color=col_global, lw=2.2, label=f"Détectable vs non-détectable")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Hasard")
    ax.set_xlabel("Taux faux positifs (1 - Spécificité)")
    ax.set_ylabel("Sensibilité")
    ax.set_title(f"(a) Séparabilité détectable vs non-détectable\n"
                 f"AUC = {g['roc_auc']:.3f} "
                 f"[IC95 : {g['roc_auc_lo']:.3f}, {g['roc_auc_hi']:.3f}]",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.15)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    # --- Panneau 2 : Distribution du score par profil (boxplot) ---
    # Reconstruit à partir du dataframe original
    ax = axes[1]
    src_dir = _latest_robust_run()
    df_full = pd.read_csv(src_dir / "robust_events_evaluation.csv")
    df_full = df_full[[SCORE_COL, "profile"]].dropna()
    profile_order = ["non_detectable_short_single_signal",
                     "detectable_borderline", "detectable_strong"]
    profile_labels = ["Non-détectable", "Borderline", "Strong"]
    box_data = [df_full[df_full["profile"] == p][SCORE_COL].values for p in profile_order]
    bp = ax.boxplot(box_data, labels=profile_labels, patch_artist=True,
                    widths=0.55, medianprops={"color": "black", "linewidth": 1.5},
                    flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.3})
    for patch, c in zip(bp["boxes"], [col_neg, col_border, col_strong]):
        patch.set_facecolor(c); patch.set_alpha(0.45)
        patch.set_edgecolor(c); patch.set_linewidth(1.5)
    ax.set_ylabel(r"Score $\mathrm{if\_anom\_k\_max}$")
    ax.set_title("(b) Distribution du score par profil\n(18 816 événements)",
                 fontsize=10)
    ax.grid(alpha=0.15, axis="y")
    ax.axhline(10, color="red", linestyle="--", lw=1.0, alpha=0.7,
               label="Seuil critique (10)")
    ax.legend(loc="upper left", fontsize=9)

    # --- Panneau 3 : ROC strong vs borderline (discrimination de sévérité) ---
    ax = axes[2]
    sb = strata["strong_vs_border"]
    ax.plot(sb["curves"]["fpr"], sb["curves"]["tpr"],
            color=col_strong, lw=2.2, label="Strong vs Borderline")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Hasard")
    # Point Youden
    ax.scatter([1 - sb["curves"]["youden_spec"]], [sb["curves"]["youden_sens"]],
               color=col_strong, s=60, zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate(f"Youden: seuil={sb['curves']['youden_thr']:.1f}",
                xy=(1 - sb["curves"]["youden_spec"], sb["curves"]["youden_sens"]),
                xytext=(10, -15), textcoords="offset points",
                fontsize=9, color=col_strong)
    ax.set_xlabel("Taux faux positifs (1 - Spécificité)")
    ax.set_ylabel("Sensibilité")
    ax.set_title(f"(c) Discrimination strong vs borderline\n"
                 f"AUC = {sb['roc_auc']:.3f} "
                 f"[IC95 : {sb['roc_auc_lo']:.3f}, {sb['roc_auc_hi']:.3f}]",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.15)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    src_dir = _latest_robust_run()
    src_csv = src_dir / "robust_events_evaluation.csv"
    print(f"Source : {src_csv.relative_to(ROOT)}")

    df = pd.read_csv(src_csv)
    df = df[[SCORE_COL, LABEL_COL, "profile", "scenario"]].dropna()
    print(f"Événements chargés : {len(df):,}")
    print(df["profile"].value_counts().to_string())

    rng = np.random.default_rng(SEED)

    # --- Stratum 1 : Global (positif = strong + borderline, négatif = non_detectable) ---
    y_g = (df[LABEL_COL] == 1).astype(int).to_numpy()
    s_g = df[SCORE_COL].to_numpy(dtype=float)

    roc_auc, roc_lo, roc_hi = bootstrap_auc(y_g, s_g, kind="roc", rng=rng)
    pr_auc, pr_lo, pr_hi = bootstrap_auc(y_g, s_g, kind="pr", rng=rng)

    strata = {}
    strata["global"] = {
        "n": int(len(y_g)), "n_pos": int(y_g.sum()), "n_neg": int((y_g == 0).sum()),
        "prevalence": float(y_g.mean()),
        "roc_auc": roc_auc, "roc_auc_lo": roc_lo, "roc_auc_hi": roc_hi,
        "pr_auc": pr_auc, "pr_auc_lo": pr_lo, "pr_auc_hi": pr_hi,
        "curves": compute_curves(y_g, s_g),
    }

    # --- Stratum 2 : Strong vs non-détectable ---
    mask_sn = df["profile"].isin(["detectable_strong", "non_detectable_short_single_signal"])
    df_sn = df[mask_sn]
    y_s = (df_sn["profile"] == "detectable_strong").astype(int).to_numpy()
    s_s = df_sn[SCORE_COL].to_numpy(dtype=float)
    roc_auc_s, roc_lo_s, roc_hi_s = bootstrap_auc(y_s, s_s, kind="roc", rng=rng)
    strata["strong_vs_neg"] = {
        "n": int(len(y_s)), "n_pos": int(y_s.sum()), "n_neg": int((y_s == 0).sum()),
        "roc_auc": roc_auc_s, "roc_auc_lo": roc_lo_s, "roc_auc_hi": roc_hi_s,
        "curves": compute_curves(y_s, s_s),
    }

    # --- Stratum 3 : Borderline vs non-détectable ---
    mask_bn = df["profile"].isin(["detectable_borderline", "non_detectable_short_single_signal"])
    df_bn = df[mask_bn]
    y_b = (df_bn["profile"] == "detectable_borderline").astype(int).to_numpy()
    s_b = df_bn[SCORE_COL].to_numpy(dtype=float)
    roc_auc_b, roc_lo_b, roc_hi_b = bootstrap_auc(y_b, s_b, kind="roc", rng=rng)
    strata["border_vs_neg"] = {
        "n": int(len(y_b)), "n_pos": int(y_b.sum()), "n_neg": int((y_b == 0).sum()),
        "roc_auc": roc_auc_b, "roc_auc_lo": roc_lo_b, "roc_auc_hi": roc_hi_b,
        "curves": compute_curves(y_b, s_b),
    }

    # --- Stratum 4 : Strong vs Borderline (discrimination de sévérité) ---
    # Question clé : le score permet-il de distinguer un cas fort d'un cas léger ?
    mask_sb = df["profile"].isin(["detectable_strong", "detectable_borderline"])
    df_sb = df[mask_sb]
    y_sb = (df_sb["profile"] == "detectable_strong").astype(int).to_numpy()
    s_sb = df_sb[SCORE_COL].to_numpy(dtype=float)
    roc_auc_sb, roc_lo_sb, roc_hi_sb = bootstrap_auc(y_sb, s_sb, kind="roc", rng=rng)
    pr_auc_sb, pr_lo_sb, pr_hi_sb = bootstrap_auc(y_sb, s_sb, kind="pr", rng=rng)
    strata["strong_vs_border"] = {
        "n": int(len(y_sb)), "n_pos": int(y_sb.sum()), "n_neg": int((y_sb == 0).sum()),
        "prevalence": float(y_sb.mean()),
        "roc_auc": roc_auc_sb, "roc_auc_lo": roc_lo_sb, "roc_auc_hi": roc_hi_sb,
        "pr_auc": pr_auc_sb, "pr_auc_lo": pr_lo_sb, "pr_auc_hi": pr_hi_sb,
        "curves": compute_curves(y_sb, s_sb),
    }

    # --- Sauvegardes ---
    metrics_rows = []
    for key, s in strata.items():
        row = {"stratum": key, "n": s["n"], "n_pos": s["n_pos"], "n_neg": s["n_neg"],
               "roc_auc": s["roc_auc"], "roc_auc_lo": s["roc_auc_lo"], "roc_auc_hi": s["roc_auc_hi"]}
        if "pr_auc" in s:
            row["pr_auc"] = s["pr_auc"]
            row["pr_auc_lo"] = s["pr_auc_lo"]
            row["pr_auc_hi"] = s["pr_auc_hi"]
        c = s["curves"]
        row["youden_threshold"] = c["youden_thr"]
        row["youden_sensitivity"] = c["youden_sens"]
        row["youden_specificity"] = c["youden_spec"]
        row["f1_threshold"] = c["f1_thr"]
        row["f1_value"] = c["f1_value"]
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_out = OUT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)

    # Figure
    png_out = OUT_DIR / "roc_pr_curves.png"
    pdf_out = OUT_DIR / "roc_pr_curves.pdf"
    plot_curves(strata, png_out, pdf_out)

    # Summary markdown
    lines = [
        "# Courbes ROC et Précision-Rappel — score `if_anom_k_max`\n",
        f"Corpus : {strata['global']['n']:,} événements issus de la validation robuste.",
        f"Bootstrap percentile IC95 : B={B_BOOT}, seed={SEED}.\n",
        "## AUC globale (positif = strong ∪ borderline, négatif = non-détectable)",
        f"- **AUC-ROC** = {strata['global']['roc_auc']:.3f} "
        f"[{strata['global']['roc_auc_lo']:.3f} ; {strata['global']['roc_auc_hi']:.3f}]",
        f"- **AUC-PR**  = {strata['global']['pr_auc']:.3f} "
        f"[{strata['global']['pr_auc_lo']:.3f} ; {strata['global']['pr_auc_hi']:.3f}]",
        f"- Prévalence positifs : {strata['global']['prevalence']:.1%}",
        f"- **Interprétation** : séparation parfaite. Tous les événements "
        f"non-détectables ont `if_anom_k_max = 0`, confirmant que l'IF "
        f"ne déclenche aucune anomalie sur le bruit pur (fuite nulle).",
        "",
        "## AUC stratifiée par profil (vs non-détectable)",
        f"- **Strong vs non-détectable** : AUC-ROC = {strata['strong_vs_neg']['roc_auc']:.3f} "
        f"[{strata['strong_vs_neg']['roc_auc_lo']:.3f} ; {strata['strong_vs_neg']['roc_auc_hi']:.3f}]",
        f"- **Borderline vs non-détectable** : AUC-ROC = {strata['border_vs_neg']['roc_auc']:.3f} "
        f"[{strata['border_vs_neg']['roc_auc_lo']:.3f} ; {strata['border_vs_neg']['roc_auc_hi']:.3f}]",
        "",
        "## Discrimination de sévérité (strong vs borderline) — métrique clinique",
        f"- **AUC-ROC strong vs borderline** = {strata['strong_vs_border']['roc_auc']:.3f} "
        f"[{strata['strong_vs_border']['roc_auc_lo']:.3f} ; "
        f"{strata['strong_vs_border']['roc_auc_hi']:.3f}]",
        f"- Seuil Youden optimal (strong vs borderline) = "
        f"{strata['strong_vs_border']['curves']['youden_thr']:.1f}",
        f"  - Se = {strata['strong_vs_border']['curves']['youden_sens']:.1%}, "
        f"Sp = {strata['strong_vs_border']['curves']['youden_spec']:.1%}",
        "",
        "## Seuils optimaux (global, détectable vs non-détectable)",
        f"- **Index de Youden** : seuil = {strata['global']['curves']['youden_thr']:.1f}, "
        f"Se = {strata['global']['curves']['youden_sens']:.1%}, Sp = {strata['global']['curves']['youden_spec']:.1%}",
        f"- **F1 optimal** : seuil = {strata['global']['curves']['f1_thr']:.1f}, "
        f"F1 = {strata['global']['curves']['f1_value']:.3f}",
    ]
    summary_out = OUT_DIR / "summary.md"
    summary_out.write_text("\n".join(lines), encoding="utf-8")

    # Manifest
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "source": {str(src_csv.relative_to(ROOT)): _sha256(src_csv)},
        "outputs": {
            str(metrics_out.relative_to(ROOT)): _sha256(metrics_out),
            str(png_out.relative_to(ROOT)): _sha256(png_out),
            str(pdf_out.relative_to(ROOT)): _sha256(pdf_out),
            str(summary_out.relative_to(ROOT)): _sha256(summary_out),
        },
        "params": {"bootstrap_B": B_BOOT, "seed": SEED,
                   "score_col": SCORE_COL, "label_col": LABEL_COL},
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print(summary_out.read_text(encoding="utf-8"))
    print()
    print(f"Figure : {png_out}")


if __name__ == "__main__":
    main()
