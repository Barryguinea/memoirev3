#!/usr/bin/env python3
"""SHAP feature importance pour l'Isolation Forest du pipeline de détection.

Entraîne l'IF sur les features historiques (*_rrz, *_d1_per_hour, *_vs_mm7,
sin/cos time) et calcule les valeurs de Shapley avec `shap.TreeExplainer`
pour identifier les features qui contribuent le plus au score d'anomalie.

Pour des raisons d'interprétabilité et de temps de calcul, l'analyse est
conduite :
  - sur une vache représentative (par défaut : 8081, utilisée dans
    la chronologie fig:timeline_8081)
  - sur un échantillon de 1000 bins d'intervalle (15 min) si n > 1000
  - TreeExplainer (exact pour forêts d'arbres)

Produit :
  - data/shap/shap_feature_importance.csv  (moyenne |SHAP| par feature)
  - data/shap/shap_summary_plot.png        (top 15 features)
  - data/shap/shap_bar_plot.png            (barres, top 15)
  - data/shap/summary.md                   (résumé Markdown)
  - data/shap/manifest.json                (SHA-256 input/output)

Déterministe : seed=42.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.features import build_interval_features  # noqa: E402
from core.model_if import _default_feature_cols, N_ESTIMATORS, MAX_FEATURES  # noqa: E402
from core.io import COW, TIME, load_csv  # noqa: E402


SRC_CSV = ROOT / "data" / "brut.csv"
OUT_DIR = ROOT / "data" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COW = 8081         # Vache utilisée dans fig:timeline_8081
INTERVAL = "15T"
WINDOW_BASELINE = 48      # 12h de baseline
SAMPLE_SIZE = 1000
SEED = 42

BASE_SENSOR_COLS = ["Steps", "Motion Index", "Lying Time", "Standing Time",
                    "Transitions", "Transitions Up", "Transitions Down"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pretty_feature_name(f: str) -> str:
    """Rend lisible un nom de feature technique pour le mémoire.

    Exemples :
      - Steps_sum_rrz             -> Steps sum (z-score roul.)
      - Motion Index_mean_d1_per_hour -> MI moy. (dérivée/h)
      - Lying Time_mean_vs_mm7    -> Lying moy. (écart 7-bins)
      - Motion Index_mean_log     -> MI moy. (log)
      - sin_hour                  -> sin(heure)
    """
    short = {
        "Steps": "Steps",
        "Motion Index": "MI",
        "Lying Time": "Lying",
        "Standing Time": "Standing",
        "Transitions Up": "Trans.Up",
        "Transitions Down": "Trans.Down",
        "Transitions": "Trans.",
    }
    # Features de temps cycliques
    time_map = {
        "sin_hour": "sin(heure)", "cos_hour": "cos(heure)",
        "sin_dow": "sin(jour sem.)", "cos_dow": "cos(jour sem.)",
    }
    if f in time_map:
        return time_map[f]

    # 1. Extraire le nom du capteur (en cherchant le plus long match d'abord)
    sensor_name = None
    sensor_short = None
    for k in sorted(short.keys(), key=len, reverse=True):
        if f.startswith(k + "_"):
            sensor_name = k
            sensor_short = short[k]
            break
    if sensor_name is None:
        return f  # fallback

    # 2. Le reste après le capteur, ex: "_mean_d1_per_hour", "_sum_rrz", "_mean_log_vs_mm7"
    rest = f[len(sensor_name):]  # commence par "_"

    # 3. Agrégation
    if rest.startswith("_sum_"):
        agg = "sum"
        rest = rest[len("_sum_"):]
    elif rest.startswith("_mean_"):
        agg = "moy."
        rest = rest[len("_mean_"):]
    else:
        agg = ""
        rest = rest.lstrip("_")

    # 4. Transformation / suffixe (ordre important : vérifier les plus longs d'abord)
    transform_map = [
        ("d1_per_hour", "dérivée/h"),
        ("log_vs_mm7", "log, écart 7-bins"),
        ("log_rrz", "log, z-score roul."),
        ("vs_mm7", "écart 7-bins"),
        ("log", "log"),
        ("rrz", "z-score roul."),
        ("mm7", "moyenne 7-bins"),
        ("d1", "dérivée"),
    ]
    transform = None
    for k, v in transform_map:
        if rest == k:
            transform = v
            break

    if transform is not None:
        return f"{sensor_short} {agg} ({transform})".strip()
    # fallback : garder le suffixe brut
    return f"{sensor_short} {agg} {rest}".strip()


def main() -> None:
    print(f"Source : {SRC_CSV}")
    # load_csv() renomme vers Cow/T canoniques et parse H:MM:SS → minutes (Lying, Standing)
    df = load_csv(str(SRC_CSV))
    df[COW] = df[COW].astype(str)

    target_str = str(TARGET_COW)
    if target_str not in df[COW].values:
        print(f"⚠ Vache {TARGET_COW} absente, utilisation première vache disponible")
        cow = df[COW].unique()[0]
    else:
        cow = target_str
    df_cow = df[df[COW] == cow].copy()
    print(f"Vache {cow} : {len(df_cow):,} bins 15-min bruts")

    # Build interval features (core.features utilise TIME='T' par défaut)
    interval_df = build_interval_features(
        df_cow, time_col=TIME, interval=INTERVAL,
        cols=BASE_SENSOR_COLS, window_baseline=WINDOW_BASELINE,
    )
    feat_cols = _default_feature_cols(interval_df)
    print(f"Features IF : {len(feat_cols)} → {feat_cols[:6]} ...")

    # Clean (drop NaN which occur at start of rolling windows)
    X = interval_df[feat_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    print(f"Bins après nettoyage : {len(X):,}")

    # Fit IF (same hyperparams as memoir)
    scaler = RobustScaler(quantile_range=(10, 90))
    Xs = scaler.fit_transform(X)
    iforest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_features=MAX_FEATURES,
        contamination=0.06,
        random_state=SEED, n_jobs=-1,
    )
    iforest.fit(Xs)
    print(f"IF entraîné : {N_ESTIMATORS} arbres, contamination=0.06")

    # SHAP (TreeExplainer exact pour forêts d'arbres)
    rng = np.random.default_rng(SEED)
    if len(X) > SAMPLE_SIZE:
        idx = rng.choice(len(X), SAMPLE_SIZE, replace=False)
        X_sample = Xs[idx]
        X_orig_sample = X.iloc[idx].reset_index(drop=True)
    else:
        X_sample = Xs
        X_orig_sample = X.reset_index(drop=True)

    print(f"Calcul SHAP sur {len(X_sample):,} bins ...")
    explainer = shap.TreeExplainer(iforest)
    # Sortie : shap_values est un array (n_samples, n_features)
    # IF = anomaly score ; valeurs négatives = plus anormal
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Moyenne |SHAP| par feature → importance globale
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feat_cols,
        "feature_pretty": [pretty_feature_name(f) for f in feat_cols],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1
    importance_df["cumulative_share"] = (
        importance_df["mean_abs_shap"].cumsum() / importance_df["mean_abs_shap"].sum()
    )

    out_csv = OUT_DIR / "shap_feature_importance.csv"
    importance_df.to_csv(out_csv, index=False)

    # Summary plot (top 15)
    top_k = min(15, len(feat_cols))
    top_features = importance_df.head(top_k)["feature"].values
    top_idx = [feat_cols.index(f) for f in top_features]

    # Bar plot simple (horizontal)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.4]})

    # --- Panneau 1 : Bar plot importance ---
    ax = axes[0]
    top_df = importance_df.head(top_k).iloc[::-1]  # reverse for top-at-top
    y_pos = np.arange(len(top_df))
    bars = ax.barh(y_pos, top_df["mean_abs_shap"].values,
                   color="#1f4e79", alpha=0.8, edgecolor="#1f4e79")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["feature_pretty"].values, fontsize=9)
    ax.set_xlabel(r"Importance moyenne $|\mathrm{SHAP}|$", fontsize=10)
    ax.set_title(f"(a) Top {top_k} features par importance globale", fontsize=11)
    ax.grid(alpha=0.15, axis="x")
    for bar, val in zip(bars, top_df["mean_abs_shap"].values):
        ax.text(val + max(top_df["mean_abs_shap"]) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color="#1f4e79")

    # --- Panneau 2 : Beeswarm (manuel pour plus de contrôle) ---
    ax = axes[1]
    # Pour chaque top feature, plot jittered SHAP values colored by feature value
    for i, (fidx, fname_pretty) in enumerate(zip(top_idx[::-1], top_df["feature_pretty"].values)):
        sv = shap_values[:, fidx]
        fv = X_orig_sample.iloc[:, fidx].values
        # Normalise la valeur pour la couleur (z-score à l'affichage)
        fv_norm = (fv - np.nanmin(fv)) / (np.nanmax(fv) - np.nanmin(fv) + 1e-9)
        # Jitter y
        jitter = rng.uniform(-0.35, 0.35, size=len(sv))
        sc = ax.scatter(sv, np.full(len(sv), i) + jitter,
                        c=fv_norm, cmap="coolwarm", s=5, alpha=0.5, edgecolor="none")
    ax.set_yticks(np.arange(len(top_df)))
    ax.set_yticklabels(top_df["feature_pretty"].values, fontsize=9)
    ax.axvline(0, color="gray", lw=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel(r"Valeur SHAP (contribution au score IF)", fontsize=10)
    ax.set_title("(b) Distribution des contributions (couleur = valeur feature)", fontsize=11)
    cbar = plt.colorbar(sc, ax=ax, label="Valeur feature (normalisée)", shrink=0.85)
    cbar.ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, axis="x")

    fig.suptitle(f"Importance des features — Isolation Forest (vache {cow}, {len(X_sample):,} bins)",
                 fontsize=12, fontweight="bold")
    png_out = OUT_DIR / "shap_summary_plot.png"
    pdf_out = OUT_DIR / "shap_summary_plot.pdf"
    fig.savefig(png_out, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)

    # Markdown summary
    lines = [
        f"# SHAP Feature Importance — Isolation Forest\n",
        f"Vache analysée : **{cow}**  ",
        f"Bins d'intervalle : {len(X_sample):,} (échantillon aléatoire, seed {SEED})  ",
        f"Features IF totales : {len(feat_cols)}  ",
        f"Hyperparamètres : n_estimators={N_ESTIMATORS}, max_features={MAX_FEATURES}, "
        f"contamination=0.06  ",
        "",
        f"## Top 10 features par importance globale",
        "",
        "| Rang | Feature | |SHAP| moyen | Part cumul. |",
        "|---|---|---|---|",
    ]
    for _, row in importance_df.head(10).iterrows():
        lines.append(
            f"| {int(row['rank'])} | `{row['feature']}` → *{row['feature_pretty']}* | "
            f"{row['mean_abs_shap']:.4f} | {row['cumulative_share']:.1%} |"
        )
    lines.append("")
    # Part des 3 premières
    top3_share = importance_df.head(3)["mean_abs_shap"].sum() / importance_df["mean_abs_shap"].sum()
    top10_share = importance_df.head(10)["mean_abs_shap"].sum() / importance_df["mean_abs_shap"].sum()
    lines.append(f"**Concentration** : les 3 premières features expliquent "
                 f"{top3_share:.1%} de l'importance totale ; les 10 premières "
                 f"{top10_share:.1%}.")

    summary_out = OUT_DIR / "summary.md"
    summary_out.write_text("\n".join(lines), encoding="utf-8")

    # Manifest
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "source": {str(SRC_CSV.relative_to(ROOT)): _sha256(SRC_CSV)},
        "outputs": {
            str(out_csv.relative_to(ROOT)): _sha256(out_csv),
            str(png_out.relative_to(ROOT)): _sha256(png_out),
            str(pdf_out.relative_to(ROOT)): _sha256(pdf_out),
            str(summary_out.relative_to(ROOT)): _sha256(summary_out),
        },
        "params": {
            "target_cow": int(cow),
            "interval": INTERVAL,
            "window_baseline_bins": WINDOW_BASELINE,
            "sample_size": len(X_sample),
            "n_features": len(feat_cols),
            "seed": SEED,
            "iforest": {"n_estimators": N_ESTIMATORS, "max_features": MAX_FEATURES,
                        "contamination": 0.06},
        },
        "top_features": importance_df.head(10).to_dict(orient="records"),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                            encoding="utf-8")

    print()
    print(summary_out.read_text(encoding="utf-8"))
    print()
    print(f"Figure : {png_out}")


if __name__ == "__main__":
    main()
