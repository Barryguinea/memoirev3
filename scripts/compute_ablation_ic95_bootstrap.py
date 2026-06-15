#!/usr/bin/env python3
"""Compute 95% percentile bootstrap confidence intervals for the expanded ablation.

Produces the IC95% values shown in the memoire (chapitre 4,
tableau tab:ablation_ic95) for the two decision-critical metrics:
  - detection_any
  - fausses_notif_cow_day

Outputs:
  - data/ablation_ic95_bootstrap.csv
  - data/ablation_ic95_bootstrap.manifest.json (SHA-256 of inputs and outputs)

Deterministic: B = 20_000 resamples, seed = 42 for detection_any, seed = 43 for
fausses_notif_cow_day (distinct streams avoid spurious correlation between the
two percentile intervals).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "ablation_expanded_results.csv"
OUT_CSV = ROOT / "data" / "ablation_ic95_bootstrap.csv"
OUT_MANIFEST = ROOT / "data" / "ablation_ic95_bootstrap.manifest.json"

B = 20_000
ORDER = ["A. Complet", "B. Sans IF", "C. Sans regles", "D. LOF + regles"]
LABELS = {
    "A. Complet": "A. IF + regles",
    "B. Sans IF": "B. Regles seules",
    "C. Sans regles": "C. IF seul",
    "D. LOF + regles": "D. LOF + regles",
}
SEEDS = {"detection_any": 42, "fausses_notif_cow_day": 43}


def percentile_ic95(values: np.ndarray, *, rng: np.random.Generator) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(v), size=(B, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(v.mean()), float(lo), float(hi)


def main() -> None:
    df = pd.read_csv(SRC)
    rows = []
    for variante in ORDER:
        sub = df[df["variante"] == variante]
        for metric, seed in SEEDS.items():
            vals = sub[metric].dropna().to_numpy()
            if len(vals) == 0:
                rows.append({
                    "variante": LABELS[variante], "metric": metric,
                    "n": 0, "mean": None, "ic95_low": None, "ic95_high": None,
                })
                continue
            mean, lo, hi = percentile_ic95(vals, rng=np.random.default_rng(seed))
            rows.append({
                "variante": LABELS[variante], "metric": metric,
                "n": int(len(vals)), "mean": mean,
                "ic95_low": lo, "ic95_high": hi,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    manifest = {
        "script": "compute_ablation_ic95_bootstrap.py",
        "method": "percentile",
        "B": B,
        "seeds": SEEDS,
        "n_cases_per_variant": 90,
        "source_csv": str(SRC.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
        "output_csv": str(OUT_CSV.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUT_CSV.read_bytes()).hexdigest(),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(out.to_string(index=False))
    print()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
