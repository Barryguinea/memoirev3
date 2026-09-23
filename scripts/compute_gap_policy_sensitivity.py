# ruff: noqa: E402
"""Effet des trous de donnees sur HYPO, sous les deux politiques de totaux glissants.

Un intervalle de 15 minutes sans mesure vaut zero apres agregation. Dans le code
du manuscrit v3 (politique ``historical``), les totaux glissants de 12 heures
additionnent ces zeros : un trou ressemble alors a une baisse d'activite. La
politique ``coverage_aware`` ecarte les intervalles vides et suspend le total tant
que moins de 90 % de la fenetre est mesuree.

Le script mesure trois choses, sous les deux politiques :

1. l'effet d'un trou seul : un bloc de 12 heures est retire a trois positions de
   la periode surveillee de chaque vache, sans aucune baisse injectee, et l'on
   compte les nouveaux departs HYPO (reference d'apprentissage figee) ;
2. l'ablation principale (44 evenements, cinq variantes) ;
3. le test de stress a reference fixe (198 evenements, cinq variantes).

Les resultats sont ecrits dans un dossier separe muni de sa propre provenance et
de son propre manifeste ; les artefacts scelles ne sont jamais modifies. Voir
docs/politique_trous_de_donnees.md.

Usage : ``python scripts/compute_gap_policy_sensitivity.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

# Permet l'execution directe: `python scripts/compute_gap_policy_sensitivity.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from core import config as C
from core.early_warning import GAP_POLICIES, apply_behavioral_early_warning
from core.features import build_interval_features
from core.io import COW, TIME, available_base_cols, load_csv
from core.model_if import run_if_core
from validation_hypo.ablation import ablation_summary, run_clean_ablation
from validation_hypo.campaign import (
    DEFAULT_MATCH_TOLERANCE_HOURS,
    _heldout_start_time,
    final_params,
    has_informative_heldout_signals,
)
from validation_hypo.stress_campaign import compare_variants, run_stress_campaign, summarize

ROOT = PROJECT_ROOT
RAW_CSV = ROOT / "data/brut.csv"
OUTPUT = ROOT / "data/validation/gap_policy_sensitivity"
SEALED = (
    ROOT / "data/validation/hypo_module",
    ROOT / "data/validation/hypo_stress",
    ROOT / "data/validation/stress_fixed_reference",
    ROOT / "data/validation/derived_metrics",
)
GAP_HOURS = 12
POSITIONS = (0.18, 0.5, 0.82)
ARTIFACTS = (
    "dropout_blocks.csv",
    "ablation_summary.csv",
    "stress_scenario_summary.csv",
    "stress_variant_summary.csv",
    "stress_paired_comparisons.csv",
    "provenance.json",
)


def _hypo(raw: pd.DataFrame, params: dict, policy: str, reference_times=None) -> pd.DataFrame:
    features = build_interval_features(
        raw, time_col=TIME, interval=str(params["interval"]),
        cols=available_base_cols(raw), window_baseline=int(params["window_baseline"]),
    )
    split = run_if_core(
        features, time_col="T", contamination=float(params["contamination"]),
        random_state=int(params["random_state"]), baseline_ratio=float(params["baseline_ratio"]),
        coverage_min_pct=float(params["coverage_min_pct"]),
        sensor_warmup_bins=C.DEFAULT_SENSOR_WARMUP_BINS, reference_times=reference_times,
    )
    out = apply_behavioral_early_warning(split, interval=str(params["interval"]), gap_policy=policy)
    out["T"] = pd.to_datetime(out["T"])
    return out


def dropout_blocks(raw_all: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Nouveaux departs HYPO provoques par un trou de donnees seul."""
    tolerance = pd.Timedelta(hours=DEFAULT_MATCH_TOLERANCE_HOURS)
    rows = []
    for cow in sorted(raw_all[COW].unique()):
        raw = raw_all[raw_all[COW] == cow].sort_values(TIME)
        if (raw[TIME].max() - raw[TIME].min()).total_seconds() / 86400.0 < 14:
            continue
        heldout = _heldout_start_time(
            raw, interval=str(params["interval"]),
            window_baseline=int(params["window_baseline"]),
            baseline_ratio=float(params["baseline_ratio"]),
            coverage_min_pct=float(params["coverage_min_pct"]),
        )
        if not has_informative_heldout_signals(raw, heldout_start=heldout):
            continue
        future = raw[raw[TIME] >= heldout]
        first, last = future[TIME].min(), future[TIME].max()
        for policy in GAP_POLICIES:
            clean = _hypo(raw, params, policy)
            reference = pd.DatetimeIndex(clean.loc[clean["dataset_split"].eq("baseline"), "T"])
            clean_starts = clean.loc[clean["behavioral_warning_start"].eq(1), "T"]
            for position in POSITIONS:
                gap_start = first + (last - first) * position
                gap_end = gap_start + pd.Timedelta(hours=GAP_HOURS)
                kept = raw[~((raw[TIME] >= gap_start) & (raw[TIME] < gap_end))]
                run = _hypo(kept, params, policy, reference_times=reference)
                starts = run.loc[run["behavioral_warning_start"].eq(1), "T"]
                new = [
                    t for t in starts
                    if gap_start <= t <= gap_end + pd.Timedelta(hours=24)
                    and not ((clean_starts - t).abs() <= tolerance).any()
                ]
                rows.append({
                    "gap_policy": policy,
                    "cow": cow,
                    "position": position,
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                    "new_hypo_starts": len(new),
                    "first_start_delay_after_gap_h": (
                        round((new[0] - gap_end).total_seconds() / 3600.0, 4) if new else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def write_provenance(output: Path) -> None:
    sources = sorted({
        *ROOT.glob("core/*.py"),
        *ROOT.glob("validation_hypo/*.py"),
        ROOT / "validation_hypo/stress_protocol.json",
        Path(__file__).resolve(),
    })
    provenance = {
        "scope": "Sensibilite a la politique des trous de donnees ; les artefacts scelles sont intacts.",
        "gap_policies": list(GAP_POLICIES),
        "gap_hours": GAP_HOURS,
        "positions": list(POSITIONS),
        "stress_reference_policy": "clean_training_timestamps_fixed",
        "python_version": platform.python_version(),
        "packages": {
            name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "input_sha256": hashlib.sha256(RAW_CSV.read_bytes()).hexdigest(),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if any(output == sealed.resolve() for sealed in SEALED):
        raise SystemExit(f"Refus d'ecrire dans un dossier scelle : {output}")
    if not RAW_CSV.exists():
        raise SystemExit("data/brut.csv est requis ; le corpus confidentiel n'est pas versionne.")
    output.mkdir(parents=True, exist_ok=True)
    params = final_params()
    raw_all = load_csv(str(RAW_CSV))
    raw_all[COW] = raw_all[COW].astype(str)

    blocks = dropout_blocks(raw_all, params)
    blocks.to_csv(output / "dropout_blocks.csv", index=False)

    ablations, scenarios, variants, comparisons = [], [], [], []
    for policy in GAP_POLICIES:
        events = run_clean_ablation(str(RAW_CSV), verbose=False, gap_policy=policy)
        summary = ablation_summary(events).set_index("variante")
        unique = events.drop_duplicates(["event_id", "variante"])
        summary["couverture_attribuable"] = unique.groupby("variante")["episode_overlap"].mean()
        ablations.append(summary.reset_index().round(6).assign(gap_policy=policy))

        stress = run_stress_campaign(
            str(RAW_CSV), verbose=False, fixed_reference=True, gap_policy=policy
        )
        scenario_summary, variant_summary = summarize(stress)
        scenarios.append(scenario_summary.assign(gap_policy=policy))
        variants.append(variant_summary.assign(gap_policy=policy))
        comparisons.append(compare_variants(stress).assign(gap_policy=policy))

    for frames, name in (
        (ablations, "ablation_summary.csv"),
        (scenarios, "stress_scenario_summary.csv"),
        (variants, "stress_variant_summary.csv"),
        (comparisons, "stress_paired_comparisons.csv"),
    ):
        table = pd.concat(frames, ignore_index=True)
        table.insert(0, "gap_policy", table.pop("gap_policy"))
        table.to_csv(output / name, index=False)

    write_provenance(output)
    (output / "artifacts.sha256").write_text(
        "".join(
            f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
            for name in ARTIFACTS
        ),
        encoding="utf-8",
    )
    print(blocks.groupby("gap_policy")["new_hypo_starts"].apply(lambda s: int((s > 0).sum())))
    print(f"Ecrit : {output}")


if __name__ == "__main__":
    main()
