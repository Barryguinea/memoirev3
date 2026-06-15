#!/usr/bin/env python3
"""CLI du benchmark de performance du pipeline."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from core.config import (
    DEFAULT_ALERT_MIN,
    DEFAULT_BASELINE_RATIO,
    DEFAULT_CONTAMINATION,
    DEFAULT_COOLDOWN_HOURS,
    DEFAULT_COVERAGE_MIN_PCT,
    DEFAULT_INTERVAL,
    DEFAULT_MIX_MODE,
    DEFAULT_MIX_RATE_THR,
    DEFAULT_MI_Z_HIGH_THR,
    DEFAULT_PERSIST_HOURS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_WINDOW_BASELINE,
    DEFAULT_Z_HIGH_THR,
    DEFAULT_Z_LOW_THR,
)
from core.io import COW, load_csv
from scripts._benchmark_performance.engine import (
    PipelineParams,
    _aggregate_runs,
    _now_stamp,
    _parse_fractions,
    _subset_by_cows,
    run_benchmark_herd,
)


def _build_parser() -> argparse.ArgumentParser:
    """Construit et retourne l'analyseur d'arguments du CLI de benchmark."""
    p = argparse.ArgumentParser(description="Benchmark performance/scalabilite du pipeline V3")
    p.add_argument("--input", type=str, default="data/brut.csv", help="CSV brut a benchmarker")
    p.add_argument("--output-root", type=str, default="data/performance", help="Dossier de sortie")
    p.add_argument("--fractions", type=str, default="0.25,0.5,0.75,1.0", help="Fractions de vaches a tester (ex: 0.25,0.5,1.0)")
    p.add_argument("--repeats", type=int, default=3, help="Nb repetitions par taille")
    p.add_argument("--collect-output", action="store_true", help="Inclure le cout de concat des sorties intervalle (plus realiste, plus lent)")
    p.add_argument("--max-cows-total", type=int, default=None, help="Limiter le nb de vaches du dataset source (smoke test)")
    p.add_argument("--interval", type=str, default=DEFAULT_INTERVAL)
    p.add_argument("--window-baseline", type=int, default=DEFAULT_WINDOW_BASELINE)
    p.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION)
    p.add_argument("--baseline-ratio", type=float, default=DEFAULT_BASELINE_RATIO)
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--persist-hours", type=int, default=DEFAULT_PERSIST_HOURS)
    p.add_argument("--alert-min", type=int, default=DEFAULT_ALERT_MIN)
    p.add_argument("--mix-mode", type=str, default=DEFAULT_MIX_MODE)
    p.add_argument("--mix-rate-thr", type=float, default=DEFAULT_MIX_RATE_THR)
    p.add_argument("--z-low-thr", type=float, default=DEFAULT_Z_LOW_THR)
    p.add_argument("--z-high-thr", type=float, default=DEFAULT_Z_HIGH_THR)
    p.add_argument("--cooldown-hours", type=int, default=DEFAULT_COOLDOWN_HOURS)
    p.add_argument("--mi-z-high-thr", type=float, default=DEFAULT_MI_Z_HIGH_THR)
    p.add_argument("--coverage-min-pct", type=float, default=DEFAULT_COVERAGE_MIN_PCT)
    p.add_argument("--quiet", action="store_true", help="Moins de logs console")
    p.add_argument(
        "--canonical-json",
        type=str,
        default="data/revalidation/performance_v3_full_corpus.json",
        help="Résumé canonique de la plus grande fraction évaluée",
    )
    return p


def _build_params_from_args(args: argparse.Namespace) -> PipelineParams:
    """Construit l'objet paramètres pipeline à partir des arguments CLI."""
    return PipelineParams(
        interval=args.interval,
        window_baseline=args.window_baseline,
        contamination=args.contamination,
        baseline_ratio=args.baseline_ratio,
        random_state=args.random_state,
        persist_hours=args.persist_hours,
        alert_min=args.alert_min,
        mix_mode=str(args.mix_mode).upper(),
        mix_rate_thr=args.mix_rate_thr,
        z_low_thr=args.z_low_thr,
        z_high_thr=args.z_high_thr,
        cooldown_hours=args.cooldown_hours,
        mi_z_high_thr=args.mi_z_high_thr,
        coverage_min_pct=args.coverage_min_pct,
    )


def main() -> int:
    """Point d'entrée CLI : charge les données, lance les benchmarks par fraction et écrit les résultats."""
    args = _build_parser().parse_args()
    fractions = _parse_fractions(args.fractions)
    out_root = Path(args.output_root)
    out_dir = out_root / f"performance_benchmark_{_now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    params = _build_params_from_args(args)

    if not args.quiet:
        print(f"[perf] loading csv: {args.input}")
    t_load0 = perf_counter()
    df_all = load_csv(args.input)
    load_seconds = perf_counter() - t_load0

    cows_all = sorted(df_all[COW].astype(str).unique().tolist())
    if args.max_cows_total is not None:
        cows_keep = set(cows_all[: int(args.max_cows_total)])
        df_all = df_all[df_all[COW].astype(str).isin(cows_keep)].copy()
        cows_all = sorted(df_all[COW].astype(str).unique().tolist())

    total_cows = len(cows_all)
    total_rows = len(df_all)
    if total_cows == 0:
        raise SystemExit("[perf] dataset vide apres filtrage.")

    if not args.quiet:
        print(f"[perf] loaded rows={total_rows:,} cows={total_cows} in {load_seconds:.3f}s")
        print(f"[perf] fractions={fractions} repeats={args.repeats} collect_output={args.collect_output}")
        print(f"[perf] params={asdict(params)}")

    run_rows = []
    bench_started_at = datetime.now().isoformat(timespec="seconds")
    run_id = 0
    total_planned = len(fractions) * int(args.repeats)

    for frac in fractions:
        n_cows = max(1, int(math.ceil(total_cows * frac)))
        n_cows = min(n_cows, total_cows)
        df_subset = _subset_by_cows(df_all, n_cows)
        for rep in range(1, int(args.repeats) + 1):
            run_id += 1
            if not args.quiet:
                print(f"[perf] run {run_id}/{total_planned} | fraction={frac:.2f} | cows={n_cows} | repeat={rep}")
            t0 = perf_counter()
            metrics = run_benchmark_herd(df_subset, params, max_cows=n_cows, collect_output=args.collect_output)
            elapsed = perf_counter() - t0
            row = {
                "run_idx": int(run_id),
                "fraction": float(frac),
                "repeat": int(rep),
                "elapsed_wrapper_seconds": float(elapsed),
                **metrics,
            }
            run_rows.append(row)
            if not args.quiet:
                print(
                    "[perf] result "
                    f"total={metrics['total_seconds']:.3f}s "
                    f"rows/s={metrics['rows_per_sec']:.1f} "
                    f"cows/s={metrics['cows_per_sec']:.2f}"
                )

    runs_df = pd.DataFrame(run_rows)
    summary_df = _aggregate_runs(runs_df)

    runs_file = out_dir / "performance_runs.csv"
    summary_file = out_dir / "performance_summary.csv"
    runs_df.to_csv(runs_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    meta = {
        "benchmark_type": "pipeline_v3_hybrid_performance_scalability",
        "started_at": bench_started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(Path(args.input)),
        "output_dir": str(out_dir),
        "load_seconds": float(load_seconds),
        "dataset_rows": int(total_rows),
        "dataset_cows": int(total_cows),
        "fractions": fractions,
        "repeats": int(args.repeats),
        "collect_output": bool(args.collect_output),
        "params": asdict(params),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "files": {
            "runs": str(runs_file),
            "summary": str(summary_file),
        },
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if summary_df.empty:
        raise RuntimeError("Le benchmark V3 n'a produit aucun résumé.")
    full = summary_df.loc[summary_df["fraction"].eq(summary_df["fraction"].max())].iloc[0]
    canonical_path = Path(args.canonical_json)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical = {
        "benchmark_type": meta["benchmark_type"],
        "input_path": str(Path(args.input)),
        "dataset_rows": int(total_rows),
        "dataset_cows": int(total_cows),
        "repeats": int(args.repeats),
        "collect_output": bool(args.collect_output),
        "measurement": "median over repeated full-corpus runs",
        "total_seconds_median": float(full["total_seconds_median"]),
        "total_seconds_min": float(full["total_seconds_min"]),
        "total_seconds_max": float(full["total_seconds_max"]),
        "rows_per_sec_median": float(full["rows_per_sec_median"]),
        "features_seconds_median": float(full["features_seconds_median"]),
        "if_seconds_median": float(full["if_seconds_median"]),
        "legacy_alerts_seconds_median": float(full["alerts_seconds_median"]),
        "hybrid_seconds_median": float(full["hybrid_seconds_median"]),
        "features_share_of_core_median": float(full["features_share_of_core_median"]),
        "if_share_of_core_median": float(full["if_share_of_core_median"]),
        "legacy_alerts_share_of_core_median": float(full["alerts_share_of_core_median"]),
        "hybrid_share_of_core_median": float(full["hybrid_share_of_core_median"]),
        "params": asdict(params),
        "run_directory": str(out_dir),
    }
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(canonical, f, indent=2, ensure_ascii=False)

    print(f"[perf] done. out_dir={out_dir}")
    print(f"[perf] runs:    {runs_file}")
    print(f"[perf] summary: {summary_file}")
    print(f"[perf] meta:    {out_dir / 'run_meta.json'}")
    print(f"[perf] canonical: {canonical_path}")

    if not summary_df.empty:
        cols = [
            c
            for c in [
                "fraction",
                "n_cows",
                "n_rows_raw",
                "repeats",
                "total_seconds_median",
                "rows_per_sec_mean",
                "cows_per_sec_mean",
                "features_seconds_mean",
                "if_seconds_mean",
                "alerts_seconds_mean",
                "hybrid_seconds_mean",
                "seconds_per_1000_rows_median",
                "time_ratio_vs_full",
            ]
            if c in summary_df.columns
        ]
        print(summary_df[cols].to_string(index=False))
    return 0


__all__ = ["PipelineParams", "run_benchmark_herd", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
