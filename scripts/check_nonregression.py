#!/usr/bin/env python3
"""Outil de verification d'integrite et de non-regression des resultats.

Usages principaux:
1) Geler un manifest SHA256 des resultats historiques (integrite locale)
2) Verifier un manifest SHA256
3) Comparer 2 CSV (tri + tolerance numerique)
4) Comparer 2 dossiers de run (CSV) pour une non-regression rapide

Ce script ne modifie aucun resultat. Il lit uniquement des fichiers et
ecrit eventuellement un manifest (texte) dans le chemin demande.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOTS = [
    PROJECT_ROOT / "data" / "validation_robuste",
    PROJECT_ROOT / "data" / "validation_manual",
    PROJECT_ROOT / "data" / "performance",
]
DEFAULT_EXTENSIONS = [".csv", ".json"]


def _resolve_root(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _iter_files(roots: Sequence[Path], exts: Sequence[str]) -> Iterable[Path]:
    wanted = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in wanted:
                yield p


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _rel_for_manifest(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def _parse_manifest(manifest_path: Path) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "  " in line:
            digest, rel = line.split("  ", 1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Ligne de manifest invalide: {raw_line!r}")
            digest, rel = parts
        entries[rel.strip()] = digest.strip()
    return entries


def _normalize_sort(df: pd.DataFrame, *, sort_cols: Sequence[str] | None, ignore_row_order: bool) -> pd.DataFrame:
    out = df.copy()
    if not ignore_row_order or len(out) == 0:
        return out.reset_index(drop=True)

    cols = [c for c in (sort_cols or []) if c in out.columns]
    if not cols:
        cols = list(out.columns)
    if not cols:
        return out.reset_index(drop=True)

    tmp = out.copy()
    helper_cols: List[str] = []
    for i, c in enumerate(cols):
        helper = f"__sort_key_{i}"
        helper_cols.append(helper)
        tmp[helper] = tmp[c].astype("string").fillna("<NA>")
    tmp = tmp.sort_values(helper_cols, kind="mergesort", na_position="first")
    tmp = tmp.drop(columns=helper_cols)
    return tmp.reset_index(drop=True)


def _compare_series(
    left: pd.Series,
    right: pd.Series,
    *,
    atol: float,
    rtol: float,
) -> np.ndarray:
    """Retourne un masque booléen True si égal (exact ou tolérance numérique)."""
    if len(left) != len(right):
        raise ValueError("Series length mismatch")

    l = left.reset_index(drop=True)
    r = right.reset_index(drop=True)

    l_num = pd.to_numeric(l, errors="coerce")
    r_num = pd.to_numeric(r, errors="coerce")
    both_num = l_num.notna() & r_num.notna()

    equal = np.zeros(len(l), dtype=bool)
    if both_num.any():
        equal[both_num.to_numpy()] = np.isclose(
            l_num[both_num].to_numpy(dtype=float),
            r_num[both_num].to_numpy(dtype=float),
            atol=float(atol),
            rtol=float(rtol),
            equal_nan=True,
        )

    remaining = ~both_num.to_numpy()
    if remaining.any():
        l_str = l.astype("string")
        r_str = r.astype("string")
        both_na = l.isna().to_numpy() & r.isna().to_numpy()
        str_eq = (l_str.fillna("<NA>") == r_str.fillna("<NA>")).to_numpy()
        equal[remaining] = (both_na | str_eq)[remaining]

    return equal


def compare_csv_files(
    left_path: Path,
    right_path: Path,
    *,
    atol: float,
    rtol: float,
    ignore_row_order: bool = True,
    sort_cols: Sequence[str] | None = None,
    ignore_cols: Sequence[str] | None = None,
    max_diffs: int = 20,
) -> Tuple[bool, List[str]]:
    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)
    ignore = set(ignore_cols or [])
    left = left.drop(columns=[c for c in ignore if c in left.columns], errors="ignore")
    right = right.drop(columns=[c for c in ignore if c in right.columns], errors="ignore")

    if set(left.columns) != set(right.columns):
        return False, [
            f"Colonnes differentes: left={sorted(left.columns.tolist())}, right={sorted(right.columns.tolist())}"
        ]

    ordered_cols = sorted(left.columns.tolist())
    left = left[ordered_cols]
    right = right[ordered_cols]

    left = _normalize_sort(left, sort_cols=sort_cols, ignore_row_order=ignore_row_order)
    right = _normalize_sort(right, sort_cols=sort_cols, ignore_row_order=ignore_row_order)

    if left.shape != right.shape:
        return False, [f"Shape differente apres normalisation: left={left.shape}, right={right.shape}"]

    msgs: List[str] = []
    for col in ordered_cols:
        eq = _compare_series(left[col], right[col], atol=atol, rtol=rtol)
        if bool(np.all(eq)):
            continue
        bad_idx = np.flatnonzero(~eq)
        msgs.append(f"Colonne differente: {col} ({len(bad_idx)} lignes)")
        for row_i in bad_idx[: max(0, max_diffs - len(msgs) + 1)]:
            msgs.append(
                f"  row={int(row_i)} left={left.iloc[row_i][col]!r} right={right.iloc[row_i][col]!r}"
            )
            if len(msgs) >= max_diffs:
                break
        if len(msgs) >= max_diffs:
            break

    return (len(msgs) == 0), msgs


def _cmd_hash_freeze(args: argparse.Namespace) -> int:
    roots = [_resolve_root(r) for r in (args.root or [])] or DEFAULT_RESULT_ROOTS
    files = list(_iter_files(roots, args.extensions))
    out_path = _resolve_root(args.output) if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for p in files:
        digest = _sha256_file(p)
        rel = _rel_for_manifest(p, PROJECT_ROOT)
        lines.append(f"{digest}  {rel}")

    header = [
        "# Manifest SHA256 des resultats (genere par scripts/check_nonregression.py)",
        f"# Base dir: {PROJECT_ROOT}",
        f"# Fichiers: {len(lines)}",
    ]
    text = "\n".join(header + lines) + ("\n" if header or lines else "")
    out_path.write_text(text, encoding="utf-8")

    print(f"Manifest ecrit: {out_path}")
    print(f"Fichiers hashes: {len(lines)}")
    for root in roots:
        status = "OK" if root.exists() else "absent"
        print(f"- root {root}: {status}")
    return 0


def _cmd_hash_verify(args: argparse.Namespace) -> int:
    manifest = _resolve_root(args.manifest) if not Path(args.manifest).is_absolute() else Path(args.manifest)
    if not manifest.exists():
        print(f"Manifest introuvable: {manifest}", file=sys.stderr)
        return 2

    entries = _parse_manifest(manifest)
    changed: List[str] = []
    missing: List[str] = []

    for rel, expected in entries.items():
        p = (PROJECT_ROOT / rel) if not Path(rel).is_absolute() else Path(rel)
        if not p.exists():
            missing.append(rel)
            continue
        got = _sha256_file(p)
        if got != expected:
            changed.append(rel)

    extras: List[str] = []
    if args.check_extra:
        roots = [_resolve_root(r) for r in (args.root or [])] or DEFAULT_RESULT_ROOTS
        current = {_rel_for_manifest(p, PROJECT_ROOT) for p in _iter_files(roots, args.extensions)}
        extras = sorted(current - set(entries.keys()))

    print(f"Manifest verifie: {manifest}")
    print(f"Entrees: {len(entries)}")
    print(f"Missing: {len(missing)} | Changed: {len(changed)} | Extra: {len(extras)}")

    if missing:
        print("Fichiers manquants (max 20):")
        for x in missing[:20]:
            print(f"  - {x}")
    if changed:
        print("Fichiers modifies (max 20):")
        for x in changed[:20]:
            print(f"  - {x}")
    if extras:
        print("Fichiers supplementaires (max 20):")
        for x in extras[:20]:
            print(f"  - {x}")

    return 0 if not (missing or changed or extras) else 1


def _cmd_compare_csv(args: argparse.Namespace) -> int:
    left = _resolve_root(args.left) if not Path(args.left).is_absolute() else Path(args.left)
    right = _resolve_root(args.right) if not Path(args.right).is_absolute() else Path(args.right)
    sort_cols = [c.strip() for c in args.sort_cols.split(",")] if args.sort_cols else []
    ignore_cols = [c.strip() for c in args.ignore_cols.split(",")] if args.ignore_cols else []

    ok, msgs = compare_csv_files(
        left,
        right,
        atol=float(args.atol),
        rtol=float(args.rtol),
        ignore_row_order=not args.strict_order,
        sort_cols=sort_cols,
        ignore_cols=ignore_cols,
        max_diffs=int(args.max_diffs),
    )
    if ok:
        print(f"OK CSV: {left} == {right}")
        return 0

    print(f"DIFF CSV: {left} != {right}")
    for m in msgs:
        print(m)
    return 1


def _cmd_compare_run(args: argparse.Namespace) -> int:
    left_run = _resolve_root(args.left_run) if not Path(args.left_run).is_absolute() else Path(args.left_run)
    right_run = _resolve_root(args.right_run) if not Path(args.right_run).is_absolute() else Path(args.right_run)
    if not left_run.is_dir() or not right_run.is_dir():
        print("Les deux chemins doivent etre des dossiers de run.", file=sys.stderr)
        return 2

    left_csv = {
        p.name: p
        for p in left_run.glob("*.csv")
        if p.is_file() and not p.name.endswith(".partial.csv")
    }
    right_csv = {
        p.name: p
        for p in right_run.glob("*.csv")
        if p.is_file() and not p.name.endswith(".partial.csv")
    }
    common = sorted(set(left_csv) & set(right_csv))
    only_left = sorted(set(left_csv) - set(right_csv))
    only_right = sorted(set(right_csv) - set(left_csv))

    print(f"Comparaison run A: {left_run}")
    print(f"Comparaison run B: {right_run}")
    print(f"CSV communs: {len(common)} | only_left: {len(only_left)} | only_right: {len(only_right)}")

    failed: List[str] = []
    for name in common:
        ok, msgs = compare_csv_files(
            left_csv[name],
            right_csv[name],
            atol=float(args.atol),
            rtol=float(args.rtol),
            ignore_row_order=not args.strict_order,
            sort_cols=[],
            ignore_cols=[],
            max_diffs=int(args.max_diffs),
        )
        if ok:
            print(f"  OK   {name}")
        else:
            print(f"  DIFF {name}")
            for m in msgs[: min(5, len(msgs))]:
                print(f"     {m}")
            failed.append(name)

    if only_left:
        print("CSV presents seulement dans run A (max 20):")
        for name in only_left[:20]:
            print(f"  - {name}")
    if only_right:
        print("CSV presents seulement dans run B (max 20):")
        for name in only_right[:20]:
            print(f"  - {name}")

    # JSONs (run_meta, summaries de comparaison, etc.) peuvent changer (timestamps/paths).
    skipped_json_left = sorted(p.name for p in left_run.glob("*.json") if p.is_file())
    skipped_json_right = sorted(p.name for p in right_run.glob("*.json") if p.is_file())
    if skipped_json_left or skipped_json_right:
        print("Note: JSON non compares automatiquement (timestamps/chemins volatils possibles).")

    has_structural_diff = bool(only_left or only_right)
    return 0 if not failed and not has_structural_diff else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verification d'integrite (SHA256) et non-regression (CSV/runs) des resultats.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_freeze = sub.add_parser(
        "hash-freeze",
        help="Genere un manifest SHA256 des fichiers CSV/JSON de resultats.",
    )
    p_freeze.add_argument(
        "--output",
        default="data/nonregression_results.sha256",
        help="Chemin de sortie du manifest (defaut: data/nonregression_results.sha256).",
    )
    p_freeze.add_argument(
        "--root",
        action="append",
        default=[],
        help="Dossier a inclure (repeter l'option). Defaut: data/validation_robuste, data/validation_manual, data/performance.",
    )
    p_freeze.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="Extensions a hasher (defaut: .csv .json).",
    )
    p_freeze.set_defaults(func=_cmd_hash_freeze)

    p_verify = sub.add_parser(
        "hash-verify",
        help="Verifie qu'un manifest SHA256 correspond encore aux fichiers locaux.",
    )
    p_verify.add_argument("manifest", help="Chemin du manifest SHA256.")
    p_verify.add_argument(
        "--check-extra",
        action="store_true",
        help="Verifie aussi les fichiers CSV/JSON supplementaires dans les roots.",
    )
    p_verify.add_argument(
        "--root",
        action="append",
        default=[],
        help="Roots pour --check-extra (repeter l'option).",
    )
    p_verify.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="Extensions a verifier pour --check-extra (defaut: .csv .json).",
    )
    p_verify.set_defaults(func=_cmd_hash_verify)

    p_csv = sub.add_parser(
        "compare-csv",
        help="Compare deux CSV avec tri (par defaut) et tolerance numerique.",
    )
    p_csv.add_argument("left", help="CSV A")
    p_csv.add_argument("right", help="CSV B")
    p_csv.add_argument("--atol", type=float, default=1e-9, help="Tolerance absolue numerique.")
    p_csv.add_argument("--rtol", type=float, default=1e-6, help="Tolerance relative numerique.")
    p_csv.add_argument(
        "--strict-order",
        action="store_true",
        help="Desactive le tri des lignes avant comparaison.",
    )
    p_csv.add_argument(
        "--sort-cols",
        default="",
        help="Colonnes de tri (comma-separated). Par defaut: tri sur toutes les colonnes.",
    )
    p_csv.add_argument(
        "--ignore-cols",
        default="",
        help="Colonnes a ignorer (comma-separated), ex: run_ts,output_dir.",
    )
    p_csv.add_argument("--max-diffs", type=int, default=20, help="Nombre max de differences affichees.")
    p_csv.set_defaults(func=_cmd_compare_csv)

    p_run = sub.add_parser(
        "compare-run",
        help="Compare les CSV top-level de deux dossiers de run (non-regression rapide).",
    )
    p_run.add_argument("left_run", help="Dossier run A")
    p_run.add_argument("right_run", help="Dossier run B")
    p_run.add_argument("--atol", type=float, default=1e-9, help="Tolerance absolue numerique.")
    p_run.add_argument("--rtol", type=float, default=1e-6, help="Tolerance relative numerique.")
    p_run.add_argument(
        "--strict-order",
        action="store_true",
        help="Desactive le tri des lignes avant comparaison des CSV.",
    )
    p_run.add_argument("--max-diffs", type=int, default=10, help="Nombre max de diffs affichees par CSV.")
    p_run.set_defaults(func=_cmd_compare_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

