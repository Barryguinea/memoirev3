"""Helpers fichiers/JSON/discovery pour la comparaison inter-versions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def _is_complete(run_dir: Path, required_files: List[str]) -> bool:
    """Retourne ``True`` si tous les fichiers requis existent dans le dossier de run donné."""
    return all((run_dir / name).exists() for name in required_files)


def _resolve_existing_file(run_dir: Path, candidates: List[str]) -> Optional[Path]:
    """Retourne le chemin du premier fichier candidat existant dans ``run_dir``, sinon ``None``."""
    for name in candidates:
        p = run_dir / name
        if p.exists():
            return p
    return None


def _is_complete_with_candidates(
    run_dir: Path,
    base_required: List[str],
    summary_candidates: List[str],
) -> bool:
    """Retourne ``True`` si les fichiers de base requis et au moins un summary candidat existent."""
    return _is_complete(run_dir, base_required) and (_resolve_existing_file(run_dir, summary_candidates) is not None)


def _discover_latest_run(
    validation_root: Path,
    required_files: List[str],
    *,
    summary_candidates: Optional[List[str]] = None,
    exclude_dirs: Optional[List[Path]] = None,
) -> Optional[Path]:
    """Trouve le dossier de run de validation complet le plus récemment modifié."""
    exclude_resolved = {p.resolve() for p in (exclude_dirs or [])}
    candidates = sorted(
        [p for p in validation_root.glob("robust_validation_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        if c.resolve() in exclude_resolved:
            continue
        if summary_candidates is None and _is_complete(c, required_files):
            return c
        if summary_candidates is not None and _is_complete_with_candidates(c, required_files, summary_candidates):
            return c
    return None


def _read_optional_json(path: Path) -> Dict[str, object]:
    """Lit un fichier JSON et retourne son contenu, ou un dict vide s'il est absent/invalide."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

