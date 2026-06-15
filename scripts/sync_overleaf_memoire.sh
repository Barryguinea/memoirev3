#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${PROJECT_ROOT}/memoire/redaction_latex"

OVERLEAF_GIT_URL="${OVERLEAF_GIT_URL:-}"
OVERLEAF_BRANCH="${OVERLEAF_BRANCH:-master}"
OVERLEAF_DEST_SUBDIR="${OVERLEAF_DEST_SUBDIR:-redaction_latex}"
DEFAULT_MESSAGE="chore(memoire): sync redaction_latex $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
COMMIT_MESSAGE="${1:-$DEFAULT_MESSAGE}"

if [[ -z "${OVERLEAF_GIT_URL}" ]]; then
  echo "Erreur: definir OVERLEAF_GIT_URL"
  echo "Exemple: export OVERLEAF_GIT_URL='https://git.overleaf.com/<project_id>'"
  exit 1
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Erreur: dossier source introuvable: ${SOURCE_DIR}"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Erreur: git n'est pas installe."
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "Erreur: rsync n'est pas installe."
  exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/overleaf-sync.XXXXXX")"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

echo "Clonage Overleaf..."
git clone --depth=1 --branch "${OVERLEAF_BRANCH}" "${OVERLEAF_GIT_URL}" "${TMP_DIR}/repo"

DEST_DIR="${TMP_DIR}/repo/${OVERLEAF_DEST_SUBDIR}"
mkdir -p "${DEST_DIR}"

echo "Synchronisation des fichiers..."
rsync -a --delete \
  --exclude ".git/" \
  --exclude "*.aux" \
  --exclude "*.bbl" \
  --exclude "*.bcf" \
  --exclude "*.blg" \
  --exclude "*.fdb_latexmk" \
  --exclude "*.fls" \
  --exclude "*.log" \
  --exclude "*.out" \
  --exclude "*.run.xml" \
  --exclude "*.synctex.gz" \
  "${SOURCE_DIR}/" "${DEST_DIR}/"

cd "${TMP_DIR}/repo"
git add -A

if git diff --cached --quiet; then
  echo "Aucun changement a pousser vers Overleaf."
  exit 0
fi

echo "Commit..."
git commit -m "${COMMIT_MESSAGE}"

echo "Push vers Overleaf..."
git push origin "${OVERLEAF_BRANCH}"

echo "Termine: Overleaf est synchronise."
