#!/usr/bin/env bash
set -euo pipefail

# Supprime uniquement les artefacts locaux temporaires ; ne touche pas aux sorties de runs.
ROOT_DIR="."
PRUNE_STREAMLIT_SAMPLES="0"

for arg in "$@"; do
  case "$arg" in
    --prune-streamlit-samples)
      PRUNE_STREAMLIT_SAMPLES="1"
      ;;
    *)
      ROOT_DIR="$arg"
      ;;
  esac
done

cd "${ROOT_DIR}"

find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
find . -type f -name ".DS_Store" -delete

if [[ "${PRUNE_STREAMLIT_SAMPLES}" == "1" ]]; then
  if [[ -d "./data/validation_robuste" ]]; then
    # Optionnel : supprime les dossiers lourds de samples Streamlit par run,
    # tout en conservant les résumés et meilleurs artefacts.
    find ./data/validation_robuste -type d -name "streamlit_samples" -prune -exec rm -rf {} +
    echo "Dossiers lourds streamlit_samples supprimés sous data/validation_robuste."
  else
    echo "Aucun dossier data/validation_robuste trouvé ; rien à supprimer."
  fi
fi

echo "Artefacts locaux temporaires nettoyés."
