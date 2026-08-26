#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "===== MACHINE-SPECIFIC PATHS ====="
if grep -RIn \
  --exclude-dir=.git \
  --exclude-dir=__pycache__ \
  --exclude=check_release.sh \
  -E '/home/evelyn|/Users/evelyn|fdhg-icl-paper-greedy|fdhg-icl-paper' \
  README.md docs configs scripts src tests
then
  echo
  echo "ERROR: machine-specific/internal path found"
  exit 1
fi
echo "PASS"

echo
echo "===== BACKUP FILES ====="
backups="$(
  find . \
    -path './.git' -prune -o \
    \( -name '*.bak' -o -name '*.bak.*' -o -name '*~' \) \
    -print
)"
if [[ -n "$backups" ]]; then
  echo "$backups"
  echo "ERROR: backup files found"
  exit 1
fi
echo "PASS"

echo
echo "===== SHELL SYNTAX ====="
for f in scripts/reproduce_*.sh scripts/check_release.sh; do
  bash -n "$f"
  echo "PASS $f"
done

echo
echo "===== PYTHON SYNTAX ====="
python -m py_compile \
  scripts/ablations/*.py \
  scripts/evaluate/evaluate_frozen_gbdt.py \
  scripts/experiments/export_frozen_selected_batch.py \
  scripts/experiments/run_predictor_generalization.py \
  scripts/experiments/run_frozen_gbdt_batch.py \
  scripts/experiments/collect_predictor_generalization.py \
  scripts/experiments/run_efficiency_one.py \
  scripts/experiments/collect_efficiency_final.py
echo "PASS"

echo
echo "===== GIT DIFF CHECK ====="
git diff --check
echo "PASS"

echo
echo "===== IMPORT SMOKE ====="
python - <<'PY'
import fdhg
print("PASS import fdhg")
PY

echo
echo "===== GIT STATUS ====="
git status --short

echo
echo "LIGHTWEIGHT RELEASE CHECK PASS"
