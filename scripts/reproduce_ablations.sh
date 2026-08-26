#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"

ABLATION_ROOT="${ABLATION_ROOT:-outputs/ablations}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-}"

if [[ -z "$ARTIFACT_ROOT" ]]; then
  echo "ERROR: ARTIFACT_ROOT must point to the preserved paper artifact repository." >&2
  echo "Example:" >&2
  echo '  ARTIFACT_ROOT=/path/to/fdhg-paper-artifacts scripts/reproduce_ablations.sh' >&2
  exit 2
fi

FINAL_GATE_ROOT="${FINAL_GATE_ROOT:-$ARTIFACT_ROOT/outputs/final-gate-51task-v2}"
CANONICAL_ONBOARDING_ROOT="${CANONICAL_ONBOARDING_ROOT:-${FINAL_GATE_ROOT}/_canonical_onboarding}"

echo "===== Cross-fold consistency ====="
"$PY" scripts/ablations/run_cross_fold_consistency.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-root "$ABLATION_ROOT/cross-fold-consistency"

"$PY" scripts/ablations/collect_cross_fold_consistency.py \
  --input-root "$ABLATION_ROOT/cross-fold-consistency"

echo
echo "===== Auto-budget sensitivity ====="
"$PY" scripts/ablations/run_auto_budget_sensitivity.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-root "$ABLATION_ROOT/auto-budget" \
  --canonical-onboarding-root "$CANONICAL_ONBOARDING_ROOT"

"$PY" scripts/ablations/collect_auto_budget_sensitivity.py \
  --input-root "$ABLATION_ROOT/auto-budget"

echo
echo "===== Random-K ====="
"$PY" scripts/ablations/run_random_k.py \
  --canonical-root "$FINAL_GATE_ROOT" \
  --canonical-onboarding-root "$CANONICAL_ONBOARDING_ROOT" \
  --output-root "$ABLATION_ROOT/random-k"

"$PY" scripts/ablations/collect_random_k.py \
  --input-root "$ABLATION_ROOT/random-k"

echo
echo "Ablation reproduction complete."
