#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"

ABLATION_ROOT="${ABLATION_ROOT:-outputs/ablations}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-}"
RUN_EXPENSIVE_ABLATIONS="${RUN_EXPENSIVE_ABLATIONS:-0}"

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
echo "===== Pairwise initialization artifact verification ====="
"$PY" scripts/ablations/verify_pairwise_initialization_artifact.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --output "$ABLATION_ROOT/pairwise-initialization/artifact_verification.json"

echo
echo "===== Independent vs Greedy ====="
"$PY" scripts/ablations/collect_independent_vs_greedy.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --output "$ABLATION_ROOT/independent-vs-greedy/summary.csv"

echo
if [[ "$RUN_EXPENSIVE_ABLATIONS" == "1" ]]; then
  echo "===== Pairwise initialization ====="
  "$PY" scripts/ablations/run_pairwise_initialization.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --output-root "$ABLATION_ROOT/pairwise-initialization"

  echo
  echo "===== Candidate-pool sensitivity ====="
  ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  OUTPUT_ROOT="$ABLATION_ROOT/candidate-pool-sensitivity" \
    scripts/ablations/run_candidate_pool_sensitivity.sh
else
  echo "===== Extended ablations skipped ====="
  echo "Pairwise initialization and candidate-pool sensitivity are computationally expensive."
  echo "Run them with:"
  echo "  RUN_EXPENSIVE_ABLATIONS=1 ARTIFACT_ROOT=/path/to/paper-artifacts scripts/reproduce_ablations.sh"
fi

echo
echo "Ablation reproduction complete."
