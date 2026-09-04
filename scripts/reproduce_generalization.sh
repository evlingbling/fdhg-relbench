#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FINAL_GATE_ROOT="${FINAL_GATE_ROOT:-outputs/final-gate-51task-v2}"
GENERALIZATION_ROOT="${GENERALIZATION_ROOT:-outputs/predictor-generalization}"
FROZEN_ROOT="${FROZEN_ROOT:-${GENERALIZATION_ROOT}/frozen-matrices}"
RESULT_ROOT="${RESULT_ROOT:-${GENERALIZATION_ROOT}/frozen-gbdt}"
EXPECT_COMPLETED="${EXPECT_COMPLETED:-51}"
JOBS="${JOBS:-1}"
THREADS_PER_MODEL="${THREADS_PER_MODEL:-2}"

echo "===== 1. PRE-FLIGHT ====="

python scripts/experiments/run_predictor_generalization.py \
  --search-root "$FINAL_GATE_ROOT" \
  --output-root "$GENERALIZATION_ROOT" \
  --expect-completed "$EXPECT_COMPLETED" \
  --write-manifest

echo
echo "===== 2. FROZEN SELECTED MATRICES ====="

python scripts/experiments/export_frozen_selected_batch.py \
  --snapshot "${GENERALIZATION_ROOT}/evaluation_manifest.csv" \
  --output-root "$FINAL_GATE_ROOT" \
  --export-root "$FROZEN_ROOT" \
  --write

echo
echo "===== 3. PREDICTOR EVALUATION ====="

python scripts/experiments/run_frozen_gbdt_batch.py \
  --matrix-root "$FROZEN_ROOT" \
  --result-root "$RESULT_ROOT" \
  --jobs "$JOBS" \
  --threads-per-model "$THREADS_PER_MODEL" \
  --write

echo
echo "===== 4. DIRECT FROZEN HGB EVALUATION ====="

python scripts/evaluate/evaluate_frozen_hgb.py \
  --matrix-root "$FROZEN_ROOT" \
  --output-root "${GENERALIZATION_ROOT}/hgb-frozen" \
  --expect-completed "$EXPECT_COMPLETED"

echo
echo "===== 5. GBDT COMPLETENESS ====="

python scripts/experiments/collect_predictor_generalization.py \
  --result-root "$RESULT_ROOT"

echo
echo "Predictor generalization reproduction complete."
