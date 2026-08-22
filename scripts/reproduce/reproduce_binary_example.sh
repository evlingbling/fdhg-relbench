#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export TABPFN_ALLOW_CPU_LARGE_DATASET="${TABPFN_ALLOW_CPU_LARGE_DATASET:-1}"

DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-41}"
FORCE_PREP="${FORCE_PREP:-0}"

DATASET="rel-event"
TASK="event_interest-interested"
WORK_ROOT="outputs/e2e/${DATASET}_${TASK}"

INSPECT_PARENT="${WORK_ROOT}/inspect"
RAW_INSPECT="${INSPECT_PARENT}/${DATASET}_${TASK}"
ENRICHED_INSPECT="${WORK_ROOT}/inspect_enriched"
FDHG_INSPECT="${WORK_ROOT}/inspect_fdhg"

DFS_DIR="${WORK_ROOT}/dfs"
AFD_DIR="${WORK_ROOT}/afd"
FDHG_DIR="${WORK_ROOT}/fdhg"

RESULT_ROOT="results/${DATASET}_${TASK}_tabpfn"
LOG_ROOT="logs/${DATASET}_${TASK}_tabpfn"

require_python_module() {
  local module="$1"
  local install_hint="$2"

  if ! python -c "import ${module}" >/dev/null 2>&1; then
    echo "ERROR: Python module '${module}' is not installed."
    echo "Install it with:"
    echo "  ${install_hint}"
    exit 1
  fi
}

require_python_module \
  "relbench" \
  'python -m pip install -e ".[dev,relbench,tabpfn]"'

require_python_module \
  "tabpfn" \
  'python -m pip install -e ".[dev,relbench,tabpfn]"'

if [[ "$FORCE_PREP" == "1" ]]; then
  echo "[prepare] FORCE_PREP=1: removing generated binary artifacts"
  rm -rf \
    "$INSPECT_PARENT" \
    "$ENRICHED_INSPECT" \
    "$FDHG_INSPECT" \
    "$DFS_DIR" \
    "$AFD_DIR" \
    "$FDHG_DIR"
fi

mkdir -p "$WORK_ROOT" "$RESULT_ROOT" "$LOG_ROOT"

echo
echo "============================================================"
echo "Step 1/6: Inspect RelBench bundle"
echo "============================================================"

if [[ ! -f "${RAW_INSPECT}/target_train.parquet" ]]; then
  python scripts/prepare/inspect_relbench_bundle.py \
    --dataset "$DATASET" \
    --task "$TASK" \
    --out-dir "$INSPECT_PARENT"
else
  echo "[skip] inspect bundle already exists: $RAW_INSPECT"
fi

echo
echo "============================================================"
echo "Step 2/6: Attach user key to target splits"
echo "============================================================"

if [[ ! -f "${ENRICHED_INSPECT}/target_train.parquet" ]] || \
   [[ ! -f "${ENRICHED_INSPECT}/target_val.parquet" ]]; then

  rm -rf "$ENRICHED_INSPECT"

  python scripts/prepare/enrich_target_from_primary_key.py \
    --inspect-dir "$RAW_INSPECT" \
    --out-dir "$ENRICHED_INSPECT" \
    --source-table event_interest \
    --primary-key-col primary_key \
    --entity-key user \
    --source-entity-col user \
    --verify-time-col timestamp \
    --verify-label-col interested \
    --splits train val
else
  echo "[skip] enriched target splits already exist: $ENRICHED_INSPECT"
fi

echo
echo "============================================================"
echo "Step 3/6: Build temporal DFS features"
echo "============================================================"

if [[ ! -f "${DFS_DIR}/target_with_dfs_agg_train.parquet" ]] || \
   [[ ! -f "${DFS_DIR}/target_with_dfs_agg_val.parquet" ]]; then

  rm -rf "$DFS_DIR"

  python scripts/prepare/build_relbench_features.py \
    --inspect-dir "$ENRICHED_INSPECT" \
    --out-dir "$DFS_DIR" \
    --mode dfs \
    --max-train 14442 \
    --max-val 2000 \
    --seed "$SEED" \
    --target-key user \
    --target-time-col timestamp \
    --child-table event_interest \
    --numeric-col user \
    --splits train val
else
  echo "[skip] DFS features already exist: $DFS_DIR"
fi

echo
echo "============================================================"
echo "Step 4/6: Prepare entity table and compute AFD statistics"
echo "============================================================"

if [[ ! -f "${FDHG_INSPECT}/table_users.parquet" ]]; then
  rm -rf "$FDHG_INSPECT"

  python - <<'PY'
from pathlib import Path
import shutil
import pandas as pd

src = Path(
    "outputs/e2e/rel-event_event_interest-interested/"
    "inspect_enriched"
)
dst = Path(
    "outputs/e2e/rel-event_event_interest-interested/"
    "inspect_fdhg"
)

shutil.copytree(src, dst)

users_path = dst / "table_users.parquet"
users = pd.read_parquet(users_path)

if "user" not in users.columns:
    if "user_id" not in users.columns:
        raise KeyError(
            "Neither 'user' nor 'user_id' exists in table_users.parquet"
        )
    users["user"] = users["user_id"]

users.to_parquet(users_path, index=False)

print("saved:", users_path)
print("shape:", users.shape)
PY
else
  echo "[skip] FDHG inspect bundle already exists: $FDHG_INSPECT"
fi

mkdir -p "$AFD_DIR"

if [[ ! -f "${AFD_DIR}/users_afd_dmax1.csv" ]]; then
  python scripts/prepare/compute_afd_dmax1.py \
    --table "${FDHG_INSPECT}/table_users.parquet" \
    --out "${AFD_DIR}/users_afd_dmax1.csv" \
    --seed "$SEED" \
    --columns locale birthyear gender joinedAt location timezone \
    --exclude-columns user user_id
else
  echo "[skip] AFD statistics already exist: ${AFD_DIR}/users_afd_dmax1.csv"
fi

echo
echo "============================================================"
echo "Step 5/6: Build FDHG ambiguity features"
echo "============================================================"

if [[ ! -f "${FDHG_DIR}/target_with_dfs_agg_train.parquet" ]] || \
   [[ ! -f "${FDHG_DIR}/target_with_dfs_agg_val.parquet" ]]; then

  rm -rf "$FDHG_DIR"

  python scripts/prepare/build_fdhg_ambiguity.py \
    --inspect-dir "$FDHG_INSPECT" \
    --dfs-dir "$DFS_DIR" \
    --afd-stats "${AFD_DIR}/users_afd_dmax1.csv" \
    --out-dir "$FDHG_DIR" \
    --entity-key user \
    --target-table users
else
  echo "[skip] FDHG features already exist: $FDHG_DIR"
fi

echo
echo "============================================================"
echo "Step 6/6: Evaluate DFS and FDHG with TabPFN"
echo "============================================================"

for variant in dfs fdhg_dmax1; do
  if [[ "$variant" == "dfs" ]]; then
    feature_root="$DFS_DIR"
  else
    feature_root="$FDHG_DIR"
  fi

  out="${RESULT_ROOT}/${variant}/seed${SEED}"
  log="${LOG_ROOT}/${variant}_seed${SEED}.log"

  mkdir -p "$out"
  mkdir -p "$(dirname "$log")"

  echo
  echo "------------------------------------------------------------"
  echo "${DATASET}/${TASK} | ${variant} | seed=${SEED}"
  echo "------------------------------------------------------------"

  python scripts/evaluate/evaluate_binary_tabpfn.py \
    --train-parquet \
    "${feature_root}/target_with_dfs_agg_train.parquet" \
    --val-parquet \
    "${feature_root}/target_with_dfs_agg_val.parquet" \
    --output-dir "$out" \
    --dataset "$DATASET" \
    --task "$TASK" \
    --variant "$variant" \
    --label-col interested \
    --drop-cols "timestamp,primary_key,user,f_event_interest_count" \
    --device "$DEVICE" \
    --seed "$SEED" \
    2>&1 | tee "$log"
done

echo
echo "============================================================"
echo "Binary reproduction summary"
echo "============================================================"

python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path(
    "results/rel-event_event_interest-interested_tabpfn"
)

dfs = pd.read_csv(root / "dfs/seed41/metrics.csv").iloc[0]
fdhg = pd.read_csv(root / "fdhg_dmax1/seed41/metrics.csv").iloc[0]

print("DFS")
print("  accuracy:", float(dfs["accuracy"]))
print("  AUROC:", float(dfs["roc_auc"]))
print("  AP:", float(dfs["average_precision"]))
print("  log_loss:", float(dfs["log_loss"]))

print()
print("FDHG dmax1")
print("  accuracy:", float(fdhg["accuracy"]))
print("  AUROC:", float(fdhg["roc_auc"]))
print("  AP:", float(fdhg["average_precision"]))
print("  log_loss:", float(fdhg["log_loss"]))

print()
print("FDHG - DFS")
print("  accuracy gain:", float(fdhg["accuracy"] - dfs["accuracy"]))
print("  AUROC gain:", float(fdhg["roc_auc"] - dfs["roc_auc"]))
print(
    "  AP gain:",
    float(fdhg["average_precision"] - dfs["average_precision"]),
)
print(
    "  log-loss reduction:",
    float(dfs["log_loss"] - fdhg["log_loss"]),
)
PY

echo
echo "============================================================"
echo "END-TO-END BINARY REPRODUCTION COMPLETE"
echo "============================================================"
