#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/final-gate-51task-v2}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/final-gate-51task-v2}"
TASK_FILE="${TASK_FILE:-$OUTPUT_ROOT/task_inventory.tsv}"
STATUS_FILE="${STATUS_FILE:-$OUTPUT_ROOT/run_status.tsv}"

CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/configs/benchmark_tasks.csv}"

FORCE="${FORCE:-0}"
EXPECTED_TASKS="${EXPECTED_TASKS:-51}"

cd "$REPO_ROOT" || exit 1

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

echo "REPO_ROOT=$REPO_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "FORCE=$FORCE"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: missing task config: $CONFIG_FILE"
  exit 1
fi

###############################################################################
# Stage 1: Build the frozen 51-task inventory
###############################################################################

python - "$CONFIG_FILE" "$TASK_FILE" "$EXPECTED_TASKS" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


config_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
expected_count = int(sys.argv[3])

df = pd.read_csv(config_path)

dataset_col = next(
    (
        column
        for column in ["dataset", "dataset_name"]
        if column in df.columns
    ),
    None,
)
task_col = next(
    (
        column
        for column in ["task", "task_name"]
        if column in df.columns
    ),
    None,
)

if dataset_col is None or task_col is None:
    raise SystemExit(
        f"Missing dataset/task columns in {config_path}. "
        f"Available columns: {list(df.columns)}"
    )

pairs = (
    df[[dataset_col, task_col]]
    .dropna()
    .astype(str)
    .apply(lambda column: column.str.strip())
    .drop_duplicates()
)

pairs = pairs[
    (pairs[dataset_col] != "")
    & (pairs[task_col] != "")
]

print(f"DISCOVERED_TASK_COUNT={len(pairs)}")

if len(pairs) != expected_count:
    raise SystemExit(
        f"Expected {expected_count} unique tasks, "
        f"but found {len(pairs)} in {config_path}"
    )

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w") as handle:
    handle.write("dataset\ttask\n")

    for dataset, task in pairs.itertuples(
        index=False,
        name=None,
    ):
        handle.write(f"{dataset}\t{task}\n")
        print(f"{dataset}\t{task}")

print(f"WROTE_TASK_INVENTORY={output_path}")
PY

if [[ $? -ne 0 ]]; then
  echo "ERROR: could not build a verified 51-task inventory."
  exit 1
fi

TASK_COUNT="$(
  awk 'NR > 1 && NF >= 2 {count += 1} END {print count + 0}' \
    "$TASK_FILE"
)"

if [[ "$TASK_COUNT" -ne "$EXPECTED_TASKS" ]]; then
  echo "ERROR: task inventory count is $TASK_COUNT, expected $EXPECTED_TASKS"
  exit 1
fi

echo
echo "VERIFIED_TASK_COUNT=$TASK_COUNT"
echo "TASK_FILE=$TASK_FILE"

###############################################################################
# Stage 2: Initialize status table
###############################################################################

if [[ ! -f "$STATUS_FILE" ]]; then
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "dataset" \
    "task" \
    "status" \
    "exit_code" \
    "started_at" \
    "finished_at" \
    "log_path" \
    > "$STATUS_FILE"
fi

###############################################################################
# Helpers
###############################################################################

is_complete() {
  local dataset="$1"
  local task="$2"
  local slug="${dataset}_${task}"

  local joint_file="$OUTPUT_ROOT/$slug/strategies/joint/$slug/joint_selection.json"

  if [[ ! -f "$joint_file" ]]; then
    return 1
  fi

  python - "$joint_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

try:
    payload = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)

required = {
    "selected_variant",
    "selected_score",
    "baseline_variant",
    "selection_reason",
    "official_validation_was_used_for_selection",
    "test_split_accessed",
}

if not required.issubset(payload):
    raise SystemExit(1)

if payload.get("official_validation_was_used_for_selection") is not False:
    raise SystemExit(1)

if payload.get("test_split_accessed") is not False:
    raise SystemExit(1)

if payload.get("same_candidate_pool_verified") is not True:
    raise SystemExit(1)

raise SystemExit(0)
PY
}

record_status() {
  local dataset="$1"
  local task="$2"
  local status="$3"
  local exit_code="$4"
  local started_at="$5"
  local finished_at="$6"
  local log_path="$7"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$dataset" \
    "$task" \
    "$status" \
    "$exit_code" \
    "$started_at" \
    "$finished_at" \
    "$log_path" \
    >> "$STATUS_FILE"
}

###############################################################################
# Stage 3: Run all tasks sequentially
###############################################################################

SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0
INDEX=0

while IFS=$'\t' read -r dataset task; do
  if [[ "$dataset" == "dataset" ]]; then
    continue
  fi

  if [[ -z "$dataset" || -z "$task" ]]; then
    continue
  fi

  INDEX=$((INDEX + 1))
  slug="${dataset}_${task}"
  log_path="$LOG_ROOT/${slug}.log"

  echo
  echo "================================================================================"
  echo "TASK $INDEX / $TASK_COUNT"
  echo "DATASET: $dataset"
  echo "TASK:    $task"
  echo "================================================================================"

  if [[ "$FORCE" != "1" ]] && is_complete "$dataset" "$task"; then
    echo "SKIP: verified complete artifact already exists"

    now="$(date -Iseconds)"
    record_status \
      "$dataset" \
      "$task" \
      "skipped_complete" \
      "0" \
      "$now" \
      "$now" \
      "$log_path"

    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    continue
  fi

  started_at="$(date -Iseconds)"

  {
    echo "STARTED_AT=$started_at"
    echo "DATASET=$dataset"
    echo "TASK=$task"
    echo "OUTPUT_ROOT=$OUTPUT_ROOT"
    echo
  } > "$log_path"

  python -u -m fdhg.cli.run_fdhg_end_to_end \
    --dataset "$dataset" \
    --task "$task" \
    --output-root "$OUTPUT_ROOT" \
    --budget-policy train_only_grid \
    --auto-budget-grid 4 8 12 16 \
    --selection-folds 3 \
    --min-delta 0.0 \
    --max-fdhg-edges 32 \
    --max-selected-fdhg-edges 32 \
    --edge-screening-rule fixed_count \
    --edge-screening-min-delta 0.0 \
    --edge-screening-min-positive-folds 2 \
    --continuous-fdhg-mode exclude \
    --classification-epsilon 0.001 \
    --regression-relative-epsilon 0.001 \
    --exact-tie-tolerance 1e-12 \
    --no-download \
    --overwrite \
    2>&1 | tee -a "$log_path"

  exit_code="${PIPESTATUS[0]}"
  finished_at="$(date -Iseconds)"

  if [[ "$exit_code" -eq 0 ]] && is_complete "$dataset" "$task"; then
    status="completed"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo "COMPLETED: $dataset / $task"
  else
    status="failed"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    echo "FAILED: $dataset / $task — exit code $exit_code"
  fi

  record_status \
    "$dataset" \
    "$task" \
    "$status" \
    "$exit_code" \
    "$started_at" \
    "$finished_at" \
    "$log_path"

done < "$TASK_FILE"

###############################################################################
# Stage 4: Final summary
###############################################################################

echo
echo "================================================================================"
echo "51-TASK SWEEP FINISHED"
echo "================================================================================"
echo "TOTAL=$TASK_COUNT"
echo "COMPLETED_THIS_RUN=$SUCCESS_COUNT"
echo "SKIPPED_ALREADY_COMPLETE=$SKIPPED_COUNT"
echo "FAILED_THIS_RUN=$FAILED_COUNT"
echo "STATUS_FILE=$STATUS_FILE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "LOG_ROOT=$LOG_ROOT"

python - "$TASK_FILE" "$OUTPUT_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

task_file = Path(sys.argv[1])
root = Path(sys.argv[2])

rows = []

for line in task_file.read_text().splitlines()[1:]:
    if not line.strip():
        continue

    dataset, task = line.split("\t")[:2]
    slug = f"{dataset}_{task}"

    joint_path = (
        root
        / slug
        / "strategies"
        / "joint"
        / slug
        / "joint_selection.json"
    )

    if not joint_path.exists():
        rows.append({
            "dataset": dataset,
            "task": task,
            "status": "missing",
        })
        continue

    try:
        payload = json.loads(joint_path.read_text())
    except Exception as error:
        rows.append({
            "dataset": dataset,
            "task": task,
            "status": f"invalid_json:{type(error).__name__}",
        })
        continue

    safe = (
        payload.get(
            "official_validation_was_used_for_selection"
        ) is False
        and payload.get("test_split_accessed") is False
        and payload.get("same_candidate_pool_verified") is True
    )

    rows.append({
        "dataset": dataset,
        "task": task,
        "status": "complete" if safe else "unsafe_or_incomplete",
        "selected_variant": payload.get("selected_variant"),
        "selected_edges": payload.get("selected_edge_count"),
        "baseline_variant": payload.get("baseline_variant"),
    })

status_counts = Counter(row["status"] for row in rows)
variant_counts = Counter(
    row.get("selected_variant")
    for row in rows
    if row.get("status") == "complete"
)

print("\nARTIFACT_STATUS_COUNTS")
for key, value in sorted(status_counts.items()):
    print(f"{key}: {value}")

print("\nSELECTED_VARIANT_COUNTS")
for key, value in sorted(
    variant_counts.items(),
    key=lambda item: str(item[0]),
):
    print(f"{key}: {value}")

print("\nNONCOMPLETE_TASKS")
noncomplete = [
    row for row in rows
    if row["status"] != "complete"
]

if not noncomplete:
    print("none")
else:
    for row in noncomplete:
        print(
            f'{row["dataset"]}\t'
            f'{row["task"]}\t'
            f'{row["status"]}'
        )
PY

if [[ "$FAILED_COUNT" -gt 0 ]]; then
  exit 2
fi
