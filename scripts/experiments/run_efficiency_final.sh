#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

ROOT="outputs/efficiency-final"
RAW="$ROOT/raw"
TIME="$ROOT/time"
LOGS="$ROOT/logs"

mkdir -p "$RAW" "$TIME" "$LOGS"

# ------------------------------------------------------------------
# Provenance snapshot
# ------------------------------------------------------------------

git rev-parse HEAD > "$ROOT/git_commit.txt"
git status --short > "$ROOT/git_status.txt"
git diff > "$ROOT/code_diff.patch"
hostname > "$ROOT/hostname.txt"

# Use "|" rather than tabs so shell parsing is unambiguous.
FINAL_GATE_ROOT="${FINAL_GATE_ROOT:-outputs/final-gate-51task-v2}"
CANONICAL_ROOT="${CANONICAL_ROOT:-${FINAL_GATE_ROOT}/_canonical_onboarding}"

TASK_SPECS=(
"rel-event|user-attendance|${FINAL_GATE_ROOT}/rel-event_user-attendance|${CANONICAL_ROOT}"
"rel-f1|driver-position|${FINAL_GATE_ROOT}/rel-f1_driver-position|${CANONICAL_ROOT}"
"rel-trial|studies-enrollment|${FINAL_GATE_ROOT}/rel-trial_studies-enrollment|${CANONICAL_ROOT}"
"rel-trial|study-outcome|${FINAL_GATE_ROOT}/rel-trial_study-outcome|${CANONICAL_ROOT}"
)

METHODS=(
    dfs
    auto
    all
    independent
    greedy
)

TOTAL=$(( ${#TASK_SPECS[@]} * ${#METHODS[@]} ))
INDEX=0

for spec in "${TASK_SPECS[@]}"; do
    IFS='|' read -r dataset task task_root canonical_root <<< "$spec"

    slug="${dataset}_${task}"
    auto_root="${task_root}/auto"
    candidate="${task_root}/candidates/fixed_candidate_edges.json"
    auto_file="${auto_root}/${slug}/selected_features.json"
    canonical_task="${canonical_root}/relbench-v1-${slug}"

    echo
    echo "######################################################################"
    echo "TASK: ${dataset}/${task}"
    echo "AUTO: ${auto_root}"
    echo "DFS : ${canonical_root}"
    echo "CAND: ${candidate}"
    echo "######################################################################"

    # --------------------------------------------------------------
    # Strict provenance preflight
    # --------------------------------------------------------------

    missing=0

    if [[ ! -f "$auto_file" ]]; then
        echo "ERROR: MISSING AUTO: $auto_file"
        missing=1
    fi

    if [[ ! -f "$candidate" ]]; then
        echo "ERROR: MISSING CANDIDATE FILE: $candidate"
        missing=1
    fi

    if [[ ! -d "$canonical_task" ]]; then
        echo "ERROR: MISSING CANONICAL DFS: $canonical_task"
        missing=1
    fi

    if [[ "$missing" -ne 0 ]]; then
        echo "SKIPPING TASK DUE TO FAILED PREFLIGHT"
        continue
    fi

    echo "PREFLIGHT: OK"
    echo

    for method in "${METHODS[@]}"; do
        INDEX=$((INDEX + 1))

        method_root="${RAW}/${method}"
        out="${method_root}/${slug}"
        timefile="${TIME}/${method}__${slug}.time.txt"
        logfile="${LOGS}/${method}__${slug}.log"

        mkdir -p "$method_root"

        echo
        echo "======================================================================"
        echo "[${INDEX}/${TOTAL}] ${method} :: ${dataset}/${task}"
        echo "======================================================================"

        rm -rf "$out"
        rm -f "$timefile" "$logfile"

        /usr/bin/time -v \
            -o "$timefile" \
            "$PY" scripts/experiments/run_efficiency_one.py \
                --dataset "$dataset" \
                --task "$task" \
                --method "$method" \
                --output-root "$method_root" \
                --auto-root "$auto_root" \
                --canonical-root "$canonical_root" \
                --candidate-file "$candidate" \
            2>&1 | tee "$logfile"

        rc=${PIPESTATUS[0]}

        if [[ "$rc" -eq 0 ]]; then
            echo "SUCCESS: ${method} ${dataset}/${task}"
        else
            echo "FAILED rc=${rc}: ${method} ${dataset}/${task}"
        fi

        # Reduce cross-run memory/cache interference.
        sleep 5
    done
done

echo
echo "======================================================================"
echo "EFFICIENCY RUNNER FINISHED"
echo "======================================================================"
echo "Requested profiles: $TOTAL"
echo "Timing files:"
find "$TIME" -maxdepth 1 -type f -name '*.time.txt' | wc -l
