#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/ablations/candidate-pool-sensitivity}"
FINAL_GATE_ROOT="${FINAL_GATE_ROOT:-}"

if [[ -z "$ARTIFACT_ROOT" ]]; then
    echo "ERROR: ARTIFACT_ROOT must point to the preserved paper artifact repository." >&2
    echo "Example:" >&2
    echo '  ARTIFACT_ROOT=/path/to/fdhg-paper-artifacts scripts/ablations/run_candidate_pool_sensitivity.sh' >&2
    exit 2
fi

if [[ -z "$FINAL_GATE_ROOT" ]]; then
    FINAL_GATE_ROOT="$ARTIFACT_ROOT/outputs/final-gate-51task-v2"
fi

BASE="$OUTPUT_ROOT"
MASTER="$OUTPUT_ROOT/candidate-pools"
LOGS="$OUTPUT_ROOT/logs"

mkdir -p "$BASE" "$MASTER" "$LOGS"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OUTPUT_ROOT


TASKS=(
  "rel-trial|study-adverse"
  "rel-f1|driver-position"
  "rel-trial|studies-enrollment"
  "rel-trial|study-outcome"
)


section () {
    echo
    echo "================================================================================"
    echo "$*"
    echo "TIME: $(date)"
    echo "================================================================================"
}


is_completed () {
    local manifest="$1"

    [[ -f "$manifest" ]] || return 1

    "$PY" - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])

try:
    x = json.loads(p.read_text())
except Exception:
    raise SystemExit(1)

ok = (
    x.get("status") == "completed"
    and x.get("test_split_accessed") is False
)

raise SystemExit(0 if ok else 1)
PY
}


# =============================================================================
# STEP 1
# Bcand=64:
# rediscover candidates from earliest inner-training fold and run Greedy.
# This run is itself the Bcand=64 sensitivity result.
# =============================================================================

section "STEP 1/3 — Bcand=64 DISCOVERY + GREEDY"


for spec in "${TASKS[@]}"; do

    IFS='|' read -r dataset task <<< "$spec"

    slug="${dataset}_${task}"

    auto_root="$FINAL_GATE_ROOT/$slug/auto"
    canonical_root="$FINAL_GATE_ROOT/_canonical_onboarding"

    [[ -f "$auto_root/$slug/selected_features.json" ]] || {
        echo "ERROR: missing frozen Auto representation:"
        echo "$auto_root/$slug/selected_features.json"
        exit 1
    }

    run_root="$BASE/b64"
    run_dir="$run_root/$slug"

    manifest="$run_dir/manifest.json"

    log="$LOGS/${slug}_b64.log"

    master="$MASTER/${slug}_master64.json"


    [[ -d "$auto_root" ]] || {
        echo "ERROR: missing Auto root:"
        echo "$auto_root"
        exit 1
    }


    if is_completed "$manifest"; then
        section "SKIP COMPLETED B64: $dataset/$task"
    else
        section "RUN B64: $dataset/$task"

        "$PY" -m fdhg.cli.auto_fdhg_relbench \
          --dataset "$dataset" \
          --task "$task" \
          --output-root "$run_root" \
          --auto-output-root "$auto_root" \
          --canonical-onboarding-root "$canonical_root" \
          --selection-folds 3 \
          --feature-budget 32 \
          --max-fdhg-edges 64 \
          --max-selected-fdhg-edges 32 \
          --edge-selection-strategy greedy \
          --edge-screening-rule fixed_count \
          --edge-screening-min-delta 0 \
          --edge-screening-min-positive-folds 2 \
          --continuous-fdhg-mode exclude \
          --no-download \
          --write \
          --overwrite \
          2>&1 | tee "$log"
    fi


    # -------------------------------------------------------------------------
    # Export the ordered B64 candidate pool.
    # This same file will be replayed with prefixes 16 and 32.
    # -------------------------------------------------------------------------

    section "EXPORT MASTER POOL: $dataset/$task"

    "$PY" -m fdhg.cli.export_fdhg_candidate_edges \
      --input-output-dir "$run_dir" \
      --output-file "$master"


    "$PY" - "$master" "$dataset" "$task" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
dataset = sys.argv[2]
task = sys.argv[3]

x = json.loads(p.read_text())

edges = (
    x["edges"]
    if isinstance(x, dict) and "edges" in x
    else x
)

if not isinstance(edges, list):
    raise SystemExit("MASTER FILE IS NOT AN EDGE LIST")

ids = [str(e["edge_id"]) for e in edges]

if len(ids) != len(set(ids)):
    raise SystemExit("DUPLICATE EDGE IDS")

print(
    f"MASTER_OK dataset={dataset} task={task} "
    f"candidate_count={len(ids)}"
)

print("FIRST5 =", ids[:5])
print("LAST5  =", ids[-5:])
PY

done



# =============================================================================
# STEP 2
# Replay exact same ordered master pool under Bcand=16 and Bcand=32.
# Bsel stays fixed at 32.
# =============================================================================

section "STEP 2/3 — REPLAY Bcand=16 / 32"


for budget in 16 32; do

    for spec in "${TASKS[@]}"; do

        IFS='|' read -r dataset task <<< "$spec"

        slug="${dataset}_${task}"

        auto_root="$FINAL_GATE_ROOT/$slug/auto"
        canonical_root="$FINAL_GATE_ROOT/_canonical_onboarding"

        [[ -f "$auto_root/$slug/selected_features.json" ]] || {
            echo "ERROR: missing frozen Auto representation:"
            echo "$auto_root/$slug/selected_features.json"
            exit 1
        }

        master="$MASTER/${slug}_master64.json"

        run_root="$BASE/b${budget}"
        run_dir="$run_root/$slug"

        manifest="$run_dir/manifest.json"

        log="$LOGS/${slug}_b${budget}.log"


        [[ -f "$master" ]] || {
            echo "ERROR: missing master candidate file:"
            echo "$master"
            exit 1
        }


        if is_completed "$manifest"; then
            section "SKIP COMPLETED B${budget}: $dataset/$task"
            continue
        fi


        section "RUN Bcand=${budget}, Bsel=32: $dataset/$task"


        "$PY" -m fdhg.cli.auto_fdhg_relbench \
          --dataset "$dataset" \
          --task "$task" \
          --output-root "$run_root" \
          --auto-output-root "$auto_root" \
          --canonical-onboarding-root "$canonical_root" \
          --fdhg-candidate-edges-file "$master" \
          --selection-folds 3 \
          --feature-budget 32 \
          --max-fdhg-edges "$budget" \
          --max-selected-fdhg-edges 32 \
          --edge-selection-strategy greedy \
          --edge-screening-rule fixed_count \
          --edge-screening-min-delta 0 \
          --edge-screening-min-positive-folds 2 \
          --continuous-fdhg-mode exclude \
          --no-download \
          --write \
          --overwrite \
          2>&1 | tee "$log"

    done
done



# =============================================================================
# STEP 3
# Build paper-ready summary and verify candidate-prefix nesting.
# =============================================================================

section "STEP 3/3 — BUILD SUMMARY"


"$PY" - <<'PY'
from pathlib import Path
import json
import os
import pandas as pd
import numpy as np


root = Path(os.environ["OUTPUT_ROOT"]).resolve()


tasks = [
    ("rel-trial", "study-adverse"),
    ("rel-f1", "driver-position"),
    ("rel-trial", "studies-enrollment"),
    ("rel-trial", "study-outcome"),
]


def read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


rows = []
candidate_pools = {}
selected_sets = {}


for dataset, task in tasks:

    slug = f"{dataset}_{task}"

    candidate_pools[slug] = {}
    selected_sets[slug] = {}

    for budget in [16, 32, 64]:

        d = root / f"b{budget}" / slug

        manifest = read_json(
            d / "manifest.json"
        )

        selected = read_json(
            d / "selected_variant.json"
        )

        discovery = read_json(
            d / "candidate_discovery.json"
        )


        # ------------------------------------------------------------
        # Candidate pool
        # ------------------------------------------------------------

        candidate_ids = (
            discovery.get(
                "ordered_candidate_edge_ids",
                []
            )
        )

        if not candidate_ids:
            candidate_ids = [
                str(e.get("edge_id"))
                for e in discovery.get(
                    "accepted_edges",
                    []
                )
                if e.get("edge_id") is not None
            ]


        candidate_pools[slug][budget] = candidate_ids


        # ------------------------------------------------------------
        # Inner scores
        # ------------------------------------------------------------

        scores = selected.get(
            "mean_scores",
            {}
        )

        auto_score = scores.get(
            "auto_only"
        )

        greedy_score = scores.get(
            "auto_plus_fdhg"
        )

        if greedy_score is None:
            greedy_score = scores.get(
                "auto_plus_fdhg_greedy"
            )


        direction = (
            selected.get("metric_direction")
            or manifest.get("metric_direction")
        )


        gain = np.nan

        if (
            auto_score is not None
            and greedy_score is not None
        ):
            if direction in {
                "lower",
                "lower_is_better",
            }:
                gain = (
                    float(auto_score)
                    - float(greedy_score)
                )
            else:
                gain = (
                    float(greedy_score)
                    - float(auto_score)
                )


        # ------------------------------------------------------------
        # Selected edges
        # ------------------------------------------------------------

        selected_ids = (
            manifest.get(
                "strategy_selected_edge_ids",
                []
            )
        )

        if not selected_ids:
            selected_ids = selected.get(
                "selected_edge_ids",
                []
            )

        selected_ids = [
            str(x)
            for x in selected_ids
        ]

        selected_sets[slug][budget] = set(
            selected_ids
        )


        rows.append({
            "dataset": dataset,
            "task": task,

            "Bcand_requested": budget,

            "candidate_count":
                len(candidate_ids),

            "selected_edge_count":
                len(selected_ids),

            "metric":
                (
                    selected.get(
                        "primary_metric"
                    )
                    or manifest.get(
                        "primary_metric"
                    )
                    or manifest.get(
                        "metric"
                    )
                ),

            "metric_direction":
                direction,

            "inner_auto":
                auto_score,

            "inner_greedy":
                greedy_score,

            "gain_vs_auto":
                gain,

            "selected_variant":
                selected.get(
                    "selected_variant"
                ),

            "selected_edge_ids":
                " | ".join(
                    selected_ids
                ),

            "test_split_accessed":
                manifest.get(
                    "test_split_accessed"
                ),
        })


# =====================================================================
# Prefix verification
# =====================================================================

for dataset, task in tasks:

    slug = f"{dataset}_{task}"

    p16 = candidate_pools[
        slug
    ][16]

    p32 = candidate_pools[
        slug
    ][32]

    p64 = candidate_pools[
        slug
    ][64]


    if p16 != p64[:len(p16)]:
        raise RuntimeError(
            f"{slug}: B16 is not "
            "a prefix of B64"
        )

    if p32 != p64[:len(p32)]:
        raise RuntimeError(
            f"{slug}: B32 is not "
            "a prefix of B64"
        )

    print(
        "PREFIX_OK",
        slug,
        f"{len(p16)} <= "
        f"{len(p32)} <= "
        f"{len(p64)}"
    )


df = pd.DataFrame(rows)


# =====================================================================
# Selected-edge overlap relative to default Bcand=32
# =====================================================================

df[
    "selected_overlap_with_b32"
] = np.nan

df[
    "selected_jaccard_with_b32"
] = np.nan


for dataset, task in tasks:

    slug = f"{dataset}_{task}"

    base = selected_sets[
        slug
    ][32]

    for budget in [16, 32, 64]:

        cur = selected_sets[
            slug
        ][budget]

        inter = base & cur
        union = base | cur

        mask = (
            (df["dataset"] == dataset)
            & (df["task"] == task)
            & (
                df["Bcand_requested"]
                == budget
            )
        )

        df.loc[
            mask,
            "selected_overlap_with_b32",
        ] = len(inter)

        df.loc[
            mask,
            "selected_jaccard_with_b32",
        ] = (
            len(inter) / len(union)
            if union
            else 1.0
        )


# =====================================================================
# Save
# =====================================================================

out = (
    root
    / "candidate_budget_sensitivity_summary.csv"
)

df.to_csv(
    out,
    index=False,
)


pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    350,
)

pd.set_option(
    "display.max_colwidth",
    100,
)


print()
print("=" * 140)
print(
    "CANDIDATE-BUDGET "
    "SENSITIVITY FINAL"
)
print("=" * 140)

print(
    df[
        [
            "dataset",
            "task",
            "Bcand_requested",
            "candidate_count",
            "selected_edge_count",
            "metric",
            "inner_auto",
            "inner_greedy",
            "gain_vs_auto",
            "selected_overlap_with_b32",
            "selected_jaccard_with_b32",
            "test_split_accessed",
        ]
    ].to_string(index=False)
)


print()
print("SAVED:", out)


print()
print("=" * 140)
print("TEST ACCESS AUDIT")
print("=" * 140)

print(
    df[
        "test_split_accessed"
    ].value_counts(
        dropna=False
    )
)

PY


section "ALL CANDIDATE-BUDGET SENSITIVITY RUNS COMPLETE"
