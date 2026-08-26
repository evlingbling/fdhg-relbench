#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export REPO_ROOT="$ROOT"

bash scripts/experiments/run_final_gate_51_tasks_v2.sh
