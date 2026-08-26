#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export REPO_ROOT="$ROOT"

bash scripts/experiments/run_efficiency_final.sh
python scripts/experiments/collect_efficiency_final.py
