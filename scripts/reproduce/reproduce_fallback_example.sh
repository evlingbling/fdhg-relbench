#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cpu}"

SEEDS=(41)

for variant in dfs fdhg_dmax1; do
  if [[ "$variant" == "dfs" ]]; then
    root="outputs/dfs_agg/rel-ratebeer_beer_ratings-total_score_sample"
  else
    root="outputs/fdhg_heuristic/rel-ratebeer_beer_ratings-total_score_sample"
  fi

  for seed in "${SEEDS[@]}"; do
    out="results/rel-ratebeer_beer_ratings-total_score_tabpfn/${variant}/seed${seed}"
    log="logs/rel-ratebeer_beer_ratings-total_score_tabpfn/${variant}_seed${seed}.log"

    mkdir -p "$out"
    mkdir -p "$(dirname "$log")"

    echo
    echo "============================================================"
    echo "rel-ratebeer/beer_ratings-total_score | ${variant} | seed=${seed}"
    echo "============================================================"

    python scripts/evaluate/evaluate_regression_tabpfn.py \
      --train-parquet \
      "${root}/target_with_dfs_agg_train.parquet" \
      --val-parquet \
      "${root}/target_with_dfs_agg_val.parquet" \
      --output-dir "$out" \
      --dataset rel-ratebeer \
      --task beer_ratings-total_score \
      --variant "$variant" \
      --label-col total_score \
      --drop-cols "created_at,rating_id,beer_id" \
      --device "$DEVICE" \
      --seed "$seed" \
      2>&1 | tee "$log"
  done
done

echo
echo "REL-RATEBEER TOTAL-SCORE COMPLETE"
