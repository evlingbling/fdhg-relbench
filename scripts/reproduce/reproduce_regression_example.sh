#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cpu}"

SEEDS=(41)

for variant in dfs fdhg_dmax1; do
  if [[ "$variant" == "dfs" ]]; then
    root="outputs/dfs_agg/rel-hm_transactions-price_article_sample"
  else
    root="outputs/fdhg_heuristic/rel-hm_transactions-price_sample"
  fi

  for seed in "${SEEDS[@]}"; do
    out="results/rel-hm_transactions-price_tabpfn/${variant}/seed${seed}"
    log="logs/rel-hm_transactions-price_tabpfn/${variant}_seed${seed}.log"

    mkdir -p "$out"
    mkdir -p "$(dirname "$log")"

    echo
    echo "============================================================"
    echo "rel-hm/transactions-price | ${variant} | seed=${seed}"
    echo "============================================================"

    python scripts/evaluate/evaluate_regression_tabpfn.py \
      --train-parquet \
      "${root}/target_with_dfs_agg_train.parquet" \
      --val-parquet \
      "${root}/target_with_dfs_agg_val.parquet" \
      --output-dir "$out" \
      --dataset rel-hm \
      --task transactions-price \
      --variant "$variant" \
      --label-col price \
      --drop-cols "t_dat,primary_key,article_id" \
      --device "$DEVICE" \
      --seed "$seed" \
      2>&1 | tee "$log"
  done
done

echo
echo "REL-HM TRANSACTIONS-PRICE COMPLETE"
