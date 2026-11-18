#!/bin/bash
# Distribution and membership inference on the drift-aware coreset (mia_check.py; needs scripts/run_main.sh):
# the coreset of the main configuration, its matched random federations and FedAvg on all clients, main encoder,
# 8 reference models per partition.
#   main  the paper's budget (20 rounds), seeds 42-44
#   r100  100 rounds, seed 42 (more memorization)
# -> mia_check/<partition root>/<encoder>/{main,r100}/; tables: paper/make_mia_tables.py
source "$(dirname "$0")/common.sh"
for ds in $DATASETS; do
  ( for s in $SEEDS; do $PY -u mia_check.py $(part $ds $s) --encoders $(main_enc $ds); done
    $PY -u mia_check.py $(part $ds 42) --encoders $(main_enc $ds) --rounds 100 --tag r100 ) > logs/mia_$ds.log 2>&1 &
done
wait
echo "ALL DONE $(date)"
