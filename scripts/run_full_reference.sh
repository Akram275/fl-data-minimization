#!/bin/bash
# FedAvg on all clients and centralized training on all records, for every partition of scripts/run_main.sh
# (full_reference.py) -> results/<partition root>/full/
source "$(dirname "$0")/common.sh"
P="$PY -u full_reference.py"
for s in $SEEDS; do
  for ds in $DATASETS; do
    $P $(part $ds $s) > logs/full_${ds}_$s.log 2>&1 &
  done
  $P images --dataset celeba --max-samples 0 --num-clients 100 --partition-on attr --partition-seed $s > logs/full_celeba_attr_$s.log 2>&1 &
  $P images --dataset fairface --max-samples 80000 --num-clients 100 --partition-on attr --partition-seed $s > logs/full_fairface_attr_$s.log 2>&1 &
  $P tabular --task Sent140 --partition groups --tab-root conv_results_tabular_sent140_groups --partition-seed $s \
      > logs/full_sent_groups_$s.log 2>&1 &
  wait
done
$P tabular --task ACSIncome --partition states > logs/full_acs_states.log 2>&1
echo "ALL DONE $(date)"
