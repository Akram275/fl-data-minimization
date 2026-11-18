#!/bin/bash
# The inference audit of scripts/run_mia.sh with record-level DP-SGD in training (mia_check.py --dp-eps), for every
# model (targets and reference models): Poisson sampling, clipping norm 1, noise calibrated per client to
# (eps2, 1e-5), momentum SGD with expected batch 1024 and learning rate 2 (the best of a small grid at eps2 = 8;
# Adam at the non-private settings stays at chance level). The selection is unchanged (eps1 = inf).
# -> mia_check/<partition root>/<encoder>/dp<eps2>/
source "$(dirname "$0")/common.sh"
EPS=${EPS:-"8 1"}
for ds in $DATASETS; do
  ( for e in $EPS; do for s in $SEEDS; do
      $PY -u mia_check.py $(part $ds $s) --encoders $(main_enc $ds) --dp-eps $e --dp-opt sgd --dp-batch 1024 \
          --dp-lr 2 --tag dp$e
    done; done ) > logs/mia_dp_$ds.log 2>&1 &
done
wait
echo "ALL DONE $(date)"
