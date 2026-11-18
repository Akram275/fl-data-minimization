#!/bin/bash
# Selection-only checks and centralized controls of the paper's appendices -> checks/
#   coverage_single  stopping rule, one draw at eps1 = 1, gamma = 0.1, D = 256: uncalibrated vs calibrated rule
#   coverage         100 independent sample + DP-noise draws, gamma = 0.5, D = 128: calibrated rule and the margin
#                    extended to the sampling error (no DP and eps1 in {1, 2, 4})
#   central          why the coreset's records train below all records (learning curve, label mix, step-matched
#                    control; images); needs the results of scripts/run_main.sh
#   robustness       quadratic penalty, random-feature seeds, sample seeds; needs scripts/run_main.sh
#   privacy          independent accounting check (Google's dp_accounting)
#   overhead         client and server cost of the protocol (run alone on the GPU, timings)
source "$(dirname "$0")/common.sh"
for s in $SEEDS; do
  for ds in $DATASETS; do
    ( $PY -u stopping_check.py $(part $ds $s) --encoders $(encs $ds) --eps 1 --gammas 0.1 --rffs 256 \
          --rules current calibrated --out coverage_single
      $PY -u stopping_check.py $(part $ds $s) --encoders $(encs $ds) --eps inf 1 2 4 --gammas 0.5 --rffs 128 \
          --taus 0.99 0.95 --rules calibrated ucb_all --reps 100 --out coverage
      $PY -u selection_checks.py $(part $ds $s) --encoders $(encs $ds) ) > logs/checks_${ds}_$s.log 2>&1 &
  done
done
wait
for s in $SEEDS; do
  for ds in celeba fairface; do
    ( [ $s = 42 ] && $PY -u central_check.py $(part $ds $s) --checks curve
      $PY -u central_check.py $(part $ds $s) --checks steps
      $PY -u central_check.py $(part $ds $s) --checks labelmatch
      for m in ddem fedbary coreset_plain coreset_neq; do
        $PY -u central_check.py $(part $ds $s) --checks steps --modes step-matched --methods $m --tag _$m
      done ) > logs/central_${ds}_$s.log 2>&1 &
  done
done
wait
$PY -u privacy_check.py > logs/privacy_check.log 2>&1
$PY -u overhead_benchmark.py celeba fairface acs_label acs_states sent_label > logs/overhead.log 2>&1
echo "ALL DONE $(date)"
