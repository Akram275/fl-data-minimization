#!/bin/bash
# The paper's selection and FL results (run_protocol.py, gamma = 0.5, D = 128, calibrated stopping rule) -> results/
#   stage 1: no DP. Label skew: coresets, tau/lambda grid (main encoder), DDEM, FedBary, DivFL, and FedProx and
#            SCAFFOLD on the drift-aware coreset and its matched random federations. Attribute skew and natural
#            clients: coresets and DDEM.
#   stage 2: private selection, eps1 in {4, 2, 1}: drift-aware coresets (tau 0.99 and 0.95), IID-equivalent rule,
#            private DDEM.
#   stage 3: ablations on the main encoder: gamma in {0.1, 0.2, 1} (D = 128) and D in {64, 256} (gamma = 0.5),
#            without DP and at eps1 = 1.
# FedProx mu and SCAFFOLD lr per dataset: from scripts/run_tuning.sh.
source "$(dirname "$0")/common.sh"
P="$PY -u run_protocol.py"
declare -A MU=( [celeba]=0.1 [fairface]=0.01 [acs]=1 [sent]=0.001 )
declare -A LR=( [celeba]=0.03 [fairface]=0.01 [acs]=0.03 [sent]=0.3 )
CORE="coreset_plain coreset_dp coreset_neq"
# ---------------- stage 1
for s in $SEEDS; do
  for ds in $DATASETS; do
    $P $(part $ds $s) --encoders $(encs $ds) --methods $CORE coreset_dp95 grid ddem fedbary divfl drift scaffold \
        --grid-encoders $(main_enc $ds) --prox-mu ${MU[$ds]} --scaffold-lr ${LR[$ds]} > logs/main_s1_${ds}_$s.log 2>&1 &
  done
done
wait
for s in $SEEDS; do
  $P images --dataset celeba --max-samples 0 --num-clients 100 --partition-on attr --partition-seed $s \
      --methods $CORE ddem > logs/main_s1_celeba_attr_$s.log 2>&1 &
  $P images --dataset fairface --max-samples 80000 --num-clients 100 --partition-on attr --partition-seed $s \
      --methods $CORE ddem > logs/main_s1_fairface_attr_$s.log 2>&1 &
  $P tabular --task Sent140 --partition groups --encoders minilm mpnet --tab-root conv_results_tabular_sent140_groups \
      --partition-seed $s --methods $CORE ddem > logs/main_s1_sent_groups_$s.log 2>&1 &
done
$P tabular --task ACSIncome --partition states --encoders raw --methods $CORE ddem > logs/main_s1_acs_states.log 2>&1 &
wait
echo "stage 1 done $(date)"
# ---------------- stage 2
for s in $SEEDS; do
  for ds in $DATASETS; do
    $P $(part $ds $s) --encoders $(encs $ds) --eps 4 2 1 --methods coreset_dp coreset_dp95 coreset_neq ddem \
        > logs/main_s2_${ds}_$s.log 2>&1 &
  done
done
$P tabular --task ACSIncome --partition states --encoders raw --eps 4 2 1 --methods coreset_dp coreset_neq \
    > logs/main_s2_acs_states.log 2>&1 &
wait
echo "stage 2 done $(date)"
# ---------------- stage 3
for s in $SEEDS; do
  for ds in $DATASETS; do
    ( $P $(part $ds $s) --encoders $(main_enc $ds) --gammas 0.1 0.2 1 --eps inf 1 --methods coreset_dp coreset_neq
      $P $(part $ds $s) --encoders $(main_enc $ds) --rff-features 64 --eps inf 1 --methods coreset_dp
      $P $(part $ds $s) --encoders $(main_enc $ds) --rff-features 256 --eps inf 1 --methods coreset_dp ) \
        > logs/main_s3_${ds}_$s.log 2>&1 &
  done
done
wait
echo "ALL DONE $(date)"
