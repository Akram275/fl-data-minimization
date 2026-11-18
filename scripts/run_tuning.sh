#!/bin/bash
# Drift correctors on all clients (seed-42 label-skew partitions): FedProx for mu in {0.001, 0.01, 0.1, 1} and
# SCAFFOLD (local SGD) for lr in {0.01, 0.03, 0.1, 0.3}. The best final balanced accuracy (ties: the smaller value)
# chooses mu and lr per dataset; the values are fixed in scripts/run_main.sh and paper/collect.py.
# -> results/<partition root>/-/full/fl_{fedprox_mu,scaffoldsgd_lr}*.csv
source "$(dirname "$0")/common.sh"
for ds in $DATASETS; do
  $PY -u run_protocol.py $(part $ds 42) --encoders $(main_enc $ds) --methods mugrid lrgrid > logs/tuning_$ds.log 2>&1 &
done
wait
$PY - <<'PY'
import glob, re
import pandas as pd
for ds, root in [('celeba', 'conv_results_embedding/celeba/*labelskew/seed_42'),
                 ('fairface', 'conv_results_embedding/fairface/*labelskew/seed_42'),
                 ('acs', 'conv_results_tabular/ACSIncome/label_seed42'),
                 ('sent', 'conv_results_tabular_sent140_label/Sent140/label_seed42')]:
    for name, pat in (('mu', 'fl_fedprox_mu'), ('lr', 'fl_scaffoldsgd_lr')):
        best = None
        for f in sorted(glob.glob(f"results/{root}/-/full/{pat}*.csv")):
            v = float(re.search(pat + r"([\d.e-]+)\.csv", f).group(1))
            a = pd.read_csv(f).balanced_accuracy.iloc[-1]
            if best is None or a > best[1] or (a == best[1] and v < best[0]):
                best = (v, a)
        print(f"{name}[{ds}] = {best[0]:g}" if best else f"{name}[{ds}]: no results")
PY
