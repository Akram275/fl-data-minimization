"""
Why do the drift-aware coreset's records train below all records when trained centrally (the paper's appendix of
the same name, and the step-matched column centr.* of the label-skew table)?
Two checks with the paper's centralized control (fl.run_pooled_training: same CNN, Adam, batch size, 60% of each
selected client's records, same test set):

  curve   learning curve: centralized training on IID random samples of ALL records (fraction f of the pooled
          clients' data, treated as one client), 20 epochs, 3 sample seeds. If f = coreset size gives about the
          coreset's centralized accuracy, the gap to all records is the volume, not the selection.
  labelmatch  IID samples of each coreset's size, with the coreset's label proportions or the population's.
  steps   step-matched control: each drift-aware coreset of run_protocol.py (label skew, no DP) trained centrally for
          as many gradient steps as 20 epochs on all records (more epochs; per-epoch lr decay rescaled so the total
          decay over training is the same as in the paper's 20 epochs).

Output: checks/central/<tag>.csv
  python central_check.py images --dataset celeba --max-samples 0 --num-clients 100 --partition-on label --partition-seed 42
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import src.federated_learning.torch_fl as fl
from src.experiment import add_partition_args, load_partition


FRACTIONS = (0.027, 0.05, 0.10, 0.25, 1.0)


def n_trained(n):
    """records of an n-record client that run_pooled_training trains on (75/25 split, then 80%)."""
    n_tr = n - math.ceil(fl.LOCAL_TEST_SIZE * n)
    return int(math.floor(n_tr * (1.0 - fl.VALIDATION_SPLIT)))


def run(args, names, client_indices, labels, attrs, attr_names, attr_dims, datasets, test_x, test_y, encoders, root):
    out = Path("checks") / "central"
    out.mkdir(parents=True, exist_ok=True)
    tag = str(root).replace('/', '_')
    rows = []
    X_all = pd.concat([d[0] for d in datasets], ignore_index=True)
    y_all = pd.concat([d[1] for d in datasets], ignore_index=True)
    N = len(X_all)
    full_train = sum(n_trained(len(d[0])) for d in datasets)
    steps_full = args.epochs * math.ceil(full_train / fl.BATCH_SIZE)
    if 'curve' in args.checks:
        for f in args.fractions:
            for s in (range(args.curve_seeds) if f < 1 else range(1)):
                idx = np.random.default_rng(1000 + s).choice(N, int(round(f * N)), replace=False)
                sc = fl.run_pooled_training([[X_all.iloc[idx], y_all.iloc[idx]]], test_x, test_y, args.epochs)
                rows.append(dict(check='curve', frac=f, seed=s, records=len(idx), epochs=args.epochs,
                                 bacc=100 * sc[-1][1]))
                print(f">> curve f={f} seed={s}: {rows[-1]['bacc']:.1f}", flush=True)
                pd.DataFrame(rows).to_csv(out / f"{tag}{args.tag}.csv", index=False)
    if 'steps' in args.checks:
        for f in sorted(f for m in args.methods
                        for f in Path(args.results).glob(f"{root}/*/g0.5_einf_D128/{m}/selection.json")):
            sel = json.load(open(f))
            if not sel.get('trained'):
                continue
            ds = [datasets[names.index(c)] for c in sel['selection']]
            tr = sum(n_trained(len(d[0])) for d in ds)
            epochs = max(args.epochs, int(round(steps_full / math.ceil(tr / fl.BATCH_SIZE))))
            decay0 = fl.DECAY_RATE
            for mode, E in (('paper', args.epochs), ('step-matched', epochs)):
                if mode not in args.modes:
                    continue
                fl.DECAY_RATE = decay0 ** (args.epochs / E)          # same total decay over training
                sc = fl.run_pooled_training(ds, test_x, test_y, E)
                fl.DECAY_RATE = decay0
                rows.append(dict(check='steps', mode=mode, method=f.parts[-2], enc=f.parts[-4], records=tr, epochs=E,
                                 records_frac=100 * sel['records_frac'], bacc=100 * sc[-1][1],
                                 best_bacc=100 * max(s_[1] for s_ in sc)))
                print(f">> steps {f.parts[-4]} {mode} E={E}: {rows[-1]['bacc']:.1f} (best {rows[-1]['best_bacc']:.1f})",
                      flush=True)
                pd.DataFrame(rows).to_csv(out / f"{tag}{args.tag}.csv", index=False)
    if 'labelmatch' in args.checks:
        # same size as each coreset, records drawn at random from the population within each label, with the coreset's
        # pooled label proportions (labelmatch) or the population's (iid); 20 epochs, as in the paper
        yv = y_all.to_numpy().astype(int)
        by_label = {k: np.where(yv == k)[0] for k in np.unique(yv)}
        for f in sorted(Path(args.results).glob(f"{root}/*/g0.5_einf_D128/coreset_dp/selection.json")):
            sel = json.load(open(f))
            if not sel.get('trained'):
                continue
            ys = np.concatenate([datasets[names.index(c)][1].to_numpy().astype(int) for c in sel['selection']])
            n = len(ys)
            rng = np.random.default_rng(7)
            for mode in ('labelmatch', 'iid'):
                if mode == 'labelmatch':
                    idx = np.concatenate([rng.choice(by_label[k], (ys == k).sum(), replace=False) for k in by_label])
                else:
                    idx = rng.choice(N, n, replace=False)
                sc = fl.run_pooled_training([[X_all.iloc[idx], y_all.iloc[idx]]], test_x, test_y, args.epochs)
                rows.append(dict(check='labelmatch', mode=mode, enc=f.parts[-4], records=n,
                                 pos_frac_coreset=float(ys.mean()), pos_frac_pop=float(yv.mean()),
                                 records_frac=100 * sel['records_frac'], bacc=100 * sc[-1][1]))
                print(f">> labelmatch {f.parts[-4]} {mode} pos={ys.mean():.2f}/{yv.mean():.2f}: {rows[-1]['bacc']:.1f}",
                      flush=True)
                pd.DataFrame(rows).to_csv(out / f"{tag}_labelmatch.csv", index=False)
    print(f">> wrote {out / tag}.csv (steps_full={steps_full})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_partition_args(p, encoders=['clip-b32'])
    p.add_argument('--checks', nargs='+', default=['curve', 'steps'], choices=['curve', 'steps', 'labelmatch'])
    p.add_argument('--fractions', type=float, nargs='+', default=list(FRACTIONS), help="curve: shares of the records")
    p.add_argument('--curve-seeds', type=int, default=3)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--results', default='results', help="results root of run_protocol.py")
    p.add_argument('--methods', nargs='+', default=['coreset_dp'], help="steps: selections to train")
    p.add_argument('--modes', nargs='+', default=['paper', 'step-matched'], help="steps: budgets to run")
    p.add_argument('--tag', default='', help="suffix of the output CSV")
    args = p.parse_args()
    run(args, *load_partition(args))


if __name__ == '__main__':
    main()
