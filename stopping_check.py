"""
Coverage of the stopping rule (the paper's coverage table), selection only. For each partition and encoder: the
ground truth is the noise-free sketches of ALL records (gamma = 1, no DP, D = --d-true); the protocol's sketches
(rates --gammas, budgets --eps, D in --rffs, the seeds of run_protocol.py) select the drift-aware coreset
(tau in --taus, lambda = 1) under the stopping rules --rules:
  current     uncalibrated rule: plain-mean reference, resolution floor
  calibrated  the paper's rule: size-weighted debiased reference, resolution floor, 95% upper bound (z = --z) on
              the DP noise of the coreset's discrepancy
  ucb_all     as calibrated, with the bound extended to the sampling error (worst case over directions)
--reps R repeats every setting over R independent sample and DP-noise draws (seeds + 0..R-1), which measures the
coverage of the complete adaptive procedure. Each selection is scored with the ground-truth sketches (true score
under the plain and the weighted reference). Output: checks/<--out>/<partition tag>.csv

  python stopping_check.py images --dataset celeba --max-samples 0 --num-clients 100 --partition-on label --partition-seed 42
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment import add_partition_args, load_partition
from src.selection.embedding_selectors import ClientCoresetSelector
from src.selection.protocol import protocol_snapshots


N_LABELS = 2


def true_scores(mu, n, idx, W):
    w = n / n.sum()
    mu_U = (w[:, None] * mu).sum(0)
    dev2 = ((mu - mu_U) ** 2).sum(1)
    mix = (W[:, None] * mu[idx]).sum(0) / W.sum()
    e = float(((mix - mu_U) ** 2).sum())
    return 1 - e / dev2.mean(), 1 - e / (w * dev2).sum(), e


def run(args, names, client_indices, labels, attrs, attr_names, attr_dims, datasets, test_x, test_y, encoders, root):
    rows = []
    sizes = np.array([len(client_indices[c]) for c in names], float)
    out = Path("checks") / args.out
    out.mkdir(parents=True, exist_ok=True)
    tag = str(root).replace('/', '_')
    for enc, Z in encoders.items():
        # ground truth: all records, no noise, many features (true scores comparable across D)
        T = protocol_snapshots(Z, client_indices, names, labels, N_LABELS, 1.0, float('inf'), args.delta,
                               rff_features=args.d_true)
        mu_true = np.stack([T['snaps'][c] for c in names])
        for eps, rep in ((e, r) for e in args.eps for r in range(args.reps)):
            for gamma in args.gammas:
                for D in args.rffs:
                    P = protocol_snapshots(Z, client_indices, names, labels, N_LABELS, gamma, eps, args.delta,
                                           sample_seed=args.sample_seed + rep, noise_seed=args.noise_seed + rep,
                                           rff_features=D)
                    dp = np.isfinite(eps)
                    wref = dict(ref_mode='weighted') if dp else {}
                    rules = {'current': {},
                             'calibrated': {**wref, 'ucb_z': args.z, 'client_dp_noise': P['client_dp_noise']},
                             'ucb_all': {**wref, 'ucb_z': args.z, 'client_dp_noise': P['client_dp_noise'],
                                         'ucb_sampling': True}}     # bound also on the sampling error
                    for rule in args.rules:
                        for tau in args.taus:
                            sel = ClientCoresetSelector(P['snaps'], P['sizes'], seed=args.seed, drift_lambda=1.0,
                                                        client_noise=P['client_noise'], **rules[rule])
                            subset = sel.fit(tau)
                            idx = np.array([names.index(c) for c in subset])
                            W = np.array([sel.weights[c] for c in subset])
                            ts_plain, ts_w, e = true_scores(mu_true, sizes, idx, W)
                            rows.append(dict(enc=enc, eps=eps, rep=rep, gamma=gamma, D=D, rule=rule, tau=tau, k=len(subset),
                                             records_frac=100 * sizes[idx].sum() / sizes.sum(),
                                             est_score=sel.diagnostics['score'], true_score_plain=ts_plain,
                                             true_score_weighted=ts_w, true_mmd2=e, z_psi=P['z']['snapshot'],
                                             selection=" ".join(subset)))
                            print(f">> {enc} eps={eps:g} g={gamma:g} D={D} {rule} tau={tau}: k={len(subset)} "
                                  f"rec={rows[-1]['records_frac']:.1f}% true={ts_plain:.3f}", flush=True)
                    pd.DataFrame(rows).to_csv(out / f"{tag}.csv", index=False)
    print(f">> wrote {out / tag}.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_partition_args(p)
    p.add_argument('--eps', type=float, nargs='+', default=[float('inf'), 4.0, 2.0, 1.0])
    p.add_argument('--delta', type=float, default=1e-5)
    p.add_argument('--z', type=float, default=1.645, help="one-sided 95% normal quantile of the calibrated rule")
    p.add_argument('--reps', type=int, default=1, help="independent sample + DP-noise draws (seeds + 0..reps-1)")
    p.add_argument('--gammas', type=float, nargs='+', default=[0.1])
    p.add_argument('--rffs', type=int, nargs='+', default=[256])
    p.add_argument('--taus', type=float, nargs='+', default=[0.99])
    p.add_argument('--rules', nargs='+', default=['current', 'calibrated'], choices=['current', 'calibrated', 'ucb_all'])
    p.add_argument('--d-true', type=int, default=1024, help="random features of the ground-truth sketches")
    p.add_argument('--out', default='coverage_single', help="sub-directory of checks/")
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--noise-seed', type=int, default=0)
    args = p.parse_args()
    run(args, *load_partition(args))


if __name__ == '__main__':
    main()
