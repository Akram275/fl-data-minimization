"""
Robustness of the drift-aware coreset (tau = 0.99, lambda = 1, calibrated rule, gamma = 0.5, D = 128), selection
only, on the label-skew partitions (the paper's appendix "Robustness of the configuration"):
  base       the main selection (sample/noise seed 0, random-feature seed 42); checked against run_protocol.py's
  quad_l<l>  the quadratic drift penalty lambda * W_r * delta_r^2 / delta_bar (penalty='quadratic'), lambda = 1, 4, 16
  rff<s>     the linear penalty with random-feature seed s (s = 0..9): stability to the random-feature draw
  smp<s>     the linear penalty with sample seed s (s = 1..5): reference variability from the sampling alone
Each selection is scored with noise-free sketches of all records (gamma = 1, no DP, D = 1024): true score (plain
reference, as in the coverage table) and true representatives' heterogeneity sum_r alpha_r delta_r^2 / delta_bar^2.
Output: checks/robustness/<tag>.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment import add_partition_args, load_partition
from src.selection.embedding_selectors import ClientCoresetSelector
from src.selection.protocol import protocol_snapshots


N_LABELS = 2


def run(args, names, client_indices, labels, attrs, attr_names, attr_dims, datasets, test_x, test_y, encoders, root):
    out = Path("checks") / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    tag = str(root).replace('/', '_')
    sizes = np.array([len(client_indices[c]) for c in names], float)
    w = sizes / sizes.sum()
    rows = []
    for enc, Z in encoders.items():
        T = protocol_snapshots(Z, client_indices, names, labels, N_LABELS, 1.0, float('inf'), rff_features=1024)
        mu = np.stack([T['snaps'][c] for c in names])
        mu_U = (w[:, None] * mu).sum(0)
        dev2 = ((mu - mu_U) ** 2).sum(1)
        ref_plain, ref_w = dev2.mean(), (w * dev2).sum()
        for eps in args.eps:
            stored = Path(args.results) / root / enc / f"g0.5_e{eps:g}_D128" / "coreset_dp" / "selection.json"
            main_sel = sorted(json.load(open(stored))['selection']) if stored.exists() else None
            variants = [('base', dict(), dict(), 1.0)] + [(f'quad_l{l:g}', dict(), dict(penalty='quadratic'), l)
                                                          for l in (1.0, 4.0, 16.0)]
            variants += [(f'rff{s}', dict(rff_seed=s), dict(), 1.0) for s in range(10)]
            variants += [(f'smp{s}', dict(sample_seed=s, noise_seed=s), dict(), 1.0) for s in range(1, 6)]
            base = None
            for name, pkw, skw, lam in variants:
                P = protocol_snapshots(Z, client_indices, names, labels, N_LABELS, 0.5, eps, 1e-5,
                                       rff_features=128, **pkw)
                sel = ClientCoresetSelector(P['snaps'], P['sizes'], seed=args.seed, drift_lambda=lam,
                                            client_noise=P['client_noise'], ref_mode='weighted', ucb_z=1.645,
                                            client_dp_noise=P['client_dp_noise'], **skw)
                subset = sel.fit(0.99)
                idx = np.array([names.index(c) for c in subset])
                W = np.array([sel.weights[c] for c in subset])
                a = W / W.sum()
                e = float((((a[:, None] * mu[idx]).sum(0) - mu_U) ** 2).sum())
                if name == 'base':
                    base = sorted(subset)
                rows.append(dict(enc=enc, eps=eps, variant=name, k=len(subset),
                                 records_frac=100 * sizes[idx].sum() / sizes.sum(),
                                 true_score=1 - e / ref_plain, true_het=float((a * dev2[idx]).sum() / ref_w),
                                 same_as_base=sorted(subset) == base,
                                 jaccard_base=len(set(subset) & set(base)) / len(set(subset) | set(base)),
                                 same_as_main=(sorted(subset) == main_sel) if main_sel is not None else None,
                                 selection=" ".join(subset)))
            print(f">> {enc} eps={eps:g}: base==main {rows[-len(variants)]['same_as_main']}", flush=True)
        pd.DataFrame(rows).to_csv(out / f"{tag}.csv", index=False)
    print(f">> wrote {out / tag}.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_partition_args(p)
    p.add_argument('--eps', type=float, nargs='+', default=[float('inf'), 1.0])
    p.add_argument('--results', default='results', help="results root of run_protocol.py (to check the base run)")
    args = p.parse_args()
    run(args, *load_partition(args))


if __name__ == '__main__':
    main()
