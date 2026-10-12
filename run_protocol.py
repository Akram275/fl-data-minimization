"""
Main pipeline of the paper: for one client partition and each encoder, run the selection protocol
(src/selection/protocol.py: sampled encoding, no server data, optionally differentially private), every selection
method on its sketches, and train each selection with FedAvg, centrally on the same records, and against
`--trials` random federations matched on client count and volume.

Methods (--methods):
  coreset_plain  plain coreset, tau = 0.95, lambda = 0
  coreset_dp     drift-aware coreset, tau = 0.99, lambda = 1 (the paper's main configuration)
  coreset_dp95   drift-aware coreset with the lax target tau = 0.95
  coreset_neq    IID-equivalent rule n_eq >= 0.5% N, lambda = 1
  grid           tau in {0.95, 0.99, 0.999} x lambda in {0, 0.25, 1, 4} (as coreset_t<tau>_l<lambda>)
Published baselines and drift correctors (they need coreset_dp's selection on the same partition; K = its k):
  ddem           DDEM (Xu et al., IEEE Access 2024): top-K clients by a label-count score, chosen once before
                 training (src/selection/ddem.py). With a finite --eps, from per-client class counts released
                 with the Gaussian mechanism (sensitivity 1, the whole budget eps, delta), so that private DDEM
                 and the private coreset spend the same selection budget.
  fedbary        FedBary (Li et al. 2024, no validation set): the K clients closest to the Wasserstein
                 barycenter of the clients (src/selection/fedbary.py)
  divfl          DivFL (Balakrishnan et al., ICLR 2022): K clients per round by facility location on client
                 gradients (every client sends a gradient before round 1), against FedAvg with K random clients
                 per round (--trials seeds)
  drift          FedProx (mu = --prox-mu) on the drift-aware coreset and its matched random federations; with
                 --full, also on all clients
  scaffold       SCAFFOLD (local SGD, option II, lr = --scaffold-lr) on the same federations; with --full, also on
                 all clients
  mugrid         FedProx on all clients for mu in --mu-grid (chooses --prox-mu per dataset)
  lrgrid         SCAFFOLD on all clients for lr in --lr-grid (chooses --scaffold-lr per dataset)
K = coreset_dp's k when that selection was trained, otherwise coreset_neq's (e.g. FairFace attribute skew, where
the drift-aware coreset needs about half of the clients).

  python run_protocol.py images --dataset celeba --max-samples 0 --num-clients 100 --partition-on label --partition-seed 42
  python run_protocol.py tabular --task ACSIncome --partition label --partition-seed 42 --encoders raw
  python run_protocol.py images ... --eps 4 2 1 --methods coreset_dp coreset_neq ddem      (private selection)
  python run_protocol.py images ... --gammas 0.1 0.2 1 --methods coreset_dp coreset_neq    (sampling-rate ablation)
Outputs: <out>/<partition root>/<encoder>/g<gamma>_e<eps>[_D<D>]/<method>/{selection.json, fl.csv, cen.csv,
         rand_trial_<t>_{fl,cen}.csv}, and <out>/<partition root>/-/full/ for the runs on all clients.
scripts/run_main.sh runs every configuration of the paper.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

import src.federated_learning.torch_fl as fl
from src.embedding.snapshots import unit_norm
from src.experiment import add_partition_args, load_partition
from src.selection.embedding_selectors import ClientCoresetSelector
from src.selection.protocol import protocol_snapshots
from src.selection.random_baselines import count_volume_matched
from src.selection.subset_metrics import SubsetEvaluator

N_LABELS = 2
CORESETS = {'coreset_plain': dict(target=0.95, lam=0.0), 'coreset_dp': dict(target=0.99, lam=1.0),
            'coreset_neq': dict(neq=0.005, lam=1.0), 'coreset_dp95': dict(target=0.95, lam=1.0)}
GRID = [(t, l) for t in (0.95, 0.99, 0.999) for l in (0.0, 0.25, 1.0, 4.0)]
# The matched random federations are drawn from a string seed that contains the results path. It is fixed to
# the root of the paper's runs, so that any --out reproduces the paper's random federations.
RANDOM_SEED_ROOT = "results_v3"


def config_dir(out_root: Path, enc: str, gamma: float, eps: float, rff_features: int) -> Path:
    return out_root / enc / (f"g{gamma:g}_e{eps:g}" + (f"_D{rff_features}" if rff_features != 256 else ""))


def run(args, names, client_indices, labels, attrs, attr_names, attr_dims, datasets, test_x, test_y, encoders, root):
    sizes = {c: len(client_indices[c]) for c in names}
    N = sum(sizes.values())
    evaluator = SubsetEvaluator(client_indices, labels, attrs, attr_names, attr_dims, N_LABELS)
    out_root = Path(args.out) / root

    def train(subset, out, weights=None):
        """FedAvg (group-mass weights for a coreset), the same records trained centrally, and the count/volume
        matched random federations (each also trained both ways). CSVs that exist are not recomputed."""
        def both(sub, stem, w=None):
            ii = [names.index(c) for c in sub]
            f_fl, f_cen = out / f"{stem}fl.csv", out / f"{stem}cen.csv"
            if not f_fl.exists():
                fl.save_scores_to_csv(fl.run_exp(ii, datasets, test_x, test_y, epochs=args.local_epochs,
                                                 max_iterations=args.rounds, aggregation_method='fedavg', lr_decay=True,
                                                 agg_weights=None if w is None else [w[c] for c in sub]), f_fl)
            if not f_cen.exists():
                fl.save_scores_to_csv(fl.run_pooled_training([datasets[i] for i in ii], test_x, test_y, args.rounds), f_cen)
        both(subset, "", weights)
        rr = random.Random(f"v2-{Path(RANDOM_SEED_ROOT) / out.relative_to(args.out)}")
        subsets, errs, n_ok = count_volume_matched(names, sizes, subset, args.trials, 0.10, 20000, rr)
        for t, s_ in enumerate(subsets, 1):
            both(s_, f"rand_trial_{t}_")
        return {'random_subsets': subsets, 'random_rel_volume_error': errs, 'n_within_tol': n_ok}

    def both_algo(subset, f, algo, weights=None, mu=0.0, sgd_lr=0.1):
        """FedAvg variant `algo` on `subset` -> CSV f (skipped if it exists)."""
        if not f.exists():
            ii = [names.index(c) for c in subset]
            fl.save_scores_to_csv(fl.run_exp(ii, datasets, test_x, test_y, epochs=args.local_epochs,
                                             max_iterations=args.rounds, aggregation_method=algo, lr_decay=True,
                                             agg_weights=None if weights is None else [weights[c] for c in subset],
                                             mu=mu, sgd_lr=sgd_lr), f)

    def record(out, method, subset, weights=None, extra=None):
        """Saves a selection and trains it, unless it holds more than --max-record-frac of the records or every
        client (no minimization: reported as not trained)."""
        out.mkdir(parents=True, exist_ok=True)
        rec = sum(sizes[c] for c in subset) / N
        trained = rec <= args.max_record_frac and len(subset) < len(names)
        info = train(subset, out, weights) if trained and not args.no_train else {}
        json.dump({'method': method, 'selection': subset, 'weights': weights, 'k': len(subset),
                   'records': sum(sizes[c] for c in subset), 'records_frac': rec, 'trained': trained,
                   'metrics': evaluator.evaluate(subset), **(extra or {}), **info},
                  open(out / "selection.json", 'w'), indent=1, default=float)

    for eps in args.eps:
        for gamma in args.gammas:
            for enc, Z in encoders.items():
                P = protocol_snapshots(Z, client_indices, names, labels, N_LABELS, gamma, eps, args.delta,
                                       sample_seed=args.sample_seed, noise_seed=args.noise_seed,
                                       rff_features=args.rff_features, gamma0=args.gamma0)
                base = config_dir(out_root, enc, gamma, eps, args.rff_features)
                prot = {'gamma': gamma, 'eps': eps, 'rff_features': args.rff_features,
                        'encoded_frac': P['encoded_frac'], 'noise_multipliers': P['z'],
                        'sample_seed': args.sample_seed, 'noise_seed': args.noise_seed, 'gamma0': args.gamma0}
                cfgs = {m: CORESETS[m] for m in CORESETS if m in args.methods}
                if 'grid' in args.methods and (not args.grid_encoders or enc in args.grid_encoders):
                    cfgs.update({f"coreset_t{t:g}_l{l:g}": dict(target=t, lam=l) for t, l in GRID})
                Ns = sum(P['sizes'].values())
                for m, cfg in cfgs.items():
                    if (base / m / "selection.json").exists():
                        continue
                    # The paper's stopping rule: size-weighted debiased reference, 95% upper bound on the DP noise
                    # of the coreset's discrepancy (zero without DP), resolution floor for the sampling error.
                    sel = ClientCoresetSelector(P['snaps'], P['sizes'], seed=args.seed, drift_lambda=cfg['lam'],
                                                client_noise=P['client_noise'], ref_mode='weighted', ucb_z=1.645,
                                                client_dp_noise=P['client_dp_noise'])
                    subset = sel.fit(n_target=cfg['neq'] * Ns, feature_var=P['feature_var']) if 'neq' in cfg \
                        else sel.fit(cfg['target'])
                    reached = sel.diagnostics['k'] < len(names) or (
                        sel.diagnostics['score'] >= cfg.get('target', 0) if 'target' in cfg else True)
                    record(base / m, m, subset, sel.weights, {**prot, 'diagnostics': sel.diagnostics,
                                                              'target_reached': bool(reached)})
    published(args, names, sizes, client_indices, labels, datasets, test_x, test_y, encoders, out_root, both_algo,
              record)
    print(f">> done {out_root}")


def published(args, names, sizes, client_indices, labels, datasets, test_x, test_y, encoders, out_root, both_algo,
              record):
    """Published baselines and drift correctors (see the module docstring)."""
    want = set(args.methods) & {'ddem', 'divfl', 'drift', 'mugrid', 'fedbary', 'lrgrid', 'scaffold'}
    if not want:
        return
    full = out_root / '-' / 'full'
    if want & {'mugrid', 'lrgrid'} or (want & {'drift', 'scaffold'} and args.full):
        full.mkdir(parents=True, exist_ok=True)
    if 'mugrid' in want:
        for mu in args.mu_grid:
            both_algo(names, full / f"fl_fedprox_mu{mu:g}.csv", 'fedprox', mu=mu)
    if 'lrgrid' in want:
        for lr in args.lr_grid:
            both_algo(names, full / f"fl_scaffoldsgd_lr{lr:g}.csv", 'scaffold_sgd', sgd_lr=lr)
    if 'scaffold' in want and args.full:
        both_algo(names, full / f"fl_scaffoldsgd_lr{args.scaffold_lr:g}.csv", 'scaffold_sgd', sgd_lr=args.scaffold_lr)
    if 'drift' in want and args.full:
        both_algo(names, full / f"fl_fedprox_mu{args.prox_mu:g}.csv", 'fedprox', mu=args.prox_mu)
    for eps in args.eps:
        for gamma in args.gammas:
            for enc in encoders:
                base = config_dir(out_root, enc, gamma, eps, args.rff_features)
                cs, k_from = None, None
                for m in ('coreset_dp', 'coreset_neq'):
                    f = base / m / 'selection.json'
                    if f.exists() and json.load(open(f)).get('trained'):
                        cs, k_from = json.load(open(f)), m
                        break
                if cs is None:
                    print(f"  [published] no trained coreset selection in {base}; skipped")
                    continue
                K = cs['k']
                if 'ddem' in want and not (base / 'ddem' / 'selection.json').exists():
                    from src.selection.ddem import ddem_scores, ddem_select
                    extra = {'K_from': k_from, 'b': 3.0}
                    if np.isfinite(eps):
                        from src.selection.dp_accounting import sigma_closed_form
                        z = sigma_closed_form(eps, args.delta, 1)
                        nrng = np.random.default_rng(1000 * args.noise_seed + 11)
                        counts = np.stack([np.bincount(labels[client_indices[c]], minlength=N_LABELS) for c in names])
                        noisy = np.maximum(counts + nrng.normal(0, z, counts.shape), 0.0)
                        order = np.argsort(-ddem_scores(noisy, 3.0), kind='stable')
                        sel = [names[i] for i in order[:K]]
                        extra.update(noise_multiplier=z, eps=eps, delta=args.delta)
                    else:
                        sel = ddem_select(client_indices, names, labels, N_LABELS, K)
                    record(base / 'ddem', 'ddem', sel, None, extra)
                if 'fedbary' in want and not (base / 'fedbary' / 'selection.json').exists():
                    from src.selection.fedbary import fedbary_select
                    Pr = protocol_snapshots(encoders[enc], client_indices, names, labels, N_LABELS, gamma, eps,
                                            args.delta, sample_seed=args.sample_seed, noise_seed=args.noise_seed,
                                            rff_features=args.rff_features, gamma0=args.gamma0)
                    # FedBary works on the clients' sampled records in the shared space of the protocol
                    Hc = {c: Pr['space'].transform(unit_norm(encoders[enc][Pr['sample'][c]])) if len(Pr['sample'][c])
                          else np.zeros((0, Pr['space'].n_components)) for c in names}
                    yc = {c: labels[Pr['sample'][c]] for c in names}
                    sel, dist = fedbary_select(Hc, yc, names, K, seed=args.seed)
                    record(base / 'fedbary', 'fedbary', sel, None,
                           {'K_from': k_from, 'distance': dict(zip(names, map(float, dist))),
                            'n_supp': 100, 'mode': 'exact barycenter (limit of Alg. 1)'})
                if 'divfl' in want and k_from == 'coreset_dp':
                    out = base / 'divfl'
                    out.mkdir(parents=True, exist_ok=True)
                    for t in range(0, args.trials + 1):          # t = 0: DivFL, t >= 1: K random clients per round
                        f = out / ("fl.csv" if t == 0 else f"rand_round_{t}_fl.csv")
                        if f.exists():
                            continue
                        sc, hist = fl.run_per_round(datasets, test_x, test_y, K, args.rounds,
                                                    selection='divfl' if t == 0 else 'random',
                                                    epochs=args.local_epochs, sample_seed=t)
                        fl.save_scores_to_csv(sc, f)
                        json.dump({'K': K, 'selection': 'divfl' if t == 0 else 'random',
                                   'history': [[names[i] for i in r] for r in hist],
                                   'ever_trained': sorted({names[i] for r in hist for i in r}),
                                   'ever_trained_records': sum(sizes[names[i]] for i in {i for r in hist for i in r}),
                                   'N': sum(sizes.values()),
                                   'gradient_clients': len(names) if t == 0 else 0},
                                  open(f.with_suffix('.json'), 'w'), indent=1)
                if 'scaffold' in want and k_from == 'coreset_dp':
                    d = base / 'coreset_dp'
                    tag = f"scaffoldsgd_lr{args.scaffold_lr:g}"
                    both_algo(cs['selection'], d / f"fl_{tag}.csv", 'scaffold_sgd', cs['weights'], sgd_lr=args.scaffold_lr)
                    for t, sub in enumerate(cs.get('random_subsets', []), 1):
                        both_algo(sub, d / f"rand_trial_{t}_fl_{tag}.csv", 'scaffold_sgd', None, sgd_lr=args.scaffold_lr)
                if 'drift' in want and k_from == 'coreset_dp':
                    d = base / 'coreset_dp'
                    tag = f"fedprox_mu{args.prox_mu:g}"
                    both_algo(cs['selection'], d / f"fl_{tag}.csv", 'fedprox', cs['weights'], args.prox_mu)
                    for t, sub in enumerate(cs.get('random_subsets', []), 1):
                        both_algo(sub, d / f"rand_trial_{t}_fl_{tag}.csv", 'fedprox', None, args.prox_mu)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_partition_args(p)
    p.add_argument('--methods', nargs='+', default=['coreset_plain', 'coreset_dp', 'coreset_neq'])
    p.add_argument('--grid-encoders', nargs='+', default=None, help="encoders that run the tau/lambda grid (default all)")
    p.add_argument('--gammas', type=float, nargs='+', default=[0.5], help="Round 1 sampling rates gamma_1")
    p.add_argument('--gamma0', type=float, default=None,
                   help="Round 0 on an independent sample of this rate (default: Round 0 uses the Round 1 sample, "
                        "the configuration of the paper's results)")
    p.add_argument('--rff-features', type=int, default=128, help="D; results go to g<gamma>_e<eps>_D<D> if not 256")
    p.add_argument('--eps', type=float, nargs='+', default=[float('inf')], help="selection budgets eps_1 (inf: no DP)")
    p.add_argument('--delta', type=float, default=1e-5)
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--noise-seed', type=int, default=0)
    p.add_argument('--trials', type=int, default=5, help="matched random federations per selection")
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--local-epochs', type=int, default=1)
    p.add_argument('--max-record-frac', type=float, default=0.5, help="selections above this share are not trained")
    p.add_argument('--no-train', action='store_true', help="selection only")
    p.add_argument('--out', default="results")
    p.add_argument('--prox-mu', type=float, default=0.01, help="FedProx mu for --methods drift")
    p.add_argument('--mu-grid', type=float, nargs='+', default=[0.001, 0.01, 0.1, 1.0])
    p.add_argument('--full', action='store_true', help="drift/scaffold: also on all clients")
    p.add_argument('--scaffold-lr', type=float, default=0.1, help="local SGD lr of SCAFFOLD (--methods scaffold)")
    p.add_argument('--lr-grid', type=float, nargs='+', default=[0.01, 0.03, 0.1, 0.3])
    args = p.parse_args()
    run(args, *load_partition(args))


if __name__ == '__main__':
    main()
