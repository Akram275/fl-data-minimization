"""
Distribution and membership inference on the coreset (Section "A representative hides in its group" and its
appendix): attacks on the models trained in the paper's main configuration (run_protocol.py: gamma = 0.5,
eps = inf, D = 128, drift-aware coreset tau = 0.99, lambda = 1), optionally retrained with record-level DP-SGD.

Targets (same clients, weights, seeds and training code as the paper's runs; the model is captured from
torch_fl's own training loop, so its final accuracy must match the saved fl.csv):
  coreset   FedAvg on the drift-aware coreset, group-mass weights
  rand_<t>  FedAvg on the saved count/volume-matched random federations (t = 1..--n-random)
  full      FedAvg on all clients (same budget of rounds)
Groups: every client is assigned to a participant by the selector's own rule, argmin_r Delta_ir + lambda delta_r,
restricted to the participants (for the coreset these are exactly its groups G_r; for a random federation the same
rule defines "pseudo-groups", the control for whether dilution comes from selection or only from size).

Attacker (black box, per-record confidence p(y|x) of the final model):
  loss   Yeom et al. (2018): score = -loss
  rmia   RMIA offline (Zarifzadeh et al., ICML 2024), --n-ref reference models, gamma = 1, a = --rmia-a
  lira   LiRA offline (Carlini et al., S&P 2022), global variance (few reference models)
Reference models are trained centrally (torch_fl.run_pooled_training recipe) on random subsets of size --ref-size
(default: the coreset's number of trained records) of a reference pool that no target ever sees: the 80% of the
test split that the paper's universal test pool (generate_universal_testdata, 20%) leaves unused. A disjoint part of
it (--n-z records) provides RMIA's population samples z. Every candidate record is OUT of every reference model.

Record roles (per client, the split torch_fl trains on): member = the first floor(0.8 n_tr) rows of the local
75% split; heldout = the remaining 20% of it + the local 25% test split (never trained on); a client outside the
federation has only non-members.

Measurements, per target and attack (summary.csv; one row per target x attack x measurement):
  (1) record      members vs heldout of the same participants, pooled: AUC, TPR at 1% and 0.1% FPR. Per-participant
                  AUC in clients.csv. Coreset vs random vs full = how much record-level risk concentrates.
  (2) dilution    heldout of r vs records of the other members of its group G_r (all non-members): AUC ~ 0.5 means
                  the model treats r's distribution and its group's as one.
      nonpart     records of non-participating clients vs the population test pool (neither ever trained on):
                  does the model single out the clients that were represented but did not train?
  (3) client      client-level membership ("did client i train?"), client score = mean record score over
                  --client-records random records of the client (all if 0): AUC over all clients, top1 = fraction
                  of groups (|G_r| >= 2) where r has the group's highest score (chance: mean 1/|G_r|), and the
                  Spearman correlation, over non-participants, between client score and Delta_{i,r(i)}.
  (4) population  the attacker scores every record of the population, threshold at 0.1% / 1% FPR over all
                  non-members: members identified (tp), precision, share of the population identified, quantiles of
                  the members' scores (coreset vs random vs all clients)
  (5) vulnerable  the 1% most exposed members of all-clients FedAvg: fraction that are members of the target vs
                  its share of the full model's members (does selection leave the most exposed records out?)
  The held-out vs group comparison of (2) is the paper's distribution inference (pair_heldout_auc in clients.csv,
  per non-participant and its representative), and pair_member_auc the membership inference against the group.
DP training (--dp-eps): record-level DP-SGD in every local solver (targets and reference models), see below.
Outputs: mia_check/<partition root>/<enc>/<tag>/{targets/*.pt, refs/*.pt, scores.npz, clients.csv, summary.csv,
         config.json}

Partition arguments as in scripts/run_main.sh (launchers: scripts/run_mia.sh, scripts/run_mia_dp.sh; tables:
paper/make_mia_tables.py):
  python mia_check.py images --dataset celeba --num-clients 100 --partition-seed 42 --encoders clip-b32
  python mia_check.py images --dataset fairface --max-samples 80000 --num-clients 50 --partition-seed 42 --encoders clip-b32
  python mia_check.py tabular --task ACSIncome --partition label --partition-seed 42 --encoders raw
  python mia_check.py tabular --task Sent140 --partition label --tab-root conv_results_tabular_sent140_label --encoders minilm
  python mia_check.py ... --rounds 100 --tag r100        (longer training: more memorization; no paper-run sanity check)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import src.federated_learning.torch_fl as fl
from src.experiment import add_partition_args, load_partition
from src.selection.embedding_selectors import ClientCoresetSelector
from src.selection.protocol import protocol_snapshots

N_LABELS = 2
ATTACKS = ('loss', 'rmia', 'lira')

# ---------------------------------------------------------------------------------------------------------------
# Hooks into torch_fl: keep the reference pool the universal test pool leaves unused, and capture the trained model
# from the paper's own training functions (they call fl._evaluate on the final model after the last round/epoch).
# ---------------------------------------------------------------------------------------------------------------
REF_POOL = {}
_orig_gen, _orig_eval = fl.generate_universal_testdata, fl._evaluate
_last = {}


def _gen(X_test, y_test, val_ratio=0.2):
    x_rest, _, y_rest, _ = train_test_split(X_test, y_test, test_size=val_ratio, random_state=42, stratify=y_test)
    REF_POOL['x'], REF_POOL['y'] = np.asarray(x_rest, dtype=np.float32), np.asarray(y_rest)
    return _orig_gen(X_test, y_test, val_ratio)


def _eval(model, *a):
    _last['model'] = model
    return _orig_eval(model, *a)


fl.generate_universal_testdata, fl._evaluate = _gen, _eval


# ---------------------------------------------------------------------------------------------------------------
# Record-level DP-SGD in the local solver (--dp-eps finite): replaces torch_fl._fit_epochs for every training call
# (FedAvg clients, the single-client centralized path and, in DP mode, the reference models). Poisson sampling with
# rate q = B / n_train, per-record clipping to C, Gaussian noise sigma C on the sum, divided by the expected batch
# q n_train; Adam is post-processing. sigma is calibrated per client (RDP accountant, opacus) so that each record,
# which belongs to one client, gets (eps2, delta) over all of its client's steps: rounds x local epochs x ceil(1/q).
# ---------------------------------------------------------------------------------------------------------------
DP = {}


def _dp_sigma(n_train):
    """Noise multiplier giving (eps2, delta) to the records of a client with n_train training rows over all its
    steps (cached per size)."""
    key = n_train
    if key not in DP['sigma']:
        from opacus.accountants.utils import get_noise_multiplier
        q = min(1.0, fl.BATCH_SIZE / n_train)
        steps = DP['epochs_total'] * int(math.ceil(1 / q))
        DP['sigma'][key] = float(get_noise_multiplier(target_epsilon=DP['eps'], target_delta=DP['delta'],
                                                      sample_rate=q, steps=steps, accountant='rdp',
                                                      epsilon_tolerance=0.005))
    return DP['sigma'][key]


def _dp_fit_epochs(model, opt, x, y, epochs, gen, prox=None, correction=None):
    from torch.func import functional_call, grad, vmap
    if prox is not None or correction is not None:
        raise NotImplementedError("DP-SGD is implemented for FedAvg only")
    n_train = int(math.floor(len(x) * (1.0 - fl.VALIDATION_SPLIT)))
    x, y = x[:n_train], y[:n_train]
    q = min(1.0, fl.BATCH_SIZE / n_train)
    sigma, C = _dp_sigma(n_train), DP['clip']

    def loss1(p, xi, yi):
        return F.binary_cross_entropy_with_logits(functional_call(model, p, (xi.unsqueeze(0),)), yi.unsqueeze(0))

    g1 = vmap(grad(loss1), in_dims=(None, 0, 0), randomness='different')
    model.train()
    params = dict(model.named_parameters())
    steps = 0
    for _ in range(epochs):
        for _ in range(int(math.ceil(1 / q))):
            b = (torch.rand(n_train, generator=gen) < q).nonzero().squeeze(1).to(x.device)
            opt.zero_grad(set_to_none=True)
            summed = {k: torch.zeros_like(v) for k, v in params.items()}
            for s in range(0, len(b), 1024):          # per-record gradients, clipped, summed
                bb = b[s:s + 1024]
                gs = g1({k: v.detach() for k, v in params.items()}, x[bb], y[bb])
                nrm = torch.sqrt(sum(g.flatten(1).pow(2).sum(1) for g in gs.values()))
                f = (C / (nrm + 1e-12)).clamp(max=1.0)
                for k, g in gs.items():
                    summed[k] += (g * f.view(-1, *[1] * (g.dim() - 1))).sum(0)
            for k, v in params.items():
                v.grad = (summed[k] + sigma * C * torch.randn_like(v)) / (q * n_train)
            opt.step()
            steps += 1
    return steps


def enable_dp(eps, delta, clip, epochs_total):
    DP.update(eps=eps, delta=delta, clip=clip, epochs_total=epochs_total, sigma={})
    fl._fit_epochs = _dp_fit_epochs


def train_and_capture(fn, *a, **kw):
    scores = fn(*a, **kw)
    return _last.pop('model'), scores


# ---------------------------------------------------------------------------------------------------------------
# Record roles: the rows torch_fl trains on (same splits, by index)
# ---------------------------------------------------------------------------------------------------------------
def local_roles(n, single):
    """(member, heldout) row indices of a client of size n. single: the client is the whole federation, which
    torch_fl.run_exp trains centrally on the first floor(0.8 n) rows (no 75/25 split)."""
    if single:
        m = int(math.floor(n * (1 - fl.VALIDATION_SPLIT)))
        return np.arange(m), np.arange(m, n)
    tr, te = train_test_split(np.arange(n), test_size=fl.LOCAL_TEST_SIZE, random_state=42)
    m = int(math.floor(len(tr) * (1 - fl.VALIDATION_SPLIT)))
    return np.sort(tr[:m]), np.sort(np.concatenate([tr[m:], te]))


@torch.no_grad()
def true_label_prob(model, x, y, bs=8192):
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for s in range(0, len(x), bs):
        xb = torch.as_tensor(x[s:s + bs], dtype=torch.float32, device=dev)
        out.append(torch.sigmoid(model(xb)).double().cpu().numpy())
    p = np.concatenate(out)
    pr = np.where(np.asarray(y) == 1, p, 1 - p)
    return np.clip(pr, 1e-12, 1 - 1e-12)


# ---------------------------------------------------------------------------------------------------------------
# Attacks (higher score = more likely member)
# ---------------------------------------------------------------------------------------------------------------
def rmia_scores(pr_t, pr_ref, pz_t, pz_ref, a, gamma=1.0):
    """Offline RMIA: Pr(x) = ((1 + a) mean_ref Pr(x|theta') + (1 - a)) / 2 (references are all OUT);
    score(x) = fraction of population samples z with LR(x) / LR(z) > gamma, LR(u) = Pr(u|theta) / Pr(u)."""
    lr_x = pr_t / (0.5 * ((1 + a) * pr_ref.mean(0) + (1 - a)))
    lr_z = np.sort(pz_t / (0.5 * ((1 + a) * pz_ref.mean(0) + (1 - a))))
    return np.searchsorted(lr_z, lr_x / gamma, side='left') / len(lr_z)


def lira_scores(pr_t, pr_ref):
    """Offline LiRA, global variance: P(phi_OUT < phi_target), phi = logit of the true-label probability."""
    phi = lambda p: np.log(p) - np.log1p(-p)
    out = phi(pr_ref)
    mu, sd = out.mean(0), max(float(out.std(0, ddof=1).mean()), 1e-6)
    return norm.cdf((phi(pr_t) - mu) / sd)


def roc_metrics(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return dict(auc=np.nan, tpr1=np.nan, tpr01=np.nan, n_pos=len(pos), n_neg=len(neg))
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    fpr, tpr, _ = roc_curve(y, s)
    at = lambda f: float(tpr[np.searchsorted(fpr, f, side='right') - 1])
    return dict(auc=float(roc_auc_score(y, s)), tpr1=at(0.01), tpr01=at(0.001), n_pos=len(pos), n_neg=len(neg))


# ---------------------------------------------------------------------------------------------------------------
def run(args, names, client_indices, labels, attrs, attr_names, attr_dims, datasets, test_x, test_y, encoders, root):
    sizes = {c: len(client_indices[c]) for c in names}
    dev = fl._device()
    if DP:   # DP-SGD hyperparameters (after the dataset loaders have set the paper's batch size and learning rate)
        fl.BATCH_SIZE = args.dp_batch or fl.BATCH_SIZE
        fl.BASE_LR = args.dp_lr or fl.BASE_LR
        if args.dp_opt == 'sgd':   # momentum SGD (fresh per round/run, like the paper's Adam) instead of Adam
            fl._adam = lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9)
        DP.update(batch=fl.BATCH_SIZE, lr=fl.BASE_LR, opt=args.dp_opt)
        print(f"  [dp] eps2 = {DP['eps']:g}, delta = {DP['delta']:g}, clip = {DP['clip']:g}, batch = {fl.BATCH_SIZE}, "
              f"lr = {fl.BASE_LR:g}, opt = {args.dp_opt}")
    # every client record, in client order (row j of client c = row j of its torch_fl dataset)
    Xc = np.concatenate([np.asarray(d[0], dtype=np.float32) for d in datasets])
    Yc = np.concatenate([np.asarray(d[1]) for d in datasets])
    rec_client = np.concatenate([[c] * sizes[c] for c in names])
    rec_row = np.concatenate([np.arange(sizes[c]) for c in names])
    for enc, Z in encoders.items():
        sel_dir = (Path(args.results) / root / enc /
                   (f"g{args.gamma:g}_einf" + (f"_D{args.rff_features}" if args.rff_features != 256 else "")) / args.method)
        saved = json.load(open(sel_dir / "selection.json"))
        if not saved.get('trained'):
            print(f"  [mia] {sel_dir}: selection not trained in the paper's runs; skipped")
            continue
        out = Path(args.out) / root / enc / (args.tag or "main")
        (out / "targets").mkdir(parents=True, exist_ok=True)
        (out / "refs").mkdir(parents=True, exist_ok=True)

        # --- groups: redo the selection on the same snapshots (deterministic) and check it is the saved one -----
        P = protocol_snapshots(Z, client_indices, names, labels, N_LABELS, args.gamma, float('inf'), 1e-5,
                               sample_seed=saved.get('sample_seed', 0), noise_seed=saved.get('noise_seed', 0),
                               rff_features=args.rff_features, gamma0=saved.get('gamma0'))
        lam = saved['diagnostics']['lambda']
        sel = ClientCoresetSelector(P['snaps'], P['sizes'], seed=args.seed, drift_lambda=lam,
                                    client_noise=P['client_noise'], ref_mode='weighted', ucb_z=1.645,
                                    client_dp_noise=P['client_dp_noise'])
        subset = sel.fit(args.target)
        if sorted(subset) != sorted(saved['selection']):
            raise RuntimeError(f"recomputed selection {subset} != saved {saved['selection']}")
        cidx = {c: i for i, c in enumerate(sel.clients)}

        def groups_of(participants):
            """client -> participant it is assigned to, by the selector's rule restricted to the participants."""
            med = np.array([cidx[c] for c in participants])
            a = sel._assign(med)
            return {c: participants[a[cidx[c]]] for c in names}

        targets = {'coreset': (saved['selection'], saved['weights'])}
        for t, s_ in enumerate(saved.get('random_subsets', [])[:args.n_random], 1):
            targets[f"rand_{t}"] = (s_, None)
        if not args.no_full:
            targets['full'] = (names, None)

        # --- train (or load) the targets with the paper's code -----------------------------------------------
        models, accs = {}, {}
        for tname, (subset, w) in targets.items():
            f = out / "targets" / f"{tname}.pt"
            ii = [names.index(c) for c in subset]
            if f.exists():
                ck = torch.load(f, map_location=dev)
                m = fl.new_model(fl.SEED).to(dev)
                m.load_state_dict(ck['state'])
                models[tname], accs[tname] = m, ck['bacc']
                continue
            m, sc = train_and_capture(fl.run_exp, ii, datasets, test_x, test_y, epochs=args.local_epochs,
                                      max_iterations=args.rounds, aggregation_method='fedavg', lr_decay=True,
                                      agg_weights=None if w is None else [w[c] for c in subset])
            torch.save({'state': m.state_dict(), 'bacc': sc[-1][1], 'subset': subset, 'weights': w}, f)
            models[tname], accs[tname] = m, sc[-1][1]
        # sanity: the paper's saved run of the same federation (GPU nondeterminism: a few points at most)
        ref_csv = {'coreset': sel_dir / "fl.csv", **{f"rand_{t}": sel_dir / f"rand_trial_{t}_fl.csv"
                                                     for t in range(1, args.n_random + 1)}}
        for tname, a in accs.items():
            if DP:
                print(f"  [mia] {tname}: bacc {a * 100:.1f} (DP-SGD eps2 = {DP['eps']:g})")
            elif args.rounds == 20 and tname in ref_csv and ref_csv[tname].exists():
                paper = pd.read_csv(ref_csv[tname])['balanced_accuracy'].iloc[-1]
                print(f"  [mia] {tname}: bacc {a * 100:.1f} (paper run {paper * 100:.1f})")

        # --- reference models: OUT for every candidate record ------------------------------------------------
        rx, ry = REF_POOL['x'], REF_POOL['y']
        prng = np.random.default_rng(args.seed + 7)
        perm = prng.permutation(len(rx))
        z_idx, pool_idx = perm[:args.n_z], perm[args.n_z:]
        cs_n = sum(len(local_roles(sizes[c], len(saved['selection']) == 1)[0]) for c in saved['selection'])
        ref_size = min(args.ref_size or cs_n, len(pool_idx))
        refs = []
        for j in range(args.n_ref):
            f = out / "refs" / f"ref_{j}.pt"
            m = fl.new_model(fl.SEED + 1000 + j).to(dev)
            if f.exists():
                m.load_state_dict(torch.load(f, map_location=dev)['state'])
            else:
                # run_pooled_training trains on the first 60% of each "client" (its 75/25 + 80/20 splits): pass
                # the subset whole as one dataset of size ref_size / 0.6 so that ~ref_size records are trained on
                k = min(len(pool_idx), int(math.ceil(ref_size / ((1 - fl.LOCAL_TEST_SIZE) * (1 - fl.VALIDATION_SPLIT)))))
                idx = np.random.default_rng(args.seed + 100 + j).choice(pool_idx, k, replace=False)
                if DP:   # same DP-SGD as the targets: run_centralized_training trains on the first 80% (patched solver)
                    idx = idx[:int(math.ceil(ref_size / (1 - fl.VALIDATION_SPLIT)))]
                    m, _ = train_and_capture(fl.run_centralized_training, (pd.DataFrame(rx[idx]), pd.Series(ry[idx])),
                                             test_x, test_y, args.rounds, seed=fl.SEED + 1000 + j)
                else:
                    m, _ = train_and_capture(fl.run_pooled_training, [[pd.DataFrame(rx[idx]), pd.Series(ry[idx])]],
                                             test_x, test_y, args.rounds, seed=fl.SEED + 1000 + j)
                torch.save({'state': m.state_dict(), 'n_records': k}, f)
            refs.append(m)

        # --- per-record confidences on every client record and on z -------------------------------------------
        zx, zy = rx[z_idx], ry[z_idx]
        tx, ty = np.asarray(test_x, dtype=np.float32), np.asarray(test_y)    # population test pool: never trained on
        pr_ref = np.stack([true_label_prob(m, Xc, Yc) for m in refs])
        pz_ref = np.stack([true_label_prob(m, zx, zy) for m in refs])
        pt_ref = np.stack([true_label_prob(m, tx, ty) for m in refs])

        rows, crow, dump = [], [], {}
        crng = np.random.default_rng(args.seed + 3)
        for tname, (subset, _) in targets.items():
            m = models[tname]
            single = len(subset) == 1
            pr_t = true_label_prob(m, Xc, Yc)
            pz_t, ptest_t = true_label_prob(m, zx, zy), true_label_prob(m, tx, ty)
            S = {'loss': np.log(pr_t), 'rmia': rmia_scores(pr_t, pr_ref, pz_t, pz_ref, args.rmia_a),
                 'lira': lira_scores(pr_t, pr_ref)}
            St = {'loss': np.log(ptest_t), 'rmia': rmia_scores(ptest_t, pt_ref, pz_t, pz_ref, args.rmia_a),
                  'lira': lira_scores(ptest_t, pt_ref)}
            role = np.full(len(rec_client), 'nonpart', dtype=object)
            off = np.cumsum([0] + [sizes[c] for c in names])
            for c in subset:
                mem, held = local_roles(sizes[c], single)
                o = off[names.index(c)]
                role[o + mem], role[o + held] = 'member', 'heldout'
            dump[tname] = {'role': role.astype(str), **{f"{a}": S[a].astype(np.float32) for a in ATTACKS},
                           **{f"test_{a}": St[a].astype(np.float32) for a in ATTACKS}}
            grp = groups_of(list(subset)) if tname != 'full' else {c: c for c in names}
            gcol = np.array([grp[c] for c in rec_client])
            part = set(subset)
            for a in ATTACKS:
                s = S[a]
                base = dict(target=tname, attack=a, k=len(subset), bacc=accs[tname])
                rows.append({**base, 'measure': 'record', **roc_metrics(s[role == 'member'], s[role == 'heldout'])})
                if tname != 'full':
                    dil = []
                    for r in subset:
                        own = (rec_client == r) & (role == 'heldout')
                        grp_r = (gcol == r) & (rec_client != r)
                        if grp_r.any():
                            dil.append(roc_metrics(s[own], s[grp_r]))
                    rows.append({**base, 'measure': 'dilution',
                                 **{k: float(np.nanmean([d[k] for d in dil])) if dil else np.nan
                                    for k in ('auc', 'tpr1', 'tpr01', 'n_pos', 'n_neg')}})
                    rows.append({**base, 'measure': 'nonpart', **roc_metrics(s[role == 'nonpart'], St[a])})
                # client level
                cs = {}
                for c in names:
                    ii = np.where(rec_client == c)[0]
                    if args.client_records and len(ii) > args.client_records:
                        ii = crng.choice(ii, args.client_records, replace=False)
                    cs[c] = float(s[ii].mean())
                if tname != 'full':
                    is_p = np.array([c in part for c in names])
                    v = np.array([cs[c] for c in names])
                    top1, chance = [], []
                    for r in subset:
                        G = [c for c in names if grp[c] == r]
                        if len(G) >= 2:
                            top1.append(max(G, key=lambda c: cs[c]) == r)
                            chance.append(1 / len(G))
                    npart = [c for c in names if c not in part]
                    dist = [sel.D[cidx[c], cidx[grp[c]]] for c in npart]
                    rho = spearmanr(dist, [cs[c] for c in npart])[0] if len(npart) > 2 else np.nan
                    rows.append({**base, 'measure': 'client', **roc_metrics(v[is_p], v[~is_p]),
                                 'top1': float(np.mean(top1)) if top1 else np.nan,
                                 'top1_chance': float(np.mean(chance)) if chance else np.nan,
                                 'spearman_dist': float(rho)})
                for c in names:
                    o = rec_client == c
                    rec_auc = (roc_metrics(s[o & (role == 'member')], s[o & (role == 'heldout')])['auc']
                               if c in part else np.nan)
                    # pairwise, for a non-participant c and its representative r = grp[c]: r's held-out records vs c's
                    # (distribution part, ~ Delta_{c,r}) and r's training records vs c's (what a record-level attacker
                    # sees when c's records are the alternative)
                    pair_held = pair_mem = np.nan
                    if tname != 'full' and c not in part:
                        r_ = rec_client == grp[c]
                        pair_held = roc_metrics(s[r_ & (role == 'heldout')], s[o])['auc']
                        pair_mem = roc_metrics(s[r_ & (role == 'member')], s[o])['auc']
                    crow.append(dict(target=tname, attack=a, client=c, n=sizes[c], participant=c in part,
                                     group=grp[c], is_rep=grp[c] == c, delta_to_rep=float(sel.D[cidx[c], cidx[grp[c]]]),
                                     dev=float(sel.dev[cidx[c]]), weight=(targets[tname][1] or {}).get(c, np.nan),
                                     client_score=cs[c], record_auc=rec_auc, pair_heldout_auc=pair_held,
                                     pair_member_auc=pair_mem,
                                     pos_frac=float(Yc[rec_client == c].mean())))
            print(pd.DataFrame([r for r in rows if r['target'] == tname])
                  [['target', 'attack', 'measure', 'auc', 'tpr1']].to_string(index=False))
        # --- population level: the attacker scores EVERY record of the population ---------------------------------
        # (4) population  threshold at a false-positive rate `fpr` over all non-member records of the population
        #                 (held-out records of participants + every record of non-participants); tp = members
        #                 flagged, precision = tp / (tp + fp), tp_frac_N = tp / N (the share of the population an
        #                 attacker identifies). Member-score quantiles describe the per-record exposure profile.
        # (5) vulnerable  the 1% most exposed members of all-clients FedAvg (highest score under that model): the
        #                 fraction that are members of the target (in_target) vs the target's share of the full
        #                 model's members (base_rate). below the base rate = selection leaves the most exposed
        #                 records out of training.
        N_pop = len(rec_client)
        for tname, d in dump.items():
            mem = d['role'] == 'member'
            for a in ATTACKS:
                s = d[a]
                q = np.quantile(s[mem], [0.5, 0.9, 0.99, 0.999]) if mem.any() else [np.nan] * 4
                for fpr in (1e-3, 1e-2):
                    thr = np.quantile(s[~mem], 1 - fpr)
                    tp, fp = int((s[mem] > thr).sum()), int((s[~mem] > thr).sum())
                    rows.append(dict(target=tname, attack=a, k=len(targets[tname][0]), bacc=accs[tname],
                                     measure='population', fpr=fpr, tp=tp, fp=fp, n_members=int(mem.sum()),
                                     precision=tp / max(tp + fp, 1), tp_frac_N=tp / N_pop,
                                     q50=q[0], q90=q[1], q99=q[2], q999=q[3], max=float(s[mem].max()) if mem.any() else np.nan))
        if 'full' in dump:
            fmem = dump['full']['role'] == 'member'
            for a in ATTACKS:
                sf = np.where(fmem, dump['full'][a], -np.inf)
                n_top = max(int(0.01 * fmem.sum()), 1)
                top = np.argsort(-sf, kind='stable')[:n_top]
                for tname, d in dump.items():
                    if tname == 'full':
                        continue
                    tmem = d['role'] == 'member'
                    rows.append(dict(target=tname, attack=a, k=len(targets[tname][0]), bacc=accs[tname],
                                     measure='vulnerable', n_top=n_top, in_target=float(tmem[top].mean()),
                                     base_rate=float(tmem[fmem].mean()),
                                     # the same records' exposure under the target (members only), vs the full model's
                                     top_score_target=float(np.nanmean(np.where(tmem[top], d[a][top], np.nan)))
                                     if tmem[top].any() else np.nan,
                                     top_score_full=float(dump['full'][a][top].mean())))
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
        pd.DataFrame(crow).to_csv(out / "clients.csv", index=False)
        np.savez_compressed(out / "scores.npz", client=rec_client.astype(str), row=rec_row,
                            **{f"{t}__{k}": v for t, d in dump.items() for k, v in d.items()})
        json.dump({'selection_dir': str(sel_dir), 'ref_size': ref_size, 'n_ref': args.n_ref, 'n_z': args.n_z,
                   'rmia_a': args.rmia_a, 'rounds': args.rounds, 'targets': {t: v[0] for t, v in targets.items()},
                   'bacc': accs, 'dp': {k: v for k, v in DP.items()} if DP else None},
                  open(out / "config.json", 'w'), indent=1, default=str)
        print(f">> done {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_partition_args(p, encoders=['clip-b32'])
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--local-epochs', type=int, default=1)
    # which paper selection
    p.add_argument('--results', default="results", help="results root of run_protocol.py")
    p.add_argument('--method', default='coreset_dp')
    p.add_argument('--target', type=float, default=0.99, help="tau of --method (to recompute its groups)")
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--rff-features', type=int, default=128)
    # targets and attacker
    p.add_argument('--n-random', type=int, default=5)
    p.add_argument('--no-full', action='store_true')
    p.add_argument('--n-ref', type=int, default=8)
    p.add_argument('--ref-size', type=int, default=0, help="records per reference model (0: the coreset's)")
    p.add_argument('--n-z', type=int, default=2000)
    p.add_argument('--rmia-a', type=float, default=0.3)
    p.add_argument('--client-records', type=int, default=0, help="records per client for client-level MIA (0: all)")
    p.add_argument('--out', default="mia_check")
    p.add_argument('--tag', default="", help="subdirectory (default 'main'), e.g. r100 for --rounds 100")
    p.add_argument('--dp-eps', type=float, default=float('inf'), help="record-level DP-SGD budget eps2 of training")
    p.add_argument('--dp-delta', type=float, default=1e-5)
    p.add_argument('--dp-clip', type=float, default=1.0)
    p.add_argument('--dp-opt', choices=['adam', 'sgd'], default='adam', help="local optimizer under DP-SGD")
    p.add_argument('--dp-batch', type=int, default=0, help="expected batch of DP-SGD (0: the paper's)")
    p.add_argument('--dp-lr', type=float, default=0.0, help="base learning rate under DP-SGD (0: the paper's)")
    args = p.parse_args()
    if np.isfinite(args.dp_eps):
        enable_dp(args.dp_eps, args.dp_delta, args.dp_clip, args.rounds * args.local_epochs)
    run(args, *load_partition(args))


if __name__ == '__main__':
    main()
