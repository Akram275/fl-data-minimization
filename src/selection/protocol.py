"""
The selection protocol (no server data, sampled encoding, optional DP), Sections 4-5 of the paper.

Each client draws a Poisson sample of its records (each record independently with probability gamma = gamma_1)
and encodes only those, with a frozen public encoder E. With gamma0 set, Round 0 uses its own, independent
Poisson sample of rate gamma0 (the paper's two-sample design, accounted by calibrate_two_samples); with
gamma0=None (default, the configuration of the paper's results), both rounds use the one sample of rate gamma
(accounted by calibrate_subsampled).

  Round 0  noisy count n_i (all records), and, on the sample, the sum and the upper triangle of
           the second-moment matrix of the unit-norm encodings x = E(r)/||E(r)||; the server
           estimates the population moments by Horvitz-Thompson (divide by gamma) and fits the
           standardize -> whitened PCA space (FederatedSpace).
  Round 1  on the sample, the sum S_i of psi(x, y) = phi(h(x)) (x) e_y with paired random features
           (||psi|| = 1, D features), plus noise; the server forms the (kernel mean) sketch
           mu_i = S_i / (gamma * n_i).

Per-client expected squared sketch error (unbiased estimator; exact for unit-norm features):
  q_i = (1 - gamma) / (gamma n_i)  [sampling]  +  z_psi^2 L D / (gamma n_i)^2  [DP noise]
which the selectors subtract from every squared MMD of their stopping rules.
eps = inf gives the non-private protocol (no noise); gamma = 1 encodes every record.
"""
from typing import Dict, List

import numpy as np

from src.embedding.snapshots import FederatedSpace, analytic_rff, unit_norm
from src.selection.dp_accounting import calibrate_subsampled, calibrate_two_samples
from src.selection.private_selection import budget_shares

GAMMA, RFF_FEATURES, PCA_DIM = 0.1, 256, 32


def poisson_sample(client_indices: Dict[str, np.ndarray], gamma: float, rng: np.random.Generator):
    if gamma >= 1:
        return {c: idx for c, idx in client_indices.items()}
    return {c: idx[rng.random(len(idx)) < gamma] for c, idx in client_indices.items()}


def protocol_snapshots(Z: np.ndarray, client_indices: Dict[str, np.ndarray], names: List[str], labels: np.ndarray,
                       n_labels: int, gamma: float = GAMMA, eps: float = float('inf'), delta: float = 1e-5,
                       sample_seed: int = 0, noise_seed: int = 0, rff_features: int = RFF_FEATURES,
                       pca_dim: int = PCA_DIM, rff_seed: int = 42, gamma0: float = None) -> dict:
    """Simulates both rounds for every client. Z holds the encodings of all records, but only sampled rows are
    used (the client encodes only those). Seeds: the Round 1 sample from 10_000 + sample_seed, the Round 0 sample
    (gamma0 set) from 20_000 + sample_seed, all DP noise from 1000 * noise_seed + 7.

    Returns what the server holds (sketches 'snaps', noisy sizes, the per-client expected squared sketch errors
    'client_noise' and their DP part 'client_dp_noise', v = 'feature_var', the noise multipliers 'z'), plus the
    client samples and the shared space (used by FedBary, which needs per-record features)."""
    private = np.isfinite(eps)
    if not private:
        z = dict.fromkeys(('count', 'sum', 'outer', 'snapshot'), 0.0)
    elif gamma0 is None:
        z = calibrate_subsampled(eps, delta, gamma, budget_shares())
    else:
        z = calibrate_two_samples(eps, delta, gamma0, gamma, budget_shares())
    srng, nrng = np.random.default_rng(10_000 + sample_seed), np.random.default_rng(1000 * noise_seed + 7)
    sample = poisson_sample(client_indices, gamma, srng)
    # gamma0 (optional): Round 0 uses its own sample, independent of Round 1's, so that the shared space does not
    # depend on the sketch sample. Default: one sample for both rounds.
    g0 = gamma if gamma0 is None else gamma0
    sample0 = sample if gamma0 is None else poisson_sample(client_indices, gamma0, np.random.default_rng(20_000 + sample_seed))
    n_t = {c: max(len(client_indices[c]) + (nrng.normal(0, z['count']) if private else 0.0), 1.0) for c in names}
    # Round 0 (sample-based moments; Horvitz-Thompson scaling by 1/gamma is post-processing)
    p = Z.shape[1]
    moments = []
    for c in names:
        X = unit_norm(Z[sample0[c]])
        s, O = X.sum(0), X.T @ X
        if private:
            s = s + nrng.normal(0, z['sum'], p)
            G = np.triu(nrng.normal(0, z['outer'], (p, p)))
            O = O + G + np.triu(G, 1).T
        moments.append({'n': n_t[c], 'sum': s / g0, 'outer': O / g0})
    space = FederatedSpace(pca_dim).fit_moments(moments)
    rff = analytic_rff(pca_dim, rff_features, seed=rff_seed, paired=True)
    LD = n_labels * rff_features
    snaps, q, q_dp = {}, {}, {}
    for c in names:
        idx = sample[c]
        S = np.zeros((n_labels, rff_features))
        if len(idx):
            np.add.at(S, labels[idx], rff.transform(space.transform(unit_norm(Z[idx]))))
        S = S.ravel()
        if private:
            S = S + nrng.normal(0, z['snapshot'], LD)
        snaps[c] = S / (gamma * n_t[c])
        q_dp[c] = z['snapshot'] ** 2 * LD / (gamma * n_t[c]) ** 2       # DP part: isotropic Gaussian, exact
        q[c] = (1 - gamma) / (gamma * n_t[c]) + q_dp[c]
    N = sum(n_t.values())
    mu = sum(n_t[c] * snaps[c] for c in names) / N
    bias = sum((n_t[c] / N) ** 2 * q[c] for c in names)
    v = 1.0 - (float((mu ** 2).sum()) - bias)            # ||psi|| = 1
    return {'snaps': snaps, 'sizes': n_t, 'client_noise': q, 'client_dp_noise': q_dp, 'feature_var': v, 'z': z, 'gamma': gamma,
            'eps': eps, 'sample': sample, 'space': space, 'rff': rff,
            'gamma0': gamma0, 'encoded_frac': sum(len(np.union1d(sample[c], sample0[c])) for c in names)
            / sum(len(client_indices[c]) for c in names)}
