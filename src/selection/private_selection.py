"""
Budget allocation of the private selection phase, and the all-records variant of the protocol (every record
encoded, no sampling), used by the cost benchmark (overhead_benchmark.py) and one figure. The paper's protocol,
with sampled encoding, is src/selection/protocol.py; dp_accounting.py has the privacy analysis.

  Round 0  each client releases a noisy count and the noisy sum and second-moment matrix of its unit-norm
           encoder outputs; the server fits the standardize -> whitened PCA space on the noisy totals.
  Round 1  each client releases the noisy sum of its paired-RFF joint features; the server divides by the noisy
           count from Round 0.

The RDP budget gives 70% to the Round 1 sketch and 10% to each Round 0 release (budget_shares).
"""
from typing import Dict, List

import numpy as np

from src.embedding.snapshots import (FederatedSpace, analytic_rff, dp_client_moments, dp_count, dp_snapshot_sum,
                                     unit_norm)
from src.selection.dp_accounting import calibrate

SNAPSHOT_SHARE = 0.7
RFF_FEATURES = 256


def budget_shares(snapshot_share: float = SNAPSHOT_SHARE) -> Dict[str, float]:
    """RDP budget split: snapshot_share to the snapshot, the rest equally to Round 0's
    count, sum and second moment (0.25 = the reference paper's equal split)."""
    r = (1 - snapshot_share) / 3
    return {'count': r, 'sum': r, 'outer': r, 'snapshot': snapshot_share}


def private_snapshots(Z: np.ndarray, client_indices: Dict[str, np.ndarray], names: List[str], labels: np.ndarray,
                      n_labels: int, eps: float, delta: float = 1e-5, snapshot_share: float = SNAPSHOT_SHARE,
                      pca_dim: int = 32, rff_features: int = RFF_FEATURES, rff_seed: int = 42,
                      noise_seed: int = 0) -> dict:
    """Simulates both private rounds. eps = inf runs the same pipeline without noise.
    Returns what the server holds: snapshots (noisy sum / noisy count), noisy
    counts, the noise variance q = z^2 * dim of every snapshot sum (for the
    debiased stopping rule) and v = 1 - ||mu||^2 (debiased; ||psi|| = 1)."""
    private = np.isfinite(eps)
    z = calibrate(eps, delta, budget_shares(snapshot_share)) if private else dict.fromkeys(
        ('count', 'sum', 'outer', 'snapshot'), 0.0)
    rng = np.random.default_rng(1000 * noise_seed + 7)
    n_t = {c: dp_count(len(client_indices[c]), z['count'], rng) for c in names}
    mom = [dp_client_moments(Z[client_indices[c]], z['sum'], z['outer'], rng) for c in names]
    for c, m in zip(names, mom):
        m['n'] = n_t[c]
    E = FederatedSpace(pca_dim).fit_moments(mom).transform(unit_norm(Z))
    rff = analytic_rff(E.shape[1], rff_features, seed=rff_seed, paired=True)
    snaps = {c: dp_snapshot_sum(E[client_indices[c]], rff, labels[client_indices[c]], n_labels, z['snapshot'], rng)
             / n_t[c] for c in names}
    q = z['snapshot'] ** 2 * n_labels * rff_features
    N_t = sum(n_t.values())
    mu = sum(n_t[c] * snaps[c] for c in names) / N_t
    v = 1.0 - (float((mu ** 2).sum()) - q * len(names) / N_t ** 2)
    return {'snaps': snaps, 'sizes': n_t, 'noise_var': q, 'feature_var': v, 'z': z}
