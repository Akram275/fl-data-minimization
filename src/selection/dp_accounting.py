"""
Differential privacy of the selection phase (everything a client sends before FL training), Section 5 and the
privacy-analysis appendix of the paper, following the structure of "Choose Wisely and Privately" (arXiv 2605.20975), Sec. V:

  - neighbouring datasets: one record added to / removed from a client's data;
  - each client perturbs its OWN releases with the Gaussian mechanism, so the guarantee holds against the
    semi-honest server for every client separately;
  - the releases are composed with Renyi DP, rho(alpha) = sum_j alpha / (2 z_j^2)
    for a query with L2 sensitivity Delta_j and noise sigma_j = z_j Delta_j, and
    converted with eps = min_alpha rho(alpha) + ln(1/delta) / (alpha - 1);
  - with M queries at the same noise multiplier z this gives the closed form of
    their Eq. (6), z = (sqrt(2 M ln(1/delta)) + sqrt(2 M (ln(1/delta) + eps))) / (2 eps);
  - the selection (k-medoids, stopping rule, group weights) is post-processing
    of the noisy releases; FL training is a separate budget eps_2 that only the
    selected clients spend (eps_total = eps_1, or eps_1 + eps_2 if selected).

Selection-phase queries (Delta = add/remove L2 sensitivity):

  Round 0 (shared space)
    n_i            record count                      Delta = 1
    sum x          x = E(r) / ||E(r)||  (unit norm)   Delta = 1
    sum x x^T      upper triangle incl. diagonal      Delta = ||x||^2 = 1
  Round 1 (sketch)
    sum psi        psi = phi(h) (x) e_y, paired cos/sin RFF, ||psi|| = 1, Delta = 1

Raw encoder outputs are unbounded, hence the unit-norm map in Round 0 (a
clipping that never clips). Round 1 needs no clipping: RFF features are
bounded whatever the (noisy) Round 0 space is. The count is released once and
reused in Round 1; sum ||psi||^2 = n_i with paired features, so the separate
scalar of the non-private protocol is no longer sent; the per-dimension
second moments are the diagonal of sum x x^T.
"""
from typing import Dict, Sequence

import numpy as np

QUERIES = ('count', 'sum', 'outer', 'snapshot')       # one Gaussian release each, Delta = 1


def sigma_closed_form(eps: float, delta: float, M: int = 1) -> float:
    """Noise multiplier for M equal Gaussian queries under (eps, delta), Eq. (6) of the reference."""
    L = np.log(1.0 / delta)
    return (np.sqrt(2 * M * L) + np.sqrt(2 * M * (L + eps))) / (2 * eps)


def rdp_epsilon(multipliers: Sequence[float], delta: float) -> float:
    """(eps, delta) of independent Gaussian releases with noise multipliers z_j
    (sigma_j / Delta_j), RDP composition + the standard conversion (exact minimum over alpha)."""
    A = 0.5 * sum(1.0 / z ** 2 for z in multipliers)
    B = np.log(1.0 / delta)
    alpha = 1.0 + np.sqrt(B / A)
    return float(A * alpha + B / (alpha - 1))


def calibrate(eps: float, delta: float = 1e-5, shares: Dict[str, float] = None) -> Dict[str, float]:
    """Noise multiplier per query for a total selection budget (eps, delta).

    shares: fraction of the RDP budget given to each query (sum 1); query j then
    gets z_j = z_1 / sqrt(share_j) with z_1 the single-query multiplier. Equal
    shares (default) reproduce the reference's equal-noise allocation, i.e.
    z = sigma_closed_form(eps, delta, M=len(QUERIES))."""
    shares = shares or {q: 1.0 / len(QUERIES) for q in QUERIES}
    if abs(sum(shares.values()) - 1) > 1e-9:
        raise ValueError("shares must sum to 1")
    z1 = sigma_closed_form(eps, delta, 1)
    z = {q: z1 / np.sqrt(s) for q, s in shares.items()}
    assert abs(rdp_epsilon(list(z.values()), delta) - eps) < 1e-6 * max(eps, 1)
    return z


# ---------------------------------------------------------------------------
# Privacy amplification by Poisson subsampling (sampled encoding)
# ---------------------------------------------------------------------------
# With sampled encoding, each client encodes every record independently with probability gamma
# and computes all its encoding-based releases (Round 0 sum and second moment, Round 1 snapshot
# sum) on that one sample. Their adaptive composition is a Gaussian mechanism with multiplier
# z_joint, 1/z_joint^2 = sum_q 1/z_q^2 (Gaussian DP composition), and Poisson subsampling of a
# mechanism with a Gaussian dominating pair is the sampled Gaussian mechanism (Mironov, Talwar
# and Zhang 2019; Zhu, Dong and Wang 2022). The record count is computed on all records and is
# not amplified. Orders are integers (the exact sampled-Gaussian formula), and the conversion to
# (eps, delta) is the same as above.
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

SAMPLED = ('sum', 'outer', 'snapshot')       # releases computed on the sample
ORDERS = np.arange(2, 1025)


def rdp_sampled_gaussian(gamma: float, z: float, alpha: int) -> float:
    """RDP at integer order alpha of the Poisson-subsampled Gaussian mechanism (add/remove)."""
    if gamma >= 1:
        return alpha / (2 * z ** 2)
    k = np.arange(alpha + 1)
    logc = gammaln(alpha + 1) - gammaln(k + 1) - gammaln(alpha - k + 1)
    return float(logsumexp(logc + (alpha - k) * np.log1p(-gamma) + k * np.log(gamma) + (k ** 2 - k) / (2 * z ** 2))
                 / (alpha - 1))


def joint_multiplier(z: Dict[str, float]) -> float:
    return float(1 / np.sqrt(sum(1 / z[q] ** 2 for q in SAMPLED)))


def epsilon_subsampled(z: Dict[str, float], gamma: float, delta: float) -> float:
    """(eps, delta) of the selection phase with sampling rate gamma and noise multipliers z."""
    if gamma >= 1:
        return rdp_epsilon(list(z.values()), delta)
    zj, zc = joint_multiplier(z), z['count']
    rho = np.array([rdp_sampled_gaussian(gamma, zj, a) + a / (2 * zc ** 2) for a in ORDERS])
    return float(np.min(rho + np.log(1 / delta) / (ORDERS - 1)))


def calibrate_subsampled(eps: float, delta: float = 1e-5, gamma: float = 1.0,
                         shares: Dict[str, float] = None) -> Dict[str, float]:
    """Noise multipliers for a total selection budget (eps, delta) at sampling rate gamma: the
    multipliers of calibrate(eps, delta, shares), all scaled by one factor c chosen so that the
    amplified guarantee equals eps (c = 1 for gamma = 1)."""
    base = calibrate(eps, delta, shares)
    if gamma >= 1:
        return base
    c = brentq(lambda s: epsilon_subsampled({q: s * v for q, v in base.items()}, gamma, delta) - eps, 1e-3, 10.0)
    return {q: c * v for q, v in base.items()}


# ---------------------------------------------------------------------------
# Two independent samples (option gamma0 of protocol_snapshots; the paper's "Privacy of the selection phase")
# ---------------------------------------------------------------------------
# Round 0's sum and second moment are computed on a Poisson sample of rate gamma0, Round 1's snapshot on an
# independent Poisson sample of rate gamma1. Each block is a Poisson-subsampled Gaussian mechanism (the Round 0 block
# with the joint multiplier of its two releases), the count is not amplified, and the three compose adaptively in RDP.
ROUND0 = ('sum', 'outer')


def epsilon_two_samples(z: Dict[str, float], gamma0: float, gamma1: float, delta: float) -> float:
    z0 = float(1 / np.sqrt(sum(1 / z[q] ** 2 for q in ROUND0)))
    rho = np.array([rdp_sampled_gaussian(gamma0, z0, a) + rdp_sampled_gaussian(gamma1, z['snapshot'], a)
                    + a / (2 * z['count'] ** 2) for a in ORDERS])
    return float(np.min(rho + np.log(1 / delta) / (ORDERS - 1)))


def calibrate_two_samples(eps: float, delta: float = 1e-5, gamma0: float = 1.0, gamma1: float = 1.0,
                          shares: Dict[str, float] = None) -> Dict[str, float]:
    """As calibrate_subsampled, for independent Round 0 (rate gamma0) and Round 1 (rate gamma1) samples."""
    base = calibrate(eps, delta, shares)
    c = brentq(lambda s: epsilon_two_samples({q: s * v for q, v in base.items()}, gamma0, gamma1, delta) - eps,
               1e-3, 10.0)
    return {q: c * v for q, v in base.items()}
