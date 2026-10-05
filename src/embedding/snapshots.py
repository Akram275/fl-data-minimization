"""
Client sketches: the fixed-size vectors a client computes from its own encoder outputs and releases instead of
its data (Section 4 of the paper).

  Round 0  client_moments / dp_client_moments: count, sum and second-moment matrix of the unit-norm encoder
           outputs; the server fits the shared standardize -> whitened PCA space from their totals
           (FederatedSpace).
  Round 1  kernel mean sketch: mean of the paired random Fourier features of the shared-space embeddings, placed
           in the record's label block (kernel_mean_snapshot / dp_snapshot_sum), i.e. an estimate of the kernel
           mean embedding of the client's labeled data, so MMDs between clients are Euclidean distances.

The random-feature bandwidth follows analytically from the whitened dimension (analytic_rff), so no server data
is needed. src/selection/protocol.py simulates both rounds with sampled encoding.
"""
from typing import Optional

import numpy as np


class RandomFourierFeatures:
    """
    phi(z) = sqrt(2/D) cos(W z + b), W ~ N(0, 1/bandwidth^2), b ~ U[0, 2pi], so
    phi(z).phi(z') ~= exp(-||z-z'||^2 / (2 bandwidth^2)) (Gaussian kernel).
    The seed (hence W, b) is public and shared by the server and all clients.
    Build it with analytic_rff.
    """

    def __init__(self, n_features: int = 1024, bandwidth: Optional[float] = None, seed: int = 0,
                 paired: bool = False):
        self.n_features = n_features
        self.bandwidth = bandwidth
        self.seed = seed
        self.paired = paired

    @property
    def n_freq(self) -> int:
        """Paired features use D/2 frequencies, each giving a (cos, sin) pair."""
        return self.n_features // 2 if self.paired else self.n_features

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """paired=False: sqrt(2/D) cos(Wz + b), ||phi||^2 <= 2 (varies per record).
        paired=True: sqrt(2/D) [cos(Wz), sin(Wz)], same kernel, ||phi|| = 1 exactly,
        so a record moves a feature sum by at most 1 (DP sensitivity)."""
        if self.paired:
            P = Z @ self.W
            return np.sqrt(2.0 / self.n_features) * np.concatenate([np.cos(P), np.sin(P)], axis=1)
        return np.sqrt(2.0 / self.n_features) * np.cos(Z @ self.W + self.b)


def kernel_mean_snapshot(Z: np.ndarray, rff: RandomFourierFeatures,
                         labels: Optional[np.ndarray] = None, n_labels: int = 1) -> np.ndarray:
    """
    Mean feature map of the client's records. With labels, uses the product
    kernel k((z,y),(z',y')) = k(z,z') * [y == y'], whose feature map is
    phi(z) placed in label y's block - so the snapshot is the per-label SUM of
    phi divided by the client's total count (blocks keep label proportions).
    """
    phi = rff.transform(Z)
    n = max(len(Z), 1)
    if labels is None:
        return phi.sum(axis=0) / n
    out = np.zeros((n_labels, phi.shape[1]), dtype=np.float32)
    np.add.at(out, labels, phi)
    return (out / n).ravel()


# ---------------------------------------------------------------------------
# Round 0: the shared space from the clients' moments
# ---------------------------------------------------------------------------
def client_moments(Z: np.ndarray) -> dict:
    """What a client sends in Round 0 without DP: its count and the first and second moments of its encoder
    outputs (the server only needs their population totals)."""
    Z = Z.astype(np.float64)
    return {'n': len(Z), 'sum': Z.sum(0), 'sumsq': (Z ** 2).sum(0), 'outer': Z.T @ Z}


class FederatedSpace:
    """Standardize -> whitened PCA, computed from the population totals of the Round 0 moments (the same space
    as fitting it on the pooled client data, without any client record leaving its owner)."""

    def __init__(self, n_components: int = 32):
        self.n_components = n_components

    def fit_moments(self, moments) -> "FederatedSpace":
        N = sum(m['n'] for m in moments)
        s1 = sum(m['sum'] for m in moments)
        S = sum(m['outer'] for m in moments)
        s2 = sum(m['sumsq'] for m in moments) if 'sumsq' in moments[0] else np.diag(S)
        self.mu = s1 / N
        var = s2 / N - self.mu ** 2
        # DP noise can make a variance tiny or negative: floor it (post-processing).
        self.sd = np.sqrt(np.maximum(var, 1e-3 * max(np.median(var), 1e-12))) + 1e-6
        cov = (S / N - np.outer(self.mu, self.mu)) / np.outer(self.sd, self.sd)   # covariance of standardized data
        evals, evecs = np.linalg.eigh(cov)
        top = np.argsort(evals)[::-1][:min(self.n_components, len(evals))]
        self.components, self.scale = evecs[:, top].T, np.sqrt(np.maximum(evals[top], 1e-12))
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        return ((((Z - self.mu) / self.sd) @ self.components.T) / self.scale).astype(np.float32)


def analytic_rff(d: int, n_features: int = 1024, seed: int = 0, paired: bool = False) -> RandomFourierFeatures:
    """Random Fourier features with bandwidth sqrt(2d): in a whitened d-dim space two independent records satisfy
    E||h - h'||^2 = 2d, so no data is needed. W and b are drawn in this order from the seed (b is unused by
    paired features but drawn anyway, which keeps W identical for both variants)."""
    rff = RandomFourierFeatures(n_features, bandwidth=float(np.sqrt(2 * d)), seed=seed, paired=paired)
    rng = np.random.default_rng(seed)
    rff.W = rng.normal(0, 1.0 / rff.bandwidth, size=(d, rff.n_freq)).astype(np.float32)
    rff.b = rng.uniform(0, 2 * np.pi, size=rff.n_freq).astype(np.float32)
    return rff


def unit_norm(Z: np.ndarray) -> np.ndarray:
    """E(x) / ||E(x)||: bounds every record's contribution to the Round 0 moments
    (norm 1) without a data-dependent clipping threshold."""
    Z = Z.astype(np.float64)
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def dp_client_moments(Z: np.ndarray, z_sum: float, z_outer: float, rng: np.random.Generator) -> dict:
    """Round 0 release under DP (src/selection/dp_accounting.py): the sum and the second-moment matrix of the
    unit-norm encoder outputs, each with Gaussian noise of std z * Delta, Delta = 1 (one record adds x, resp.
    x x^T, of L2 / Frobenius norm 1). The symmetric matrix is released as its upper triangle (with the diagonal),
    so the noise is drawn once per entry and mirrored. The count is released once, separately (dp_count), and
    reused in Round 1."""
    X = unit_norm(Z)
    p = X.shape[1]
    G = np.triu(rng.normal(0.0, z_outer, size=(p, p)))
    return {'sum': X.sum(0) + rng.normal(0.0, z_sum, size=p), 'outer': X.T @ X + G + np.triu(G, 1).T}


def dp_count(n: int, z_count: float, rng: np.random.Generator) -> float:
    """Noisy record count (Delta = 1), floored at 1 (post-processing)."""
    return max(n + rng.normal(0.0, z_count), 1.0)


def dp_snapshot_sum(Z: np.ndarray, rff: RandomFourierFeatures, labels: np.ndarray, n_labels: int,
                    z_snap: float, rng: np.random.Generator) -> np.ndarray:
    """Round 1 release under DP: sum of psi(x, y) = phi(h(x)) in label y's block,
    paired RFF so ||psi|| = 1 = Delta, plus N(0, z^2) per coordinate. The server
    divides by the noisy count to get the snapshot (post-processing)."""
    if not rff.paired:
        raise ValueError("DP snapshots need paired RFF (bounded, constant-norm features)")
    phi = rff.transform(Z)
    out = np.zeros((n_labels, phi.shape[1]))
    np.add.at(out, labels, phi)
    return out.ravel() + rng.normal(0.0, z_snap, size=out.size)


def snapshot_sq_norm(Z: np.ndarray, rff: RandomFourierFeatures) -> float:
    """sum ||psi(x, y)||^2 = sum ||phi(h(x))||^2 over a client's records (equal to its count for paired
    features), for v = E||psi||^2 - ||mu||^2 of the IID-equivalent rule."""
    return float((rff.transform(Z) ** 2).sum())
