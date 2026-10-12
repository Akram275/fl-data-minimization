"""
FedBary: Li, Fu, Zhang and Pang, "Data Valuation and Detections in Federated Learning" (arXiv:2311.05304),
the setting without a server validation set: the server approximates the Wasserstein
barycenter Q of the clients' data distributions, federatedly, and values client i by its distance
W(P_i, Q) (smaller = more valuable). Used as a selector: the K clients closest to the barycenter, then
FedAvg.

Official code (github.com/muz1lee/MOTdata) releases only the pairwise client/server distance (FedOT,
with a validation set); the barycenter loop of the paper's Algorithm 1 is not released. We implement
Algorithm 1 with the released building blocks, copied from cifar10_fed.ipynb:
  - interp(X, Y, t): exact OT plan G0 between X and Y (ot.emd on the max-normalized cost) and the
    approximate interpolating measure Z = (1 - t) X + t (G0 * |X|) Y, cost = sum G0 * M  (InterpMeas,
    approx_interp=True); metric 'euclidean' (p = 1), t = 0.5, random N(0, 1) initial supports, as in
    the released example (cal_distance);
  - per iteration and client i (FedOT.fit's update):  G = interp(gamma_i, P_i)  [client],
    H = interp(gamma_i, Q)  [server],  gamma_i <- interp(H, G)  [server/client];
  - final distance W(P_i, gamma_i) + W(gamma_i, Q)  (FedOT's cost = cost_g + cost_h).
What we run (mode='exact', the default): Theorem 1 of the paper states that Algorithm 1 converges to
sum_i W(P_i, Q) with Q the Wasserstein barycenter of the clients, so we compute that limit directly, in
simulation: Q = free-support barycenter of the clients' point clouds (uniform lambda_i), and the exact
distances W(P_i, Q). This is the best FedBary's federated approximation can do, i.e. it favours the
baseline. Our port of the federated loop (mode='federated', below) did not converge in 10 iterations on
our data (Spearman -0.15 with the exact distances on Sent140, seed 42), so it is kept for reference only.
Choices the paper leaves open, ours:
  - barycenter update (Alg. 1 line 7, "transport samples from gamma_i to the common distribution"):
    a free-support Wasserstein barycenter of the gamma_i with uniform weights (lambda_i = 1, as in the
    paper), ot.lp.free_support_barycenter warm-started at the previous Q;
  - support size S = 100 for Q and every gamma_i, K_iter = 10 iterations (the paper reports convergence
    within 10 iterations and little sensitivity to S);
  - records: each client's Poisson-sampled records in the protocol's shared whitened space h(x), each
    augmented with its class mean and per-coordinate class std on that client, x~ = [h; m_y; s_y]
    (Eq. 8, the otdd augmented representation with diagonal covariances, as in the released code).
"""
import os

for _b in ("TENSORFLOW", "JAX", "CUPY"):      # POT probes these backends on import; TF needs AVX
    os.environ.setdefault(f"POT_BACKEND_DISABLE_{_b}", "1")

from typing import Dict, List

import numpy as np
import ot


def augmented(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x~ = [h; m_y; s_y] with the client's own per-class mean and std."""
    out = np.zeros((len(H), 3 * H.shape[1]))
    out[:, :H.shape[1]] = H
    for c in np.unique(y):
        m = y == c
        out[m, H.shape[1]:2 * H.shape[1]] = H[m].mean(0)
        out[m, 2 * H.shape[1]:] = H[m].std(0) if m.sum() > 1 else 0.0
    return out


def _interp(X, Y, t=0.5, metric='euclidean'):
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)
    M = ot.dist(X, Y, metric=metric)
    norm = M.max() if M.max() > 1 else 1
    G0 = ot.emd(a, b, M / norm)
    return (1 - t) * X + t * (G0 * len(X)) @ Y, float((G0 * M).sum())


def _w(X, Y, metric='euclidean'):
    return float(ot.emd2(np.ones(len(X)) / len(X), np.ones(len(Y)) / len(Y), ot.dist(X, Y, metric=metric)))


def fedbary_distances(clouds: List[np.ndarray], n_supp: int = 100, n_iter: int = 10, t: float = 0.5,
                      seed: int = 0, bary_iter: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = next(P.shape[1] for P in clouds if len(P))
    Q = rng.standard_normal((n_supp, dim))
    gam = [rng.standard_normal((n_supp, dim)) for _ in clouds]
    w = np.ones(n_supp) / n_supp
    for _ in range(n_iter):
        for i, P in enumerate(clouds):
            if len(P) == 0:
                continue
            G, _ = _interp(gam[i], P, t)        # client i
            H, _ = _interp(gam[i], Q, t)        # server
            gam[i], _ = _interp(H, G, t)
        live = [g for g, P in zip(gam, clouds) if len(P)]
        Q = ot.lp.free_support_barycenter(live, [w] * len(live), Q, numItermax=bary_iter)
    return np.array([_w(P, g) + _w(g, Q) if len(P) else np.inf for P, g in zip(clouds, gam)])


def exact_distances(clouds: List[np.ndarray], n_supp: int = 100, seed: int = 0, bary_iter: int = 50) -> np.ndarray:
    """Limit of Algorithm 1: the free-support Wasserstein barycenter of the clients (support initialized
    at n_supp pooled points), then the exact distance of every client to it (euclidean cost, as above)."""
    rng = np.random.default_rng(seed)
    live = [P for P in clouds if len(P)]
    pool = np.concatenate(live)
    X0 = pool[rng.choice(len(pool), min(n_supp, len(pool)), replace=False)]
    Q = ot.lp.free_support_barycenter(live, [np.ones(len(P)) / len(P) for P in live], X0, numItermax=bary_iter)
    return np.array([_w(P, Q) if len(P) else np.inf for P in clouds])


def fedbary_select(H_by_client: Dict[str, np.ndarray], y_by_client: Dict[str, np.ndarray], names: List[str],
                   k: int, seed: int = 0, mode: str = 'exact', max_points: int = 2000) -> (List[str], np.ndarray):
    """max_points: each client's cloud is a uniform random subset of at most max_points of its (sampled,
    augmented) records, to keep the exact barycenter tractable (with gamma = 0.5 the largest clients hold
    tens of thousands of sampled records); smaller clients keep all their points."""
    rng = np.random.default_rng(seed)
    clouds = []
    for c in names:
        if not len(H_by_client[c]):
            clouds.append(np.zeros((0, 1)))
            continue
        A = augmented(H_by_client[c], y_by_client[c])
        if len(A) > max_points:
            A = A[rng.choice(len(A), max_points, replace=False)]
        clouds.append(A)
    d = exact_distances(clouds, seed=seed) if mode == 'exact' else fedbary_distances(clouds, seed=seed)
    order = np.argsort(d, kind='stable')
    return [names[i] for i in order[:k]], d
