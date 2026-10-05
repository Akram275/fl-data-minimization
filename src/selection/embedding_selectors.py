"""
Client coreset selection (Section 4 of the paper): size-weighted k-medoids on the MMDs between the clients'
kernel mean sketches, with the drift penalty, the debiased stopping rules and their DP calibration.
"""
from typing import Dict, List, Tuple

import numpy as np


class ClientCoresetSelector:
    """
    One representative client per group of statistically redundant clients.

      - distance: Delta_ij = ||mu_i - mu_j||, the MMD between the clients' data, from their kernel mean sketches;
      - k: size-weighted k-medoids for k = 1, 2, ..., stopping at the first k whose group-weighted mixture of
        representatives sum_r (W_r / N) mu_r, with W_r the record mass of r's group, reaches the target
        (score = 1 - MMD^2(mixture, population) / delta_bar^2, or the IID-equivalent rule);
      - weights: each representative stands for its group, so FedAvg weights it by W_r (self.weights).

    k = M reproduces the population exactly (score 1), so a k always exists. k-medoids is a heuristic
    (alternating assignment and medoid update from size-weighted k-medoids++ seeds, best of n_init restarts),
    with no approximation guarantee.

    Drift penalty (drift_lambda = lambda > 0): each client pays its MMD to its representative plus lambda times
    that representative's MMD to the population,
        C_lambda(R, a) = sum_i n_i (Delta_{i,a(i)} + lambda delta_{a(i)}),
    and both k-medoids steps minimize C_lambda, which pulls the representatives towards clients that are
    individually close to the population. lambda = 0 is the plain coreset. The mixture bound
    ||mu_hat - mu|| <= sum_i n_i Delta_{i,a(i)} / N holds for any assignment, and the stopping rule is unchanged.
    """

    def __init__(self, kernel_means: Dict[str, np.ndarray], sizes: Dict[str, int], n_init: int = 10, seed: int = 0,
                 drift_lambda: float = 0.0, noise_var: float = 0.0, client_noise: Dict[str, float] = None,
                 ref_mode: str = 'mean', ucb_z: float = 0.0, client_dp_noise: Dict[str, float] = None,
                 ucb_sampling: bool = False, penalty: str = 'linear'):
        """kernel_means, sizes: the sketches mu_i and the (noisy) record counts n_i the server holds.

        noise_var (DP sketches of the all-records variant, private_selection.py): q = sigma^2 * dim, the expected squared norm
        of the Gaussian noise on each client's snapshot SUM; the snapshots are
        then noisy_sum / noisy_count and sizes the noisy counts. k-medoids runs on
        the raw noisy distances: their bias q / n_r^2 for a representative r
        penalizes noisy (small) representatives, which is wanted. The stopping
        rule (score, n_eq) uses debiased squared MMDs: E||sum_i c_i z_i||^2 =
        q sum_i c_i^2 is subtracted from every squared norm of a linear
        combination sum_i c_i mu_i of the snapshots (counts treated as exact).

        client_noise (sampled encoding): per-client q_i = E||mu_i_hat - mu_i||^2, the
        expected squared error of each client's SNAPSHOT (not its sum), e.g. the
        sampling error of a snapshot computed from a sample of the client's records.
        The bias of ||sum_i c_i mu_i||^2 is then sum_i c_i^2 q_i; noise_var = q is the
        special case q_i = q / n_i^2. Both may be given (independent noises add).

        penalty='quadratic': penalize delta_r^2 / delta_bar instead of delta_r (robustness check in the paper's appendix).

        Noise-calibrated stopping (the paper's rule is ref_mode='weighted', ucb_z=1.645 with client_dp_noise;
        the defaults give the uncalibrated rule, kept for the stopping-rule comparison in the paper's appendix):
        ref_mode='weighted': the reference discrepancy is the size-weighted mean sum_i (n_i/N) delta_i^2 (the
          discrepancy of a random record's client), debiased, instead of the plain mean over clients. Under DP
          q_i ~ 1/n_i^2, so the plain mean is dominated by the noisiest clients and cannot be resolved; the
          weighted one is dominated by large clients, whose noise is small.
        ucb_z > 0: the stopping rule uses an upper confidence bound on the coreset's squared discrepancy,
          estimate + ucb_z * SE, with SE from the known noise: for a = sum_i c_i mu_i and the debiased estimator
          ||sum_i c_i mu_hat_i||^2 - s, s = sum_i c_i^2 q_i, treating the errors as isotropic Gaussian in the
          d = dim(mu) coordinates, Var = 4 ||a||^2 s / d + 2 s^2 / d. With client_dp_noise given, s counts only
          the DP noise (isotropic Gaussian, for which the formula is exact) and the resolution floor is kept for
          the sampling error (anisotropic, the formula would underestimate it); without DP the rule is then
          exactly the original one.
        ucb_sampling=True (with ucb_z and client_dp_noise; off by default, the extended margin of the paper's appendix): the bound also covers the
          sampling error, s_smp = sum_i c_i^2 (q_i - q_dp_i). Its errors are anisotropic, so we use the worst case
          over directions: a^T Cov_i a <= ||a||^2 (1 - gamma) / (gamma n_i) because ||psi|| = 1, i.e.
          Var += 4 ||a||^2 s_smp + 2 s_smp^2 (effective dimension 1) + 4 s_dp s_smp / d."""
        self.clients = list(kernel_means.keys())
        self.mu = np.stack([kernel_means[c] for c in self.clients]).astype(np.float64)
        self.n = np.array([sizes[c] for c in self.clients], dtype=float)
        self.q = float(noise_var)
        # per-client expected squared snapshot error, all noise sources
        self.qi = self.q / self.n ** 2 + (np.array([client_noise[c] for c in self.clients], float)
                                          if client_noise is not None else 0.0)
        self.mu_U = (self.n[:, None] * self.mu).sum(0) / self.n.sum()
        N = self.n.sum()
        w = self.n / N
        # coefficients of mu_i - mu_U: e_i - w  ->  bias q_i (1 - 2 w_i) + sum_j w_j^2 q_j
        dev2 = ((self.mu - self.mu_U) ** 2).sum(1) - (self.qi * (1 - 2 * w) + float((w ** 2 * self.qi).sum()))
        # The mean single-client discrepancy cannot be resolved below the clients' mean estimation
        # error (when clients barely differ, e.g. attribute skew with sampled encoding, its unbiased
        # estimate can be ~0 or negative); q = 0 leaves it unchanged.
        self.ref_mode, self.ucb_z = ref_mode, float(ucb_z)
        self.ucb_sampling = ucb_sampling
        self.qdp = (np.array([client_dp_noise[c] for c in self.clients], float)
                    if client_dp_noise is not None else None)
        if ref_mode == 'weighted':
            self.ref = float(max((w * dev2).sum(), 1e-12))
        else:
            self.ref = float(max(dev2.mean(), self.qi.mean(), 1e-12))
        self.ref_plain_debiased = float(dev2.mean())
        self.ref_weighted_debiased = float((w * dev2).sum())
        sq = (self.mu ** 2).sum(1)
        self.D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * self.mu @ self.mu.T, 0.0))
        self.n_init = n_init
        self.seed = seed
        self.lam = drift_lambda
        self.dev = np.sqrt(((self.mu - self.mu_U) ** 2).sum(1))   # MMD(P_i, P) per client
        # drift penalty per representative: delta_r (default), or delta_r^2 / delta_bar (penalty='quadratic', the
        # quantity of the drift proposition, rescaled to MMD units so that both equal delta_bar at delta_r = delta_bar)
        self.pen = self.dev if penalty == 'linear' else self.dev ** 2 / np.sqrt(self.ref)
        self.weights: Dict[str, float] = {}
        self.diagnostics: Dict[str, float] = {}

    def _kmedoids(self, k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        M = len(self.clients)
        medoids = [int(rng.choice(M, p=self.n / self.n.sum()))]
        while len(medoids) < k:                       # k-medoids++ seeding, size-weighted
            d2 = self.D[:, medoids].min(1) ** 2 * self.n
            if d2.sum() <= 0:
                rest = [i for i in range(M) if i not in medoids]
                medoids.append(int(rng.choice(rest)))
            else:
                medoids.append(int(rng.choice(M, p=d2 / d2.sum())))
        medoids = np.array(medoids)
        for _ in range(100):
            assign = self._assign(medoids)
            new = medoids.copy()
            for c in range(k):
                members = np.where(assign == c)[0]
                if len(members):
                    cost = (self.D[np.ix_(members, members)] * self.n[members][:, None]).sum(0) \
                        + self.lam * self.n[members].sum() * self.pen[members]
                    new[c] = members[np.argmin(cost)]
            if np.array_equal(np.sort(new), np.sort(medoids)):
                break
            medoids = new
        return medoids, self._assign(medoids)

    def _cost(self, medoids) -> float:
        return float(((self.D[:, medoids] + self.lam * self.pen[medoids][None, :]).min(1) * self.n).sum())

    def _assign(self, medoids: np.ndarray) -> np.ndarray:
        return (self.D[:, medoids] + self.lam * self.pen[medoids][None, :]).argmin(1)

    def _fit_k(self, k: int):
        rng = np.random.default_rng(self.seed + k)
        best = None
        for _ in range(self.n_init):
            medoids, assign = self._kmedoids(k, rng)
            r = medoids[assign]
            cost = float(((self.D[np.arange(len(self.n)), r] + self.lam * self.pen[r]) * self.n).sum())
            if best is None or cost < best[0]:
                best = (cost, medoids, assign)
        _, medoids, assign = best
        W = np.array([self.n[assign == c].sum() for c in range(k)])
        # With lambda > 0 a medoid can lose every client (itself included) to a
        # representative that is closer to the population: it then stands for
        # nobody (W = 0), so it is dropped rather than selected with zero weight.
        keep = W > 0
        medoids, W = medoids[keep], W[keep]
        mix = (W[:, None] * self.mu[medoids]).sum(0) / W.sum()
        # coefficients of mix - mu_U on the snapshots: alpha_r on medoids, minus n_i / N on every client
        c = -self.n / self.n.sum()
        c[medoids] += W / W.sum()
        mmd2 = float(((mix - self.mu_U) ** 2).sum()) - float((c ** 2 * self.qi).sum())
        # Resolution floor: the server cannot certify a discrepancy below the coreset's own estimation
        # noise sum_r alpha_r^2 q_r (unbiased estimates of small discrepancies from few sampled or
        # noisy records can be negative). Zero noise (q = 0) leaves the rule unchanged.
        alpha = W / W.sum()
        if self.ucb_z > 0 and self.qdp is not None:
            floor = float((alpha ** 2 * self.qi[medoids]).sum())
            s = float((c ** 2 * self.qdp).sum())
            d = self.mu.shape[1]
            var = 4 * max(mmd2, 0.0) * s / d + 2 * s ** 2 / d
            if self.ucb_sampling:
                ss = float((c ** 2 * (self.qi - self.qdp)).sum())
                var += 4 * max(mmd2, 0.0) * ss + 2 * ss ** 2 + 4 * s * ss / d
            mmd2 = max(mmd2, floor) + self.ucb_z * np.sqrt(var)
        elif self.ucb_z > 0:
            s = float((c ** 2 * self.qi).sum())
            d = self.mu.shape[1]
            se = np.sqrt(4 * max(mmd2, 0.0) * s / d + 2 * s ** 2 / d)
            mmd2 = max(mmd2, 0.0) + self.ucb_z * se
        else:
            mmd2 = max(mmd2, float((alpha ** 2 * self.qi[medoids]).sum()))
        score = 1.0 - mmd2 / self.ref
        return medoids, W, score

    def fit(self, target_score: float = 0.95, max_k: int = None, n_target: float = None,
            feature_var: float = None) -> List[str]:
        """Smallest k whose coreset mixture reaches the target.

        Default stopping rule: score = 1 - MMD^2(mixture, population) / MMD^2_ref
        >= target_score, relative to the mean single-client discrepancy.

        IID-equivalent rule (n_target given): the coreset mixture must be as close
        to the population as a random sample of n_target population records,
        i.e. n_eq = v / MMD^2(mixture, population) >= n_target, where
        v = E||psi||^2 - ||mu||^2 is the trace of the feature covariance
        (feature_var, estimated from the sketches, protocol.py). For an IID sample
        of n records E||mu_hat - mu||^2 = v / n, so n_eq is the sample size whose
        sampling noise equals the coreset's discrepancy; a coreset cannot do much
        better than its own sampling noise, so the rule also demands volume.
        """
        max_k = max_k or len(self.clients)
        if n_target is not None and feature_var is None:
            raise ValueError("n_target needs feature_var (v)")
        for k in range(1, max_k + 1):
            medoids, W, score = self._fit_k(k)
            mmd2 = (1.0 - score) * self.ref
            n_eq = feature_var / max(mmd2, 1e-12) if feature_var is not None else float('nan')
            if (n_target is not None and n_eq >= n_target) or (n_target is None and score >= target_score):
                break
        selected = [self.clients[i] for i in medoids]
        self.weights = {self.clients[i]: float(w) for i, w in zip(medoids, W)}
        alpha = W / W.sum()
        # Heterogeneity of the representatives under group-mass weights
        # (sum_r alpha_r MMD(P_r, P)^2), relative to the mean single client.
        self.diagnostics = {'k': int(k), 'score': float(score), 'lambda': float(self.lam),
                            'cost': self._cost(list(medoids)),
                            'records': int(self.n[medoids].sum()),
                            'rep_heterogeneity': float((alpha * self.dev[medoids] ** 2).sum() / self.ref),
                            'n_eq': float(n_eq), 'n_eff': float(1.0 / (alpha ** 2 / self.n[medoids]).sum())}
        print(f"  [coreset] lambda={self.lam:g}: k={k} medoids, weighted-mixture score={score:.4f}, "
              f"{int(self.n[medoids].sum()):,} records (clusters stand for {int(W.sum()):,}), "
              f"rep. heterogeneity={self.diagnostics['rep_heterogeneity']:.3f}")
        return selected
