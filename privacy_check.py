"""
Independent check of the selection-phase privacy accounting (the accountant table of the paper's privacy appendix).

For the noise multipliers our protocol uses (calibrate_two_samples, 70/10/10/10 shares), the
(eps, delta) guarantee is recomputed with Google's dp_accounting library (v0.6), which shares no code with
src/selection/dp_accounting.py:
  - RDP accountant (fractional orders, its own subsampled-Gaussian bound and RDP -> DP conversion);
  - PLD accountant (privacy loss distributions; numerically tight, the reference value).
Mechanism, add/remove one record of one client, with independent Round 0 (rate gamma0) and Round 1 (rate gamma1)
samples (the paper's design; gamma0 = 0.2, gamma1 = 0.5):
  PoissonSampled(gamma0, Gaussian(z_0))  o  PoissonSampled(gamma1, Gaussian(z_psi))  o  Gaussian(z_count),
  1/z_0^2 = 1/z_sum^2 + 1/z_outer^2.
Checks also: (i) the Gaussian-composition identity behind z_joint (three Gaussians = one Gaussian, PLD at
gamma = 1); (ii) the per-record L2 sensitivities used by the protocol (unit-norm sum, upper triangle of x x^T,
paired-RFF joint feature psi), on random and adversarial inputs; (iii) the private DDEM baseline's noise.
Output: printed, and written to checks/privacy_check.txt
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
import numpy as np
import dp_accounting as dpa
from dp_accounting import pld, rdp

from src.embedding.snapshots import analytic_rff, unit_norm
from src.selection.dp_accounting import (ROUND0, calibrate_two_samples, epsilon_two_samples, sigma_closed_form)
from src.selection.private_selection import budget_shares

DELTA = 1e-5
ORDERS = np.concatenate([np.linspace(1.01, 10, 500), np.arange(11, 2049)])


GAMMA0 = 0.2


def z0(z):
    return float(1 / np.sqrt(sum(1 / z[q] ** 2 for q in ROUND0)))


def sampled(gamma, zz):
    return dpa.GaussianDpEvent(zz) if gamma >= 1 else dpa.PoissonSampledDpEvent(gamma, dpa.GaussianDpEvent(zz))


def event(z, gamma0, gamma1):
    return dpa.ComposedDpEvent([sampled(gamma0, z0(z)), sampled(gamma1, z['snapshot']), dpa.GaussianDpEvent(z['count'])])


def eps_rdp(ev):
    return rdp.RdpAccountant(ORDERS).compose(ev).get_epsilon(DELTA)


def eps_pld(ev, interval=1e-4):
    return pld.PLDAccountant(value_discretization_interval=interval).compose(ev).get_epsilon(DELTA)


def main(out):
    p = lambda *a: print(*a, file=out, flush=True)
    p(f"== (A) selection phase: eps for the multipliers the protocol uses (delta = 1e-5, gamma0 = {GAMMA0})")
    p(f"{'gamma1':>6} {'eps':>4} | {'z_cnt':>6} {'z_sum':>6} {'z_out':>6} {'z_snap':>6} {'z_0':>6} | "
      f"{'ours(RDP)':>9} {'dpa RDP':>8} {'dpa PLD':>8} {'PLD/ours':>8}")
    worst = 0.0
    for gamma in (0.5, 0.1, 0.2, 1.0):
        for eps in (1.0, 2.0, 4.0):
            z = calibrate_two_samples(eps, DELTA, GAMMA0, gamma, budget_shares())
            ev = event(z, GAMMA0, gamma)
            e_ours, e_rdp, e_pld = epsilon_two_samples(z, GAMMA0, gamma, DELTA), eps_rdp(ev), eps_pld(ev)
            worst = max(worst, e_pld / eps, e_rdp / eps)
            p(f"{gamma:6.1f} {eps:4.0f} | {z['count']:6.2f} {z['sum']:6.2f} {z['outer']:6.2f} {z['snapshot']:6.2f} "
              f"{z0(z):6.2f} | {e_ours:9.3f} {e_rdp:8.3f} {e_pld:8.3f} {e_pld / e_ours:8.2f}")
    p(f"max over the grid of (independent eps / claimed eps): {worst:.3f}  (<= 1: claim is conservative)")

    p("\n== (B) Gaussian composition identity behind z_0 (no sampling, PLD, eps = 1 calibration)")
    z = calibrate_two_samples(1.0, DELTA, 1.0, 1.0, budget_shares())
    two = dpa.ComposedDpEvent([dpa.GaussianDpEvent(z[q]) for q in ROUND0])
    p(f"two Gaussians {[round(z[q], 3) for q in ROUND0]}: eps = {eps_pld(two):.4f};"
      f" one Gaussian z_0 = {z0(z):.3f}: eps = {eps_pld(dpa.GaussianDpEvent(z0(z))):.4f}")

    p("\n== (E) amplification table: eps with the no-sampling multipliers, and multipliers calibrated to eps = 1")
    zb = calibrate_two_samples(1.0, DELTA, 1.0, 1.0, budget_shares())
    for g0, g1 in ((1.0, 1.0), (GAMMA0, 1.0), (GAMMA0, 0.5), (GAMMA0, 0.2), (GAMMA0, 0.1), (GAMMA0, 0.05)):
        zc = calibrate_two_samples(1.0, DELTA, g0, g1, budget_shares())
        p(f"gamma0 = {g0:4.2f} gamma1 = {g1:4.2f}: same-noise eps = {epsilon_two_samples(zb, g0, g1, DELTA):.2f}; "
          f"z_cnt = {zc['count']:.2f} z_0 = {z0(zc):.2f} z_psi = {zc['snapshot']:.2f} z_psi/gamma1 = {zc['snapshot'] / g1:.1f}")

    p("\n== (C) per-record L2 sensitivities (add/remove one record; must be <= 1)")
    rng = np.random.default_rng(0)
    d, D = 512, 128
    X = unit_norm(np.vstack([rng.standard_normal((2000, d)), rng.standard_t(1, (2000, d)),
                             np.eye(d)[:200] * 1e6, np.ones((1, d))]))
    p(f"sum query  ||x||        max = {np.linalg.norm(X, axis=1).max():.6f}")
    tri = np.array([np.linalg.norm(np.triu(np.outer(x, x))) for x in X[::10]])
    p(f"outer query ||triu(xx^T)||_F max = {tri.max():.6f}")
    for dim in (32,):
        rff = analytic_rff(dim, D, seed=42, paired=True)
        H = np.vstack([rng.standard_normal((5000, dim)) * s for s in (0.01, 1, 100, 1e4)])
        phi = rff.transform(H)
        p(f"snapshot ||psi|| = ||phi(h) (x) e_y|| over {len(H)} points, h scales 1e-2..1e4: "
          f"min = {np.linalg.norm(phi, axis=1).min():.6f}, max = {np.linalg.norm(phi, axis=1).max():.6f}")

    p("\n== (D) private DDEM baseline: one Gaussian release of the label-count vector (Delta = 1)")
    for eps in (1.0, 2.0, 4.0):
        zd = sigma_closed_form(eps, DELTA, 1)
        p(f"eps = {eps:.0f}: z = {zd:.3f}, PLD eps = {eps_pld(dpa.GaussianDpEvent(zd)):.3f}")


if __name__ == '__main__':
    Path('checks').mkdir(exist_ok=True)
    with open('checks/privacy_check.txt', 'w') as f:
        class Tee:
            def write(self, s): f.write(s); sys.stdout.write(s)
            def flush(self): f.flush(); sys.stdout.flush()
        main(Tee())
