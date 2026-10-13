"""
Figure of the client sketches (paper, Section 4): CelebA under label skew (seed-42 partition, CLIP), the clients'
kernel mean sketches on their first two principal components, with the plain coreset (lambda = 0), the drift-aware
coreset (lambda = 1) and lambda = 2. The sketches are those of the protocol (run_protocol.py: gamma = 0.5,
D = 128, no DP); the lambda = 0 and 1 selections are read from run_protocol.py's results, lambda = 2 (not part of
the runs) is computed with the same rule.

  .venv/bin/python paper/make_figures.py
Writes paper/figures/fig_snapshots_pca_celeba.{pdf,png}.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "figures"
RESULTS = ROOT / "results"

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.data.partitions import create_dirichlet_partitions, heterogeneity_dir, partition_key  # noqa: E402
from src.selection.embedding_selectors import ClientCoresetSelector  # noqa: E402
from src.selection.protocol import protocol_snapshots  # noqa: E402

# Style: blue / orange poles, neutral gray midpoint, recessive axes.
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#9c9b95', '#e4e3df'
BLUE, ORANGE, NEUTRAL = '#2a78d6', '#eb6834', '#d9d8d2'
DIVERGING = LinearSegmentedColormap.from_list('pos_share', [BLUE, NEUTRAL, ORANGE])
plt.rcParams.update({'font.size': 7, 'axes.edgecolor': INK2, 'axes.linewidth': 0.6, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'legend.frameon': False, 'pdf.fonttype': 42})
N_LABELS = 2
GAMMA, D = 0.5, 128


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_celeba():
    """The seed-42 label-skew partition of CelebA (100 clients, alpha = 0.2) and its CLIP encodings."""
    from src.data.image_metadata import DATASET_LOADERS
    from src.embedding.encoders import encode_cached
    train, _ = DATASET_LOADERS['celeba'](max_samples=0, n_public=2000)
    np.random.seed(42)
    parts = create_dirichlet_partitions(partition_key(train, 'celeba', 'label'), 100, 0.2, max_attempts=2000000)
    names = [f"client_{i:02d}" for i in range(len(parts))]
    enc = 'clip-b32'
    root = Path("conv_results_embedding") / "celeba" / heterogeneity_dir(0.2, 'label') / "seed_42"
    return dict(names=names, ci={c: np.asarray(p, dtype=int) for c, p in zip(names, parts)}, y=train.labels,
                Z=encode_cached(train.paths, enc, tag="celeba_train"), enc=enc, root=root)


def sketches(S):
    """The protocol's sketches (run_protocol.py, no DP, sample seed 0) and the clients' sizes and label mix."""
    P = protocol_snapshots(S['Z'], S['ci'], S['names'], S['y'], N_LABELS, GAMMA, float('inf'), 1e-5, sample_seed=0,
                           noise_seed=0, rff_features=D)
    mu = np.stack([P['snaps'][c] for c in S['names']])
    n = np.array([len(S['ci'][c]) for c in S['names']], float)
    pos = np.array([S['y'][S['ci'][c]].mean() for c in S['names']])
    return dict(P=P, mu=mu, n=n, pos=pos, pop_pos=float(S['y'].mean()))


def saved_selection(S, method):
    """A selection of run_protocol.py (no DP, gamma = 0.5, D = 128): representative indices and group weights."""
    f = RESULTS / S['root'] / S['enc'] / f"g{GAMMA:g}_einf_D{D}" / method / "selection.json"
    j = json.load(open(f))
    return np.array([S['names'].index(c) for c in j['selection']]), np.array([j['weights'][c] for c in j['selection']])


def live_selection(S, snap, lam, tau):
    """Drift-aware coreset computed as run_protocol.py does (calibrated rule, seed 42), for configurations that
    were not part of the runs."""
    P = snap['P']
    sel = ClientCoresetSelector(P['snaps'], P['sizes'], seed=42, drift_lambda=lam, client_noise=P['client_noise'],
                                ref_mode='weighted', ucb_z=1.645, client_dp_noise=P['client_dp_noise'])
    sub = sel.fit(tau)
    return np.array([S['names'].index(c) for c in sub]), np.array([sel.weights[c] for c in sub])


def assignment(mu, n, reps, lam):
    """a(i) = argmin_r Delta_ir + lam * delta_r (the selector's assignment step)."""
    mu_U = (n[:, None] * mu).sum(0) / n.sum()
    Dm = np.linalg.norm(mu[:, None, :] - mu[None, reps, :], axis=2)
    dev = np.linalg.norm(mu[reps] - mu_U, axis=1)
    return reps[(Dm + lam * dev[None, :]).argmin(1)]


def pca2(mu, n):
    """2-D PCA of the sketches around the population kernel mean (size-weighted)."""
    mu_U = (n[:, None] * mu).sum(0) / n.sum()
    X = mu - mu_U
    w = n / n.sum()
    C = (X * w[:, None]).T @ X
    evals, evecs = np.linalg.eigh(C)
    V = evecs[:, ::-1][:, :2]
    return lambda M: (M - mu_U) @ V


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def style(ax):
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2, labelsize=6)


def sizes(n, smin=6, smax=90):
    r = np.sqrt(n / n.max())
    return smin + (smax - smin) * r


def client_map(ax, P, snap, reps, weights, lam, title):
    """Clients as dots (area ~ records, color = share of positive labels, diverging at the population rate), the
    population kernel mean (star), links from each client to its representative, representatives ringed, and the
    coreset mixture (open star)."""
    norm = TwoSlopeNorm(vmin=0, vcenter=snap['pop_pos'], vmax=1)
    xy = P(snap['mu'])
    a = assignment(snap['mu'], snap['n'], reps, lam)
    for i, r in enumerate(a):
        if i != r:
            ax.plot([xy[i, 0], xy[r, 0]], [xy[i, 1], xy[r, 1]], color=MUTED, lw=0.4, alpha=0.6, zorder=1)
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=sizes(snap['n']), c=snap['pos'], cmap=DIVERGING, norm=norm,
                    edgecolors='white', linewidths=0.5, zorder=2)
    pop = P(((snap['n'][:, None] * snap['mu']).sum(0) / snap['n'].sum())[None])[0]
    ax.scatter(*pop, marker='*', s=110, color=INK, edgecolors='white', linewidths=0.6, zorder=5)
    ax.scatter(xy[reps, 0], xy[reps, 1], s=sizes(snap['n'][reps]) + 40, marker='s', facecolors='none',
               edgecolors=INK, linewidths=1.1, zorder=4)
    mix = P(((weights[:, None] * snap['mu'][reps]).sum(0) / weights.sum())[None])[0]
    ax.scatter(*mix, marker='*', s=110, facecolors='white', edgecolors=INK, linewidths=0.9, zorder=6)
    ax.set_title(title, fontsize=7, color=INK, loc='left')
    return sc


def legend_items():
    return [Line2D([], [], marker='o', ls='', color=NEUTRAL, mec=MUTED, mew=0.5, ms=6, label='client (area ~ records)'),
            Line2D([], [], marker='*', ls='', color=INK, ms=8, label='population'),
            Line2D([], [], marker='s', ls='', mfc='none', mec=INK, mew=1.1, ms=7, label='representative'),
            Line2D([], [], marker='*', ls='', mfc='white', mec=INK, mew=0.9, ms=8, label='coreset mixture')]


def fig_sketch_map():
    """Single-column figure: three panels (lambda = 0, 1, 2), no tick values, legend below."""
    S = load_celeba()
    snap = sketches(S)
    P = pca2(snap['mu'], snap['n'])
    FS, TS = 7.5, 6.5
    panels = [(0.0, 'plain, $\\lambda=0$', lambda: saved_selection(S, 'coreset_plain')),
              (1.0, '$\\lambda=1$', lambda: saved_selection(S, 'coreset_dp')),
              (2.0, '$\\lambda=2$ ($\\tau=0.98$)', lambda: live_selection(S, snap, 2.0, 0.98))]
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.55), sharex=True, sharey=True)
    for ax, (lam, title, get) in zip(axes, panels):
        reps, w = get()
        sc = client_map(ax, P, snap, reps, w, lam, title=title)
        style(ax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.title.set_fontsize(FS)
        for c in ax.collections:          # smaller markers at single-column size
            if c.get_sizes() is not None and len(c.get_sizes()):
                c.set_sizes(np.asarray(c.get_sizes()) * 0.45)
    axes[0].set_ylabel('PC 2', fontsize=FS)
    axes[1].set_xlabel('PC 1', fontsize=FS)
    fig.subplots_adjust(left=0.06, right=0.83, top=0.86, bottom=0.12, wspace=0.06)
    cax = fig.add_axes([0.85, 0.12, 0.022, 0.74])
    cb = fig.colorbar(sc, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=TS, length=2, color=INK2)
    cb.ax.axhline(snap['pop_pos'], color=INK, lw=0.6)
    cb.set_label('share positive', color=INK2, fontsize=FS)
    h = legend_items()
    h.insert(1, Line2D([], [], color=MUTED, lw=1.0, label='assignment'))
    for x in h:
        x.set_markersize(x.get_markersize() * 0.75)
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.45, 0.02), ncol=3, fontsize=TS,
               handletextpad=0.3, columnspacing=0.9, labelspacing=0.3)
    OUT.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f"fig_snapshots_pca_celeba.{ext}", bbox_inches='tight', dpi=200 if ext == 'png' else None)
    plt.close(fig)
    print(f">> {OUT / 'fig_snapshots_pca_celeba'}.pdf")


if __name__ == '__main__':
    fig_sketch_map()
