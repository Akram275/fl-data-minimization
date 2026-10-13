"""
Tables and number macros of the paper from results/ (run_protocol.py, full_reference.py) and checks/
(stopping_check.py, central_check.py), printed as LaTeX blocks (%%%%%%%% <name> %%%%%%%%); writes the figures
paper/figures/pareto_label.pdf and paper/figures/fig_private.pdf.

  .venv/bin/python paper/make_tables.py > tables.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper.collect import RESULTS, collect, full_reference  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
CHECKS = Path(__file__).resolve().parent.parent / "checks"
G = 0.5          # one configuration for every run: gamma = 0.5, D = 128, calibrated stopping rule
DMAIN = 128
METHODS = ['ddem', 'fedbary', 'divfl', 'coreset_plain', 'coreset_dp', 'coreset_neq']
MTEX = {'ddem': r'\DDEM{}', 'fedbary': r'\FedBary{}', 'divfl': r'\DivFL{}',
        'coreset_plain': r'\Coreset{} ($\lambda{=}0$)', 'coreset_dp': r'\Coreset{} ($\tau{=}0.99,\lambda{=}1$)',
        'coreset_neq': r'\Coreset{} ($n_{\mathrm{eq}},\lambda{=}1$)'}
DTEX = {'celeba': 'CelebA', 'fairface': 'FairFace', 'ACSIncome': 'ACSIncome', 'Sent140': 'Sent140'}


def pm(x, nd=1):
    x = pd.Series(x).dropna()
    return f"{x.mean():.{nd}f}" if len(x) else "---"


def sg(v, nd=1):
    if v is None or not np.isfinite(v):
        return "---"
    s = f"{v:+.{nd}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def wins(g, col='d_fl'):
    t = g.dropna(subset=[col])
    return f"{int((t[col] > 0).sum())}/{len(t)}"


def dcell(t, col):
    return "---" if t[col].notna().sum() == 0 else f"{sg(t[col].mean())} ({wins(t, col)})"


def write(name, text):
    print(f"%%%%%%%% {name} %%%%%%%%\n{text}")


DF = collect(RESULTS)
DF['D'] = DF['D'].fillna(DMAIN)            # DivFL rows carry no D (selection-independent of it)
FULL = full_reference()


def sel(dataset, regime, gamma=G, eps=np.inf, methods=None, enc=None, D=DMAIN):
    d = DF[(DF.dataset == dataset) & (DF.regime == regime)]
    d = d[(d.gamma == gamma) & (d.eps == eps) & (d.D == D)]
    if methods is not None:
        d = d[d.method.isin(methods)]
    if enc is not None:
        d = d[d.enc == enc]
    return d


def step_matched():
    """Centralized control trained for as many gradient steps as 20 epochs on all records (central_check.py,
    label skew, images): one row per dataset, partition, encoder and method."""
    rows = []
    for f in (CHECKS / "central").glob("*labelskew_seed_??*.csv"):
        if f.name.endswith('_labelmatch.csv'):
            continue
        d = pd.read_csv(f)
        d = d[(d.check == 'steps') & (d['mode'] == 'step-matched')]
        if 'method' not in d:
            d['method'] = 'coreset_dp'
        d['dataset'] = 'celeba' if 'celeba' in f.name else 'fairface'
        rows.append(d[['dataset', 'method', 'enc', 'bacc']])
    return pd.concat(rows) if rows else pd.DataFrame(columns=['dataset', 'method', 'enc', 'bacc'])


STEPS = step_matched()


def method_row(g, label, shade=False, enc=None, steps=None):
    t = g[g.fl.notna()]
    skipped = int((~g.trained).sum())
    sup = f"$^{{{skipped}}}$" if skipped else ""
    cells = (f"{g.k.mean():.1f} & {g.records_frac.mean():.1f} & {pm(t.fl)} & {dcell(t, 'd_fl')}{sup} & "
             f"{pm(t.cen)} & {dcell(t, 'd_cen')} & {sg((t.fl - t.cen).mean())}")
    if steps is not None:
        cells += f" & {pm(steps)}"
    row = f" & {label}{'' if enc is None else ' & ' + enc} & {cells} \\\\"
    return (r"\rowcolor{black!6}" + row) if shade else row


def full_row(dataset, regime, k_text, extra_col=False, steps=False):
    f = FULL[(FULL.dataset == dataset) & (FULL.regime == regime)]
    if f.empty:
        return None
    mid = " & ---" if extra_col else ""
    tail = f" & {pm(f.cen)}" if steps else ""          # 20 epochs on all records define the step budget
    return (f" & \\textit{{all clients}}{mid} & {k_text} & 100 & {pm(f.fl)} & --- & {pm(f.cen)} & --- & "
            f"{sg((f.fl - f.cen).mean())}{tail} \\\\")


HEAD = (r"Method & $k$ & \% rec. & FL acc. & $\Delta_{\mathrm{FL}}$ & centr. & $\Delta_{\mathrm{cen}}$ & FL$-$centr. \\")


def table_images(regime, steps=False):
    """steps: add the step-matched centralized control (centr.$^\\star$) as a last column."""
    head = HEAD.replace(r" \\", r" & centr.$^\star$ \\") if steps else HEAD
    lines = [r"\begin{tabular}{llrrrrrrr" + ("r" if steps else "") + "}", r"\toprule", " & " + head, r"\midrule"]
    for di, ds in enumerate(['celeba', 'fairface']):
        d = sel(ds, regime)
        first = True
        for m in METHODS:
            g = d[d.method == m]
            if g.empty:
                continue
            st = STEPS[(STEPS.dataset == ds) & (STEPS.method == m)].bacc if steps else None
            if steps and m == 'divfl':
                st = pd.Series(dtype=float)
            row = method_row(g, MTEX[m], shade=(m == 'coreset_dp'), steps=st)
            lines.append(row.replace(" & ", f"{DTEX[ds] if first else ''} & ", 1) if not row.startswith(r"\rowcolor")
                         else row.replace(r"\rowcolor{black!6} & ", r"\rowcolor{black!6}" + (DTEX[ds] if first else '') + " & ", 1))
            first = False
        fr = full_row(ds, regime, '100' if (ds == 'celeba' or regime == 'attr') else '50', steps=steps)
        if fr:
            lines.append(fr)
        lines.append(r"\midrule" if di == 0 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def table_tabtext(blocks):
    """blocks: [(dataset, regime, label, encoders, k_all)]; one row per method x encoder."""
    lines = [r"\begin{tabular}{lllrrrrrrr}", r"\toprule", " & Method & Enc. & " + HEAD.split(" & ", 1)[1], r"\midrule"]
    etex = {'raw': 'raw', 'minilm': 'MiniLM', 'mpnet': 'MPNet', '-': '---'}
    for bi, (ds, regime, label, encs, kall) in enumerate(blocks):
        d = sel(ds, regime)
        first = True
        for m in METHODS:
            for e in encs:
                g = d[(d.method == m) & (d.enc == e)]
                if g.empty:
                    continue
                t = g[g.fl.notna()]
                sk = int((~g.trained).sum())
                sup = f"$^{{{sk}}}$" if sk else ""
                row = (f"{label if first else ''} & {MTEX[m] if e in ('-', encs[0]) else ''} & {etex[e]} & "
                       f"{g.k.mean():.1f} & {g.records_frac.mean():.1f} & {pm(t.fl)} & {dcell(t, 'd_fl')}{sup} & "
                       f"{pm(t.cen)} & {dcell(t, 'd_cen')} & {sg((t.fl - t.cen).mean())} \\\\")
                lines.append((r"\rowcolor{black!6}" if m == 'coreset_dp' else "") + row)
                first = False
        if first:
            lines.append(f"{label} & \\multicolumn{{9}}{{c}}{{\\pending{{runs in progress}}}} \\\\")
        fr = full_row(ds, regime, kall, extra_col=True)
        if fr:
            lines.append(fr)
        lines.append(r"\midrule" if bi < len(blocks) - 1 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def table_grid():
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & $\tau$ & $\lambda$ & $k$ & \% rec. & het. & FL acc. & $\Delta_{\mathrm{FL}}$ & FL$-$centr. \\", r"\midrule"]
    for di, ds in enumerate(['celeba', 'fairface']):
        d = DF[(DF.dataset == ds) & (DF.regime == 'label') & (DF.gamma == G) & (DF.eps == np.inf) & (DF.D == DMAIN) &
               DF.method.str.startswith('coreset_t') & (DF.enc == 'clip-b32')].copy()
        # the shaded configuration (tau = 0.99, lambda = 1) is the main drift-aware coreset: use its runs (coreset_dp,
        # same selections) so that this table matches every other table
        main = DF[(DF.dataset == ds) & (DF.regime == 'label') & (DF.gamma == G) & (DF.eps == np.inf) &
                  (DF.D == DMAIN) & (DF.method == 'coreset_dp') & (DF.enc == 'clip-b32')]
        d = pd.concat([d[d.method != 'coreset_t0.99_l1'], main.assign(method='coreset_t0.99_l1')])
        d['tau'] = d.method.str.extract(r"_t([\d.]+)_")[0].astype(float)
        d['lam'] = d.method.str.extract(r"_l([\d.]+)$")[0].astype(float)
        first = True
        for (tau, lam), g in d.groupby(['tau', 'lam']):
            t = g[g.fl.notna()]
            sk = int((~g.trained).sum())
            sup = f"$^{{{sk}}}$" if sk else ""
            row = (f"{DTEX[ds] if first else ''} & {tau:g} & {lam:g} & {g.k.mean():.1f} & {g.records_frac.mean():.1f} & "
                   f"{g.het.mean():.2f} & {pm(t.fl)} & {sg(t.d_fl.mean())} ({wins(t)}){sup} & {sg((t.fl - t.cen).mean())} \\\\")
            lines.append((r"\rowcolor{black!6}" if (tau, lam) == (0.99, 1.0) else "") + row)
            first = False
        lines.append(r"\midrule" if di == 0 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


SETTINGS = [('celeba', 'label', 'CelebA'), ('fairface', 'label', 'FairFace'), ('ACSIncome', 'label', 'ACSIncome'),
            ('Sent140', 'label', 'Sent140')]


def table_stop():
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{score $\ge\tau=0.99$} & \multicolumn{4}{c}{$n_{\mathrm{eq}}\ge0.5\%\,N$} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Setting & $k$ & \% rec. & FL acc. & $\Delta_{\mathrm{FL}}$ & $k$ & \% rec. & FL acc. & $\Delta_{\mathrm{FL}}$ \\", r"\midrule"]
    for ds, regime, name in SETTINGS:
        cells = []
        for m in ('coreset_dp', 'coreset_neq'):
            g = sel(ds, regime, methods=[m])
            t = g[g.fl.notna()]
            sk = int((~g.trained).sum())
            sup = f"$^{{{sk}}}$" if sk else ""
            cells.append(f"{g.k.mean():.1f} & {g.records_frac.mean():.1f} & {pm(t.fl)} & {sg(t.d_fl.mean())} ({wins(t)}){sup}")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_private():
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{4}{c}{drift-aware ($\tau{=}0.99$)} & $n_{\mathrm{eq}}$ & \multicolumn{2}{c}{\DDEM{}} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){8-9}",
             r"Setting & $\varepsilon_1$ & \% rec. & FL acc. & $\Delta_{\mathrm{FL}}$ & wins & \% rec. & \% rec. & FL acc. \\",
             r"\midrule"]
    for si, (ds, regime, name) in enumerate(SETTINGS + [('ACSIncome', 'states', 'ACS states')]):
        for i, eps in enumerate([np.inf, 4.0, 2.0, 1.0]):
            g = sel(ds, regime, eps=eps, methods=['coreset_dp'])
            h = sel(ds, regime, eps=eps, methods=['coreset_neq'])
            if g.empty:
                continue
            t = g[g.fl.notna()]
            neq = (f"{h.records_frac.mean():.0f}" + (f"$^{{{int((~h.trained).sum())}}}$" if (~h.trained).any() else "")
                   if len(h) else "---")
            q = sel(ds, regime, eps=eps, methods=['ddem'])
            dd = (f"{q.records_frac.mean():.1f} & {pm(q[q.fl.notna()].fl)}" if len(q) else "--- & ---")
            lines.append(f"{name if i == 0 else ''} & {'none' if eps == np.inf else f'{eps:g}'} & {g.records_frac.mean():.1f} & "
                         f"{pm(t.fl)} & {sg(t.d_fl.mean())} & {wins(t)} & {neq} & {dd} \\\\")
        lines.append(r"\midrule" if si < 4 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def table_drift():
    """FedAvg, FedProx and SCAFFOLD on the drift-aware coreset's matched random federations, on all clients and on
    the coreset itself (label skew); one column per dataset."""
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             "Federation & Training & " + " & ".join(n for _, _, n in SETTINGS) + r" \\", r"\midrule"]
    col = lambda d, c: pm(d[c]) if c in d else "---"
    data = []
    for ds, regime, name in SETTINGS:
        c = sel(ds, regime, methods=['coreset_dp'])
        data.append((c[c.fl.notna()], FULL[(FULL.dataset == ds) & (FULL.regime == regime)]))
    blocks = [("matched random", lambda c, f, x: col(c, {'avg': 'fl_rand', 'prox': 'fl_rand_prox', 'scaf': 'fl_rand_scaf'}[x])),
              ("all clients", lambda c, f, x: col(f, {'avg': 'fl', 'prox': 'fl_prox', 'scaf': 'fl_scaf'}[x])),
              ("drift-aware coreset", lambda c, f, x: col(c, {'avg': 'fl', 'prox': 'fl_prox', 'scaf': 'fl_scaf'}[x]))]
    for bi, (fed, get) in enumerate(blocks):
        for i, (x, alg) in enumerate([('avg', 'FedAvg'), ('prox', 'FedProx'), ('scaf', 'SCAFFOLD')]):
            row = f"{fed if i == 0 else ''} & {alg} & " + " & ".join(get(c, f, x) for c, f in data) + r" \\"
            lines.append((r"\rowcolor{black!6}" if bi == 2 else "") + row)
        lines.append(r"\midrule" if bi < 2 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def volume_matching():
    """Mean relative volume difference between each selection and its matched random federations."""
    d = DF[(DF.gamma == G) & (DF.eps == np.inf) & DF.fl.notna() & DF.method.isin(METHODS)]
    return d.groupby('method').rand_volume_err.mean().round(1).to_string()


def table_gamma():
    """Sampling rate gamma (D = 128) and number of random features D (gamma = 0.5), without DP and at eps1 = 1:
    drift-aware coreset (tau = 0.99, lambda = 1), CLIP / raw / MiniLM."""
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{no DP} & \multicolumn{3}{c}{$\varepsilon_1=1$} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r"Setting & $\gamma_1$ & $D$ & \% rec. & FL acc. & \% rec. & FL acc. & $\Delta_{\mathrm{FL}}$ \\", r"\midrule"]
    encs = {'celeba': 'clip-b32', 'fairface': 'clip-b32', 'ACSIncome': 'raw', 'Sent140': 'minilm'}
    confs = [(0.1, 128), (0.2, 128), (0.5, 128), (1.0, 128), (0.5, 64), (0.5, 256)]
    for si, (ds, regime, name) in enumerate(SETTINGS):
        for i, (gm, D) in enumerate(confs):
            g = sel(ds, regime, gamma=gm, D=D, methods=['coreset_dp'], enc=encs[ds])
            h = sel(ds, regime, gamma=gm, D=D, eps=1.0, methods=['coreset_dp'], enc=encs[ds])
            t, u = g[g.fl.notna()], h[h.fl.notna()]
            sk = int((~h.trained).sum())
            sk0 = int((~g.trained).sum())
            row = (f"{name if i == 0 else ''} & {gm:g} & {D} & {g.records_frac.mean():.1f}{f'$^{{{sk0}}}$' if sk0 else ''} & {pm(t.fl)} & "
                   f"{h.records_frac.mean():.1f}{f'$^{{{sk}}}$' if sk else ''} & {pm(u.fl)} & {sg(u.d_fl.mean())} \\\\")
            lines.append((r"\rowcolor{black!6}" if (gm, D) == (G, DMAIN) else "") + row)
        lines.append(r"\midrule" if si < len(SETTINGS) - 1 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def table_coverage():
    """Coverage of the complete adaptive stopping procedure (stopping_check.py, selection only): share of runs whose
    true score (noise-free sketches of all records, gamma = 1, D = 1024) reaches tau = 0.99, and records selected.
    Top: one noise draw per partition and encoder at eps1 = 1 (checks/coverage_single: uncalibrated rule 'current' vs
    'calibrated', gamma = 0.1, D = 256).
    Bottom: 100 independent sample + DP-noise draws per partition and encoder (checks/coverage), gamma = 0.5,
    D = 128: the calibrated rule of the paper, and the bound extended to the sampling error (rule ucb_all)."""
    root = CHECKS
    def load(sub):
        rows = []
        for f in (root / sub).glob("*.csv"):
            d = pd.read_csv(f)
            d['ds'] = ('celeba' if 'celeba' in f.name else 'fairface' if 'fairface' in f.name
                       else 'ACSIncome' if 'ACSIncome' in f.name else 'Sent140')
            rows.append(d)
        return pd.concat(rows)
    one, rp = load("coverage_single"), load("coverage")
    # the paper's one-draw rows were computed by an earlier version of stopping_check.py, which named the
    # calibrated rule 'wref_ucbdp'
    one = one[one.eps == 1.0].replace({'rule': {'wref_ucbdp': 'calibrated'}})
    rp = rp[rp.tau == 0.99]
    epsn = lambda e: "no DP" if not np.isfinite(e) else f"$\\varepsilon_1{{=}}{e:g}$"
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             "Rule & " + " & ".join(n for _, _, n in SETTINGS) + r" \\", r"\midrule"]
    def row(name, d):
        cells = []
        for ds, _, _ in SETTINGS:
            x = d[d.ds == ds]
            cells.append(f"{100 * (x.true_score_plain >= 0.99).mean():.0f} / {x.records_frac.mean():.1f}")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines.append(r"\multicolumn{5}{l}{\emph{one draw, $\varepsilon_1{=}1$, $\gamma_1{=}0.1$, $D{=}256$}} \\")
    row("old rule", one[one.rule == 'current'])
    row("calibrated", one[one.rule == 'calibrated'])
    for rule, title in (('calibrated', 'calibrated (ours)'), ('ucb_all', 'margin on sampling too')):
        lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{100 draws, $\\gamma_1{{=}}0.5$, $D{{=}}128$: {title}}}}} \\\\")
        for e in (float('inf'), 1.0, 2.0, 4.0):
            row(epsn(e), rp[(rp.rule == rule) & (rp.eps == e)])
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def figure_pareto():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    lam_c = {0.0: '#eb6834', 0.25: '#1baf7a', 1.0: '#2a78d6', 4.0: '#4a3aa7'}
    tau_m = {0.95: 'o', 0.99: 's', 0.999: 'D'}
    base = {'ddem': ('DDEM', '^'), 'fedbary': ('FedBary', 'v'), 'divfl': ('DivFL', 'P')}
    plt.rcParams.update({'font.size': 7, 'axes.edgecolor': '#52514e', 'axes.linewidth': 0.6,
                         'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, ds in zip(axes, ['celeba', 'fairface']):
        d = DF[(DF.dataset == ds) & (DF.regime == 'label') & (DF.eps == np.inf) & (DF.D == DMAIN) & (DF.gamma == G)]
        grid = d[d.method.str.startswith('coreset_t') & (d.enc == 'clip-b32')].copy()
        main = d[(d.method == 'coreset_dp') & (d.enc == 'clip-b32')]          # same runs as Table grid
        grid = pd.concat([grid[grid.method != 'coreset_t0.99_l1'], main.assign(method='coreset_t0.99_l1')])
        grid['tau'] = grid.method.str.extract(r"_t([\d.]+)_")[0].astype(float)
        grid['lam'] = grid.method.str.extract(r"_l([\d.]+)$")[0].astype(float)
        t = grid[grid.fl.notna()]
        ax.scatter(t.records_frac, t.fl_rand, s=4, color='#c3c2b7', zorder=1)
        for m, (lab, mk) in base.items():
            b = d[(d.method == m) & d.fl.notna()]
            if len(b):
                ax.scatter(b.records_frac.mean(), b.fl.mean(), marker=mk, s=28, color='#52514e', zorder=3)
                ax.annotate(lab, (b.records_frac.mean(), b.fl.mean()), xytext=(4, -8), textcoords='offset points',
                            fontsize=5.5, color='#52514e')
        for (tau, lam), g in t.groupby(['tau', 'lam']):
            ax.errorbar(g.records_frac.mean(), g.fl.mean(), yerr=g.fl.std(), fmt=tau_m[tau], ms=5, color=lam_c[lam],
                        ecolor=lam_c[lam], elinewidth=0.6, capsize=1.5, zorder=4, mec='white', mew=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('records exposed (% of population, log scale)')
        ax.set_title({'celeba': 'CelebA', 'fairface': 'FairFace'}[ds], fontsize=8)
        ax.grid(color='#e4e3df', lw=0.5)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('FedAvg balanced accuracy (%)')
    h = [Line2D([], [], color=c, marker='o', ls='', ms=5, label=f'Coreset $\\lambda$={l:g}') for l, c in lam_c.items()]
    h += [Line2D([], [], color='#52514e', marker=mk, ls='', ms=5, label=f'$\\tau$={t:g}') for t, mk in tau_m.items()]
    h += [Line2D([], [], color='#c3c2b7', marker='o', ls='', ms=3, label='matched random')]
    fig.legend(handles=h, loc='lower center', ncol=8, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "pareto_label.pdf", bbox_inches='tight')
    plt.close(fig)


# Shared figure style: reference palette slots (validated all-pairs: blue, orange, aqua, violet), recessive axes.
INK2, GRIDC = '#52514e', '#e4e3df'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']


def _style(ax):
    ax.grid(color=GRIDC, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, labelsize=6)


def figure_private():
    """Drift-aware coreset vs matched random federations across privacy budgets, label skew."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 7, 'axes.edgecolor': INK2, 'axes.linewidth': 0.6, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    xs = [0, 1, 2, 3]
    ends = []
    for (ds, name), c in zip([('celeba', 'CelebA'), ('fairface', 'FairFace'), ('ACSIncome', 'ACSIncome'),
                              ('Sent140', 'Sent140')], CAT):
        acc, rnd = [], []
        for eps in [np.inf, 4.0, 2.0, 1.0]:
            g = sel(ds, 'label', eps=eps, methods=['coreset_dp'])
            g = g[g.fl.notna()]
            acc.append(g.fl.mean()); rnd.append(g.fl_rand.mean())
        ax.plot(xs, acc, color=c, lw=2, marker='o', ms=4, mec='white', mew=0.5)
        ax.plot(xs, rnd, color=c, lw=1, ls=':', marker='o', ms=2.5)
        ends.append((acc[-1], name))
    # end labels, spread so that no two are closer than GAP accuracy points
    GAP, ys = 1.8, []
    for y, name in sorted(ends, reverse=True):
        y_lab = min(y, ys[-1][0] - GAP) if ys else y
        ys.append((y_lab, y, name))
    for y_lab, y, name in ys:
        ax.annotate(name, (xs[-1], y), xytext=(xs[-1] + 0.12, y_lab), textcoords='data', fontsize=6, color=INK2,
                    va='center')
    ax.set_xticks(xs)
    ax.set_xticklabels(['none', '4', '2', '1'])
    ax.set_xlim(-0.2, 3.9)
    ax.set_xlabel(r'selection budget $\varepsilon_1$ ($\delta=10^{-5}$, $\gamma_1=0.5$)')
    ax.set_ylabel('FedAvg bal. acc. (%)')
    _style(ax)
    h = [Line2D([], [], color=INK2, lw=2, label='drift-aware coreset'),
         Line2D([], [], color=INK2, lw=1, ls=':', label='matched random')]
    ax.legend(handles=h, loc='center left', fontsize=6, frameon=False, bbox_to_anchor=(0, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / "fig_private.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig_private.png", bbox_inches='tight', dpi=200)
    plt.close(fig)


def macros():
    out = []

    def mac(name, val):
        out.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    for ds, key in [('celeba', 'A'), ('fairface', 'B')]:
        for m, tag in [('coreset_dp', 'DP'), ('coreset_plain', 'Plain'), ('coreset_neq', 'Neq'), ('ddem', 'Ddem'),
                       ('fedbary', 'Fbary'), ('divfl', 'Divfl')]:
            g = sel(ds, 'label', methods=[m])
            t = g[g.fl.notna()]
            if t.empty:
                continue
            mac(f"V{tag}{key}fl", pm(t.fl)); mac(f"V{tag}{key}rec", f"{g.records_frac.mean():.1f}")
            mac(f"V{tag}{key}k", f"{g.k.mean():.1f}"); mac(f"V{tag}{key}dfl", sg(t.d_fl.mean()))
            mac(f"V{tag}{key}wins", wins(t)); mac(f"V{tag}{key}cen", pm(t.cen))
            mac(f"V{tag}{key}dcen", sg(t.d_cen.mean())); mac(f"V{tag}{key}drift", sg((t.fl - t.cen).mean()))
            mac(f"V{tag}{key}het", f"{g.het.mean():.2f}")
        f = FULL[(FULL.dataset == ds) & (FULL.regime == 'label')]
        mac(f"VFull{key}fl", pm(f.fl)); mac(f"VFull{key}cen", pm(f.cen))
    for ds, key in [('ACSIncome', 'Acs'), ('Sent140', 'Sent')]:
        for m, tag in [('coreset_dp', 'DP'), ('coreset_plain', 'Plain'), ('coreset_neq', 'Neq'), ('ddem', 'Ddem'),
                       ('fedbary', 'Fbary'), ('divfl', 'Divfl')]:
            g = sel(ds, 'label', methods=[m])
            t = g[g.fl.notna()]
            if len(t):
                mac(f"V{tag}{key}fl", pm(t.fl)); mac(f"V{tag}{key}rec", f"{g.records_frac.mean():.1f}")
                mac(f"V{tag}{key}dfl", sg(t.d_fl.mean())); mac(f"V{tag}{key}wins", wins(t))
                mac(f"V{tag}{key}drift", sg((t.fl - t.cen).mean()))
        f = FULL[(FULL.dataset == ds) & (FULL.regime == 'label')]
        mac(f"VFull{key}fl", pm(f.fl)); mac(f"VFull{key}cen", pm(f.cen))
    # private, drift-aware, eps = 1
    p = DF[(DF.method == 'coreset_dp') & (DF.gamma == G) & (DF.D == DMAIN) & (DF.eps == 1.0) & (DF.regime == 'label')]
    t = p[p.fl.notna()]
    mac("VPrivWins", wins(t)); mac("VPrivN", f"{len(p)}")
    for ds, key in [('celeba', 'A'), ('fairface', 'B'), ('ACSIncome', 'Acs'), ('Sent140', 'Sent')]:
        g = t[(t.dataset == ds) & (t.regime == 'label')]
        mac(f"VPriv{key}fl", pm(g.fl)); mac(f"VPriv{key}rec", f"{p[(p.dataset == ds) & (p.regime == 'label')].records_frac.mean():.1f}")
        mac(f"VPriv{key}dfl", sg(g.d_fl.mean()))
    q = DF[(DF.method == 'coreset_neq') & (DF.gamma == G) & (DF.D == DMAIN) & (DF.eps < np.inf)]
    mac("VPrivNeqTrained", f"{int(q.trained.sum())}"); mac("VPrivNeqN", f"{len(q)}")
    # attribute skew: largest |Delta_FL| of any method
    a = DF[(DF.regime == 'attr') & (DF.gamma == G) & (DF.D == DMAIN) & (DF.eps == np.inf) & DF.fl.notna()]
    mac("VAttrMaxd", f"{a.groupby(['dataset', 'method']).d_fl.mean().abs().max():.1f}")
    # private DDEM (eps = 1)
    for ds, key in [('celeba', 'A'), ('fairface', 'B'), ('ACSIncome', 'Acs'), ('Sent140', 'Sent')]:
        q = sel(ds, 'label', eps=1.0, methods=['ddem'])
        t = q[q.fl.notna()]
        if len(t):
            mac(f"VPrivDdem{key}fl", pm(t.fl)); mac(f"VPrivDdem{key}rec", f"{q.records_frac.mean():.1f}")
    # drift correctors: on matched random federations, all clients and the coreset
    for ds, key in [('celeba', 'A'), ('fairface', 'B'), ('ACSIncome', 'Acs'), ('Sent140', 'Sent')]:
        c = sel(ds, 'label', methods=['coreset_dp'])
        f = FULL[(FULL.dataset == ds) & (FULL.regime == 'label')]
        for tag, col in (('Prox', 'prox'), ('Scaf', 'scaf')):
            mac(f"V{tag}Rand{key}", pm(c.get(f'fl_rand_{col}', pd.Series(dtype=float))))
            mac(f"V{tag}Core{key}", pm(c.get(f'fl_{col}', pd.Series(dtype=float))))
            mac(f"V{tag}Full{key}", pm(f.get(f'fl_{col}', pd.Series(dtype=float))))
        mac(f"VAvgRand{key}", pm(c.fl_rand))
    return "\n".join(out) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"  {len(DF)} selections, {int(DF.fl.notna().sum())} trained", file=sys.stderr)
    write("tab_label.tex", table_images('label', steps=True))
    write("tab_attr.tex", table_images('attr'))
    write("tab_acs.tex", table_tabtext([('ACSIncome', 'label', 'ACSIncome', ['raw'], '100'),
                                        ('Sent140', 'label', 'Sent140', ['minilm', 'mpnet'], '100')]))
    write("tab_natural.tex", table_tabtext([('ACSIncome', 'states', 'ACS states', ['raw'], '50'),
                                            ('Sent140', 'groups', 'Sent140 groups', ['minilm', 'mpnet'], '100')]))
    write("tab_grid.tex", table_grid())
    write("tab_stop.tex", table_stop())
    write("tab_dpfl.tex", table_private())
    write("tab_gamma.tex", table_gamma())
    write("tab_coverage.tex", table_coverage())
    write("tab_drift.tex", table_drift())
    print("  volume matching (mean |rel. volume difference|, %):\n" + volume_matching(), file=sys.stderr)
    write("numbers.tex", macros())
    figure_pareto()
    figure_private()


if __name__ == '__main__':
    main()
