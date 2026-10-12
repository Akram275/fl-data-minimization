"""
Collector for the results of run_protocol.py and full_reference.py: one row per selection, with its training
results, and one row per partition for the federation of all clients.

  from paper.collect import collect, full_reference
  df = collect()          # columns: dataset, regime, seed, enc, gamma, eps, D, method, k, records_frac,
                          #          trained, fl, cen, fl_rand, cen_rand, d_fl, d_cen, win_fl, n_rand, ...
Paths: results/<partition root>/<enc>/g<gamma>_e<eps>[_D<D>]/<method>/selection.json
       results/<partition root>/full/{fl,cen}.csv            (full_reference.py)
       results/<partition root>/-/full/fl_<corrector>.csv    (FedProx / SCAFFOLD on all clients)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MIN_RAND = 5


def acc(f):
    return 100 * pd.read_csv(f).balanced_accuracy.iloc[-1]


def partition_of(parts):
    """(dataset, regime, seed) from the partition root's path components."""
    s = "/".join(parts)
    if 'conv_results_embedding' in s:
        ds = parts[parts.index('conv_results_embedding') + 1]
        regime = 'label' if 'labelskew' in s else 'attr'
        seed = int(re.search(r"seed_(\d+)", s).group(1))
    else:
        ds = 'ACSIncome' if 'ACSIncome' in s else 'Sent140'
        m, g = re.search(r"label_seed(\d+)", s), re.search(r"groups_seed(\d+)", s)
        regime, seed = (('label', int(m.group(1))) if m else ('groups', int(g.group(1))) if g
                        else (('states' if ds == 'ACSIncome' else 'users'), 0))
    return ds, regime, seed


# Drift correctors, per dataset: FedProx mu and SCAFFOLD's local SGD lr, chosen by the grids on all clients of
# scripts/run_tuning.sh (seed-42 partitions, best final balanced accuracy).
PROX_MU = {'celeba': '0.1', 'fairface': '0.01', 'ACSIncome': '1', 'Sent140': '0.001'}
SCAFFOLD_LR = {'celeba': '0.03', 'fairface': '0.01', 'ACSIncome': '0.03', 'Sent140': '0.3'}


def _corrector_cols(d, ds):
    """FedProx / SCAFFOLD (published, SGD) on a coreset and its matched random federations."""
    r = {}
    for tag, suf in (('prox', f"_fedprox_mu{PROX_MU.get(ds, 'x')}"), ('scaf', f"_scaffoldsgd_lr{SCAFFOLD_LR.get(ds, 'x')}")):
        rnd = [d / f"rand_trial_{t}_fl{suf}.csv" for t in range(1, 6)]
        if (d / f"fl{suf}.csv").exists() and all(f.exists() for f in rnd):
            r[f"fl_{tag}"] = acc(d / f"fl{suf}.csv")
            r[f"fl_rand_{tag}"] = np.mean([acc(f) for f in rnd])
    return r


def collect(root=RESULTS):
    rows = []
    # DivFL: per-round selection (K clients per round); records = records of the clients ever trained
    for f in root.rglob("divfl/fl.json"):
        d = f.parent
        rel = d.relative_to(root).parts
        m = re.match(r"g([\d.]+)_e([^_]+)(?:_D(\d+))?$", rel[-2])
        ds, regime, seed = partition_of(list(rel[:-3]))
        j = json.load(open(f))
        rnd = sorted(d.glob("rand_round_*_fl.csv"))
        if not (d / "fl.csv").exists() or len(rnd) < MIN_RAND:
            continue
        a = acc(d / "fl.csv")
        rf = [acc(x) for x in rnd]
        ever_r = [json.load(open(x.with_suffix('.json')))['ever_trained_records'] / j['N'] for x in rnd]
        rows.append(dict(dataset=ds, regime=regime, seed=seed, enc=rel[-3], gamma=float(m.group(1)), eps=float(m.group(2)),
                         method='divfl', k=j['K'], records_frac=100 * j['ever_trained_records'] / j['N'], trained=True,
                         fl=a, fl_rand=np.mean(rf), n_rand=len(rf), d_fl=a - np.mean(rf), win_fl=a > np.mean(rf),
                         clients_ever=len(j['ever_trained']), rand_records_frac=100 * np.mean(ever_r)))
    for f in root.rglob("selection.json"):
        rel = f.relative_to(root).parts
        method_dir = rel[-2]
        m = re.match(r"g([\d.]+)_e([^_]+)(?:_D(\d+))?$", rel[-3])
        if not m:
            continue
        enc, gamma, eps, base = rel[-4], float(m.group(1)), float(m.group(2)), rel[:-4]
        rff = int(m.group(3)) if m.group(3) else 256
        ds, regime, seed = partition_of(list(base))
        j = json.load(open(f))
        d = f.parent
        r = dict(dataset=ds, regime=regime, seed=seed, enc=enc, gamma=gamma, eps=eps, D=rff, method=method_dir,
                 k=j['k'], records_frac=100 * j['records_frac'], trained=j.get('trained', False),
                 encoded_frac=j.get('encoded_frac', np.nan), target_reached=j.get('target_reached', True),
                 het=(j.get('diagnostics') or {}).get('rep_heterogeneity', np.nan))
        rnd = sorted(d.glob("rand_trial_*_fl.csv"))
        if (d / "fl.csv").exists() and (d / "cen.csv").exists() and len(rnd) >= MIN_RAND:
            rf = [acc(x) for x in rnd]
            rc = [acc(x) for x in sorted(d.glob("rand_trial_*_cen.csv"))]
            r.update(fl=acc(d / "fl.csv"), cen=acc(d / "cen.csv"), fl_rand=np.mean(rf),
                     cen_rand=np.mean(rc) if rc else np.nan, n_rand=len(rf),
                     fl_beat=sum(acc(d / "fl.csv") > x for x in rf))
            r['d_fl'], r['d_cen'] = r['fl'] - r['fl_rand'], r['cen'] - r['cen_rand']
            r['win_fl'], r['win_cen'] = r['d_fl'] > 0, r['d_cen'] > 0
            r['rand_volume_err'] = 100 * np.mean(np.abs(j.get('random_rel_volume_error') or [np.nan]))
            if method_dir == 'coreset_dp':
                r.update(_corrector_cols(d, ds))
        rows.append(r)
    return pd.DataFrame(rows)


def full_reference(root=RESULTS):
    """FedAvg on all clients and centralized training on all records (independent of the selection), from
    full_reference.py's <partition root>/full/ runs, with FedProx and SCAFFOLD on all clients when present."""
    rows = []
    for f in root.rglob("full/fl.csv"):
        d = f.parent
        if d.parent.name == '-' or not (d / "cen.csv").exists():
            continue
        ds, regime, seed = partition_of(list(d.relative_to(root).parts))
        r = dict(dataset=ds, regime=regime, seed=seed, fl=acc(d / "fl.csv"), cen=acc(d / "cen.csv"))
        corr = d.parent / '-' / 'full'
        for tag, name in (('prox', f"fl_fedprox_mu{PROX_MU.get(ds, 'x')}.csv"),
                          ('scaf', f"fl_scaffoldsgd_lr{SCAFFOLD_LR.get(ds, 'x')}.csv")):
            if (corr / name).exists():
                r[f"fl_{tag}"] = acc(corr / name)
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    df = collect()
    print(f"{len(df)} selections, {int(df.get('fl', pd.Series()).notna().sum())} trained with baselines")
    if len(df):
        print(df.groupby(['dataset', 'regime', 'method', 'gamma', 'eps']).agg(
            n=('k', 'size'), k=('k', 'mean'), rec=('records_frac', 'mean'),
            fl=('fl', 'mean'), d_fl=('d_fl', 'mean')).round(2).to_string())
    print(full_reference().groupby(['dataset', 'regime'])[['fl', 'cen']].mean().round(1))
