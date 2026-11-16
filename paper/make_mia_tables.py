"""
Numbers and LaTeX tables of the distribution- and membership-inference audit (Section "A representative hides in
its group" and its appendix) from mia_check/ (mia_check.py, scripts/run_mia.sh, scripts/run_mia_dp.sh). Averages:
over the non-participants of a federation (pairwise AUCs), over the 5 random federations of a seed, then over
seeds (3, FairFace 2).

  .venv/bin/python paper/make_mia_tables.py            # prints the three tables and the in-text numbers
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent / "mia_check"
DS = [('celeba', 'CelebA'), ('fairface', 'FairFace'), ('ACSIncome', 'ACSIncome'), ('Sent140', 'Sent140')]
ATT = [('loss', 'Loss'), ('rmia', 'RMIA'), ('lira', 'LiRA')]


def runs(tag):
    """(dataset, seed dir) -> (summary, clients, config)."""
    out = {}
    for f in ROOT.rglob(f"{tag}/summary.csv"):
        ds = next(k for k, _ in DS if k in f.parts)
        seed = next(p for p in f.parts if 'seed' in p)
        out[(ds, seed)] = (pd.read_csv(f), pd.read_csv(f.parent / "clients.csv"),
                           json.load(open(f.parent / "config.json")))
    return out


def grp(t):
    return np.where(pd.Series(t).str.startswith('rand_'), 'random', t)


def stats(tag, attack):
    """per dataset: dict of averaged quantities."""
    R = runs(tag)
    res = {}
    for ds, _ in DS:
        per_seed = []
        for (d, seed), (S, C, cfg) in R.items():
            if d != ds:
                continue
            q = {}
            S = S[S.attack == attack].assign(g=lambda x: grp(x.target))
            C = C[C.attack == attack].assign(g=lambda x: grp(x.target))
            npc = C[~C.participant.astype(bool)]
            for g in ('coreset', 'random'):
                c = npc[npc.g == g].groupby('target')
                q[f'held_{g}'] = c.pair_heldout_auc.mean().mean()
                q[f'mem_{g}'] = c.pair_member_auc.mean().mean()
                if g == 'coreset':
                    x = npc[npc.g == g]
                    q['rho'] = spearmanr(x.delta_to_rep, x.pair_heldout_auc)[0] if len(x) > 2 else np.nan
            for g in ('coreset', 'random', 'full'):
                for m, col in (('record', 'auc'), ('record', 'tpr1'), ('client', 'auc'), ('client', 'top1'),
                               ('client', 'top1_chance')):
                    s = S[(S.g == g) & (S.measure == m)]
                    q[f'{m}_{col}_{g}'] = s[col].mean() if len(s) and col in s else np.nan
                p = S[(S.g == g) & (S.measure == 'population')]
                if 'fpr' in p:
                    p = p[p.fpr == 1e-3]
                    q[f'id_{g}'], q[f'prec_{g}'] = (p.tp.mean(), p.precision.mean()) if len(p) else (np.nan, np.nan)
            b = cfg['bacc']
            q['acc_coreset'] = b['coreset']
            q['acc_random'] = np.mean([v for k, v in b.items() if k.startswith('rand_')])
            q['acc_full'] = b.get('full', np.nan)
            q['k'] = len(cfg['targets']['coreset'])
            per_seed.append(q)
        if per_seed:
            df = pd.DataFrame(per_seed)
            res[ds] = {**df.mean().to_dict(), 'seeds': len(df), 'k1': int((df.k == 1).sum())}
    return res


f2 = lambda x: "---" if pd.isna(x) else f"{x:.2f}"
pc = lambda x: "---" if pd.isna(x) else f"{100 * x:.1f}"


def main_table():
    a, d8 = stats('main', 'rmia'), stats('dp8', 'rmia')
    print("% --- main body: tab:mia (RMIA) ---")
    for ds, name in DS:
        r, r8 = a[ds], d8[ds]
        print(f"{name} & {f2(r['held_coreset'])} & {f2(r['held_random'])} & {f2(r['mem_coreset'])} & "
              f"{f2(r8['mem_coreset'])} & {f2(r['mem_random'])} \\\\")


def full_table():
    print("% --- appendix: tab:mia-full (no DP) ---")
    for ds, name in DS:
        for i, (att, an) in enumerate(ATT):
            r = stats('main', att)[ds]
            print(f"{name if i == 0 else ''} & {an} & {f2(r['held_coreset'])} & {f2(r['held_random'])} & "
                  f"{f2(r['mem_coreset'])} & {f2(r['mem_random'])} & {f2(r['record_auc_coreset'])} & "
                  f"{f2(r['record_auc_random'])} & {f2(r['record_auc_full'])} & {pc(r['record_tpr1_coreset'])} & "
                  f"{pc(r['record_tpr1_full'])} & {f2(r['client_auc_coreset'])} & {f2(r['client_auc_random'])} & "
                  f"{f2(r['client_top1_coreset'])} ({f2(r['client_top1_chance_coreset'])}) \\\\")
        if ds != DS[-1][0]:
            print(r"\midrule")


def dp_table():
    print("% --- appendix: tab:mia-dp (RMIA) ---")
    S = {t: stats(t, 'rmia') for t in ('main', 'dp8', 'dp1')}
    for ds, name in DS:
        for i, (t, e) in enumerate((('main', r'$\infty$'), ('dp8', '8'), ('dp1', '1'))):
            r = S[t][ds]
            print(f"{name if i == 0 else ''} & {e} & {pc(r['acc_coreset'])} & {f2(r['held_coreset'])} & "
                  f"{f2(r['held_random'])} & {f2(r['mem_coreset'])} & {f2(r['mem_random'])} & "
                  f"{f2(r['record_auc_coreset'])} & {pc(r['record_tpr1_coreset'])} & {f2(r['client_auc_coreset'])} & "
                  f"{f2(r['client_top1_coreset'])} \\\\")
        if ds != DS[-1][0]:
            print(r"\midrule")


def text_numbers():
    print("% --- in-text numbers ---")
    a = stats('main', 'rmia')
    for ds, name in DS:
        r = a[ds]
        print(f"% {name}: seeds {r['seeds']}, k=1 in {r['k1']}, rho(delta, held AUC) {r['rho']:.2f}, "
              f"id@0.1%FPR coreset {r['id_coreset']:.1f} random {r['id_random']:.1f} full {r['id_full']:.1f}, "
              f"precision coreset {r['prec_coreset']:.3f} full {r['prec_full']:.3f}, "
              f"acc core/rand/full {100 * r['acc_coreset']:.1f}/{100 * r['acc_random']:.1f}/{100 * r['acc_full']:.1f}")
    for t in ('dp8', 'dp1'):
        s = stats(t, 'rmia')
        print(f"% {t} acc core/rand/full: " + ", ".join(
            f"{n} {100 * s[d]['acc_coreset']:.1f}/{100 * s[d]['acc_random']:.1f}/{100 * s[d]['acc_full']:.1f}"
            for d, n in DS))
    r100 = stats('r100', 'rmia')
    print("% r100 (seed 42) held core/rand: " + ", ".join(
        f"{n} {r100[d]['held_coreset']:.2f}/{r100[d]['held_random']:.2f}" for d, n in DS if d in r100))


if __name__ == '__main__':
    main_table()
    full_table()
    dp_table()
    text_numbers()
