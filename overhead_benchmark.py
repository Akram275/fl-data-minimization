"""
Compute and communication overhead of the selection protocol, measured on the real partitions
(seed-42 label-skew partitions, ACSIncome states), next to the cost of the FedAvg training it saves.

Client side, per client (median over clients):
  encode       frozen encoder forward pass per record, GPU and CPU (8 threads), images and text
  round0       count + sum + second-moment matrix of the encodings (non-private; private adds noise)
  round1       map to the shared space + random features + per-label sum (non-private D=1024 cosine,
               private D=256 paired + noise)
  messages     floats sent in Rounds 0 and 1
Server side:
  space        aggregate the Round 0 totals and eigendecompose (whitened PCA)
  select       coreset selection (distance matrix + k-medoids loop), drift-aware tau=0.99 and the
               IID-equivalent rule, non-private and private (eps_1 = 1)
Reference (what selection saves):
  train        FedAvg local training time per record (one epoch, GPU), model size (upload per round)

  python overhead_benchmark.py celeba fairface acs_label acs_states sent_label
Outputs: checks/overhead/<setting>.csv (one row per measurement: setting, item, variant, value, unit)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src.federated_learning.torch_fl as fl
from src.data.partitions import create_dirichlet_partitions, partition_key
from src.embedding.snapshots import (FederatedSpace, analytic_rff, client_moments, dp_client_moments, dp_count,
                                     dp_snapshot_sum, kernel_mean_snapshot, unit_norm)
from src.selection.dp_accounting import calibrate
from src.selection.embedding_selectors import ClientCoresetSelector
from src.selection.private_selection import budget_shares, private_snapshots

OUT = Path("checks") / "overhead"
N_LABELS = 2


def timeit(fn, repeat=3):
    """Median wall time (s) of fn() over `repeat` runs (CUDA synchronized)."""
    ts = []
    for _ in range(repeat):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def load(key):
    if key in ('celeba', 'fairface'):
        from src.data.image_metadata import DATASET_LOADERS
        from src.embedding.encoders import encode_cached
        ms = 0 if key == 'celeba' else 80000
        train, _ = DATASET_LOADERS[key](max_samples=ms, n_public=2000)
        np.random.seed(42)
        parts = create_dirichlet_partitions(partition_key(train, key, 'label'), 100 if key == 'celeba' else 50, 0.2,
                                            max_attempts=2000000)
        if key == 'celeba':
            (X, _, _), _ = fl.load_and_preprocess_celebA(target_attribute='Smiling', max_samples=ms)
        else:
            (X, _, _), _ = fl.load_and_preprocess_fairface(max_samples=ms)
        return dict(parts=parts, y=train.labels, Z=encode_cached(train.paths, 'clip-b32', tag=f"{key}_train"),
                    enc='clip-b32', X=X.astype(np.float32), model=lambda: fl.OurModel(), batch=64, paths=train.paths,
                    encoders=['sd-vae', 'dinov2-small', 'clip-b32', 'resnet18'], kind='image')
    from src.data.acs_tabular import load as load_tab, load_embeddings
    task, part, enc = {'acs_label': ('ACSIncome', 'label', 'raw'), 'acs_states': ('ACSIncome', 'states', 'raw'),
                       'sent_label': ('Sent140', 'label', 'minilm')}[key]
    d = load_tab(task)
    cli = np.where(d['split'] == 0)[0]
    y = d['y'][cli]
    if part == 'label':
        np.random.seed(42)
        parts = create_dirichlet_partitions(y, 100, 0.2, max_attempts=2000000)
    else:
        st = d['state'][cli]
        parts = [np.where(st == s)[0] for s in np.unique(st)]
    X = d['X_model'][cli].astype(np.float32)
    return dict(parts=parts, y=y, Z=load_embeddings(task, enc, d)[cli], enc=enc, X=X,
                model=lambda: fl.TabularMLP(X.shape[1]), batch=32, encoders=['minilm', 'mpnet'] if task == 'Sent140' else [],
                kind='text' if task == 'Sent140' else 'tabular')


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def encoder_times(S, rows, key):
    """Seconds per record of the frozen encoder, on GPU and on 8 CPU threads."""
    if S['kind'] == 'image':
        from src.embedding.encoders import load_encoder
        paths = list(S['paths'][:1088])
        for name in S['encoders']:
            for dev, n in [('cuda', 1024), ('cpu', 128)]:
                torch.set_num_threads(8)
                enc = load_encoder(name, dev)
                enc.encode(paths[:64], batch_size=64, verbose=False)                      # warm-up
                t = timeit(lambda: enc.encode(paths[64:64 + n], batch_size=64, verbose=False), repeat=1)
                rows.append((key, 'encode', f'{name} ({dev})', 1e3 * t / n, 'ms/record'))
                del enc
                torch.cuda.empty_cache()
    elif S['kind'] == 'text':
        from transformers import AutoModel, AutoTokenizer
        from src.data.sent140 import ENCODERS, TEXTS
        texts = list(np.load(TEXTS, allow_pickle=True)[:2112])
        for name in S['encoders']:
            for dev, n in [('cuda', 2048), ('cpu', 256)]:
                torch.set_num_threads(8)
                tok, model = AutoTokenizer.from_pretrained(ENCODERS[name]), AutoModel.from_pretrained(ENCODERS[name]).eval().to(dev)

                @torch.no_grad()
                def run(batch):
                    for s in range(0, len(batch), 64):
                        b = tok(batch[s:s + 64], padding=True, truncation=True, max_length=128, return_tensors='pt').to(dev)
                        model(**b)
                run(texts[:64])
                t = timeit(lambda: run(texts[64:64 + n]), repeat=1)
                rows.append((key, 'encode', f'{name} ({dev})', 1e3 * t / n, 'ms/record'))
                del model
                torch.cuda.empty_cache()


def protocol_times(S, rows, key):
    names = [f"c{i}" for i in range(len(S['parts']))]
    ci = {c: np.asarray(p, dtype=int) for c, p in zip(names, S['parts'])}
    Z, y = S['Z'], S['y']
    n = np.array([len(ci[c]) for c in names])
    med = names[int(np.argsort(n)[len(n) // 2])]                      # median-size client
    d0 = Z.shape[1]
    rng = np.random.default_rng(0)
    z = calibrate(1.0, 1e-5, budget_shares())
    rows.append((key, 'clients', '', len(names), 'clients'))
    rows.append((key, 'median client', '', int(n[np.argsort(n)[len(n) // 2]]), 'records'))
    rows.append((key, 'encoding dim', S['enc'], d0, 'd0'))
    # client side, median client
    Zm, ym = Z[ci[med]], y[ci[med]]
    rows.append((key, 'round0', 'non-private', 1e3 * timeit(lambda: client_moments(Zm)), 'ms/client'))
    rows.append((key, 'round0', 'private', 1e3 * timeit(lambda: (dp_count(len(Zm), z['count'], rng),
                                                                    dp_client_moments(Zm, z['sum'], z['outer'], rng))),
                 'ms/client'))
    space = FederatedSpace(32).fit_moments([client_moments(Z[ci[c]]) for c in names])
    rff = analytic_rff(32, 1024, seed=42)
    rows.append((key, 'round1', 'non-private', 1e3 * timeit(
        lambda: kernel_mean_snapshot(space.transform(Zm), rff, ym, N_LABELS)), 'ms/client'))
    space_p = FederatedSpace(32).fit_moments([dict(n=len(ci[c]), **dp_client_moments(Z[ci[c]], 0, 0, rng)) for c in names])
    rff_p = analytic_rff(32, 256, seed=42, paired=True)
    rows.append((key, 'round1', 'private', 1e3 * timeit(
        lambda: dp_snapshot_sum(space_p.transform(unit_norm(Zm)), rff_p, ym, N_LABELS, z['snapshot'], rng)), 'ms/client'))
    # messages (floats per client)
    rows.append((key, 'message round0', 'non-private', 1 + 2 * d0 + d0 * d0, 'floats'))
    rows.append((key, 'message round0', 'private', 1 + d0 + d0 * (d0 + 1) // 2, 'floats'))
    rows.append((key, 'message round1', 'non-private', 1 + 1 + N_LABELS * 1024, 'floats'))
    rows.append((key, 'message round1', 'private', N_LABELS * 256, 'floats'))
    # server side
    mom = [client_moments(Z[ci[c]]) for c in names]
    rows.append((key, 'space', 'non-private', 1e3 * timeit(lambda: FederatedSpace(32).fit_moments(mom)), 'ms'))
    E = space.transform(Z)
    snaps = {c: kernel_mean_snapshot(E[ci[c]], rff, y[ci[c]], N_LABELS) for c in names}
    sizes = {c: len(ci[c]) for c in names}
    N = n.sum()
    mu = sum(sizes[c] * snaps[c] for c in names) / N
    v = sum(float((rff.transform(E[ci[c]]) ** 2).sum()) for c in names) / N - float((mu ** 2).sum())
    P = private_snapshots(Z, ci, names, y, N_LABELS, 1.0)
    for variant, sn, sz, vv, q in [('non-private', snaps, sizes, v, 0.0),
                                   ('private', P['snaps'], P['sizes'], P['feature_var'], P['noise_var'])]:
        for rule in ('tau', 'neq'):
            def go():
                sel = ClientCoresetSelector(sn, sz, seed=42, drift_lambda=1.0, noise_var=q)
                sel.fit(n_target=0.005 * sum(sz.values()), feature_var=vv) if rule == 'neq' else sel.fit(0.99)
                go.k = sel.diagnostics['k']
            t = timeit(go, repeat=1)
            rows.append((key, f'select ({rule})', variant, t, 's'))
            rows.append((key, f'select k ({rule})', variant, go.k, 'k'))


def training_times(S, rows, key):
    """FedAvg reference: local training time per record (one epoch, GPU) and model size."""
    model = S['model']()
    rows.append((key, 'model', '', sum(p.numel() for p in model.parameters()), 'floats'))
    names = list(range(len(S['parts'])))
    n = np.array([len(p) for p in S['parts']])
    idx = np.asarray(S['parts'][names[int(np.argsort(n)[len(n) // 2])]], dtype=int)
    dev = fl._device()
    x = torch.as_tensor(S['X'][idx], device=dev)
    yy = torch.as_tensor(S['y'][idx].astype(np.float32), device=dev)
    old = fl.BATCH_SIZE
    fl.BATCH_SIZE = S['batch']
    model = model.to(dev)
    gen = torch.Generator(device='cpu').manual_seed(0)
    opt = fl._adam(model.parameters(), 5e-4)
    fl._fit_epochs(model, opt, x, yy, 1, gen)                                          # warm-up
    t = timeit(lambda: fl._fit_epochs(model, opt, x, yy, 1, gen))
    fl.BATCH_SIZE = old
    rows.append((key, 'train', f'one epoch ({dev.type})', 1e3 * t / len(idx), 'ms/record'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('settings', nargs='+', choices=['celeba', 'fairface', 'acs_label', 'acs_states', 'sent_label'])
    p.add_argument('--skip-encoders', action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for key in args.settings:
        S = load(key)
        rows = []
        if not args.skip_encoders:
            encoder_times(S, rows, key)
        protocol_times(S, rows, key)
        training_times(S, rows, key)
        pd.DataFrame(rows, columns=['setting', 'item', 'variant', 'value', 'unit']).to_csv(OUT / f"{key}.csv", index=False)
        print(pd.DataFrame(rows, columns=['setting', 'item', 'variant', 'value', 'unit']).to_string(), flush=True)


if __name__ == '__main__':
    main()
