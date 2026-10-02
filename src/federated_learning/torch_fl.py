"""
Federated training (PyTorch): FedAvg on a federation of clients, its drift-correcting variants (FedProx,
SCAFFOLD), per-round selection (DivFL and random), and the centralized controls. The setup reproduces a Keras
pipeline (the code the experiments started from), so its conventions are Keras':

  - image model: Conv(16, 3x3, same, relu) -> MaxPool2 -> Conv(32, 3x3, same, relu) -> MaxPool2 -> Flatten (NHWC
    order) -> Dense(32, relu) -> Dense(1, sigmoid) on 32x32 inputs; tabular and text model: TabularMLP (three
    Dense(32, relu) layers). Binary cross-entropy from logits.
  - init: HeUniform(seed) for the kernels AND biases of the hidden layers, Glorot-uniform kernel and zero bias
    for the output layer (Keras defaults).
  - optimizer: Adam(beta1=0.9, beta2=0.999, eps=1e-7). FL: a fresh Adam every round at lr = 5e-4 * 0.98^(round-1)
    (BASE_LR, DECAY_RATE); centralized: one Adam for all epochs, its lr reset each epoch on the same schedule.
  - local data: train_test_split(test_size=0.25, random_state=42) per client, then validation_split=0.2, which
    holds out the LAST 20% of the (unshuffled) training rows, i.e. training uses the first floor(0.8 n) rows.
    Batch size 64 (32 for tabular data), reshuffled every epoch, last partial batch kept.
  - FedAvg: average of all parameters weighted by client sizes, or by given aggregation weights (a coreset
    representative is weighted by the record mass of its group), renormalized over the participants.
  - a federation of one client trains centrally for max_iterations epochs (run_exp).
  - evaluation after every round on the shared test pool: log loss, balanced accuracy, recall and precision of
    the rounded probabilities (save_scores_to_csv).
"""
import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, log_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split

SEED = int(os.environ.get('FL_SEED', 42))
BASE_LR = 0.0005
DECAY_RATE = 0.98
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
LOCAL_TEST_SIZE = 0.25

FAIRFACE_AGE = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
FAIRFACE_RACE = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
FAIRFACE_GENDER = ['Female', 'Male']


# ---------------------------------------------------------------------------
# Data: CelebA and FairFace images decoded to 32x32 RGB arrays in [0, 1], rows in the order of
# src/data/image_metadata.py. Decoded arrays are cached in data/fl_arrays/.
# ---------------------------------------------------------------------------

def _decode(args):
    path, img_size = args
    img = Image.open(path).convert('RGB').resize(img_size)
    return (np.array(img, dtype=np.float32) / 255.0).flatten()


def _decode_all(paths, img_size, workers=16):
    if not paths:
        return np.zeros((0, img_size[0] * img_size[1] * 3), dtype=np.float32)
    with Pool(workers) as pool:
        return np.stack(pool.map(_decode, [(p, img_size) for p in paths], chunksize=256)).astype(np.float32)


def _cached(name, build):
    path = Path("data/fl_arrays") / f"{name}.npz"
    if path.exists():
        print(f"  [fl data] loaded cached arrays: {path}")
        d = np.load(path)
        return (d['X_train'], d['y_train'], d['a_train']), (d['X_test'], d['y_test'], d['a_test'])
    (X_tr, y_tr, a_tr), (X_te, y_te, a_te) = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, X_train=X_tr, y_train=y_tr, a_train=a_tr, X_test=X_te, y_test=y_te, a_test=a_te)
    os.replace(tmp, path)
    return (X_tr, y_tr, a_tr), (X_te, y_te, a_te)


def load_and_preprocess_celebA(data_dir="data/celeba-dataset", target_attribute='Smiling', img_size=(32, 32),
                               max_samples=None):
    def build():
        img_dir = os.path.join(data_dir, "img_align_celeba", "img_align_celeba")
        if not os.path.exists(img_dir):
            img_dir = os.path.join(data_dir, "img_align_celeba")
        df = pd.merge(pd.read_csv(os.path.join(data_dir, "list_attr_celeba.csv")),
                      pd.read_csv(os.path.join(data_dir, "list_eval_partition.csv")), on='image_id')
        for col in [target_attribute, 'Male', 'Young', 'Eyeglasses']:
            if col in df.columns:
                df[col] = df[col].replace(-1, 0)
        train_df, test_df = df[df['partition'] == 0], df[df['partition'] == 2]
        if max_samples:
            train_df = train_df.iloc[:max_samples]
            test_df = test_df.iloc[:int(max_samples * 0.2)]

        def split(split_df, name):
            print(f"Processing {name} images ({len(split_df):,} total)...")
            keep = split_df[[os.path.exists(os.path.join(img_dir, f)) for f in split_df['image_id']]]
            X = _decode_all([os.path.join(img_dir, f) for f in keep['image_id']], img_size)
            return (X, keep[target_attribute].values.astype(np.int32),
                    keep[['Male', 'Young', 'Eyeglasses']].values.astype(np.int32))

        return split(train_df, "Train"), split(test_df, "Test")

    return _cached(f"celeba_{target_attribute}_{max_samples or 'all'}_{img_size[0]}", build)


def load_and_preprocess_fairface(data_dir="data/FairFace", img_size=(32, 32), max_samples=None):
    def build():
        train_df = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
        val_df = pd.read_csv(os.path.join(data_dir, "val_labels.csv"))
        if max_samples:
            train_df = train_df.iloc[:max_samples]
            val_df = val_df.iloc[:int(max_samples * 0.2)]

        def split(split_df, name):
            print(f"Processing {name} images ({len(split_df):,} total)...")
            keep = split_df[[os.path.exists(os.path.join(data_dir, f)) for f in split_df['file']]]
            X = _decode_all([os.path.join(data_dir, f) for f in keep['file']], img_size)
            y = np.array([FAIRFACE_GENDER.index(g) for g in keep['gender']], dtype=np.int32)
            a = np.array([[FAIRFACE_AGE.index(ag), FAIRFACE_RACE.index(r)]
                          for ag, r in zip(keep['age'], keep['race'])], dtype=np.int32).reshape(-1, 2)
            return X, y, a

        return split(train_df, "Train"), split(val_df, "Val")

    return _cached(f"fairface_{max_samples or 'all'}_{img_size[0]}", build)


def generate_universal_testdata(X_test, y_test, val_ratio=0.2):
    _, x_val, _, y_val = train_test_split(X_test, y_test, test_size=val_ratio, random_state=42, stratify=y_test)
    print(f">> Centralized Universal Test Pool initialized with {len(x_val):,} records.")
    return pd.DataFrame(x_val), pd.Series(y_val)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _he_uniform_(t: torch.Tensor, fan_in: int, gen: torch.Generator):
    limit = math.sqrt(6.0 / fan_in)
    with torch.no_grad():
        t.copy_(torch.empty(t.shape).uniform_(-limit, limit, generator=gen))


class OurModel(nn.Module):
    """Image CNN of the module docstring (32x32 RGB inputs)."""
    def __init__(self, seed: int = SEED):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 32, 32)
        self.out = nn.Linear(32, 1)
        gen = torch.Generator().manual_seed(seed)
        for layer, fan_in in [(self.conv1, 3 * 3 * 3), (self.conv2, 3 * 3 * 16), (self.fc1, 8 * 8 * 32)]:
            _he_uniform_(layer.weight, fan_in, gen)
            _he_uniform_(layer.bias, layer.bias.numel(), gen)
        limit = math.sqrt(6.0 / (32 + 1))
        with torch.no_grad():
            self.out.weight.copy_(torch.empty(1, 32).uniform_(-limit, limit, generator=gen))
            self.out.bias.zero_()

    def forward(self, x):                       # x: (B, 3072) flattened HWC images
        x = x.view(-1, 32, 32, 3).permute(0, 3, 1, 2)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.permute(0, 2, 3, 1).flatten(1)    # Keras Flatten order (NHWC)
        return self.out(F.relu(self.fc1(x))).squeeze(1)   # logits


class TabularMLP(nn.Module):
    """Dense(32)-ReLU x3 -> Dense(1)-sigmoid for tabular and text inputs, HeUniform(seed) kernels AND biases on
    the hidden layers, Glorot-uniform kernel and zero bias on the output layer (Keras defaults)."""

    def __init__(self, in_dim: int, seed: int = SEED, width: int = 32):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(in_dim, width), nn.Linear(width, width), nn.Linear(width, width)])
        self.out = nn.Linear(width, 1)
        gen = torch.Generator().manual_seed(seed)
        for layer in self.hidden:
            _he_uniform_(layer.weight, layer.in_features, gen)
            _he_uniform_(layer.bias, layer.bias.numel(), gen)
        limit = math.sqrt(6.0 / (width + 1))
        with torch.no_grad():
            self.out.weight.copy_(torch.empty(1, width).uniform_(-limit, limit, generator=gen))
            self.out.bias.zero_()

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.out(x).squeeze(1)   # logits


# Model used by every training function. Default: the image CNN (OurModel). The
# tabular experiments set MODEL_FACTORY = lambda seed: TabularMLP(in_dim, seed)
# and BATCH_SIZE = 32 before training (src/experiment.py).
MODEL_FACTORY = None


def new_model(seed: int = SEED) -> nn.Module:
    return MODEL_FACTORY(seed) if MODEL_FACTORY is not None else OurModel(seed)


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _fit_epochs(model, opt, x, y, epochs, gen, prox=None, correction=None):
    """Keras fit(validation_split=0.2, batch_size=64, shuffle=True) on x, y.
    prox = (mu, global params): adds FedProx's mu/2 ||w - w_global||^2 to every batch loss.
    correction: per-parameter tensors added to every batch gradient (SCAFFOLD's c - c_i)."""
    n_train = int(math.floor(len(x) * (1.0 - VALIDATION_SPLIT)))
    x, y = x[:n_train], y[:n_train]
    model.train()
    params = list(model.parameters())
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n_train, generator=gen).to(x.device)
        for s in range(0, n_train, BATCH_SIZE):
            b = perm[s:s + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x[b]), y[b])
            if prox is not None:
                mu, w0 = prox
                loss = loss + 0.5 * mu * sum(((p - q) ** 2).sum() for p, q in zip(params, w0))
            loss.backward()
            if correction is not None:
                for p, c in zip(params, correction):
                    p.grad.add_(c)
            opt.step()
            steps += 1
    return steps


def _full_gradient(model, x, y):
    """Gradient of the mean loss over the client's local training rows (the first
    floor(0.8 n) rows, as _fit_epochs), at the model's current parameters, one tensor
    per parameter. Used by DivFL."""
    n_train = int(math.floor(len(x) * (1.0 - VALIDATION_SPLIT)))
    model.train()
    model.zero_grad(set_to_none=True)
    for s in range(0, n_train, 4096):
        xb, yb = x[s:min(s + 4096, n_train)], y[s:min(s + 4096, n_train)]
        (F.binary_cross_entropy_with_logits(model(xb), yb, reduction='sum') / max(n_train, 1)).backward()
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]


@torch.no_grad()
def _evaluate(model, test_x, test_y_np):
    model.eval()
    y_pred = torch.cat([torch.sigmoid(model(test_x[s:s + 4096])) for s in range(0, len(test_x), 4096)]).cpu().numpy()
    y_pred = y_pred.astype(np.float32)
    # A diverged model (non-finite outputs, e.g. SCAFFOLD's local SGD at a too-large learning rate in its grid) is
    # scored as an uninformative one: p = 0.5, i.e. chance-level balanced accuracy.
    y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=0.5, neginf=0.5)
    y_hat = np.round(y_pred)
    return [log_loss(test_y_np, np.clip(y_pred, 1e-15, 1 - 1e-15), labels=[0, 1]),
            balanced_accuracy_score(test_y_np, y_hat),
            recall_score(test_y_np, y_hat, zero_division=0),
            precision_score(test_y_np, y_hat, zero_division=0)]


def _adam(params, lr):
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-7)


def _to_tensors(df_x, s_y, dev):
    return (torch.as_tensor(np.asarray(df_x, dtype=np.float32), device=dev),
            torch.as_tensor(np.asarray(s_y, dtype=np.float32), device=dev))


def run_centralized_training(dataset, test_features, test_labels, num_epochs, seed=SEED, base_lr=None,
                             lr_decay=True):
    base_lr = BASE_LR if base_lr is None else base_lr   # read at call time (BASE_LR can be overridden)
    dev = _device()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x, y = _to_tensors(dataset[0], dataset[1], dev)
    test_x = torch.as_tensor(test_features.to_numpy().astype('float32'), device=dev)
    test_y = test_labels.to_numpy().astype('int32')
    model = new_model(seed).to(dev)
    opt = _adam(model.parameters(), base_lr)
    scores = []
    for epoch in range(1, num_epochs + 1):
        lr = base_lr * (DECAY_RATE ** (epoch - 1)) if lr_decay else base_lr
        for g in opt.param_groups:
            g['lr'] = lr
        _fit_epochs(model, opt, x, y, 1, gen)
        scores.append(_evaluate(model, test_x, test_y))
        print(f"  Centralized Epoch {epoch:02d} (LR: {lr:.7f}) -> Loss: {scores[-1][0]:.5f} | "
              f"Balanced Acc: {scores[-1][1] * 100:.2f}%")
    return scores


def run_training(datasets, test_features, test_labels, epochs, max_iterations, lr_decay=True, seed=SEED,
                 agg_weights=None, algo='fedavg', mu=0.0, sgd_lr=0.1):
    """FedAvg on all the given clients (full participation) for max_iterations rounds of `epochs` local epochs.
    algo: 'fedavg'; 'fedprox' (Li et al. 2020: local loss + mu/2 ||w - w_t||^2, same local solver); or
    'scaffold_sgd', SCAFFOLD as published (Alg. 1 of Karimireddy et al. 2020): plain local SGD with learning rate
    sgd_lr (decayed per round like FedAvg's), corrected gradient g - c_i + c, option II control variates, global
    lr 1. Its control variates and the model are averaged with the FedAvg weights. Returns the per-round scores
    of the global model."""
    dev = _device()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    test_x = torch.as_tensor(test_features.to_numpy().astype('float32'), device=dev)
    test_y = test_labels.to_numpy().astype('int32')

    # Per-client local train split, fixed across rounds.
    local = []
    for X_i, y_i in datasets:
        x_tr, _, y_tr, _ = train_test_split(X_i, y_i, test_size=LOCAL_TEST_SIZE, random_state=42)
        local.append(_to_tensors(x_tr, y_tr, dev))
    # FedAvg weights: client sizes by default; agg_weights overrides them, e.g. a coreset representative weighted
    # by the record mass of the group it stands for.
    sizes = np.array([d[0].shape[0] for d in datasets], dtype=float)
    raw = sizes if agg_weights is None else np.asarray(agg_weights, dtype=float)
    weights = raw / raw.sum()

    global_model = new_model(seed).to(dev)
    if algo == 'scaffold_sgd':
        c_loc = [[torch.zeros_like(p) for p in global_model.parameters()] for _ in local]
        c_glob = [torch.zeros_like(p) for p in global_model.parameters()]
    scores = []
    for rnd in range(1, max_iterations + 1):
        lr = BASE_LR * (DECAY_RATE ** (rnd - 1)) if lr_decay else BASE_LR
        global_state = {k: v.clone() for k, v in global_model.state_dict().items()}
        w0 = [p.detach().clone() for p in global_model.parameters()]
        avg = {k: torch.zeros_like(v) for k, v in global_state.items()}
        new_c = []
        for i, (x, y) in enumerate(local):
            model = new_model(seed).to(dev)
            model.load_state_dict(global_state)
            prox = (mu, w0) if algo == 'fedprox' else None
            corr = [cg - ci for cg, ci in zip(c_glob, c_loc[i])] if algo == 'scaffold_sgd' else None
            if algo == 'scaffold_sgd':
                # SCAFFOLD as published: local SGD (lr = sgd_lr with the same per-round decay), option II
                # control variates c_i+ = c_i - c + (x - y_i) / (K eta_l), K = local steps
                eta = sgd_lr * (DECAY_RATE ** (rnd - 1) if lr_decay else 1.0)
                steps = _fit_epochs(model, torch.optim.SGD(model.parameters(), lr=eta), x, y, epochs, gen,
                                    correction=corr)
                new_c.append([ci - cg + (p0 - p.detach()) / (steps * eta)
                              for ci, cg, p0, p in zip(c_loc[i], c_glob, w0, model.parameters())])
            else:
                _fit_epochs(model, _adam(model.parameters(), lr), x, y, epochs, gen, prox=prox)
            for k, v in model.state_dict().items():
                avg[k] += weights[i] * v
        global_model.load_state_dict(avg)
        if algo == 'scaffold_sgd':
            c_loc = new_c
            c_glob = [sum(weights[i] * c_loc[i][j] for i in range(len(local))) for j in range(len(c_glob))]
        scores.append(_evaluate(global_model, test_x, test_y))
        print(f"  Round {rnd:02d} (LR: {lr:.7f}) -> Loss: {scores[-1][0]:.5f} | "
              f"Balanced Acc: {scores[-1][1] * 100:.2f}%")
    return scores


def run_exp(selected_client_indices, all_client_datasets, test_features, test_labels, epochs=1,
            max_iterations=30, aggregation_method='fedavg', lr_decay=True, agg_weights=None, mu=0.0, sgd_lr=0.1):
    """Trains the federation of the selected clients. aggregation_method: 'fedavg', 'fedprox' (with mu) or
    'scaffold_sgd' (with sgd_lr), see run_training. A single client trains centrally, whatever the method (no
    aggregation, hence no drift to correct)."""
    if aggregation_method not in ('fedavg', 'fedprox', 'scaffold_sgd'):
        raise NotImplementedError(aggregation_method)
    subset = [all_client_datasets[i] for i in selected_client_indices]
    if len(subset) == 1:
        print(f"  [!] Single client: centralized training for {max_iterations} epochs.")
        return run_centralized_training(subset[0], test_features, test_labels, max_iterations, lr_decay=lr_decay)
    return run_training(subset, test_features, test_labels, epochs, max_iterations, lr_decay=lr_decay,
                        agg_weights=agg_weights, algo=aggregation_method, mu=mu, sgd_lr=sgd_lr)


def facility_location_greedy(dist, k):
    """DivFL's selection (Balakrishnan et al. 2022, melodi-lab/divfl fedbase.greedy): greedily add the
    client that most decreases sum_i min_{j in S} dist[i, j] until |S| = k (lazy greedy in the
    reference code returns the same set)."""
    M = dist.shape[0]
    chosen, cur = [], None
    for _ in range(min(k, M)):
        cand = [j for j in range(M) if j not in chosen]
        cost = (dist[:, cand].sum(0) if cur is None else np.minimum(cur[:, None], dist[:, cand]).sum(0))
        j = cand[int(np.argmin(cost))]
        chosen.append(j)
        cur = dist[:, j].copy() if cur is None else np.minimum(cur, dist[:, j])
    return chosen


def run_per_round(datasets, test_features, test_labels, k, max_iterations, selection='divfl', epochs=1,
                  lr_decay=True, seed=SEED, sample_seed=0):
    """FedAvg with k clients selected per round among all clients.
    selection='divfl': DivFL as in melodi-lab/divfl (fedavg.py, clientsel_algo='submodular'): before
      round 1 every client sends the full gradient of its local loss at the initial model; every
      round the server picks k clients by facility location on the Euclidean distances between the
      clients' latest gradients; each selected client sends its full gradient at the received model
      (stale gradients are kept for the others) and trains; updates are averaged with client sizes.
    selection='random': k clients uniformly without replacement per round (DivFL's random baseline).
    The local solver, schedule and splits are those of run_training. Returns (scores, per-round
    selections)."""
    dev = _device()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(sample_seed)
    test_x = torch.as_tensor(test_features.to_numpy().astype('float32'), device=dev)
    test_y = test_labels.to_numpy().astype('int32')
    local = []
    for X_i, y_i in datasets:
        x_tr, _, y_tr, _ = train_test_split(X_i, y_i, test_size=LOCAL_TEST_SIZE, random_state=42)
        local.append(_to_tensors(x_tr, y_tr, dev))
    sizes = np.array([d[0].shape[0] for d in datasets], dtype=float)
    global_model = new_model(seed).to(dev)
    flat = lambda g: torch.cat([t.flatten() for t in g]).cpu().numpy()
    if selection == 'divfl':
        G = np.stack([flat(_full_gradient(global_model, x, y)) for x, y in local])
    scores, history = [], []
    for rnd in range(1, max_iterations + 1):
        lr = BASE_LR * (DECAY_RATE ** (rnd - 1)) if lr_decay else BASE_LR
        if selection == 'divfl':
            sq = (G ** 2).sum(1)
            dist = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * G @ G.T, 0))
            np.fill_diagonal(dist, 0)
            sel = facility_location_greedy(dist, k)
        else:
            sel = sorted(rng.choice(len(local), size=min(k, len(local)), replace=False).tolist())
        history.append([int(i) for i in sel])
        global_state = {kk: v.clone() for kk, v in global_model.state_dict().items()}
        avg = {kk: torch.zeros_like(v) for kk, v in global_state.items()}
        w = sizes[sel] / sizes[sel].sum()
        for wi, i in zip(w, sel):
            model = new_model(seed).to(dev)
            model.load_state_dict(global_state)
            if selection == 'divfl':
                G[i] = flat(_full_gradient(model, *local[i]))
            _fit_epochs(model, _adam(model.parameters(), lr), *local[i], epochs, gen)
            for kk, v in model.state_dict().items():
                avg[kk] += wi * v
        global_model.load_state_dict(avg)
        scores.append(_evaluate(global_model, test_x, test_y))
        print(f"  [{selection}] Round {rnd:02d} clients {sel} -> Balanced Acc: {scores[-1][1] * 100:.2f}%")
    return scores, history


def save_scores_to_csv(scores: list, filepath: Path):
    df = pd.DataFrame(scores, columns=['loss', 'balanced_accuracy', 'recall', 'precision'])
    df.index.name = 'round'
    df.index += 1
    df.to_csv(filepath, index=True)
    print(f">> Saved scores to {filepath}")


def run_pooled_training(datasets, test_features, test_labels, num_epochs, sample_weights=None, seed=SEED,
                        base_lr=None, lr_decay=True):
    """Centralized control for FL: trains one model on the UNION of exactly the
    records the clients would train on in run_training (per client:
    train_test_split(test_size=0.25, random_state=42), then the first
    floor(0.8 * n) rows, as Keras' validation_split=0.2 does), with the same
    model, Adam settings, batch size and per-epoch lr schedule as
    run_centralized_training (one optimizer for all epochs). FL round t and
    epoch t then see the same data and learning rate, so accuracy differences
    between the two isolate the federated aggregation (client drift).
    sample_weights: optional per-client weight applied to each of its records'
    loss (e.g. a coreset representative's group mass / its own size)."""
    base_lr = BASE_LR if base_lr is None else base_lr   # read at call time (BASE_LR can be overridden)
    dev = _device()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    xs, ys, ws = [], [], []
    for i, (X_i, y_i) in enumerate(datasets):
        x_tr, _, y_tr, _ = train_test_split(X_i, y_i, test_size=LOCAL_TEST_SIZE, random_state=42)
        n_tr = int(math.floor(len(x_tr) * (1.0 - VALIDATION_SPLIT)))
        xs.append(np.asarray(x_tr, dtype=np.float32)[:n_tr])
        ys.append(np.asarray(y_tr, dtype=np.float32)[:n_tr])
        ws.append(np.full(n_tr, 1.0 if sample_weights is None else float(sample_weights[i]), dtype=np.float32))
    x = torch.as_tensor(np.concatenate(xs), device=dev)
    y = torch.as_tensor(np.concatenate(ys), device=dev)
    w = torch.as_tensor(np.concatenate(ws), device=dev)
    w = w / w.mean()
    test_x = torch.as_tensor(test_features.to_numpy().astype('float32'), device=dev)
    test_y = test_labels.to_numpy().astype('int32')
    model = new_model(seed).to(dev)
    opt = _adam(model.parameters(), base_lr)
    scores = []
    for epoch in range(1, num_epochs + 1):
        lr = base_lr * (DECAY_RATE ** (epoch - 1)) if lr_decay else base_lr
        for g in opt.param_groups:
            g['lr'] = lr
        model.train()
        perm = torch.randperm(len(x), generator=gen).to(dev)
        for s in range(0, len(x), BATCH_SIZE):
            b = perm[s:s + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x[b]), y[b], reduction='none')
            (loss * w[b]).mean().backward()
            opt.step()
        scores.append(_evaluate(model, test_x, test_y))
    print(f"  Pooled ({len(x):,} records) epoch {num_epochs} -> Balanced Acc: {scores[-1][1] * 100:.2f}%")
    return scores
