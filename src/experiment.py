"""
Loading one client partition for the experiment scripts: the partition itself, the FL training arrays, the
shared test pool and the encoder outputs of every record.

  images   CelebA (label 'Smiling') or FairFace (label gender), 32x32 arrays for the CNN, Dirichlet partition
           (alpha) over the label (label skew) or a metadata attribute (attribute skew), frozen image encoders.
  tabular  a cache of src/data/acs_tabular.py format (ACSIncome, Sent140): Dirichlet label partition
           ('label'), one client per state or user ('states', 'users'), or random groups of users ('groups').

    parser = argparse.ArgumentParser(); add_partition_args(parser); ...; args = parser.parse_args()
    part = load_partition(args)          # Partition(names, client_indices, labels, ...)
"""
import argparse
from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np
import pandas as pd

import src.federated_learning.torch_fl as fl
from src.data.partitions import create_dirichlet_partitions, heterogeneity_dir, partition_key

IMAGE_ENCODERS = ['sd-vae', 'dinov2-small', 'clip-b32', 'resnet18']


class Partition(NamedTuple):
    names: List[str]                        # client names, in partition order
    client_indices: Dict[str, np.ndarray]   # client -> indices of its records among the client records
    labels: np.ndarray                      # task label of every client record
    attrs: np.ndarray                       # metadata attributes of every client record (evaluation only)
    attr_names: List[str]
    attr_dims: Dict[str, int]
    datasets: list                          # per client [X (DataFrame), y (Series)], the FL training data
    test_x: pd.DataFrame                    # shared test pool (no client holds it)
    test_y: pd.Series
    encoders: Dict[str, np.ndarray]         # encoder name -> encoder outputs of every client record
    root: Path                              # results sub-directory of the partition


def add_partition_args(p: argparse.ArgumentParser, encoders=IMAGE_ENCODERS):
    """Arguments shared by every experiment script (which partition, which encoders, tabular training)."""
    p.add_argument('kind', choices=['images', 'tabular'])
    p.add_argument('--dataset', default='celeba', choices=['celeba', 'fairface'], help="images")
    p.add_argument('--max-samples', type=int, default=0, help="images: first records only (0: all)")
    p.add_argument('--partition-on', choices=['attr', 'label'], default='label', help="images: skewed variable")
    p.add_argument('--encoders', nargs='+', default=list(encoders))
    p.add_argument('--task', default='ACSIncome', help="tabular: ACSIncome or Sent140")
    p.add_argument('--partition', choices=['states', 'users', 'groups', 'label'], default='label', help="tabular")
    p.add_argument('--tab-root', default="conv_results_tabular", help="tabular: results sub-directory")
    p.add_argument('--num-clients', type=int, default=100)
    p.add_argument('--alpha', type=float, default=0.2, help="Dirichlet concentration")
    p.add_argument('--partition-seed', type=int, default=42)
    p.add_argument('--partition-max-attempts', type=int, default=2000000)
    p.add_argument('--seed', type=int, default=42, help="k-medoids and random-feature seed")
    p.add_argument('--lr', type=float, default=5e-4, help="tabular: base learning rate")
    p.add_argument('--batch-size', type=int, default=32, help="tabular: batch size")
    p.add_argument('--pca-dim', type=int, default=32)


def load_partition(args) -> Partition:
    return load_images(args) if args.kind == 'images' else load_tabular(args)


def load_images(args) -> Partition:
    from src.data.image_metadata import DATASET_LOADERS
    from src.embedding.encoders import encode_cached
    train, _ = DATASET_LOADERS[args.dataset](max_samples=args.max_samples, n_public=2000)
    np.random.seed(args.partition_seed)
    parts = create_dirichlet_partitions(partition_key(train, args.dataset, args.partition_on), args.num_clients,
                                        args.alpha, max_attempts=args.partition_max_attempts)
    names = [f"client_{i:02d}" for i in range(len(parts))]
    client_indices = {c: np.array(p, dtype=int) for c, p in zip(names, parts)}
    if args.dataset == 'celeba':
        (X, y, _), (X_te, y_te, _) = fl.load_and_preprocess_celebA(target_attribute='Smiling', max_samples=args.max_samples)
    else:
        (X, y, _), (X_te, y_te, _) = fl.load_and_preprocess_fairface(max_samples=args.max_samples)
    if len(X) != len(train.paths) or not np.array_equal(y, train.labels):
        raise RuntimeError("FL loader rows don't line up with the metadata loader.")
    test_x, test_y = fl.generate_universal_testdata(X_te, y_te, val_ratio=0.2)
    datasets = [[pd.DataFrame(X[client_indices[c]]), pd.Series(y[client_indices[c]])] for c in names]
    root = (Path("conv_results_embedding") / args.dataset / heterogeneity_dir(args.alpha, args.partition_on)
            / f"seed_{args.partition_seed}")
    encoders = {e: encode_cached(train.paths, e, tag=f"{args.dataset}_train") for e in args.encoders}
    return Partition(names, client_indices, train.labels, train.attrs, train.attr_names, train.attr_dims, datasets,
                     test_x, test_y, encoders, root)


def load_tabular(args) -> Partition:
    from src.data.acs_tabular import attr_meta, load, load_embeddings
    d = load(args.task)
    attr_names, attr_dims = attr_meta(d)
    cli, test = np.where(d['split'] == 0)[0], np.where(d['split'] == 2)[0]
    y_all = d['y']
    if args.partition in ('states', 'users'):
        st = d['state'][cli]
        parts = [np.where(st == s)[0] for s in np.unique(st)]
        names = [f"{args.partition[:-1]}_{s:02d}" for s in np.unique(st)]
        tag = args.partition
    elif args.partition == 'groups':
        # Sent140: num_clients clients, each a random group of users (every user's tweets stay together)
        st = d['state'][cli]
        users = np.random.default_rng(args.partition_seed).permutation(np.unique(st))
        parts = [np.where(np.isin(st, g))[0] for g in np.array_split(users, args.num_clients)]
        names = [f"group_{i:02d}" for i in range(len(parts))]
        tag = f"groups_seed{args.partition_seed}"
    else:
        np.random.seed(args.partition_seed)
        parts = create_dirichlet_partitions(y_all[cli], args.num_clients, args.alpha,
                                            max_attempts=args.partition_max_attempts)
        names = [f"client_{i:02d}" for i in range(len(parts))]
        tag = f"label_seed{args.partition_seed}"
    client_indices = {c: np.asarray(p, dtype=int) for c, p in zip(names, parts)}
    labels, attrs = y_all[cli], d['attrs'][cli]
    # tabular training: the MLP, with the tabular batch size and learning rate (module-level settings of torch_fl)
    fl.BATCH_SIZE, fl.BASE_LR = args.batch_size, args.lr
    in_dim = d['X_model'].shape[1]
    fl.MODEL_FACTORY = lambda seed: fl.TabularMLP(in_dim, seed)
    Xc = d['X_model'][cli]
    datasets = [[pd.DataFrame(Xc[client_indices[c]]), pd.Series(labels[client_indices[c]])] for c in names]
    test_x, test_y = fl.generate_universal_testdata(d['X_model'][test], y_all[test], val_ratio=0.2)
    root = Path(args.tab_root) / args.task / tag
    encoders = {e: load_embeddings(args.task, e, d)[cli] for e in args.encoders}
    return Partition(names, client_indices, labels, attrs, attr_names, attr_dims, datasets, test_x, test_y,
                     encoders, root)
