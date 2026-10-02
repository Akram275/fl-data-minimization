"""
Non-IID client partitions: Dirichlet partitions over one variable (the task label for label skew, a metadata
attribute for attribute skew), with the directory names used for the results.
"""
from typing import List

import numpy as np

# Metadata column skewed by the attribute-skew partitions (image_metadata.ImageSplit.attrs): CelebA Male,
# FairFace race.
PARTITION_ATTR = {'celeba': 0, 'fairface': 1}


def partition_key(train, dataset: str, partition_on: str = 'attr') -> np.ndarray:
    """The variable the Dirichlet partition skews: the task label (label skew) or the dataset's metadata
    attribute (attribute skew)."""
    return train.labels if partition_on == 'label' else train.attrs[:, PARTITION_ATTR[dataset]]


def heterogeneity_dir(alpha: float, partition_on: str = 'attr') -> str:
    """Results sub-directory of a partition, e.g. 'high_heterogeneity_labelskew' for alpha = 0.2 and label skew."""
    d = 'high_heterogeneity' if alpha < 0.5 else 'moderate_heterogeneity' if alpha <= 1.0 else 'low_heterogeneity'
    return d + ('_labelskew' if partition_on == 'label' else '')


def create_dirichlet_partitions(partition_labels: np.ndarray, num_clients: int, alpha: float,
                                min_require_size: int = 10, max_attempts: int = 5000) -> List[List[int]]:
    """Dirichlet partition of the record indices over `num_clients` clients: for every value k of
    `partition_labels`, the records with that value are split by proportions ~ Dir(alpha). Draws are rejected
    until every client holds at least `min_require_size` records. Uses the global NumPy RNG (seed it with
    np.random.seed before calling), so a partition seed reproduces the same partition.

    Rejected draws only track the client sizes, and the index lists are built for the accepted draw, with the
    same RNG calls in the same order. This keeps the partitions identical while making low acceptance rates
    (about 4e-5 for CelebA, 100 clients, alpha = 0.2) affordable."""
    min_size = 0
    N = len(partition_labels)
    num_classes = len(np.unique(partition_labels))

    print(f"Generating Dirichlet partitions (alpha={alpha}) across {num_clients} clients...")
    attempts = 0
    while min_size < min_require_size:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"No Dirichlet partition with >= {min_require_size} records per client after {max_attempts} "
                f"draws ({N} records, {num_clients} clients, {num_classes} classes, alpha={alpha}). "
                f"Use fewer clients, more records, or a larger alpha.")
        sizes = np.zeros(num_clients, dtype=int)
        draws = []
        for k in range(num_classes):
            idx_k = np.where(partition_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions * (sizes < N / num_clients)
            if proportions.sum() == 0:
                proportions = np.repeat(1.0 / num_clients, num_clients)
            proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            sizes += np.diff(np.concatenate([[0], cuts, [len(idx_k)]]))
            draws.append((idx_k, cuts))
        min_size = sizes.min()
    idx_batch = [[] for _ in range(num_clients)]
    for idx_k, cuts in draws:
        for i, part in enumerate(np.split(idx_k, cuts)):
            idx_batch[i] += part.tolist()
    for i in range(num_clients):
        np.random.shuffle(idx_batch[i])
    return idx_batch
