"""
Method-agnostic description of a client subset, stored with every selection: its size and how far its pooled
label and metadata distributions are from the population's (total variation distance).
"""
from typing import Dict, List

import numpy as np


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p / max(p.sum(), 1e-12) - q / max(q.sum(), 1e-12)).sum())


class SubsetEvaluator:
    def __init__(self, client_indices: Dict[str, np.ndarray], labels: np.ndarray, attrs: np.ndarray,
                 attr_names: List[str], attr_dims: Dict[str, int], n_labels: int):
        self.clients = list(client_indices.keys())
        self.sizes = {c: len(client_indices[c]) for c in self.clients}
        self.total = sum(self.sizes.values())
        self.label_counts = {c: np.bincount(labels[client_indices[c]], minlength=n_labels).astype(float)
                             for c in self.clients}
        self.attr_counts = {c: [np.bincount(attrs[client_indices[c], j], minlength=attr_dims[a]).astype(float)
                                for j, a in enumerate(attr_names)] for c in self.clients}
        self.label_U = np.sum(list(self.label_counts.values()), axis=0)
        self.attr_U = [np.sum([self.attr_counts[c][j] for c in self.clients], axis=0)
                       for j in range(len(attr_names))]

    def evaluate(self, subset: List[str]) -> Dict[str, float]:
        """n_clients, records, record_frac, label_tv (TV between the subset's and the population's label
        distributions) and attr_tv (mean TV over the metadata attributes)."""
        if not subset:
            return {'n_clients': 0, 'records': 0, 'record_frac': 0.0}
        n_rec = sum(self.sizes[c] for c in subset)
        label_S = np.sum([self.label_counts[c] for c in subset], axis=0)
        attr_tvs = [_tv(np.sum([self.attr_counts[c][j] for c in subset], axis=0), self.attr_U[j])
                    for j in range(len(self.attr_U))]
        return {'n_clients': len(subset), 'records': n_rec, 'record_frac': n_rec / self.total,
                'label_tv': _tv(label_S, self.label_U), 'attr_tv': float(np.mean(attr_tvs))}
