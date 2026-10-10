"""
DDEM: the pre-training client selection of Xu, Liu, Li, Feng and Qi, "Pretraining Client Selection
Algorithm Based on a Data Distribution Evaluation Model in Federated Learning", IEEE Access 2024
(doi:10.1109/ACCESS.2024.3396279). No public code; implemented from the paper's Eqs. 1-3 and
Algorithm 1.

The server collects the global class counts N_c (N = sum_c N_c, E clients, C classes) and defines,
for client e with n_e^c records of class c,
    r(e, c)          = min(n_e^c / (N_c / E), 1)                                   (Eq. 1)
    phi_c_rareness   = 1 - (U - N_c) / (U + N_c),     U = N / C                     (Eq. 2)
    phi(n_e^c)       = int_0^{r(e,c)} (1 - x)^(b * phi_c_rareness) dx               (Eq. 3)
                     = (1 - (1 - r)^(a + 1)) / (a + 1),    a = b * phi_c_rareness
    s_e              = sum_c phi(n_e^c)
and selects the K clients with the largest s_e (Algorithm 1). Selected clients then train with
plain FedAvg.

Choices the paper leaves open (documented here):
  - b = 3, the value shown in the paper's Fig. 2 (the text only requires b > 0).
  - Eq. 2 is printed with square brackets, read literally (no absolute value):
    phi_c_rareness = 2 N_c / (U + N_c), which is < 1 for rare classes, so that "for a given
    volume of data, the rarer the data the higher the evaluation" (Sec. III.A.2), as the paper states.
  - K is an input; we set it to the drift-aware coreset's k on the same partition.
  - The payment step of Algorithm 1 is irrelevant here and omitted.
"""
from typing import Dict, List

import numpy as np


def ddem_scores(class_counts: np.ndarray, b: float = 3.0) -> np.ndarray:
    """class_counts: (E, C) array of n_e^c. Returns s_e for every client."""
    n = np.asarray(class_counts, dtype=float)
    E, C = n.shape
    Nc = n.sum(0)
    U = Nc.sum() / C
    rare = 1.0 - (U - Nc) / (U + Nc)                        # Eq. 2
    a = b * rare
    r = np.minimum(n / np.maximum(Nc / E, 1e-12), 1.0)      # Eq. 1
    phi = (1.0 - (1.0 - r) ** (a + 1.0)) / (a + 1.0)        # Eq. 3, closed form
    return phi.sum(1)


def ddem_select(client_indices: Dict[str, np.ndarray], names: List[str], labels: np.ndarray, n_labels: int,
                k: int, b: float = 3.0) -> List[str]:
    counts = np.stack([np.bincount(labels[client_indices[c]], minlength=n_labels) for c in names])
    s = ddem_scores(counts, b)
    order = np.argsort(-s, kind='stable')
    return [names[i] for i in order[:k]]
