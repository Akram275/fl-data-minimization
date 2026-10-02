"""
Random federations matched on client count AND record volume, the baseline of every selection: for a selection
S of k clients holding n_S records, each random federation has exactly k clients and n_S records within a
relative tolerance.
"""
import random
from typing import Dict, List, Tuple


def count_volume_matched(names: List[str], sizes: Dict[str, int], subset: List[str], trials: int, tol: float,
                         pool: int, rng: random.Random) -> Tuple[List[List[str]], List[float], int]:
    """Draws `pool` uniform random k-subsets (excluding S itself), keeps those whose volume is within `tol` of
    n_S, and samples `trials` of them. If fewer qualify, the closest ones in relative volume are used.
    Returns (subsets, their relative volume errors, number of candidates within tolerance)."""
    k, target = len(subset), sum(sizes[c] for c in subset)
    own = frozenset(subset)
    cands = {}
    for _ in range(pool):
        s = frozenset(rng.sample(names, k))
        if s != own:
            cands[s] = abs(sum(sizes[c] for c in s) - target) / target
    ok = [s for s, e in cands.items() if e <= tol]
    if len(ok) >= trials:
        chosen = rng.sample(sorted(ok, key=sorted), trials)
    else:
        chosen = sorted(cands, key=lambda s: (cands[s], sorted(s)))[:trials]
    return [sorted(s) for s in chosen], [cands[s] for s in chosen], len(ok)
