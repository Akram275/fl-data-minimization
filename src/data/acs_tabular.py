"""
ACS (folktables, 2018 1-Year person survey, all 50 states), cached as one array file per task with

  - a test split (per state, stratified on the label, 20%), from which the FL test pool is drawn,
  - a held-out slice (per state, 2% of the remaining records) on which the feature scaler is fitted; it belongs to
    no client,
  - the client records (everything else), with their state for the state-level clients.

Model inputs follow acs_preprocessing (one-hot categoricals, standardized numericals, high-cardinality OCCP/POBP
dropped). Evaluation attributes: SEX, RAC1P and binned AGEP. The same cache format holds Sent140
(src/data/sent140.py), so the tabular drivers read both through load() and load_embeddings().

Usage (repo root, after downloading the 2018 person files into data/ with folktables):
  python -m src.data.acs_tabular --task ACSIncome
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSMobility, ACSPublicCoverage, ACSTravelTime
from sklearn.model_selection import train_test_split

from src.data.acs_preprocessing import TASK_FEATURES, GLOBAL_CATEGORIES, preprocess_acs_data_with_scaler

TASK_REGISTRY = {'ACSIncome': ACSIncome, 'ACSEmployment': ACSEmployment, 'ACSPublicCoverage': ACSPublicCoverage,
                 'ACSMobility': ACSMobility, 'ACSTravelTime': ACSTravelTime}

ALL_STATES = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY',
              'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND',
              'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
AGE_EDGES = np.array([25, 50, 75])
ATTR_NAMES = ['SEX', 'RAC1P', 'AGE']
ATTR_DIMS = {'SEX': 2, 'RAC1P': 9, 'AGE': 4}
CACHE = Path("data/acs_tabular")


def cache_path(task: str) -> Path:
    return CACHE / f"{task}_2018.npz"


def build(task: str, test_frac: float = 0.2, public_frac: float = 0.02, seed: int = 42) -> Path:
    src = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir='data')
    feats = TASK_FEATURES[task]
    parts = []
    for si, st in enumerate(ALL_STATES):
        df = src.get_data(states=[st], download=False)
        X, y, _ = TASK_REGISTRY[task].df_to_pandas(df)
        X = X[feats].copy()
        y = y.iloc[:, 0].astype(int).values
        idx = np.arange(len(X))
        rest, test = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=y)
        rest, public = train_test_split(rest, test_size=public_frac, random_state=seed + 1, stratify=y[rest])
        split = np.zeros(len(X), dtype=np.int8)           # 0 client, 1 public, 2 test
        split[public], split[test] = 1, 2
        X['_state'], X['_y'], X['_split'] = si, y, split
        parts.append(X)
        print(f"  {st}: {len(X):,} records")
    df = pd.concat(parts, ignore_index=True)

    raw = df[feats]
    _, scaler = preprocess_acs_data_with_scaler(raw[df._split.values == 1])
    X_model, _ = preprocess_acs_data_with_scaler(raw, scaler=scaler)
    X_model = X_model.astype(np.float32)

    cat_idx = np.array([j for j, f in enumerate(feats) if f in GLOBAL_CATEGORIES], dtype=int)
    attrs = np.stack([df.SEX.values.astype(int) - 1, df.RAC1P.values.astype(int) - 1,
                      np.digitize(df.AGEP.values, AGE_EDGES)], axis=1).astype(int)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = cache_path(task)
    np.savez(out, X_model=X_model.values, model_columns=np.array(X_model.columns), X_raw=raw.values.astype(np.float32),
             raw_columns=np.array(feats), cat_idx=cat_idx, y=df._y.values.astype(int), state=df._state.values.astype(int),
             split=df._split.values, attrs=attrs, attr_names=np.array(ATTR_NAMES))
    print(f">> {out}: {len(df):,} records ({(df._split == 0).sum():,} client, {(df._split == 1).sum():,} public, "
          f"{(df._split == 2).sum():,} test), {X_model.shape[1]} model inputs")
    return out


def load(task: str) -> dict:
    d = np.load(cache_path(task), allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='ACSIncome', choices=list(TASK_FEATURES))
    build(p.parse_args().task)


# --- shared by the tabular and text drivers (any cache in this format) ---


def attr_meta(d: dict):
    """Evaluation attribute names and sizes, read from the cache (for ACS:
    SEX 2, RAC1P 9, AGE 4, i.e. ATTR_NAMES / ATTR_DIMS)."""
    names = [str(n) for n in d['attr_names']]
    return names, {n: int(d['attrs'][:, j].max()) + 1 for j, n in enumerate(names)}


def load_embeddings(task: str, encoder: str, d: dict) -> np.ndarray:
    """Per-row embeddings aligned with the cache rows: 'raw' is the model inputs (ACSIncome), any other encoder
    is read from data/embeddings/<task>_<encoder>{.npy,_rows.npy} (e.g. Sent140_minilm, written by
    src/data/sent140.py). Rows without an embedding are NaN and are never used."""
    if encoder == 'raw':
        return d['X_model']
    stem = f"data/embeddings/{task}_{encoder}"
    Z, rows = np.load(stem + ".npy"), np.load(stem + "_rows.npy")
    out = np.full((len(d['y']), Z.shape[1]), np.nan, dtype=np.float32)
    out[rows] = Z
    return out
