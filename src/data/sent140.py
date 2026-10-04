"""
Sentiment140 (Go et al. 2009; the LEAF Sent140 FL benchmark) with Twitter users as natural clients, cached in
the same format as src/data/acs_tabular.py so that the tabular drivers read it unchanged.

  - users with >= --min-tweets tweets (after removing duplicate (user, text)),
  - split BY USER: 2% of users are held out (they belong to no client), 10% are the TEST population (unseen
    users), the rest are CLIENT users,
  - label: positive sentiment (raw label 4) vs negative (0),
  - evaluation attributes from the timestamp: time of day (4 buckets),
    weekend, month (Apr/May/Jun 2009),
  - embeddings of every tweet with frozen sentence encoders (mean pooling, L2
    normalization, as sentence-transformers does), saved to
    data/embeddings/Sent140_<encoder>.npy (+ _rows.npy = all rows); X_model,
    the FL model's input, is the MiniLM embedding.

Usage (repo root):  python -m src.data.sent140 build
                    python -m src.data.sent140 embed --encoder minilm
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# The encoders are PyTorch-only: keep transformers from importing TensorFlow when it happens to be installed.
os.environ.setdefault("USE_TF", "0")

RAW = Path("data/sentiment140/training.1600000.processed.noemoticon.csv")
CACHE = Path("data/acs_tabular/Sent140_2018.npz")     # same location/naming scheme as acs_tabular.load
TEXTS = Path("data/sentiment140/texts.npy")
ENCODERS = {'minilm': "sentence-transformers/all-MiniLM-L6-v2", 'mpnet': "sentence-transformers/all-mpnet-base-v2"}
ATTR_NAMES = ['HOUR', 'WEEKEND', 'MONTH']


def build(min_tweets: int = 20, public_frac: float = 0.02, test_frac: float = 0.10, seed: int = 42):
    df = pd.read_csv(RAW, encoding='latin-1', header=None, names=['y', 'id', 'date', 'query', 'user', 'text'])
    df = df.drop_duplicates(['user', 'text'])
    counts = df.user.value_counts()
    df = df[df.user.isin(counts[counts >= min_tweets].index)].reset_index(drop=True)
    users = np.array(sorted(df.user.unique()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(users))
    n_pub, n_test = int(round(public_frac * len(users))), int(round(test_frac * len(users)))
    role = np.zeros(len(users), dtype=np.int8)                 # 0 client, 1 public, 2 test
    role[perm[:n_pub]], role[perm[n_pub:n_pub + n_test]] = 1, 2
    uid = pd.Series(np.arange(len(users)), index=users)[df.user].values
    ts = pd.to_datetime(df.date.str.replace(' PDT', '', regex=False), format='%a %b %d %H:%M:%S %Y')
    attrs = np.stack([ts.dt.hour.values // 6, (ts.dt.dayofweek.values >= 5).astype(int),
                      np.clip(ts.dt.month.values - 4, 0, 2)], axis=1).astype(int)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(TEXTS, df.text.values.astype(str))
    np.savez(CACHE, y=(df.y.values == 4).astype(int), state=uid, split=role[uid], attrs=attrs,
             attr_names=np.array(ATTR_NAMES), user_names=users, X_model=np.zeros((len(df), 1), np.float32))
    print(f">> {CACHE}: {len(df):,} tweets, {len(users):,} users ({(role == 0).sum():,} clients, "
          f"{(role == 1).sum():,} public, {(role == 2).sum():,} test); positive rate {(df.y == 4).mean():.3f}")


@torch.no_grad()
def embed(encoder: str, batch_size: int = 512):
    from transformers import AutoModel, AutoTokenizer
    name = ENCODERS[encoder]
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval().cuda()
    texts = list(np.load(TEXTS, allow_pickle=True))
    out = []
    for s in range(0, len(texts), batch_size):
        b = tok(texts[s:s + batch_size], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
        h = model(**b).last_hidden_state
        m = b['attention_mask'].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(e, dim=1).float().cpu().numpy())
        if (s // batch_size) % 50 == 0:
            print(f"  [{encoder}] {s + len(b['input_ids']):,}/{len(texts):,}", flush=True)
    Z = np.concatenate(out)
    path = Path(f"data/embeddings/Sent140_{encoder}.npy")
    np.save(path, Z)
    np.save(path.with_name(path.stem + "_rows.npy"), np.arange(len(Z)))
    if encoder == 'minilm':                          # the FL model's input
        d = dict(np.load(CACHE, allow_pickle=True))
        d['X_model'] = Z.astype(np.float32)
        np.savez(CACHE, **d)
    print(f">> {path}: {Z.shape}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['build', 'embed'])
    p.add_argument('--encoder', choices=list(ENCODERS), default='minilm')
    p.add_argument('--min-tweets', type=int, default=20)
    a = p.parse_args()
    build(a.min_tweets) if a.step == 'build' else embed(a.encoder)
