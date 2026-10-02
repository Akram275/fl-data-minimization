"""
Image-dataset loaders that return file PATHS, labels and metadata attributes, without decoding pixels. The
frozen encoders need full-resolution images (unlike the 32x32 arrays the FL loaders of
src/federated_learning/torch_fl.py build), and encodings are cached per path list, so only paths are needed.

Row order and filtering match torch_fl.load_and_preprocess_celebA / load_and_preprocess_fairface (same CSV order,
same max_samples slicing, same skip-if-missing rule), so record i here is record i there and a client partition
built on these indices is used verbatim for FL training.

Each loader also returns a disjoint split of n_public records (`public`), removed from the client records. The
protocol uses no server data; the split is kept so that the client records are those of the paper's runs.
"""
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class ImageSplit:
    paths: List[str]
    labels: np.ndarray            # (N,) int, the FL task label
    attrs: np.ndarray             # (N, A) int, metadata attributes (for evaluation)
    attr_names: List[str]
    attr_dims: Dict[str, int]


def _existing(df: pd.DataFrame, path_fn) -> pd.DataFrame:
    mask = [os.path.exists(path_fn(r)) for r in df.itertuples(index=False)]
    return df[np.array(mask, dtype=bool)] if len(df) else df


def load_celeba_metadata(data_dir: str = "data/celeba-dataset", target_attribute: str = 'Smiling',
                         max_samples: int = None, n_public: int = 2000):
    """Train split (partition 0) for clients, validation split (partition 1)
    as the held-out split - the FL test set (partition 2) is left untouched."""
    img_dir = os.path.join(data_dir, "img_align_celeba", "img_align_celeba")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(data_dir, "img_align_celeba")

    df = pd.merge(pd.read_csv(os.path.join(data_dir, "list_attr_celeba.csv")),
                  pd.read_csv(os.path.join(data_dir, "list_eval_partition.csv")), on='image_id')
    attr_names = ['Male', 'Young', 'Eyeglasses']
    for col in [target_attribute] + attr_names:
        df[col] = df[col].replace(-1, 0)

    train_df = df[df['partition'] == 0]
    if max_samples:
        train_df = train_df.iloc[:max_samples]
    public_df = df[df['partition'] == 1].iloc[:n_public]

    def build(split_df):
        split_df = _existing(split_df, lambda r: os.path.join(img_dir, r.image_id))
        return ImageSplit(
            paths=[os.path.join(img_dir, f) for f in split_df['image_id']],
            labels=split_df[target_attribute].values.astype(int),
            attrs=split_df[attr_names].values.astype(int),
            attr_names=attr_names,
            attr_dims={a: 2 for a in attr_names},
        )

    return build(train_df), build(public_df)


FAIRFACE_AGE = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
FAIRFACE_RACE = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
FAIRFACE_GENDER = ['Female', 'Male']  # same order as fairface_fl.GENDER_CATEGORIES


def load_fairface_metadata(data_dir: str = "data/FairFace", max_samples: int = None, n_public: int = 2000):
    """Train rows [0, max_samples) for clients (matching fairface_fl's slice);
    held-out split is drawn from the END of the train CSV, disjoint from the
    clients' rows. The val split stays the FL test set."""
    train_df = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
    client_df = train_df.iloc[:max_samples] if max_samples else train_df
    public_start = max(len(client_df), len(train_df) - n_public)
    public_df = train_df.iloc[public_start:]
    if len(public_df) < n_public:
        print(f"  [Warning] only {len(public_df)} public FairFace rows left after the client slice.")

    attr_names = ['Age', 'Race']

    def build(split_df):
        split_df = _existing(split_df, lambda r: os.path.join(data_dir, r.file))
        return ImageSplit(
            paths=[os.path.join(data_dir, f) for f in split_df['file']],
            labels=np.array([FAIRFACE_GENDER.index(g) for g in split_df['gender']], dtype=int),
            attrs=np.stack([
                [FAIRFACE_AGE.index(a) for a in split_df['age']],
                [FAIRFACE_RACE.index(r) for r in split_df['race']],
            ], axis=1).astype(int),
            attr_names=attr_names,
            attr_dims={'Age': len(FAIRFACE_AGE), 'Race': len(FAIRFACE_RACE)},
        )

    return build(client_df), build(public_df)


DATASET_LOADERS = {
    'celeba': load_celeba_metadata,
    'fairface': load_fairface_metadata,
}
