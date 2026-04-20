"""torch Datasets + Sampler.

  TabularDataset   — single tx per item (MLP, LightGBM).
  SequenceDataset  — windowed L-row slices (LSTM, Transformer, Mamba-3).

"""

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from src.data.schema import LABEL_COLUMNS, NUMERIC_FEATURES


class TabularDataset(Dataset):
    """one tx per item. used by MLP + LightGBM."""

    def __init__(self, features_path, labels_path):
        feat_table = pq.read_table(features_path)
        lab_table = pq.read_table(labels_path)

        num_cols = NUMERIC_FEATURES
        cat_cols = ["from_addr_hash", "to_addr_hash", "data_4bytes_hash"]

        self.x_num = np.column_stack(
            [feat_table.column(c).to_numpy().astype(np.float32) for c in num_cols]
        )
        self.x_cat = np.column_stack(
            [feat_table.column(c).to_numpy().astype(np.int64) for c in cat_cols]
        )
        self.labels = np.column_stack(
            [lab_table.column(c).to_numpy().astype(np.float32) for c in LABEL_COLUMNS]
        )

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        return {
            "x_num": torch.from_numpy(self.x_num[idx]),
            "x_cat": torch.from_numpy(self.x_cat[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
        }


class SequenceDataset(Dataset):
    """sliding L-row window over the chronological stream.

    each item dict:
        x_num   : (L, 11) float
        x_cat   : (L,  3) long  -- categorical hash indices
        delta_t : (L,)    float -- inter-arrival, sec (Mamba-3 phys adapter)
        labels  : (L,  3) float -- per-token
        mask    : (L,)    bool  -- False for the first warmup_tokens positions
    """

    def __init__(self, features_path, labels_path, seq_len=1024, warmup_tokens=64):
        feat_table = pq.read_table(features_path)
        lab_table = pq.read_table(labels_path)

        num_cols = NUMERIC_FEATURES
        cat_cols = ["from_addr_hash", "to_addr_hash", "data_4bytes_hash"]

        self.x_num = np.column_stack(
            [feat_table.column(c).to_numpy().astype(np.float32) for c in num_cols]
        )
        self.x_cat = np.column_stack(
            [feat_table.column(c).to_numpy().astype(np.int64) for c in cat_cols]
        )
        # timestamp_delta_s is feature index 7 in NUMERIC_FEATURES
        self.delta_t = self.x_num[:, 7].copy()
        self.labels = np.column_stack(
            [lab_table.column(c).to_numpy().astype(np.float32) for c in LABEL_COLUMNS]
        )
        # MEV labels for positive-aware sampling
        self.mev_labels = self.labels[:, 1]

        self.seq_len = seq_len
        self.warmup_tokens = warmup_tokens
        self.n_windows = max(1, (len(self.x_num) - seq_len) // seq_len + 1)

        # Pre-compute mask
        self._mask = np.zeros(seq_len, dtype=bool)
        self._mask[warmup_tokens:] = True

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len

        # Clamp to data bounds
        if end > len(self.x_num):
            start = len(self.x_num) - self.seq_len
            end = len(self.x_num)

        return {
            "x_num": torch.from_numpy(self.x_num[start:end].copy()),
            "x_cat": torch.from_numpy(self.x_cat[start:end].copy()),
            "delta_t": torch.from_numpy(self.delta_t[start:end].copy()),
            "labels": torch.from_numpy(self.labels[start:end].copy()),
            "mask": torch.from_numpy(self._mask.copy()),
        }

    def has_mev_positive(self, idx):
        """Check if window idx contains at least one MEV positive."""
        start = idx * self.seq_len
        end = min(start + self.seq_len, len(self.mev_labels))
        if end > len(self.mev_labels):
            start = len(self.mev_labels) - self.seq_len
            end = len(self.mev_labels)
        return self.mev_labels[start:end].any()


class PositiveAwareSampler(Sampler):
    """oversample MEV-positive windows. with prob p_pos pick a window
    containing >=1 victim; otherwise sample uniformly. window order
    is preserved internally."""

    def __init__(self, dataset: SequenceDataset, p_pos=0.3, seed=42):
        self.dataset = dataset
        self.p_pos = p_pos
        self.rng = np.random.RandomState(seed)
        self.n = len(dataset)

        # cache the indices of windows that contain a positive
        self.pos_indices = [
            i for i in range(self.n) if dataset.has_mev_positive(i)
        ]
        self.all_indices = list(range(self.n))

    def __iter__(self):
        indices = []
        for _ in range(self.n):
            if self.pos_indices and self.rng.rand() < self.p_pos:
                idx = self.rng.choice(self.pos_indices)
            else:
                idx = self.rng.randint(0, self.n)
            indices.append(idx)
        return iter(indices)

    def __len__(self):
        return self.n


def collate_fn(batch):
    """torch default doesn't know how to stack our dicts of mixed types."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[k] = torch.stack(vals)
        elif isinstance(vals[0], dict):
            result[k] = vals  # keep list of dicts for meta
        else:
            result[k] = vals
    return result
