"""delta t corruption for the physical-time hypothesis

we modify ONLY the delta_t fed to the mamba adapter,
modes
    shuffle: permute delta t within each window
    quantize: replace each delta t with the median of its training-set quartile bin
        (kills fine resolution, bins are fit on train so no test leakage)

both are eval-only
"""

import numpy as np
import torch


def shuffle_delta(delta_t, seed=42):
    """permute delta t within each (B, L) row."""
    rng = np.random.RandomState(seed)
    out = delta_t.clone()
    B, L = out.shape
    for b in range(B):
        out[b] = out[b, rng.permutation(L)]
    return out


def quantize_delta(delta_t, bin_edges, bin_medians):
    """snap each delta t to its bin's median value."""
    edges = torch.tensor(bin_edges, dtype=delta_t.dtype, device=delta_t.device)
    medians = torch.tensor(bin_medians, dtype=delta_t.dtype, device=delta_t.device)
    bins = torch.bucketize(delta_t, edges)   # 0..3
    return medians[bins]


def compute_quantile_bins(train_deltas):
    """3 boundaries + 4 bin medians from train delta t values."""
    train_deltas = np.asarray(train_deltas, dtype=np.float64)
    valid = train_deltas[train_deltas > 0]
    if len(valid) == 0:
        valid = train_deltas

    bin_edges = np.percentile(valid, [25, 50, 75]).astype(np.float32)

    bin_medians = np.zeros(4, dtype=np.float32)
    bins = np.digitize(valid, bin_edges)
    for i in range(4):
        mask = bins == i
        bin_medians[i] = np.median(valid[mask]) if mask.any() else np.median(valid)
    return bin_edges, bin_medians


def corrupt_batch(batch, mode, bin_edges=None, bin_medians=None, seed=42):
    """return a new batch dict with delta_t replaced; other keys passed through."""
    new = {k: v for k, v in batch.items()}

    if mode == "shuffle":
        new["delta_t"] = shuffle_delta(batch["delta_t"], seed=seed)
    elif mode == "quantize":
        assert bin_edges is not None and bin_medians is not None, "need bin_edges + bin_medians"
        new["delta_t"] = quantize_delta(batch["delta_t"], bin_edges, bin_medians)
    else:
        raise ValueError(f"unknown corruption mode: {mode}")
    return new
