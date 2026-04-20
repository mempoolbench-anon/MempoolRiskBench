"""post-hoc filters: traffic-quartile stratification + DEX subset.

both operate on already-saved (preds, labels). nothing here touches the
model.
"""

import numpy as np

from src.data.schema import DEX_SELECTORS


def compute_traffic_quartiles(mempool_pressure):
    """Q1..Q4 labels from the eval-split mempool_pressure values."""
    p = np.asarray(mempool_pressure, dtype=np.float64)
    boundaries = np.percentile(p, [25, 50, 75])
    # digitize returns 0..3; we want 1..4
    labels = np.clip(np.digitize(p, boundaries) + 1, 1, 4)
    return labels, boundaries.astype(np.float32)


def filter_dex_subset(data_4bytes):
    """bool mask: True if the 4-byte selector is in DEX_SELECTORS."""
    dex_set = set(DEX_SELECTORS)
    if hasattr(data_4bytes, "tolist"):
        data_4bytes = data_4bytes.tolist()
    return np.array([s is not None and s in dex_set for s in data_4bytes], dtype=bool)


def stratified_metrics(preds, labels, quartile_labels, metric_fn):
    """run metric_fn separately on each Q1..Q4 slice."""
    out = {}
    for q in range(1, 5):
        mask = quartile_labels == q
        if mask.sum() < 10:
            out[q] = None
            continue
        try:
            out[q] = metric_fn(preds[mask], labels[mask])
        except (ValueError, ZeroDivisionError):
            out[q] = None
    return out
