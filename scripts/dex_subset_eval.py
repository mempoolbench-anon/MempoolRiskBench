"""DEX-only MEV numbers, pulled from cached test predictions.

for each run: load predictions.parquet + dex_mask.parquet + labels.parquet,
keep only the DEX rows, and drop a `dex_subset` block into metrics.json.

same numbers as run_eval.py --subset dex, just without re-running the model.
"""

import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

DATA_DIR = "data/processed"
RESULTS_DIR = "results"

# 11 runs: logreg + 3 lgbm + 2 mlp + 4 sequence + 1 extra mamba no-cat
RUN_IDS = [
    "logreg",
    "lgbm", "lgbm_noaddr", "lgbm_nocat",
    "mlp", "mlp_nocat",
    "lstm", "transformer",
    "mamba3_const", "mamba3_phys", "mamba3_phys_nocat",
]


def precision_at_k(scores, labels, k):
    if k <= 0 or k > len(scores):
        return 0.0
    idx = np.argsort(-scores)[:k]
    return float(labels[idx].sum() / k)


def recall_at_k(scores, labels, k):
    total_pos = float(labels.sum())
    if total_pos == 0 or k <= 0:
        return 0.0
    idx = np.argsort(-scores)[:k]
    return float(labels[idx].sum() / total_pos)


def _mev_pack(scores, labels, label_count):
    """same MEV numbers as run_eval.py:_mev_pack."""
    out = {}
    if label_count >= 1 and len(np.unique(labels)) > 1:
        out["mev_pr_auc"] = float(average_precision_score(labels, scores))
    n = len(scores)
    out["mev_p_at_100"] = precision_at_k(scores, labels, min(100, n))
    out["mev_p_at_1000"] = precision_at_k(scores, labels, min(1000, n))
    out["mev_recall_at_1000"] = recall_at_k(scores, labels, min(1000, n))
    return out


def main():
    split_dir = os.path.join(DATA_DIR, "test")
    dex_mask = pq.read_table(os.path.join(split_dir, "dex_mask.parquet")).column(
        "is_dex_candidate").to_numpy().astype(bool)
    mev_labels_full = pq.read_table(os.path.join(split_dir, "labels.parquet")).column(
        "is_mev_victim").to_numpy().astype(np.float32)

    print(f"test split: {len(dex_mask):,} txs; DEX: {int(dex_mask.sum()):,} "
          f"({100*dex_mask.mean():.3f}%); MEV positives: {int(mev_labels_full.sum()):,}")
    print(f"MEV in DEX: {int((mev_labels_full * dex_mask).sum()):,} "
          f"({100*mev_labels_full[dex_mask].mean():.3f}% of DEX vs "
          f"{100*mev_labels_full.mean():.3f}% base rate = "
          f"{mev_labels_full[dex_mask].mean() / mev_labels_full.mean():.1f}x lift)\n")

    header = (f"{'model':22s} {'overall_prauc':>14s} {'dex_prauc':>11s} "
              f"{'dex_p100':>10s} {'dex_p1000':>11s} {'dex_recall1k':>14s} {'prauc_lift':>12s}")
    print(header)
    print("-" * len(header))

    for run_id in RUN_IDS:
        pred_path = os.path.join(RESULTS_DIR, run_id, "predictions.parquet")
        metrics_path = os.path.join(RESULTS_DIR, run_id, "metrics.json")
        if not (os.path.exists(pred_path) and os.path.exists(metrics_path)):
            print(f"{run_id}: missing predictions.parquet or metrics.json", file=sys.stderr)
            continue

        mev_scores_full = pq.read_table(pred_path).column(
            "mev_score").to_numpy().astype(np.float32)

        # sequence models sometimes drop the last few rows -> trim to min length
        n = min(len(mev_scores_full), len(dex_mask), len(mev_labels_full))
        dex = dex_mask[:n]
        scores = mev_scores_full[:n]
        labels = mev_labels_full[:n]

        overall  = _mev_pack(scores,        labels,        int(labels.sum()))
        dex_only = _mev_pack(scores[dex],   labels[dex],   int(labels[dex].sum()))
        non_dex  = _mev_pack(scores[~dex],  labels[~dex],  int(labels[~dex].sum()))

        dex_subset = {
            "overall": overall,
            "dex_only": dex_only,
            "non_dex": non_dex,
            "n_total": int(n),
            "n_dex": int(dex.sum()),
            "n_dex_pos": int(labels[dex].sum()),
            "n_non_dex_pos": int(labels[~dex].sum()),
        }

        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["dex_subset"] = dex_subset
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=float)

        lift = (dex_only.get("mev_pr_auc", 0.0) /
                overall["mev_pr_auc"]) if overall.get("mev_pr_auc") else 0.0
        print(f"{run_id:22s} "
              f"{overall.get('mev_pr_auc', 0):14.4f} "
              f"{dex_only.get('mev_pr_auc', 0):11.4f} "
              f"{dex_only['mev_p_at_100']:10.3f} "
              f"{dex_only['mev_p_at_1000']:11.3f} "
              f"{dex_only['mev_recall_at_1000']:14.3f} "
              f"{lift:11.2f}x")


if __name__ == "__main__":
    main()
