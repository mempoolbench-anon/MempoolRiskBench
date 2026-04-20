"""Rev AUC + MEV PR-AUC in 5 mempool-pressure buckets.

sorts test txs into 5 equal-size buckets by mempool_pressure, runs each
non-linear model on each bucket, and prints a python dict you can paste
straight into figures/gen_fig2_density.py.

skips heuristic / logreg — just the models that actually converge.
"""

import os

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_DIR = "data/processed"
RESULTS_DIR = "results"

RUNS = [
    ("LightGBM",         "lgbm"),
    ("MLP",              "mlp"),
    ("LSTM",             "lstm"),
    ("Transformer",      "transformer"),
    ("Mamba-3 (const.)", "mamba3_const"),
    ("Mamba-3 (phys.)",  "mamba3_phys"),
]


def main():
    feat_path = os.path.join(DATA_DIR, "test", "features.parquet")
    lab_path = os.path.join(DATA_DIR, "test", "labels.parquet")

    pressure = pq.read_table(feat_path).column("mempool_pressure").to_numpy().astype(np.float32)
    lab = pq.read_table(lab_path)
    rev_labels_full = lab.column("is_reverted").to_numpy().astype(np.float32)
    mev_labels_full = lab.column("is_mev_victim").to_numpy().astype(np.float32)

    # 5 equal-size buckets; q_idx in {0..4} maps to Q1..Q5
    q_edges = np.quantile(pressure, [0.2, 0.4, 0.6, 0.8])
    q_idx_full = np.digitize(pressure, q_edges)

    print(f"# quintile edges (mempool_pressure): {q_edges.tolist()}")
    for q in range(5):
        n = int((q_idx_full == q).sum())
        rev_q = rev_labels_full[q_idx_full == q]
        n_rev = int(rev_q[rev_q >= 0].sum())
        n_mev = int(mev_labels_full[q_idx_full == q].sum())
        print(f"# Q{q+1}: {n:,} txs, {n_rev:,} revert positives, {n_mev:,} MEV positives")
    print()

    print("DATA = {")
    for label, run_id in RUNS:
        pred = pq.read_table(os.path.join(RESULTS_DIR, run_id, "predictions.parquet"))
        rev_scores = pred.column("revert_score").to_numpy().astype(np.float32)
        mev_scores = pred.column("mev_score").to_numpy().astype(np.float32)

        n = min(len(pressure), len(rev_scores), len(mev_scores))
        q_idx = q_idx_full[:n]

        rev_per, mev_per = [], []
        for q in range(5):
            mask = q_idx == q

            # revert: skip the -1s, and we need both 0s and 1s to get an AUC
            rev_q = rev_labels_full[:n][mask]
            rs_q = rev_scores[mask]
            valid = rev_q >= 0
            if valid.sum() >= 10 and len(np.unique(rev_q[valid])) > 1:
                rev_per.append(float(roc_auc_score(rev_q[valid], rs_q[valid])))
            else:
                rev_per.append(0.5)

            mev_q = mev_labels_full[:n][mask]
            ms_q = mev_scores[mask]
            if mev_q.sum() >= 1 and len(np.unique(mev_q)) > 1:
                mev_per.append(float(average_precision_score(mev_q, ms_q)))
            else:
                mev_per.append(0.0)

        rev_str = "[" + ", ".join(f"{v:.3f}" for v in rev_per) + "]"
        mev_str = "[" + ", ".join(f"{v:.3f}" for v in mev_per) + "]"
        print(f'    "{label}": {{"rev": {rev_str},')
        print(f'    {" "*(len(label)+4)}"mev": {mev_str}}},')
    print("}")


if __name__ == "__main__":
    main()
