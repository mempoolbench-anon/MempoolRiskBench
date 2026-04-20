"""day-by-day DEX P@100 / PR-AUC for the triage pipeline.

cuts the test split into 7 equal-size chunks and treats each as one day.
close enough — arrivals come in roughly evenly across the test week.

used to check how steady the numbers are from one day to the next.
"""

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

N_DAYS = 7
DATA_DIR = "data/processed/test"
RESULTS_DIR = "results"

RUNS = [
    ("MLP",            "mlp"),
    ("MLP no-cat",     "mlp_nocat"),
    ("Transformer",    "transformer"),
    ("Mamba-3 const",  "mamba3_const"),
    ("Mamba-3 phys",   "mamba3_phys"),
    ("LightGBM",       "lgbm"),
]


def precision_at_k(scores, labels, k):
    if k <= 0 or k > len(scores):
        return float("nan")
    idx = np.argsort(-scores)[:k]
    return float(labels[idx].sum() / k)


def main():
    dex_mask = pq.read_table(f"{DATA_DIR}/dex_mask.parquet").column(
        "is_dex_candidate").to_numpy().astype(bool)
    mev_full = pq.read_table(f"{DATA_DIR}/labels.parquet").column(
        "is_mev_victim").to_numpy().astype(np.float32)
    N = len(mev_full)
    day_edges = np.linspace(0, N, N_DAYS + 1).astype(int)

    print(f"test rows: {N:,}; chunk: {day_edges[1]-day_edges[0]:,} rows")
    print(f"DEX candidates: {int(dex_mask.sum()):,}; MEV victims on DEX: "
          f"{int((mev_full*dex_mask).sum()):,}\n")

    for label, run_id in RUNS:
        scores_full = pq.read_table(f"{RESULTS_DIR}/{run_id}/predictions.parquet").column(
            "mev_score").to_numpy().astype(np.float32)
        n = min(N, len(scores_full))
        print(f"=== {label} ({run_id}) ===")
        print(f"  {'day':>4s} {'n_dex':>7s} {'n_pos':>6s} {'P@100':>7s} {'P@50':>7s} {'DEX PR-AUC':>11s}")
        for d in range(N_DAYS):
            a, b = day_edges[d], min(day_edges[d + 1], n)
            mask = dex_mask[a:b]
            scores = scores_full[a:b][mask]
            labels = mev_full[a:b][mask]
            n_dex = int(mask.sum())
            n_pos = int(labels.sum())
            p100 = precision_at_k(scores, labels, min(100, n_dex))
            p50 = precision_at_k(scores, labels, min(50, n_dex))
            if n_pos >= 1 and len(np.unique(labels)) > 1:
                pr_auc = float(average_precision_score(labels, scores))
            else:
                pr_auc = float("nan")
            print(f"  {d+1:>4d} {n_dex:>7,d} {n_pos:>6,d} {p100:>7.3f} {p50:>7.3f} {pr_auc:>11.4f}")
        print()


if __name__ == "__main__":
    main()
