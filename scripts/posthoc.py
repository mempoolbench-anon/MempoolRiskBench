"""extra analysis on cached val predictions.

two passes:
  1) metrics in 4 mempool_pressure buckets (Q1-Q4)
  2) DEX-only MEV metrics

reads predictions that earlier training runs already wrote — doesn't
re-run any model.
"""

import json
import os

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.data.schema import LABEL_COLUMNS
from src.evaluation.metrics import precision_at_k, recall_at_k
from src.evaluation.stratification import compute_traffic_quartiles


def _load_predictions(run_id, results_dir="results"):
    # try parquet first (neural runs), fall back to NPZ (lgbm runs)
    pq_path = os.path.join(results_dir, run_id, "predictions.parquet")
    if os.path.exists(pq_path):
        t = pq.read_table(pq_path)
        return {
            "revert": t.column("revert_score").to_numpy().astype(np.float32),
            "mev":    t.column("mev_score").to_numpy().astype(np.float32),
            "drop":   t.column("drop_score").to_numpy().astype(np.float32),
        }
    npz_path = os.path.join(results_dir, run_id, "val_predictions.npz")
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        return {
            "revert": d["revert"].astype(np.float32),
            "mev":    d["mev"].astype(np.float32),
            "drop":   d["drop"].astype(np.float32),
        }
    raise FileNotFoundError(f"no predictions for {run_id}")


def _compute_metrics(preds, labels):
    out = {}

    # revert — skip the -1 rows
    valid = labels[:, 0] >= 0
    if valid.sum() >= 10:
        rev_s = preds["revert"][:len(labels)][valid]
        rev_l = labels[:, 0][valid]
        rev_b = (rev_s > 0.5).astype(int)
        out["revert_f1"] = float(f1_score(rev_l, rev_b, zero_division=0))
        if len(np.unique(rev_l)) > 1:
            out["revert_auc"] = float(roc_auc_score(rev_l, rev_s))

    # mev
    mev_s = preds["mev"][:len(labels)]
    mev_l = labels[:, 1]
    if mev_l.sum() >= 1 and len(np.unique(mev_l)) > 1:
        out["mev_pr_auc"] = float(average_precision_score(mev_l, mev_s))
    out["mev_p_at_100"] = precision_at_k(mev_s, mev_l, 100)
    out["mev_p_at_1000"] = precision_at_k(mev_s, mev_l, 1000)
    out["mev_recall_at_1000"] = recall_at_k(mev_s, mev_l, 1000)

    # drop
    drop_s = preds["drop"][:len(labels)]
    drop_l = labels[:, 2]
    drop_b = (drop_s > 0.5).astype(int)
    out["drop_f1"] = float(f1_score(drop_l, drop_b, zero_division=0))
    if len(np.unique(drop_l)) > 1:
        out["drop_auc"] = float(roc_auc_score(drop_l, drop_s))
    return out


def _mev_only(preds_mev, labels_mev):
    out = {}
    if labels_mev.sum() >= 1 and len(np.unique(labels_mev)) > 1:
        out["mev_pr_auc"] = float(average_precision_score(labels_mev, preds_mev))
    n = len(preds_mev)
    out["mev_p_at_100"] = precision_at_k(preds_mev, labels_mev, min(100, n))
    out["mev_p_at_1000"] = precision_at_k(preds_mev, labels_mev, min(1000, n))
    out["mev_recall_at_1000"] = recall_at_k(preds_mev, labels_mev, min(1000, n))
    return out


def main():
    data_dir = "data/processed"
    results_dir = "results"

    # label -> (results subdir, model family). change if your runs are named differently.
    models = {
        "LightGBM":         ("lgbm",          "tabular"),
        "LSTM":             ("lstm_val",      "sequence"),
        "Transformer":      ("transformer_val", "sequence"),
        "Mamba-3 const":    ("mamba3_const_val", "sequence"),
        "Mamba-3 phys":     ("mamba3_phys_val", "sequence"),
    }

    feat = pq.read_table(os.path.join(data_dir, "val", "features.parquet"))
    lab = pq.read_table(os.path.join(data_dir, "val", "labels.parquet"))
    dex_tbl = pq.read_table(os.path.join(data_dir, "val", "dex_mask.parquet"))

    pressure_full = feat.column("mempool_pressure").to_numpy().astype(np.float32)
    labels_full = np.column_stack(
        [lab.column(c).to_numpy().astype(np.float32) for c in LABEL_COLUMNS]
    )
    dex_mask_full = dex_tbl.column("is_dex_candidate").to_numpy().astype(bool)

    # ---- Q1-Q4 buckets by traffic ----
    print("=" * 70)
    print("Q1-Q4 stratification (val)")
    print("=" * 70)

    strat_results = {}
    for label, (run_id, _) in models.items():
        preds = _load_predictions(run_id, results_dir)
        n_pred = len(preds["revert"])

        # trim to whatever the model actually scored (sequence models
        # can drop the last few rows if the window doesn't fit)
        pressure = pressure_full[:n_pred]
        labels = labels_full[:n_pred]

        q_labels, q_bounds = compute_traffic_quartiles(pressure)

        per_q = {}
        for q in range(1, 5):
            mask = q_labels == q
            if mask.sum() < 10:
                per_q[f"Q{q}"] = None
                continue
            q_preds = {k: v[mask] for k, v in preds.items()}
            per_q[f"Q{q}"] = _compute_metrics(q_preds, labels[mask])

        strat_results[label] = {
            "stratified": per_q,
            "boundaries": q_bounds.tolist(),
            "quartile_counts": {f"Q{q}": int((q_labels == q).sum()) for q in range(1, 5)},
        }

        print(f"\n{label}  (n={n_pred:,})")
        print(f"  q-bounds: 25={q_bounds[0]:.0f}  50={q_bounds[1]:.0f}  75={q_bounds[2]:.0f}")
        for q in range(1, 5):
            m = per_q[f"Q{q}"]
            if m:
                ra = m.get("revert_auc", "N/A")
                mp = m.get("mev_pr_auc", "N/A")
                if isinstance(ra, float): ra = f"{ra:.4f}"
                if isinstance(mp, float): mp = f"{mp:.4f}"
                print(f"  Q{q}: rev_auc={ra}  mev_pr={mp}  n={(q_labels==q).sum():,}")

    out_dir = os.path.join(results_dir, "traffic_stratification")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(strat_results, f, indent=2, default=float)
    print(f"\n-> {out_dir}/metrics.json")

    # ---- DEX-only MEV ----
    print("\n" + "=" * 70)
    print("DEX-subset MEV (val)")
    print("=" * 70)

    dex_results = {}
    for label, (run_id, _) in models.items():
        preds = _load_predictions(run_id, results_dir)
        n_pred = len(preds["revert"])

        labels = labels_full[:n_pred]
        dex_mask = dex_mask_full[:n_pred]

        overall = _mev_only(preds["mev"], labels[:, 1])
        dex_preds = preds["mev"][dex_mask]
        dex_labels = labels[:, 1][dex_mask]
        dex = _mev_only(dex_preds, dex_labels)

        non_dex_mask = ~dex_mask
        non_dex = _mev_only(preds["mev"][non_dex_mask], labels[:, 1][non_dex_mask])

        dex_results[label] = {
            "overall": overall,
            "dex_subset": dex,
            "non_dex": non_dex,
            "dex_count": int(dex_mask.sum()),
            "dex_mev_count": int(dex_labels.sum()),
            "total_mev_count": int(labels[:, 1].sum()),
        }

        op = overall.get("mev_pr_auc", "N/A")
        dp = dex.get("mev_pr_auc", "N/A")
        if isinstance(op, float): op = f"{op:.4f}"
        if isinstance(dp, float): dp = f"{dp:.4f}"
        print(f"\n{label}")
        print(f"  overall pr={op}  p@100={overall.get('mev_p_at_100', 'N/A')}")
        print(f"  dex     pr={dp}  n={dex_mask.sum():,}  pos={int(dex_labels.sum()):,}")

    out_dir = os.path.join(results_dir, "dex_subset")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(dex_results, f, indent=2, default=float)
    print(f"\n-> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
