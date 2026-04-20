"""F1 threshold calc for LightGBM + LogReg.

this script sweeps val scores (sort-and-scan, O(nlogn)), picks F1-argmax,
applies the same threshold to test, and updates metrics.json.

only touches F1 fields
"""

import json
import os

import numpy as np
import pyarrow.parquet as pq

VAL_LABELS_PATH = "data/processed/val/labels.parquet"
TEST_LABELS_PATH = "data/processed/test/labels.parquet"
LGBM_RUNS = ["lgbm", "lgbm_noaddr", "lgbm_nocat", "logreg"]
TASKS = [("revert", "is_reverted"), ("drop", "is_dropped")]


def best_f1_threshold(scores, labels):
    """O(n log n) sort-and-scan for the F1-argmax threshold."""
    valid = labels >= 0
    if valid.sum() < 10:
        return 0.5, 0.0
    scores = scores[valid]
    labels = labels[valid].astype(np.int32)
    P = labels.sum()
    if P == 0:
        return 0.5, 0.0

    # descending sort. at threshold = scores[k], predicted-positive = top k+1.
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    scores_sorted = scores[order]
    tp = np.cumsum(labels_sorted)
    pred_pos = np.arange(1, len(scores_sorted) + 1)
    prec = tp / pred_pos
    rec = tp / P
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    k = int(np.argmax(f1))
    return float(scores_sorted[k]), float(f1[k])


def test_f1_at_threshold(scores, labels, threshold):
    valid = labels >= 0
    if valid.sum() < 10:
        return 0.0
    s = scores[valid]
    y = labels[valid].astype(np.int32)
    preds = (s >= threshold).astype(np.int32)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp); r = tp / (tp + fn)
    return float(2 * p * r / (p + r))


def main():
    val_labels = pq.read_table(VAL_LABELS_PATH)
    test_labels = pq.read_table(TEST_LABELS_PATH)
    val_lab = {col: val_labels.column(col).to_numpy().astype(np.float32)
               for _, col in TASKS}
    test_lab = {col: test_labels.column(col).to_numpy().astype(np.float32)
                for _, col in TASKS}

    for run_id in LGBM_RUNS:
        val_path = f"results/{run_id}/val_predictions.npz"
        test_pred_path = f"results/{run_id}/predictions.parquet"
        metrics_path = f"results/{run_id}/metrics.json"
        if not all(os.path.exists(p) for p in (val_path, test_pred_path, metrics_path)):
            print(f"skip {run_id}: missing file"); continue

        val_preds = np.load(val_path)
        test_table = pq.read_table(test_pred_path)
        test_scores = {
            "revert": test_table.column("revert_score").to_numpy().astype(np.float32),
            "drop":   test_table.column("drop_score").to_numpy().astype(np.float32),
        }

        with open(metrics_path) as f:
            metrics = json.load(f)

        print(f"\n=== {run_id} ===")
        calib = metrics.get("f1_threshold_calibration", {})
        for task_key, lab_col in TASKS:
            t_opt, val_f1 = best_f1_threshold(val_preds[task_key], val_lab[lab_col])
            test_f1 = test_f1_at_threshold(test_scores[task_key], test_lab[lab_col], t_opt)
            old = metrics.get(f"{task_key}_f1", 0.0)
            print(f"  {task_key}: threshold={t_opt:.4f}  val_f1={val_f1:.4f}  "
                  f"test_f1={test_f1:.4f}  (prev: {old:.4f})")
            metrics[f"{task_key}_f1"] = test_f1
            calib[task_key] = {
                "val_optimal_threshold": t_opt,
                "val_f1_at_threshold": val_f1,
                "score_range_val": [float(val_preds[task_key].min()),
                                     float(val_preds[task_key].max())],
            }
        metrics["f1_threshold_calibration"] = calib

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=float)
        print(f"  wrote {metrics_path}")


if __name__ == "__main__":
    main()
