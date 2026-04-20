"""F1 threshold calc for the 6 neural checkpoints.

same math as calibrate_lgbm_f1.py, but val scores don't exist on disk so
we run val inference from each checkpoint and save the (revert, drop)
arrays under results/<run>/val_predictions.parquet.

only touches F1 fields in metrics.json
"""

import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml  # noqa: E402

# make src/ importable when running from repo root
sys.path.insert(0, os.path.abspath("."))

from src.data.datasets import SequenceDataset, TabularDataset, collate_fn  # noqa: E402
from src.evaluation.run_eval import _load_model  # noqa: E402

VAL_DIR = "data/processed/val"
TEST_LABELS_PATH = "data/processed/test/labels.parquet"

# (config_name, run_id, checkpoint)
RUNS = [
    ("mlp_no_identity",              "mlp_nocat",
     "checkpoints/mlp_nocat/last.ckpt"),
    ("lstm",                          "lstm",
     "checkpoints/lstm/last.ckpt"),
    ("transformer",                   "transformer",
     "checkpoints/transformer/last.ckpt"),
    ("mamba3_constant",               "mamba3_const",
     "checkpoints/mamba3_const/last.ckpt"),
    ("mamba3_physical",               "mamba3_phys",
     "checkpoints/mamba3_phys/last.ckpt"),
    ("mamba3_physical_no_identity",   "mamba3_phys_nocat",
     "checkpoints/mamba3_phys_nocat/last.ckpt"),
]
TASKS = [("revert", "is_reverted"), ("drop", "is_dropped")]


def best_f1_threshold(scores, labels):
    """sort-and-scan for the F1-argmax threshold."""
    valid = labels >= 0
    if valid.sum() < 10:
        return 0.5, 0.0
    s = scores[valid]
    y = labels[valid].astype(np.int32)
    P = int(y.sum())
    if P == 0:
        return 0.5, 0.0
    order = np.argsort(-s)
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted)
    pred_pos = np.arange(1, len(s_sorted) + 1)
    prec = tp / pred_pos
    rec = tp / P
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    k = int(np.argmax(f1))
    return float(s_sorted[k]), float(f1[k])


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


@torch.no_grad()
def run_val_inference(config, checkpoint_path):
    """val-split inference -> {revert, mev, drop} score arrays."""
    model = _load_model(config, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    feat_path = os.path.join(VAL_DIR, "features.parquet")
    lab_path = os.path.join(VAL_DIR, "labels.parquet")
    input_view = config["model"]["input_view"]
    if input_view == "sequence":
        seq_len = config.get("training", {}).get("seq_len", 1024)
        warmup = config.get("training", {}).get("warmup_tokens", 64)
        ds = SequenceDataset(feat_path, lab_path, seq_len=seq_len, warmup_tokens=warmup)
    else:
        ds = TabularDataset(feat_path, lab_path)

    batch_size = config.get("training", {}).get("batch_size", 32)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2,
    )

    preds = {"revert": [], "mev": [], "drop": []}
    for batch in loader:
        x_num = batch["x_num"].to(device)
        x_cat = batch["x_cat"].to(device)
        kwargs = {}
        if "delta_t" in batch:
            kwargs["delta_t"] = batch["delta_t"].to(device)
        logits = model(x_num, x_cat, **kwargs)
        for task in ("revert", "mev", "drop"):
            probs = torch.sigmoid(logits[task]).cpu().numpy()
            preds[task].append(probs.reshape(-1))
    return {k: np.concatenate(v) for k, v in preds.items()}


def main():
    val_lab_table = pq.read_table(os.path.join(VAL_DIR, "labels.parquet"))
    test_lab_table = pq.read_table(TEST_LABELS_PATH)
    val_lab = {col: val_lab_table.column(col).to_numpy().astype(np.float32)
               for _, col in TASKS}
    test_lab = {col: test_lab_table.column(col).to_numpy().astype(np.float32)
                for _, col in TASKS}

    for cfg_name, run_id, ckpt_path in RUNS:
        cfg_path = f"configs/{cfg_name}.yaml"
        test_pred_path = f"results/{run_id}/predictions.parquet"
        metrics_path = f"results/{run_id}/metrics.json"
        val_pred_cache = f"results/{run_id}/val_predictions.parquet"

        if not all(os.path.exists(p) for p in (cfg_path, ckpt_path, test_pred_path, metrics_path)):
            print(f"skip {run_id}: missing input file"); continue

        with open(cfg_path) as f:
            config = yaml.safe_load(f)

        print(f"\n=== {run_id} ===")
        if os.path.exists(val_pred_cache):
            print(f"  using cached val predictions: {val_pred_cache}")
            vp_table = pq.read_table(val_pred_cache)
            val_preds = {c: vp_table.column(c).to_numpy().astype(np.float32)
                         for c in ("revert", "drop")}
        else:
            print(f"  running val inference with {ckpt_path}...")
            val_preds = run_val_inference(config, ckpt_path)
            # cache revert + drop; mev not needed for F1 calibration
            pq.write_table(
                pa.table({"revert": val_preds["revert"], "drop": val_preds["drop"]}),
                val_pred_cache,
            )
            print(f"  cached to {val_pred_cache}")

        test_table = pq.read_table(test_pred_path)
        test_scores = {
            "revert": test_table.column("revert_score").to_numpy().astype(np.float32),
            "drop":   test_table.column("drop_score").to_numpy().astype(np.float32),
        }

        with open(metrics_path) as f:
            metrics = json.load(f)
        calib = metrics.get("f1_threshold_calibration", {})

        # align lengths (sequence models may leave a tail remainder on val)
        n_val = min(len(val_preds["revert"]), len(val_lab["is_reverted"]))
        n_test = min(len(test_scores["revert"]), len(test_lab["is_reverted"]))

        for task_key, lab_col in TASKS:
            v_s = val_preds[task_key][:n_val]
            v_y = val_lab[lab_col][:n_val]
            t_opt, val_f1 = best_f1_threshold(v_s, v_y)

            t_s = test_scores[task_key][:n_test]
            t_y = test_lab[lab_col][:n_test]
            test_f1 = test_f1_at_threshold(t_s, t_y, t_opt)
            old = metrics.get(f"{task_key}_f1", 0.0)
            print(f"  {task_key}: threshold={t_opt:.4f}  val_f1={val_f1:.4f}  "
                  f"test_f1={test_f1:.4f}  (prev: {old:.4f})")
            metrics[f"{task_key}_f1"] = test_f1
            calib[task_key] = {
                "val_optimal_threshold": t_opt,
                "val_f1_at_threshold": val_f1,
                "score_range_val": [float(v_s.min()), float(v_s.max())],
            }
        metrics["f1_threshold_calibration"] = calib

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=float)
        print(f"  wrote {metrics_path}")


if __name__ == "__main__":
    main()
