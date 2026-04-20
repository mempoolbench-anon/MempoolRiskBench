"""eval CLI: load model -> generate preds -> compute metrics.


example usage:
    # standard test eval
    python -m src.evaluation.run_eval --run-id <id> --split test \\
        --checkpoint <ckpt> --config configs/mamba3_physical.yaml

    # corruption eval (shuffled or quantized delta_t)
    python -m src.evaluation.run_eval --run-id <id> --split test \\
        --checkpoint <ckpt> --corruption shuffle

    # traffic-stratified
    python -m src.evaluation.run_eval --run-id <id> --split test \\
        --checkpoint <ckpt> --stratify traffic

    # DEX-only subset
    python -m src.evaluation.run_eval --run-id <id> --split test \\
        --checkpoint <ckpt> --subset dex
        
drop right-censoring is wired through: --drop-exclude-hours
"""

import argparse
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
import yaml
from sklearn.metrics import precision_recall_curve

# long bootstrap loops after a DataLoader with workers leak shm fds on the
# default sharing strategy -> "Too many open files". file_system uses named
# tmpfiles the OS reclaims cleanly.
mp.set_sharing_strategy("file_system")

from src.data.datasets import SequenceDataset, TabularDataset, collate_fn
from src.data.schema import LABEL_COLUMNS
from src.evaluation.corruption import compute_quantile_bins, corrupt_batch
from src.evaluation.metrics import compute_all_metrics, compute_all_metrics_with_ci
from src.evaluation.stratification import compute_traffic_quartiles, stratified_metrics
from src.training.train import _build_model


def _load_model(config, checkpoint_path):
    """Load model from config and checkpoint."""
    model = _build_model(config)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            # Lightning checkpoint
            state = {}
            for k, v in ckpt["state_dict"].items():
                # Remove 'model.' prefix from Lightning state dict
                k_clean = k.replace("model.", "", 1) if k.startswith("model.") else k
                state[k_clean] = v
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model


@torch.no_grad()
def _generate_predictions_neural(model, dataset, config, corruption=None,
                                  bin_edges=None, bin_medians=None):
    """Generate predictions from a neural model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = config.get("training", {}).get("batch_size", 32)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    all_preds = {"revert": [], "mev": [], "drop": []}
    all_labels = []

    for batch in loader:
        # Move to device
        x_num = batch["x_num"].to(device)
        x_cat = batch["x_cat"].to(device)
        labels = batch["labels"]

        kwargs = {}
        if "delta_t" in batch:
            delta_t = batch["delta_t"].to(device)
            # Apply corruption if specified
            if corruption:
                b = {"delta_t": delta_t}
                b = corrupt_batch(b, corruption, bin_edges, bin_medians)
                delta_t = b["delta_t"]
            kwargs["delta_t"] = delta_t

        logits = model(x_num, x_cat, **kwargs)

        for task in ["revert", "mev", "drop"]:
            probs = torch.sigmoid(logits[task]).cpu().numpy()
            # For sequence models: flatten (B, L, 1) -> (B*L,)
            all_preds[task].append(probs.reshape(-1))

        # Flatten labels similarly
        all_labels.append(labels.numpy().reshape(-1, 3))

    preds = {k: np.concatenate(v) for k, v in all_preds.items()}
    labels = np.concatenate(all_labels)

    return preds, labels


def main():
    parser = argparse.ArgumentParser(description="evaluate a trained model")
    parser.add_argument("--run-id", required=True, help="run id (output subdir)")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--corruption", default=None, choices=["shuffle", "quantize"])
    parser.add_argument("--stratify", default=None, choices=["traffic"])
    parser.add_argument("--subset", default=None, choices=["dex"])
    parser.add_argument("--drop-exclude-hours", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = os.path.join(args.results_dir, args.run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Save resolved config
    with open(os.path.join(output_dir, "config.resolved.yaml"), "w") as f:
        yaml.dump(config, f)

    family = config["model"]["family"]
    input_view = config["model"]["input_view"]

    # Build dataset
    split_dir = os.path.join(args.data_dir, args.split)
    feat_path = os.path.join(split_dir, "features.parquet")
    lab_path = os.path.join(split_dir, "labels.parquet")

    # Prepare corruption bins if needed
    bin_edges, bin_medians = None, None
    if args.corruption == "quantize":
        train_feat = pq.read_table(
            os.path.join(args.data_dir, "train", "features.parquet")
        )
        train_deltas = train_feat.column("timestamp_delta_s").to_numpy()
        bin_edges, bin_medians = compute_quantile_bins(train_deltas)

    if family == "neural":
        if not args.checkpoint:
            # fail loud — evaluating a random-init net silently gives ~chance
            # AUC and corrupts every downstream number.
            raise SystemExit(
                f"run_eval: neural family needs --checkpoint. got none for "
                f"run_id={args.run_id}, config={args.config}."
            )
        model = _load_model(config, args.checkpoint)

        if input_view == "sequence":
            seq_len = config.get("training", {}).get("seq_len", 1024)
            warmup = config.get("training", {}).get("warmup_tokens", 64)
            dataset = SequenceDataset(feat_path, lab_path, seq_len=seq_len, warmup_tokens=warmup)
        else:
            dataset = TabularDataset(feat_path, lab_path)

        preds, labels = _generate_predictions_neural(
            model, dataset, config,
            corruption=args.corruption,
            bin_edges=bin_edges,
            bin_medians=bin_medians,
        )
    elif family == "sklearn":
        from src.training.train import _resolve_lgbm_columns
        model_name = config["model"].get("name", "lightgbm")
        if model_name == "lightgbm":
            from src.baselines.lgbm import LightGBMBaseline
            model = LightGBMBaseline()
        elif model_name == "logreg":
            from src.baselines.logreg import LogisticRegressionBaseline
            model = LogisticRegressionBaseline()
        else:
            raise ValueError(f"unknown sklearn model name: {model_name}")
        # default to results/<run-id>/model; --model-dir (via args) can point
        # at a different run when reusing a training checkpoint for test eval.
        model_dir = getattr(args, "model_dir", None) or os.path.join(args.results_dir, args.run_id, "model")
        model.load(model_dir)

        feat_table = pq.read_table(feat_path)
        lab_table = pq.read_table(lab_path)

        numeric_cols, categorical_cols = _resolve_lgbm_columns(config)
        feature_cols = numeric_cols + categorical_cols
        X = np.column_stack([feat_table.column(c).to_numpy().astype(np.float32) for c in feature_cols])
        labels = np.column_stack(
            [lab_table.column(c).to_numpy().astype(np.float32) for c in LABEL_COLUMNS]
        )
        preds = model.predict(X)
    elif family == "heuristic":
        import polars as pl
        from src.baselines.heuristic import HeuristicBaseline

        thresholds_path = os.path.join(args.results_dir, "heuristic", "thresholds.json")
        model = HeuristicBaseline.load(thresholds_path)

        # heuristic reads the processed (already log-transformed) features
        feat_df = pl.read_parquet(feat_path)
        preds = model.predict(feat_df)
        lab_table = pq.read_table(lab_path)
        labels = np.column_stack(
            [lab_table.column(c).to_numpy().astype(np.float32) for c in LABEL_COLUMNS]
        )
    else:
        raise ValueError(f"Unknown family: {family}")

    # Save predictions as Parquet
    pred_table = pa.table({
        "revert_score": preds["revert"],
        "mev_score": preds["mev"],
        "drop_score": preds["drop"],
    })
    pq.write_table(pred_table, os.path.join(output_dir, "predictions.parquet"))

    # Compute metrics
    metrics = compute_all_metrics(preds, labels)
    print(f"\nMetrics for {args.run_id}:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Bootstrap CIs
    if args.bootstrap > 0:
        metrics_ci = compute_all_metrics_with_ci(preds, labels, n_bootstrap=args.bootstrap)
        metrics["bootstrap_ci"] = {
            k: {"point": v[0], "lower": v[1], "upper": v[2]}
            for k, v in metrics_ci.items()
        }

    # DEX-subset MEV — restrict to txs whose 4-byte selector is in DEX_SELECTORS.
    # reads dex_mask.parquet alongside features/labels.
    if args.subset == "dex":
        dex_mask_path = os.path.join(split_dir, "dex_mask.parquet")
        if not os.path.exists(dex_mask_path):
            raise FileNotFoundError(
                f"--subset dex needs {dex_mask_path}; rebuild with src.data.build_dataset."
            )
        dex_mask = pq.read_table(dex_mask_path).column("is_dex_candidate").to_numpy().astype(bool)
        # truncate to whatever the model actually predicted (sequence models
        # may skip the tail when the split doesn't divide cleanly into windows)
        dex_mask = dex_mask[:len(preds["mev"])]

        from sklearn.metrics import average_precision_score
        from src.evaluation.metrics import precision_at_k, recall_at_k

        def _mev_pack(scores, lab):
            out = {}
            if lab.sum() >= 1 and len(np.unique(lab)) > 1:
                out["mev_pr_auc"] = float(average_precision_score(lab, scores))
            n = len(scores)
            out["mev_p_at_100"] = precision_at_k(scores, lab, min(100, n))
            out["mev_p_at_1000"] = precision_at_k(scores, lab, min(1000, n))
            out["mev_recall_at_1000"] = recall_at_k(scores, lab, min(1000, n))
            return out

        mev_scores = preds["mev"][:len(dex_mask)]
        mev_labels = labels[:len(dex_mask), 1]
        metrics["dex_subset"] = {
            "overall":   _mev_pack(mev_scores, mev_labels),
            "dex_only":  _mev_pack(mev_scores[dex_mask], mev_labels[dex_mask]),
            "non_dex":   _mev_pack(mev_scores[~dex_mask], mev_labels[~dex_mask]),
            "n_dex":     int(dex_mask.sum()),
            "n_dex_pos": int(mev_labels[dex_mask].sum()),
            "n_total":   int(len(dex_mask)),
        }
        d = metrics["dex_subset"]
        print(f"\nDEX subset: {d['n_dex']:,} of {d['n_total']:,} txs "
              f"({100*d['n_dex']/d['n_total']:.3f}%); {d['n_dex_pos']:,} MEV positives")
        for k in ("overall", "dex_only", "non_dex"):
            pr = d[k].get("mev_pr_auc")
            p100 = d[k].get("mev_p_at_100")
            print(f"  {k:10s} pr_auc={pr if pr is None else f'{pr:.4f}'}  p@100={p100:.4f}")

    # Traffic stratification
    if args.stratify == "traffic":
        feat_table = pq.read_table(feat_path)
        pressure = feat_table.column("mempool_pressure").to_numpy().astype(np.float32)
        # Truncate to match predictions length
        pressure = pressure[:len(preds["revert"])]
        q_labels, q_bounds = compute_traffic_quartiles(pressure)

        from sklearn.metrics import average_precision_score
        def mev_prauc(p, l):
            if len(np.unique(l)) > 1:
                return float(average_precision_score(l, p))
            return 0.0

        strat = stratified_metrics(
            preds["mev"], labels[:, 1], q_labels, mev_prauc
        )
        metrics["traffic_stratification"] = {
            f"Q{q}": v for q, v in strat.items()
        }
        metrics["traffic_boundaries"] = q_bounds.tolist()

    # MEV PR curve
    if labels[:, 1].sum() > 0:
        precision, recall, thresholds = precision_recall_curve(
            labels[:, 1], preds["mev"]
        )
        pr_curve = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        }
        with open(os.path.join(output_dir, "pr_curve.json"), "w") as f:
            json.dump(pr_curve, f)

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
