"""train CLI. dispatches on config `family`:

    neural    -> Lightning
    sklearn   -> sklearn_trainer.train_lightgbm()
    heuristic -> sklearn_trainer.run_heuristic()

usage:
    python -m src.training.train --config configs/lgbm.yaml          --run-id lgbm
    python -m src.training.train --config configs/mamba3_physical.yaml --run-id m3p
    # resume:
    python -m src.training.train --config configs/mamba3_physical.yaml \\
        --run-id m3p_5ep --resume checkpoints/m3p/epoch_3.ckpt
"""

import argparse
import os

import yaml
import torch
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from src.data.datasets import SequenceDataset, TabularDataset, PositiveAwareSampler, collate_fn
from src.training.lightning_module import MempoolLitModule
from src.training import sklearn_trainer


def _build_model(config):
    """Instantiate model from config dict."""
    model_cfg = config["model"]
    name = model_cfg["name"]

    drop_categorical = bool(model_cfg.get("drop_categorical", False))

    if name == "mlp":
        from src.model.mlp import MLPBaseline
        return MLPBaseline(
            d_model=model_cfg.get("d_hidden", 256),
            dropout=model_cfg.get("dropout", 0.1),
            drop_categorical=drop_categorical,
        )

    elif name == "lstm":
        from src.model.lstm import LSTMBaseline
        return LSTMBaseline(
            d_model=model_cfg.get("d_hidden", 256),
            n_layers=model_cfg.get("n_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
        )

    elif name == "transformer":
        from src.model.transformer import TransformerBaseline
        return TransformerBaseline(
            d_model=model_cfg.get("d_model", 256),
            n_heads=model_cfg.get("n_heads", 8),
            n_layers=model_cfg.get("n_layers", 6),
            d_ffn=model_cfg.get("d_ffn", 1024),
            dropout=model_cfg.get("dropout", 0.1),
            max_len=model_cfg.get("seq_len", 1024),
        )

    elif name in ("mamba3_physical", "mamba3_constant"):
        from src.model.mempool_mamba import MempoolMamba
        adapter_cfg = config.get("adapter", {})
        use_physical = adapter_cfg.get("use_physical_delta", True)
        constant_delta = None
        if not use_physical:
            # Will be set to train-set median during data loading
            constant_delta = adapter_cfg.get("constant_delta", None)
            if constant_delta == "median":
                constant_delta = None  # computed later

        return MempoolMamba(
            d_model=model_cfg.get("d_model", 256),
            n_layers=model_cfg.get("n_layers", 6),
            n_heads=model_cfg.get("nheads", 8),
            use_physical_delta=use_physical,
            constant_delta=constant_delta,
            d_state=model_cfg.get("d_state", 128),
            expand=model_cfg.get("expand", 2),
            headdim=model_cfg.get("headdim", 64),
            drop_categorical=drop_categorical,
        )

    else:
        raise ValueError(f"Unknown model name: {name}")


def _train_neural(config, args):
    """Train a neural model using Lightning."""
    # Set seed
    seed = args.seed or config.get("seed", 42)
    L.seed_everything(seed)

    # Build model
    model = _build_model(config)

    # Build datasets
    train_cfg = config["training"]
    input_view = config["model"]["input_view"]
    data_dir = args.data_dir

    train_feat = os.path.join(data_dir, "train", "features.parquet")
    train_lab = os.path.join(data_dir, "train", "labels.parquet")
    val_feat = os.path.join(data_dir, "val", "features.parquet")
    val_lab = os.path.join(data_dir, "val", "labels.parquet")

    if input_view == "sequence":
        seq_len = train_cfg.get("seq_len", 1024)
        warmup_tokens = train_cfg.get("warmup_tokens", 64)
        train_ds = SequenceDataset(train_feat, train_lab, seq_len=seq_len, warmup_tokens=warmup_tokens)
        val_ds = SequenceDataset(val_feat, val_lab, seq_len=seq_len, warmup_tokens=warmup_tokens)

        # MEV-positive-aware sampler
        p_pos = config.get("sampling", {}).get("mev_positive_prob", 0.3)
        sampler = PositiveAwareSampler(train_ds, p_pos=p_pos, seed=seed)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=train_cfg.get("batch_size", 32),
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
    else:
        train_ds = TabularDataset(train_feat, train_lab)
        val_ds = TabularDataset(val_feat, val_lab)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=train_cfg.get("batch_size", 4096),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # For constant-delta Mamba: compute train-set median delta
    if hasattr(model, "constant_delta") and model.constant_delta is None and not model.use_physical_delta:
        if input_view == "sequence":
            median_delta = float(np.median(train_ds.delta_t[train_ds.delta_t > 0]))
        else:
            median_delta = 0.1  # fallback
        model.constant_delta = median_delta
        print(f"Constant-delta set to train-set median: {median_delta:.4f}s")

    # Calibrate adapter b parameter
    if hasattr(model, "adapter"):
        if input_view == "sequence":
            median_delta = float(np.median(train_ds.delta_t[train_ds.delta_t > 0]))
        else:
            median_delta = 0.1
        model.adapter.calibrate_b(median_delta)
        print(f"Adapter b calibrated with median_delta={median_delta:.4f}s")

    # Build Lightning module
    lit_module = MempoolLitModule(
        model=model,
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_fraction=train_cfg.get("warmup_fraction", 0.05),
        warmup_tokens=train_cfg.get("warmup_tokens", 64) if input_view == "sequence" else 0,
        total_epochs=train_cfg.get("epochs", 5),
    )

    # Checkpoint callback
    ckpt_dir = os.path.join(args.ckpt_dir, args.run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch_{epoch}",
        save_top_k=-1,  # save all epochs
        every_n_epochs=1,
        save_last=True,
    )

    # Trainer — stop-epoch overrides max_epochs while LR schedule uses config epochs
    max_epochs = args.stop_epoch if args.stop_epoch else train_cfg.get("epochs", 5)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_cb],
        default_root_dir=os.path.join("results", args.run_id),
        enable_progress_bar=True,
        log_every_n_steps=50,
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
    )

    # Resume from checkpoint if specified
    ckpt_path = args.resume if args.resume else None
    trainer.fit(lit_module, train_loader, val_loader, ckpt_path=ckpt_path)

    # Save final config
    results_dir = os.path.join("results", args.run_id)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.resolved.yaml"), "w") as f:
        yaml.dump(config, f)

    print(f"Training complete. Checkpoints: {ckpt_dir}")


def _resolve_lgbm_columns(config):
    """pull (numeric, categorical) feature lists from config.

    honours feature overrides in YAML so the identity-ablation variants
    can drop the address / selector hashes.
    """
    from src.data.schema import NUMERIC_FEATURES

    feat_cfg = config.get("features", {}) or {}
    numeric_cols = feat_cfg.get("numeric") or NUMERIC_FEATURES
    categorical_cols = feat_cfg.get("categorical") or []
    return list(numeric_cols), list(categorical_cols)


def _train_sklearn(config, args):
    """Train a sklearn/LightGBM model."""
    import pyarrow.parquet as pq

    data_dir = args.data_dir
    numeric_cols, categorical_cols = _resolve_lgbm_columns(config)
    feature_cols = numeric_cols + categorical_cols
    label_cols = ["is_reverted", "is_mev_victim", "is_dropped"]

    # Load data
    train_feat = pq.read_table(os.path.join(data_dir, "train", "features.parquet"))
    train_lab = pq.read_table(os.path.join(data_dir, "train", "labels.parquet"))
    val_feat = pq.read_table(os.path.join(data_dir, "val", "features.parquet"))
    val_lab = pq.read_table(os.path.join(data_dir, "val", "labels.parquet"))

    X_train = np.column_stack([train_feat.column(c).to_numpy().astype(np.float32) for c in feature_cols])
    y_train = np.column_stack([train_lab.column(c).to_numpy().astype(np.float32) for c in label_cols])
    X_val = np.column_stack([val_feat.column(c).to_numpy().astype(np.float32) for c in feature_cols])
    y_val = np.column_stack([val_lab.column(c).to_numpy().astype(np.float32) for c in label_cols])

    cat_features = list(range(len(numeric_cols), len(feature_cols)))
    output_dir = os.path.join("results", args.run_id)
    model_name = config["model"].get("name", "lightgbm")
    if model_name == "lightgbm":
        sklearn_trainer.train_lightgbm(
            config, (X_train, y_train), (X_val, y_val), output_dir,
            cat_features=cat_features,
        )
    elif model_name == "logreg":
        sklearn_trainer.train_logreg(
            config, (X_train, y_train), (X_val, y_val), output_dir,
        )
    else:
        raise ValueError(f"unknown sklearn model name: {model_name}")


def _train_heuristic(config, args):
    """Calibrate heuristic baseline."""
    import polars as pl

    data_dir = args.data_dir
    train_feat_path = os.path.join(data_dir, "train", "features.parquet")

    # Load training features for calibration
    train_feat = pl.read_parquet(train_feat_path)

    output_dir = os.path.join("results", args.run_id)
    sklearn_trainer.run_heuristic(config, train_feat, output_dir)


def main():
    parser = argparse.ArgumentParser(description="train one model on the mempool risk dataset")
    parser.add_argument("--config", required=True, help="model YAML config")
    parser.add_argument("--run-id", required=True, help="run id (output subdir under results/)")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--data-dir", default="data/processed", help="processed data dir")
    parser.add_argument("--ckpt-dir", default="checkpoints", help="Checkpoint output")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--stop-epoch", type=int, default=None,
                        help="Stop at this epoch (LR schedule uses config epochs)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    family = config["model"]["family"]
    print(f"Training {config['model']['name']} (family={family}, run={args.run_id})")

    if family == "neural":
        _train_neural(config, args)
    elif family == "sklearn":
        _train_sklearn(config, args)
    elif family == "heuristic":
        _train_heuristic(config, args)
    else:
        raise ValueError(f"Unknown model family: {family}")


if __name__ == "__main__":
    main()
