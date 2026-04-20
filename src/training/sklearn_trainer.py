"""train wrappers for the non-neural baselines (lightgbm + heuristic).

separate from Lightning because lgbm has the sklearn fit/predict shape
and the heuristic only needs threshold calibration.
"""

import os

import numpy as np

from src.baselines.lgbm import LightGBMBaseline
from src.baselines.heuristic import HeuristicBaseline
from src.baselines.logreg import LogisticRegressionBaseline


def train_lightgbm(config, train_data, val_data, output_dir, cat_features=None):
    """fit lgbm, save model + val predictions."""
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("model", {})
    model = LightGBMBaseline(
        max_depth=cfg.get("max_depth", 8),
        n_estimators=cfg.get("n_estimators", 1000),
        num_leaves=cfg.get("num_leaves", 63),
        learning_rate=cfg.get("learning_rate", 0.05),
        min_child_samples=cfg.get("min_child_samples", 200),
        feature_fraction=cfg.get("feature_fraction", 0.9),
        bagging_fraction=cfg.get("bagging_fraction", 0.9),
        bagging_freq=cfg.get("bagging_freq", 1),
        early_stopping=cfg.get("early_stopping_rounds", 50),
        use_scale_pos_weight=cfg.get("use_scale_pos_weight", True),
        cat_features=cat_features,
    )

    X_train, y_train = train_data
    X_val, y_val = val_data

    print("training lgbm...")
    model.fit(X_train, y_train, X_val, y_val)

    model_dir = os.path.join(output_dir, "model")
    model.save(model_dir)

    preds = model.predict(X_val)
    np.savez(
        os.path.join(output_dir, "val_predictions.npz"),
        **{k: v for k, v in preds.items()},
    )

    print(f"-> {model_dir}")
    return model


def train_logreg(config, train_data, val_data, output_dir):
    """fit logreg on 11 standardised numerics, save model + val predictions."""
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.get("model", {})
    model = LogisticRegressionBaseline(
        C=cfg.get("C", 1.0),
        max_iter=cfg.get("max_iter", 20),
        class_weight=cfg.get("class_weight", "balanced"),
        random_state=config.get("seed", 42),
    )

    X_train, y_train = train_data
    X_val, y_val = val_data

    print("training logreg...")
    model.fit(X_train, y_train, X_val, y_val)

    model_dir = os.path.join(output_dir, "model")
    model.save(model_dir)

    preds = model.predict(X_val)
    np.savez(
        os.path.join(output_dir, "val_predictions.npz"),
        **{k: v for k, v in preds.items()},
    )

    print(f"-> {model_dir}")
    return model


def run_heuristic(config, train_data, output_dir):
    """fit threshold-only heuristic on the processed (log-transformed) features.

    log1p is monotonic, so percentile ordering carries over from raw -> log.
    """
    os.makedirs(output_dir, exist_ok=True)

    rules = config.get("rules", {})
    pctl = rules.get("drop", {}).get("gas_fee_cap_percentile", 5)

    log_gas_fee = train_data["log_gas_fee"].to_numpy().astype(np.float64)
    log_gas_tip = train_data["log_gas_tip"].to_numpy().astype(np.float64)
    log_gas    = train_data["log_gas"].to_numpy().astype(np.float64)

    log_gas_fee_thr = float(np.nanpercentile(log_gas_fee, pctl))
    log_gas_tip_med = float(np.nanmedian(log_gas_tip))
    log_gas_p10     = float(np.nanpercentile(log_gas, 10))

    print(f"  log_gas_fee p{pctl} = {log_gas_fee_thr:.4f}")
    print(f"  log_gas_tip median = {log_gas_tip_med:.4f}")
    print(f"  log_gas p10        = {log_gas_p10:.4f}")

    model = HeuristicBaseline(
        log_gas_fee_threshold=log_gas_fee_thr,
        log_gas_tip_median=log_gas_tip_med,
        log_gas_threshold=log_gas_p10,
    )

    thresholds_path = os.path.join(output_dir, "thresholds.json")
    model.save(thresholds_path)
    print(f"-> {thresholds_path}")
    return model
