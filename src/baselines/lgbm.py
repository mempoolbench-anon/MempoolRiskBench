"""LightGBM tabular baseline (single transaction, no sequence context).

three independent binary classifiers, one per task. handles the 3
categorical hashes natively via lgb's categorical_feature path; no
embedding layer.

hyperparameters match the paper: depth=8, 1000 trees, 63 leaves,
lr=0.05, min_child_samples=200, feature/bagging fraction 0.9 with
bagging_freq=1, 50-round early stopping, and per-task scale_pos_weight
= neg/pos to handle severe class imbalance.
"""

import os
import pickle

import numpy as np


class LightGBMBaseline:
    """sklearn-style wrapper. NOT an nn.Module — runs through sklearn_trainer."""

    TASK_NAMES = ["revert", "mev", "drop"]

    def __init__(
        self,
        max_depth=8,
        n_estimators=1000,
        num_leaves=63,
        learning_rate=0.05,
        min_child_samples=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        early_stopping=50,
        use_scale_pos_weight=True,
        cat_features=None,
        random_state=42,
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.early_stopping = early_stopping
        self.use_scale_pos_weight = use_scale_pos_weight
        self.random_state = random_state
        # last 3 cols are the categorical hashes (11 numeric + 3 hashes).
        # ablation runs override this with [] or just the selector hash.
        self.cat_features = [11, 12, 13] if cat_features is None else list(cat_features)
        self.models = {}
        self.scale_pos_weights = {}

    @staticmethod
    def _pos_weight(y):
        """neg/pos ratio; 1.0 if either class is empty."""
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        if pos <= 0 or neg <= 0:
            return 1.0
        return neg / pos

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """train one classifier per task, with early stopping if val is given."""
        import lightgbm as lgb

        cat_features = self.cat_features

        for i, task in enumerate(self.TASK_NAMES):
            yi = y_train[:, i].copy()

            # revert has -1 (no receipt); drop has -1 (right-censored). mask both.
            if task in ("revert", "drop"):
                valid = yi >= 0
                Xi = X_train[valid]
                yi = yi[valid]
            else:
                Xi = X_train

            callbacks = []
            eval_set = []
            eval_metric = None
            if X_val is not None and y_val is not None:
                yi_val = y_val[:, i].copy()
                if task in ("revert", "drop"):
                    vmask = yi_val >= 0
                    Xi_val = X_val[vmask]
                    yi_val = yi_val[vmask]
                else:
                    Xi_val = X_val
                eval_set = [(Xi_val, yi_val)]
                # binary_logloss is unstable with large scale_pos_weight; use AUC/AP
                eval_metric = "auc" if task != "mev" else "average_precision"
                callbacks = [lgb.early_stopping(self.early_stopping, verbose=False)]

            spw = self._pos_weight(yi) if self.use_scale_pos_weight else 1.0
            self.scale_pos_weights[task] = spw

            model = lgb.LGBMClassifier(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                min_child_samples=self.min_child_samples,
                colsample_bytree=self.feature_fraction,
                subsample=self.bagging_fraction,
                subsample_freq=self.bagging_freq,
                scale_pos_weight=spw,
                verbose=-1,
                random_state=self.random_state,
            )
            fit_kwargs = dict(
                eval_set=eval_set,
                categorical_feature=cat_features,
                callbacks=callbacks,
            )
            if eval_metric is not None:
                fit_kwargs["eval_metric"] = eval_metric
            model.fit(Xi, yi, **fit_kwargs)
            self.models[task] = model
            print(f"  lgbm[{task}]: {model.best_iteration_} iters, spw={spw:.2f}")

    def predict(self, X):
        """returns {'revert','mev','drop'} -> (N,) float32 scores."""
        out = {}
        for task in self.TASK_NAMES:
            model = self.models[task]
            scores = model.predict_proba(X)[:, 1]
            out[task] = scores.astype(np.float32)
        return out

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for task, model in self.models.items():
            with open(os.path.join(path, f"{task}.pkl"), "wb") as f:
                pickle.dump(model, f)
        if self.scale_pos_weights:
            with open(os.path.join(path, "scale_pos_weights.pkl"), "wb") as f:
                pickle.dump(self.scale_pos_weights, f)

    def load(self, path):
        for task in self.TASK_NAMES:
            with open(os.path.join(path, f"{task}.pkl"), "rb") as f:
                self.models[task] = pickle.load(f)
        spw_path = os.path.join(path, "scale_pos_weights.pkl")
        if os.path.exists(spw_path):
            with open(spw_path, "rb") as f:
                self.scale_pos_weights = pickle.load(f)
        return self
