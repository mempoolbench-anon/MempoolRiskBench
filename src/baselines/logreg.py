"""logistic regression — the linear floor for TABLE_main.

3 binary classifiers (revert / mev / drop) on 11 standardised numerics.
categorical hashes are dropped (paper §4: the ~300K-bin one-hot blew up
training without moving the floor).
"""

import os
import pickle

import numpy as np


class LogisticRegressionBaseline:
    """sklearn-style fit/predict. NOT an nn.Module."""

    TASK_NAMES = ["revert", "mev", "drop"]

    def __init__(
        self,
        C=1.0,
        max_iter=20,
        class_weight="balanced",
        random_state=42,
    ):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.scaler = None
        self.models = {}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """standardise once, then train 3 independent SGD log-loss heads.

        SGDClassifier rather than LogisticRegression because liblinear /
        lbfgs copy X internally (OOMs on 72M x 11 float32 at ~30 GB).
        SGD streams mini-batches, stays ~200 MB, same L2 log-loss
        objective. alpha = 1 / (C * N) matches LogisticRegression's C.

        X_val / y_val are accepted but unused — LogReg has no early stop.
        """
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(X_train)

        for i, task in enumerate(self.TASK_NAMES):
            yi = y_train[:, i].copy()

            # revert has -1 for unincluded txs; drop has -1 for right-censored.
            # mask them out. mev has no -1.
            if task in ("revert", "drop"):
                valid = yi >= 0
                Xi = self.scaler.transform(X_train[valid])
                yi = yi[valid].astype(np.int32)
            else:
                Xi = self.scaler.transform(X_train)
                yi = yi.astype(np.int32)

            alpha = 1.0 / max(1.0, self.C * len(Xi))

            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
                tol=1e-4,
                n_jobs=-1,
            )
            model.fit(Xi, yi)
            self.models[task] = model
            n_iter = int(model.n_iter_) if hasattr(model, "n_iter_") else -1
            print(f"  logreg[{task}]: {n_iter} SGD epochs, N={len(Xi):,}, alpha={alpha:.2e}")
            del Xi

    def predict(self, X):
        """score X with all 3 heads. returns {'revert', 'mev', 'drop': (N,) float}."""
        X_s = self.scaler.transform(X)
        return {
            task: self.models[task].predict_proba(X_s)[:, 1].astype(np.float32)
            for task in self.TASK_NAMES
        }

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for task, model in self.models.items():
            with open(os.path.join(path, f"{task}.pkl"), "wb") as f:
                pickle.dump(model, f)
        with open(os.path.join(path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

    def load(self, path):
        for task in self.TASK_NAMES:
            with open(os.path.join(path, f"{task}.pkl"), "rb") as f:
                self.models[task] = pickle.load(f)
        with open(os.path.join(path, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        return self
