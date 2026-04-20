"""rule-based heuristic baseline. no training, no torch dep.

operates on the *processed* (log-transformed) features, not the raw
columns. rules:
    revert: log_gas < 10th-percentile of train  (low gas budget)
    mev:    has calldata AND log_gas_tip > 2x train median  (likely
            contract interaction with elevated tip)
    drop:   log_gas_fee < 5th-percentile of train log_gas_fee

thresholds come from a one-time fit on the train split.
"""

import numpy as np


class HeuristicBaseline:
    """three thresholds, one rule per task."""

    def __init__(self, log_gas_fee_threshold, log_gas_tip_median, log_gas_threshold):
        self.log_gas_fee_threshold = log_gas_fee_threshold
        self.log_gas_tip_median = log_gas_tip_median
        self.log_gas_threshold = log_gas_threshold

    def predict(self, df):
        """{'revert','mev','drop'} -> (N,) float32 scores. Accepts polars or dict."""
        if hasattr(df, "to_numpy"):
            log_gas = df["log_gas"].to_numpy().astype(np.float32)
            log_gas_fee = df["log_gas_fee"].to_numpy().astype(np.float32)
            log_gas_tip = df["log_gas_tip"].to_numpy().astype(np.float32)
            d4b_hash = df["data_4bytes_hash"].to_numpy().astype(np.int64)
        else:
            log_gas = np.asarray(df["log_gas"], dtype=np.float32)
            log_gas_fee = np.asarray(df["log_gas_fee"], dtype=np.float32)
            log_gas_tip = np.asarray(df["log_gas_tip"], dtype=np.float32)
            d4b_hash = np.asarray(df["data_4bytes_hash"], dtype=np.int64)

        # revert: low gas budget
        revert_scores = (log_gas < self.log_gas_threshold).astype(np.float32)

        # mev: has calldata (non-zero selector hash) AND elevated tip
        has_calldata = (d4b_hash != 0).astype(np.float32)
        high_tip = (log_gas_tip > 2.0 * self.log_gas_tip_median).astype(np.float32)
        mev_scores = has_calldata * high_tip

        # drop: too-low fee cap to compete in the next block
        drop_scores = (log_gas_fee < self.log_gas_fee_threshold).astype(np.float32)

        return {"revert": revert_scores, "mev": mev_scores, "drop": drop_scores}

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump({
                "log_gas_fee_threshold": float(self.log_gas_fee_threshold),
                "log_gas_tip_median": float(self.log_gas_tip_median),
                "log_gas_threshold": float(self.log_gas_threshold),
            }, f)

    @classmethod
    def load(cls, path):
        import json
        with open(path) as f:
            d = json.load(f)
        return cls(
            log_gas_fee_threshold=d["log_gas_fee_threshold"],
            log_gas_tip_median=d["log_gas_tip_median"],
            log_gas_threshold=d["log_gas_threshold"],
        )
