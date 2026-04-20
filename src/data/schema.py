"""single source of truth for column names, dtypes, features, labels.

import everything from here — never hardcode column names elsewhere or
they will drift apart over time.
"""

# raw parquet stores these as utf8 (yes, big ints serialised as strings).
RAW_STRING_NUMERIC_COLS = [
    "gas_price", "gas_tip_cap", "gas_fee_cap", "value", "nonce", "gas",
]

SORT_KEY = "timestamp_ms"

# feature whitelist — these 14 are the only model inputs anywhere.
NUMERIC_FEATURES = [
    "log_gas_price",      # log1p(int(gas_price) / 1e9)
    "log_gas_tip",        # log1p(int(gas_tip_cap) / 1e9)
    "log_gas_fee",        # log1p(int(gas_fee_cap) / 1e9)
    "log_value",          # log1p(int(value) / 1e18)
    "log_gas",            # log1p(int(gas))
    "log_nonce",          # log1p(int(nonce))
    "log_data_size",      # log1p(data_size)
    "timestamp_delta_s",  # (ts_current - ts_previous) / 1000, capped 300s
    "sender_tx_count",    # rolling expanding count per sender (causal)
    "sender_avg_gas",     # rolling avg gas price per sender, shift(1)
    "mempool_pressure",   # count of txs in trailing 60s window
]

CATEGORICAL_FEATURES = {
    "from_addr_hash":   {"n_bins": 131_072, "embed_dim": 16},
    "to_addr_hash":     {"n_bins": 131_072, "embed_dim": 16},
    "data_4bytes_hash": {"n_bins": 16_384,  "embed_dim": 8},
}

# d_input = 11 numeric + 16+16+8 embedded = 51
NUM_NUMERIC = len(NUMERIC_FEATURES)                                    # 11
TOTAL_EMBED_DIM = sum(v["embed_dim"] for v in CATEGORICAL_FEATURES.values())  # 40
D_INPUT = NUM_NUMERIC + TOTAL_EMBED_DIM                                # 51

LABEL_COLUMNS = ["is_reverted", "is_mev_victim", "is_dropped"]

# the safety rail. nothing in this list ever feeds a model — they all
# carry post-inclusion info (or, in the case of `sources`, are uniformly
# null in our snapshot).
BLACKLISTED_COLUMNS = [
    "receipt_status", "receipt_gas_used", "inclusion_delay_ms",
    "included_block_timestamp_ms", "included_at_block_height",
    "block_number", "mev_type", "sources",
]

# chronological splits, contiguous. no embargo gaps, no discarded weeks.
# leakage is prevented at the feature-engineering level (past-only
# rolling, whitelist input set) and verified by the shuffled-label audit.
SPLIT_RANGES = {
    "train": ("2026-02-01", "2026-03-25"),  # [inclusive, exclusive)
    "val":   ("2026-03-25", "2026-04-01"),
    "test":  ("2026-04-01", "2026-04-08"),
}

CAPTURE_START_DATE = "2026-02-01"
CAPTURE_END_DATE   = "2026-04-08"  # exclusive
# rows whose 24h drop-maturity window extends past CAPTURE_END_DATE
# can't have a settled is_dropped, so we right-censor them to -1.
DROP_MATURITY_HOURS = 24

# clip inter-arrival to [1ms, 300s] before log1p
DELTA_CLIP_MIN = 1e-3
DELTA_CLIP_MAX = 300.0

# 16 well-known DEX swap selectors used by the §5.7 DEX-subset eval.
# NOT a model input — only an evaluation-time mask.
DEX_SELECTORS = [
    "0x38ed1739", "0x8803dbee", "0x7ff36ab5", "0x18cbafe5", "0xfb3bdb41",
    "0x5ae401dc", "0x414bf389", "0xc04b8d59", "0xdb3e2198", "0xf28c0498",
    "0x3593564c", "0x12aa3caf", "0xe449022e", "0x0502b1c5", "0xd9627aa4",
    "0x415565b0",
]
