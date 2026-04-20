"""feature transforms: log scalars, rolling sender stats, mempool pressure.

every transform is strictly causal

"""

import numpy as np
import polars as pl

from src.data.schema import (
    CATEGORICAL_FEATURES,
    DELTA_CLIP_MAX,
    DELTA_CLIP_MIN,
)


def compute_log_features(df: pl.DataFrame) -> pl.DataFrame:
    """log1p the heavy-tailed numeric cols.

    gas_price/tip/fee live in wei -> divide by 1e9 (gwei) first.
    value is wei -> 1e18 (eth). gas/nonce/data_size are already small,
    just log1p.
    """
    return df.with_columns(
        (pl.col("gas_price").cast(pl.Float64) / 1e9).log1p().alias("log_gas_price"),
        (pl.col("gas_tip_cap").cast(pl.Float64) / 1e9).log1p().alias("log_gas_tip"),
        (pl.col("gas_fee_cap").cast(pl.Float64) / 1e9).log1p().alias("log_gas_fee"),
        (pl.col("value").cast(pl.Float64) / 1e18).log1p().alias("log_value"),
        pl.col("gas").cast(pl.Float64).log1p().alias("log_gas"),
        pl.col("nonce").cast(pl.Float64).log1p().alias("log_nonce"),
        pl.col("data_size").cast(pl.Float64).log1p().alias("log_data_size"),
    )


def compute_timestamp_delta(df: pl.DataFrame) -> pl.DataFrame:
    """inter-arrival = (ts_now - ts_prev) / 1000.

    needs the df sorted by timestamp_ms. capped at 300 s. this is also the delta t the
    Mamba-3 physical-time variant consumes.
    """
    ts = df["timestamp_ms"]
    delta_ms = ts.diff().fill_null(0)
    delta_s = (delta_ms.cast(pl.Float64) / 1000.0).clip(0.0, DELTA_CLIP_MAX)
    return df.with_columns(delta_s.alias("timestamp_delta_s"))


def compute_sender_rolling(
    df: pl.DataFrame,
    sender_snapshot: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """rolling per-sender stats — strictly causal.

    sender_tx_count : expanding count per from_addr (pre + this chunk).
    sender_avg_gas  : shift(1) running mean of the sender's past gas
                       (in gwei, log1p'd), so the *current* row's gas is
                       NOT included.

    `sender_snapshot` is the pre-aggregated state from before this chunk
    (cols: from_addr, _pre_count, _pre_gas_sum); 
    without it val/test would only see the senders that happen to appear inside the chunk rather than their full history.s
    """
    gas_col = "gas_price"
    if df.schema.get(gas_col) == pl.Utf8:
        df = df.with_columns(pl.col(gas_col).cast(pl.Float64))

    # local cumulatives within this chunk (incl current row)
    df = df.with_columns(
        pl.col("from_addr").cum_count().over("from_addr").alias("_lc"),
        (pl.col(gas_col) / 1e9).log1p().cum_sum().over("from_addr").alias("_lg"),
    )

    # left-join pre-chunk state, default to zeros if a sender is new here
    if sender_snapshot is not None and len(sender_snapshot) > 0:
        df = df.join(sender_snapshot, on="from_addr", how="left")
        df = df.with_columns(
            pl.col("_pre_count").fill_null(0),
            pl.col("_pre_gas_sum").fill_null(0.0),
        )
    else:
        df = df.with_columns(
            pl.lit(0).alias("_pre_count"),
            pl.lit(0.0).alias("_pre_gas_sum"),
        )

    # global cumulative count = pre-chunk + local
    df = df.with_columns(
        (pl.col("_pre_count") + pl.col("_lc")).cast(pl.Float32).alias("sender_tx_count"),
    )

    # for the running mean we want to *exclude* the current row, so shift
    # the local cumulatives by 1 within each sender, then add pre-chunk.
    df = df.with_columns(
        pl.col("_lc").shift(1).over("from_addr").fill_null(0).alias("_lc_prev"),
        pl.col("_lg").shift(1).over("from_addr").fill_null(0.0).alias("_lg_prev"),
    )
    df = df.with_columns(
        pl.when((pl.col("_pre_count") + pl.col("_lc_prev")) > 0)
        .then(
            (pl.col("_pre_gas_sum") + pl.col("_lg_prev"))
            / (pl.col("_pre_count") + pl.col("_lc_prev")).cast(pl.Float64)
        )
        .otherwise(0.0)
        .alias("sender_avg_gas"),
    )

    df = df.drop(["_lc", "_lg", "_pre_count", "_pre_gas_sum", "_lc_prev", "_lg_prev"])
    return df


def compute_mempool_pressure(df: pl.DataFrame) -> pl.DataFrame:
    """count of txs observed in the trailing 60s.

    sorted timestamps + searchsorted is much faster than rolling a
    polars window over 90M rows.
    """
    ts = df["timestamp_ms"].to_numpy().astype(np.int64)
    window_ms = 60_000
    left = np.searchsorted(ts, ts - window_ms, side="left")
    right = np.arange(len(ts))
    pressure = (right - left).astype(np.float32)
    return df.with_columns(pl.Series("mempool_pressure", pressure))


def compute_categorical_hashes(df: pl.DataFrame) -> pl.DataFrame:
    """hash sender, recipient, 4-byte selector into integer bins.

    bin counts come from schema.CATEGORICAL_FEATURES. nulls always map
    to bin 0 (so contract-creations and plain ETH transfers get a
    well-defined slot). non-null values use `hash % (n_bins-1) + 1`
    so they can never collide with the null bucket.
    """
    specs = {
        "from_addr_hash": ("from_addr", CATEGORICAL_FEATURES["from_addr_hash"]["n_bins"]),
        "to_addr_hash": ("to_addr", CATEGORICAL_FEATURES["to_addr_hash"]["n_bins"]),
        "data_4bytes_hash": ("data_4bytes", CATEGORICAL_FEATURES["data_4bytes_hash"]["n_bins"]),
    }
    new_cols = []
    for alias, (src_col, n_bins) in specs.items():
        hashed = (
            pl.when(pl.col(src_col).is_null())
            .then(pl.lit(0))
            .otherwise((pl.col(src_col).hash().mod(n_bins - 1) + 1))
            .cast(pl.Int64)
            .alias(alias)
        )
        new_cols.append(hashed)
    return df.with_columns(new_cols)


def compute_labels(
    df: pl.DataFrame,
    capture_end_ms: int | None = None,
    drop_maturity_hours: int = 24,
) -> pl.DataFrame:
    """derive the three binary risk labels.

    is_reverted: 1 if receipt_status==0, 0 if ==1, -1 if NULL. -1 means we never saw a receipt (probably dropped), and those rows are excluded from training / eval.
    is_mev_victim: 1 if mev_type == 'sandwich_victim', else 0. attacker rows (156 of them on mainnet) are filtered upstream
    is_dropped: 1 if no canonical-chain block ever included the tx within `drop_maturity_hours` of first arrival. 
        Same-nonce replacements get counted as drops on the displaced hash because their included_at_block_height stays NULL.

    """
    is_dropped_raw = (
        ((pl.col("included_at_block_height").cast(pl.Int64) == 0)
         | pl.col("included_at_block_height").is_null())
        & pl.col("receipt_status").is_null()
    )

    if capture_end_ms is not None:
        maturity_window_ms = drop_maturity_hours * 3_600_000
        cutoff_ms = capture_end_ms - maturity_window_ms
        is_dropped = (
            pl.when(pl.col("timestamp_ms") > cutoff_ms)
            .then(pl.lit(-1))
            .when(is_dropped_raw)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Float32)
            .alias("is_dropped")
        )
    else:
        is_dropped = (
            pl.when(is_dropped_raw)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Float32)
            .alias("is_dropped")
        )

    return df.with_columns(
        pl.when(pl.col("receipt_status").is_null())
        .then(pl.lit(-1))
        .when(pl.col("receipt_status").cast(pl.Int64) == 0)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Float32)
        .alias("is_reverted"),

        pl.when(pl.col("mev_type") == "sandwich_victim")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Float32)
        .alias("is_mev_victim"),

        is_dropped,
    )
