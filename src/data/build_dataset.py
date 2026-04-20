"""build the processed train/val/test splits from raw mempool parquet.

usage:
    python -m src.data.build_dataset                          # full
    python -m src.data.build_dataset --smoke --n-txs 50000    # quick check

"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import polars as pl

from src.data.schema import (
    BLACKLISTED_COLUMNS,
    CAPTURE_END_DATE,
    CATEGORICAL_FEATURES,
    DEX_SELECTORS,
    DROP_MATURITY_HOURS,
    LABEL_COLUMNS,
    NUMERIC_FEATURES,
    RAW_STRING_NUMERIC_COLS,
    SORT_KEY,
    SPLIT_RANGES,
)
from src.data.features import (
    compute_categorical_hashes,
    compute_labels,
    compute_log_features,
    compute_mempool_pressure,
    compute_sender_rolling,
    compute_timestamp_delta,
)


def _cast_string_numerics(df: pl.DataFrame) -> pl.DataFrame:
    """Cast string numeric columns to proper types."""
    casts = []
    for col in RAW_STRING_NUMERIC_COLS:
        if col in df.columns:
            casts.append(pl.col(col).cast(pl.Float64))
    if casts:
        df = df.with_columns(casts)
    return df


def _split_by_time(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """split into train/val/test by SPLIT_RANGES (smoke mode only)."""
    splits = {}
    for name, (start, end) in SPLIT_RANGES.items():
        start_ms = int(pl.Series([start]).str.to_datetime().dt.epoch("ms")[0])
        end_ms = int(pl.Series([end]).str.to_datetime().dt.epoch("ms")[0])
        mask = (pl.col(SORT_KEY) >= start_ms) & (pl.col(SORT_KEY) < end_ms)
        splits[name] = df.filter(mask)
    return splits


def _leakage_audit(df: pl.DataFrame, feature_cols: list[str]) -> bool:
    """train lgbm on shuffled labels; if it can predict, features leak.

    we destroy the real signal by random-permuting the labels, then split
    50/50 within the sampled rows: fit on the first half, evaluate auc on
    the second. anything > 0.55 means the model is finding leakage from
    the features alone (something is encoding the label).

    returns True iff audit passes.
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("warn: lightgbm missing, skipping leakage audit")
        return True

    # subsample for speed
    n = min(50_000, len(df))
    sample = df.head(n)

    X = sample.select(feature_cols).to_numpy().astype(np.float32)
    y_real = sample["is_reverted"].to_numpy().astype(np.float32)

    # -1 (no-receipt) rows
    valid = y_real >= 0
    if valid.sum() < 200 or y_real[valid].sum() < 10:
        print("warn: not enough labels for leakage audit; skipping")
        return True

    X_valid = X[valid]
    y_valid = y_real[valid]

    rng = np.random.RandomState(42)
    y_shuffled = y_valid.copy()
    rng.shuffle(y_shuffled)

    # NB. eval on held-out, not the rows we trained on
    split = len(y_shuffled) // 2
    X_train, X_eval = X_valid[:split], X_valid[split:]
    y_train, y_eval = y_shuffled[:split], y_shuffled[split:]

    model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, verbose=-1, n_jobs=1)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_eval)[:, 1]
    auc_shuffled = roc_auc_score(y_eval, preds)

    if auc_shuffled > 0.55:
        print(f"LEAKAGE AUDIT FAILED — shuffled AUC = {auc_shuffled:.4f} > 0.55")
        return False
    print(f"leakage audit ok: shuffled AUC = {auc_shuffled:.4f}")
    return True


def _compute_split_boundaries_ms():
    """SPLIT_RANGES (yyyy-mm-dd strings) -> {split: (start_ms, end_ms)}."""
    boundaries = {}
    for name, (start, end) in SPLIT_RANGES.items():
        start_ms = int(pl.Series([start]).str.to_datetime().dt.epoch("ms")[0])
        end_ms = int(pl.Series([end]).str.to_datetime().dt.epoch("ms")[0])
        boundaries[name] = (start_ms, end_ms)
    return boundaries


def _chunk_boundaries(start_ms: int, end_ms: int, chunk_size_ms: int) -> list[tuple[int, int]]:
    """slice [start_ms, end_ms) into contiguous sub-chunks <= chunk_size_ms.

    last sub-chunk gets the remainder so we always cover the full range.
    """
    span = end_ms - start_ms
    if span <= chunk_size_ms:
        return [(start_ms, end_ms)]
    n = (span + chunk_size_ms - 1) // chunk_size_ms
    step = span // n
    out = []
    cur = start_ms
    for i in range(n):
        nxt = end_ms if i == n - 1 else cur + step
        out.append((cur, nxt))
        cur = nxt
    return out


def _process_chunk(
    raw_dir: str,
    load_start: int,
    sub_start: int,
    sub_end: int,
    capture_end_ms: int,
    sender_snapshot: pl.DataFrame | None,
) -> pl.DataFrame:
    """load+featurise+label one sub-chunk, then trim back to (sub_start, sub_end)."""
    # only project the cols downstream actually needs
    needed_cols = [
        SORT_KEY,
        "gas_price", "gas_tip_cap", "gas_fee_cap",
        "value", "gas", "nonce", "data_size",
        "from_addr", "to_addr", "data_4bytes",
        "receipt_status", "mev_type", "included_at_block_height",
    ]
    df = (
        pl.scan_parquet(os.path.join(raw_dir, "*.parquet"))
        .filter((pl.col(SORT_KEY) >= load_start) & (pl.col(SORT_KEY) < sub_end))
        .select(needed_cols)
        .collect()
    )
    df = df.sort(SORT_KEY)
    df = _cast_string_numerics(df)
    df = compute_log_features(df)
    df = compute_timestamp_delta(df)
    df = compute_sender_rolling(df, sender_snapshot=sender_snapshot)
    df = compute_mempool_pressure(df)
    # compute is_dex_candidate from the raw selector BEFORE compute_categorical_hashes
    # drops the data_4bytes column. eval-only diagnostic, never a model input.
    df = df.with_columns(
        pl.col("data_4bytes").is_in(DEX_SELECTORS).fill_null(False).alias("is_dex_candidate")
    )
    df = compute_categorical_hashes(df)
    df = compute_labels(
        df,
        capture_end_ms=capture_end_ms,
        drop_maturity_hours=DROP_MATURITY_HOURS,
    )
    # drop the lookback rows
    # only loaded for the rolling-feature warm-up
    df = df.filter((pl.col(SORT_KEY) >= sub_start) & (pl.col(SORT_KEY) < sub_end))
    return df


def _compute_sender_snapshot(raw_dir: str, before_ms: int) -> pl.DataFrame:
    """per-sender state at `before_ms` -- used to warm-start rolling features.

    only scans from_addr + gas_price (cheap). without this, val/test splits
    would only "see" each sender's recent history within the lookback
    window instead of their full timeline.

    returns: DataFrame[from_addr, _pre_count, _pre_gas_sum].
    """
    snapshot = (
        pl.scan_parquet(os.path.join(raw_dir, "*.parquet"))
        .filter(pl.col(SORT_KEY) < before_ms)
        .select(["from_addr", "gas_price"])
        .with_columns(pl.col("gas_price").cast(pl.Float64))
        .group_by("from_addr")
        .agg(
            pl.count().cast(pl.Int64).alias("_pre_count"),
            (pl.col("gas_price") / 1e9).log1p().sum().alias("_pre_gas_sum"),
        )
        .collect()
    )
    return snapshot


def build(args):
    """Build the dataset. Two modes:
    full mode: per-split, sub-chunked, with sender_snapshot warm-up.
    smoke mode: single in-memory pass over args.n_txs rows.
    """
    raw_dir = args.raw_dir
    capture_end_ms = int(
        pl.Series([CAPTURE_END_DATE]).str.to_datetime().dt.epoch("ms")[0]
    )
    print(f"loading parquet from {raw_dir}")

    if args.smoke:
        # smoke: just slurp args.n_txs rows and run the whole thing in one go
        df = pl.scan_parquet(os.path.join(raw_dir, "*.parquet"))
        df = df.head(args.n_txs).collect()
        print(f"smoke: loaded {len(df)} rows")
        df = df.sort(SORT_KEY)
        df = _cast_string_numerics(df)
        df = compute_log_features(df)
        df = compute_timestamp_delta(df)
        df = compute_sender_rolling(df)
        df = compute_mempool_pressure(df)
        df = df.with_columns(
            pl.col("data_4bytes").is_in(DEX_SELECTORS).fill_null(False).alias("is_dex_candidate")
        )
        df = compute_categorical_hashes(df)
        df = compute_labels(
            df,
            capture_end_ms=capture_end_ms,
            drop_maturity_hours=DROP_MATURITY_HOURS,
        )

        feature_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES.keys())
        for col in feature_cols:
            assert col not in BLACKLISTED_COLUMNS, f"blacklisted column leaked: {col}"

        print("running leakage audit...")
        audit_ok = _leakage_audit(df, feature_cols)
        if not audit_ok:
            print("WARN: leakage detected!")
        splits = _split_by_time(df)
    else:
        # full mode is per-split + sub-chunked
        # train chunk (~72M rows) > 30 GB OOM
        # feature computation
        boundaries = _compute_split_boundaries_ms()
        all_start = min(s for s, _ in boundaries.values())

        feature_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES.keys())
        keep_cols = (
            list(NUMERIC_FEATURES)
            + list(CATEGORICAL_FEATURES.keys())
            + LABEL_COLUMNS
            + ["is_dex_candidate", SORT_KEY]
        )
        splits = {}
        audit_ok = True
        lookback_ms = 5 * 60 * 1000  # 5 min, enough for delta_t + 60 s pressure
        chunk_size_ms = args.max_chunk_days * 24 * 3600 * 1000

        for split_name, (start_ms, end_ms) in boundaries.items():
            print(f"\n--- {split_name} ---")
            sub_chunks = _chunk_boundaries(start_ms, end_ms, chunk_size_ms)
            print(f"  {len(sub_chunks)} sub-chunk(s)")
            chunk_dfs = []
            for ci, (sub_start, sub_end) in enumerate(sub_chunks):
                load_start = max(all_start, sub_start - lookback_ms)

                sender_snapshot = None
                if load_start > all_start:
                    print(f"  [chunk {ci+1}/{len(sub_chunks)}] sender snapshot...")
                    sender_snapshot = _compute_sender_snapshot(raw_dir, load_start)
                    print(f"    {len(sender_snapshot)} unique senders")

                print(f"  [chunk {ci+1}/{len(sub_chunks)}] loading [{load_start}, {sub_end})")
                df = _process_chunk(
                    raw_dir=raw_dir,
                    load_start=load_start,
                    sub_start=sub_start,
                    sub_end=sub_end,
                    capture_end_ms=capture_end_ms,
                    sender_snapshot=sender_snapshot,
                )
                # shed everything except the 14 features + 3 labels + ts
                # before concatenating, otherwise the train concat blows up.
                chunk_dfs.append(df.select([c for c in keep_cols if c in df.columns]))
                print(f"    kept {len(df)} rows")

            split_df = pl.concat(chunk_dfs, how="vertical_relaxed") if len(chunk_dfs) > 1 else chunk_dfs[0]
            print(f"  {split_name}: {len(split_df)} rows total")
            splits[split_name] = split_df
            chunk_dfs = None  # let the gc reclaim per-chunk frames

            # one audit on the train split is enough -- val/test inherit the same pipeline
            if split_name == "train":
                print("  leakage audit (train)...")
                audit_ok = _leakage_audit(df, feature_cols)
                if not audit_ok:
                    print("  WARN: leakage detected!")

        total = sum(len(v) for v in splits.values())
        print(f"\ntotal rows: {total}")

    # write per-split parquet + a manifest
    os.makedirs(args.out_dir, exist_ok=True)
    output_cols_features = feature_cols
    output_cols_labels = LABEL_COLUMNS

    manifest = {
        "mode": "smoke" if args.smoke else "full",
        "n_txs_requested": args.n_txs if args.smoke else None,
        "total_rows": sum(len(v) for v in splits.values()),
        "feature_columns": output_cols_features,
        "label_columns": output_cols_labels,
        "feature_hash": hashlib.md5(
            ",".join(output_cols_features).encode()
        ).hexdigest(),
        "splits": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "leakage_audit": "passed" if audit_ok else "FAILED",
    }

    for split_name, split_df in splits.items():
        split_dir = os.path.join(args.out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        feat_df = split_df.select(output_cols_features)
        lab_df = split_df.select(output_cols_labels)

        feat_path = os.path.join(split_dir, "features.parquet")
        lab_path = os.path.join(split_dir, "labels.parquet")

        feat_df.write_parquet(feat_path)
        lab_df.write_parquet(lab_path)

        # eval-only diagnostic; not a model input
        if "is_dex_candidate" in split_df.columns:
            dex_path = os.path.join(split_dir, "dex_mask.parquet")
            split_df.select(["is_dex_candidate"]).write_parquet(dex_path)

        n_rows = len(split_df)
        label_rates = {}
        for lc in output_cols_labels:
            vals = split_df[lc].to_numpy()
            valid = vals[vals >= 0]
            if len(valid) > 0:
                label_rates[lc] = float(valid.mean())
            else:
                label_rates[lc] = 0.0

        manifest["splits"][split_name] = {
            "rows": n_rows,
            "label_rates": label_rates,
            "features_path": feat_path,
            "labels_path": lab_path,
        }
        print(f"  {split_name}: {n_rows} rows")

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"done — manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="build the processed mempool risk dataset")
    parser.add_argument("--smoke", action="store_true", help="quick check on a small slice")
    parser.add_argument("--n-txs", type=int, default=50_000, help="rows when --smoke")
    parser.add_argument("--raw-dir", default="data/raw", help="raw parquet dir")
    parser.add_argument("--out-dir", default="data/processed", help="output dir")

    parser.add_argument("--max-chunk-days", type=int, default=21,
                        help="sub-chunk width for per-split processing")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
