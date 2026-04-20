# MempoolRiskBench

Code and data for our BlockSys 2026 submission *"MempoolRiskBench: A 91.7M-Transaction Pre-Inclusion Benchmark for Ethereum and a Deployable Sandwich-Victim Triage Pipeline"*.

## Downloads

Three downloads:

1. **Processed dataset** (2.74 GB):  required, this is what the models eat.
2. **Pre-trained checkpoints** (418 MB): recommended, saves ~2 days of GPU time.
3. **Raw parquet** (~15 GB): optional if want to rebuild features or retrain from scratch.

## Environment setup

- `install-local.sh` — local conda setup. Makes a python 3.12 env, installs deps via `uv pip`.
- `install-vastai.sh` — vast.ai container setup (base image `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`).
- `requirements.txt` — raw dep list. `mamba_ssm` and `causal_conv1d` need `--no-build-isolation` when pip-installing.

## 1. Processed dataset (2.74 GB)

Download `processed.tar`:

<https://osf.io/download/3rsxg/?view_only=38f4c85b38724149b404a54059e80e91>

Extract under `data/processed/` so you get:

```
data/processed/
  train/{features,labels,dex_mask}.parquet
  val/{features,labels,dex_mask}.parquet
  test/{features,labels,dex_mask}.parquet
  manifest.json
```

## 2. Pre-trained checkpoints (418 MB)

All models and variants, ready to eval. Download and extract at the repo root

<https://osf.io/download/hs6cv/?view_only=38f4c85b38724149b404a54059e80e91>

```
checkpoints/
  mlp/last.ckpt
  mlp_no_identity/last.ckpt
  lstm/last.ckpt
  transformer/last.ckpt
  mamba3_constant/last.ckpt
  mamba3_physical/last.ckpt
  mamba3_physical_no_identity/last.ckpt
results/
  lgbm/model/{revert,mev,drop,scale_pos_weights}.pkl
  lgbm_no_addr/model/{revert,mev,drop,scale_pos_weights}.pkl
  lgbm_no_identity/model/{revert,mev,drop,scale_pos_weights}.pkl
  logreg/model/{revert,mev,drop,scaler}.pkl
  heuristic/thresholds.json
```

### 2.1 Structure

Checkpoints (PyTorch Lightning):

| Checkpoint                                          | Config                                     | Model                          | Size   |
|-----------------------------------------------------|--------------------------------------------|--------------------------------|--------|
| `checkpoints/mlp/last.ckpt`                         | `configs/mlp.yaml`                         | MLP (14 features)              | 50 MB  |
| `checkpoints/mlp_no_identity/last.ckpt`             | `configs/mlp_no_identity.yaml`             | MLP (no-identity, 11 features) | 18 MB  |
| `checkpoints/lstm/last.ckpt`                        | `configs/lstm.yaml`                        | LSTM                           | 62 MB  |
| `checkpoints/transformer/last.ckpt`                 | `configs/transformer.yaml`                 | Transformer                    | 104 MB |
| `checkpoints/mamba3_constant/last.ckpt`             | `configs/mamba3_constant.yaml`             | Mamba-3 (constant delta)       | 82 MB  |
| `checkpoints/mamba3_physical/last.ckpt`             | `configs/mamba3_physical.yaml`             | Mamba-3 (physical delta)       | 82 MB  |
| `checkpoints/mamba3_physical_no_identity/last.ckpt` | `configs/mamba3_physical_no_identity.yaml` | Mamba-3 phys. (no-identity)    | 49 MB  |

Sklearn + heuristic (pickles):

| Directory                             | Config                            | Model                  | Size   |
|---------------------------------------|-----------------------------------|------------------------|--------|
| `results/lgbm/model/`                 | `configs/lgbm.yaml`               | LightGBM (14 cols)     | 2.4 MB |
| `results/lgbm_no_addr/model/`         | `configs/lgbm_no_addr.yaml`       | LightGBM (no-address)  | 260 KB |
| `results/lgbm_no_identity/model/`     | `configs/lgbm_no_identity.yaml`   | LightGBM (no-identity) | 104 KB |
| `results/logreg/model/`               | `configs/logreg.yaml`             | LogReg                 | 20 KB  |
| `results/heuristic/thresholds.json`   | `configs/heuristic.yaml`          | Gas-price heuristic    | <1 KB  |

Trained on the full 71.9M-row train split (Feb 1 – Mar 24) with seed 42.

### 2.2 Running eval

neural models: pass `--checkpoint`:

```bash
# MLP
python -m src.evaluation.run_eval --run-id mlp --split test \
    --config configs/mlp.yaml \
    --checkpoint checkpoints/mlp/last.ckpt \
    --bootstrap 100

# Mamba-3 with physical delta
python -m src.evaluation.run_eval --run-id mamba3_physical --split test \
    --config configs/mamba3_physical.yaml \
    --checkpoint checkpoints/mamba3_physical/last.ckpt \
    --bootstrap 100
```

sklearn + heuristic: use `run_eval`, make sure `--run-id` matches the extracted folder name:

```bash
# LightGBM full
python -m src.evaluation.run_eval --run-id lgbm --split test \
    --config configs/lgbm.yaml --bootstrap 100

# Logistic regression
python -m src.evaluation.run_eval --run-id logreg --split test \
    --config configs/logreg.yaml --bootstrap 100

# Heuristic
python -m src.evaluation.run_eval --run-id heuristic --split test \
    --config configs/heuristic.yaml --bootstrap 100
```

Each run write to `results/<run-id>/{predictions.parquet, metrics.json, pr_curve.json}`.

### 2.3 Extra eval flags

`run_eval` optional flags for the extra evals mentioned in the paper:

- `--subset dex` — DEX-only MEV metrics
- `--stratify traffic` — mempool-pressure quartiles
- `--corruption shuffle` or `--corruption quantize` — delta-t adapter corruption audit. Only meaningful on the Mamba-3 physical checkpoint.

Example — DEX-subset MEV with the MLP no-identity checkpoint

```bash
python -m src.evaluation.run_eval --run-id mlp_no_identity_dex \
    --split test --config configs/mlp_no_identity.yaml \
    --checkpoint checkpoints/mlp_no_identity/last.ckpt \
    --subset dex --bootstrap 100
```

## 3. Raw dataset (optional)

You only need the raw parquet if you want to audit the feature engineering, change the feature whitelist, or extend the capture window.

- **Raw dataset (~15 GB zip, 500 parquet files, CC-BY-4.0)**
  - the file is bigger than OSF's per-file limit, so we'll mirror it somewhere else and update this README later

Extract under `data/raw/` so the layout is `data/raw/joined/*.parquet` whith 500 files. Then rebuild the processed splits and retrain:

```bash
# 1. rebuild processed splits (takes ~2 min on a decent machine)
python -m src.data.build_dataset --raw-dir data/raw --out-dir data/processed

# 2. retrain from scratch ~2 days total on a single RTX 5080
python -m src.training.train --config configs/lgbm.yaml --run-id lgbm
```

Every full build runs a shuffled-label leakage check (`src/data/build_dataset.py:_leakage_audit`).
If shuffled-label AUC > 0.55, the build fails.

## Training from scratch (optional)

```bash
# train one model (config → run id)
python -m src.training.train --config configs/lgbm.yaml --run-id lgbm

# eval on test (writes results/<run-id>/{predictions.parquet, metrics.json})
python -m src.evaluation.run_eval \
    --run-id lgbm --split test --config configs/lgbm.yaml \
    --bootstrap 100
```

`--data-dir` defaults to `data/processed/`, `--bootstrap N` sets the bootstrap CI width, `100` for the paper.

Configs in `configs/`:

| Family    | Configs                                                                                                                                                  |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| heuristic | `heuristic.yaml`                                                                                                                                         |
| sklearn   | `logreg.yaml`, `lgbm.yaml`, `lgbm_no_addr.yaml`, `lgbm_no_identity.yaml`                                                                                 |
| neural    | `mlp.yaml`, `mlp_no_identity.yaml`, `lstm.yaml`, `transformer.yaml`, `mamba3_constant.yaml`, `mamba3_physical.yaml`, `mamba3_physical_no_identity.yaml`   |

system memory requirement: LightGBM and LogReg training loads the whole train split as one big numpy matrix. Full LGBM (14 cols) needs >20 GB of system RAM, LogReg (11 cols) needs less. Suggest using vastai instance with attached setup script.

All 12 configs (train + test-eval) can be run in one shot via `bash scripts/run_all.sh`; the script is resumable — skips TRAIN if a checkpoint already exists and skips EVAL if a valid `metrics.json` is already written.

## Reproducibility

Scripts under `scripts/` produces the tables and figures off the cached predictions from `run_eval`

## Hardware

Everything in the paper ran on a single consumer-grade RTX 5080. Sequence models train at batch 32, seq-len 1024 in BF16. Takes about 2 days on one GPU.

## Licence

Code: Apache 2.0. Processed dataset: CC-BY-4.0.

> OSF view-only project link if needed: <https://osf.io/hfbku/overview?view_only=38f4c85b38724149b404a54059e80e91>
