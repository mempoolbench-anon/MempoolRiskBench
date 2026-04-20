#!/bin/bash
# Self-contained install script for local machines with preinstalled conda.
# Creates a python 3.12 conda env and installs dependencies via uv pip.
# Requires: conda on PATH, NVIDIA driver supporting CUDA 13.0.
set -e

ENV_NAME="${ENV_NAME:-p312cu13}"
conda create -y -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"

echo "=== Installing uv ==="
python -m pip install --upgrade pip
python -m pip install uv

echo "=== Installing PyTorch 2.11.0 + CUDA 13.0 ==="
uv pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.11.0 torchvision torchaudio

echo "=== Installing PyTorch Geometric ==="
uv pip install --find-links https://data.pyg.org/whl/torch-2.11.0+cu130.html \
    torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster

echo "=== Installing ML essentials ==="
uv pip install \
    lightning xformers numpy pandas pyarrow scipy matplotlib scikit-learn \
    transformers datasets accelerate xgboost lightgbm tilelang \
    wandb jupyterlab tqdm packaging wheel

echo "=== Installing flash-attn ==="
uv pip install flash-attn --no-build-isolation

echo "=== Installing causal-conv1d ==="
uv pip install causal-conv1d --no-build-isolation

echo "=== Cloning and installing mamba ==="
if [ ! -d /tmp/mamba ]; then
    git clone https://github.com/state-spaces/mamba.git /tmp/mamba
fi
uv pip install -e /tmp/mamba --no-build-isolation

echo "=== All installations complete (env: $ENV_NAME) ==="
