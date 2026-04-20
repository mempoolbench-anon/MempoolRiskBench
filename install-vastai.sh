#!/bin/bash
# Self-contained install script for vast.ai instances
# Docker image: pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel
# (PyTorch, CUDA 13.0, cuDNN 9, compilers, cmake, ninja already included)
set -e

export PIP_ROOT_USER_ACTION=ignore

echo "=== Installing PyTorch Geometric ==="
pip install --find-links https://data.pyg.org/whl/torch-2.11.0+cu130.html \
    torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster

echo "=== Installing ML essentials ==="
pip install \
    lightning xformers numpy pandas pyarrow scipy matplotlib scikit-learn \
    transformers datasets accelerate xgboost lightgbm tilelang \
    wandb jupyterlab tqdm packaging wheel

echo "=== Installing flash-attn ==="
pip install flash-attn --no-build-isolation

echo "=== Installing causal-conv1d ==="
pip install causal-conv1d --no-build-isolation

echo "=== Cloning and installing mamba ==="
git clone https://github.com/state-spaces/mamba.git /tmp/mamba
pip install -e /tmp/mamba --no-build-isolation

echo "=== All installations complete ==="
