"""MLP — single transaction, no sequence context (the "L=1" baseline)."""

import torch
import torch.nn as nn

from src.model.components.embeddings import CategoricalEmbedding
from src.model.components.heads import MultiTaskHead
from src.data.schema import D_INPUT


class MLPBaseline(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, drop_categorical=False):
        super().__init__()
        self.cat_embed = CategoricalEmbedding(drop_categorical=drop_categorical)
        # 51 -> 256 -> 256, GELU + dropout after each hidden layer.
        self.mlp = nn.Sequential(
            nn.Linear(D_INPUT, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = MultiTaskHead(d_model)

    def forward(self, x_num, x_cat, **_unused):
        x_embed = self.cat_embed(x_cat)
        x = torch.cat([x_num, x_embed], dim=-1)
        return self.head(self.mlp(x))
