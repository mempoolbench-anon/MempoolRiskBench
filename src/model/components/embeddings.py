"""shared categorical embedding tables + input projection.

every neural model (MLP, LSTM, Transformer, Mamba-3) goes through these,
so feature preprocessing stays identical across variants.

embedding sizes (see schema.CATEGORICAL_FEATURES):
    from_addr   : 131072 -> 16d
    to_addr     : 131072 -> 16d
    data_4bytes :  16384 ->  8d
nulls map to bin 0.

projection: concat(11 numeric, 16+16+8 embedded) = 51d -> Linear -> LayerNorm.
"""

import torch
import torch.nn as nn

from src.data.schema import CATEGORICAL_FEATURES, D_INPUT


class CategoricalEmbedding(nn.Module):
    """embedding tables for the 3 categorical features.

    drop_categorical=True is the identity-ablation hook: returns zeros
    of the same concatenated shape so downstream tensor shapes don't
    change.
    """

    def __init__(self, embed_specs=None, drop_categorical=False):
        super().__init__()
        if embed_specs is None:
            embed_specs = CATEGORICAL_FEATURES
        specs = list(embed_specs.values())
        self.embeddings = nn.ModuleList([
            nn.Embedding(s["n_bins"], s["embed_dim"]) for s in specs
        ])
        self.drop_categorical = drop_categorical
        self.total_embed_dim = sum(s["embed_dim"] for s in specs)

    def forward(self, x_cat):
        # x_cat: (..., 3) long  ->  (..., 40) float
        if self.drop_categorical:
            shape = x_cat.shape[:-1] + (self.total_embed_dim,)
            ref = self.embeddings[0].weight
            return torch.zeros(shape, dtype=ref.dtype, device=ref.device)
        parts = [emb(x_cat[..., i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=-1)


class InputProjection(nn.Module):
    """concat(numeric, embedded) -> Linear -> LayerNorm."""

    def __init__(self, d_input=D_INPUT, d_model=256):
        super().__init__()
        self.proj = nn.Linear(d_input, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_num, x_embed):
        x = torch.cat([x_num, x_embed], dim=-1)
        return self.norm(self.proj(x))
