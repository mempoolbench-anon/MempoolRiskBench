"""causal Transformer baseline with sinusoidal positions + GELU FFN.

delta_t is not consumed here — only token order via the positional
encoding. the encoder is unusable without positions because attention
is order-invariant otherwise.
"""

import math

import torch
import torch.nn as nn

from src.model.components.embeddings import CategoricalEmbedding, InputProjection
from src.model.components.heads import MultiTaskHead


class SinusoidalPositionalEncoding(nn.Module):
    """Vaswani-2017 additive sin/cos positional encoding. fixed, non-trainable."""

    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerBaseline(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=6, d_ffn=1024,
                 dropout=0.1, max_len=4096):
        super().__init__()
        self.cat_embed = CategoricalEmbedding()
        self.input_proj = InputProjection(d_model=d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.pos_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ffn, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = MultiTaskHead(d_model)

    @staticmethod
    def _causal_mask(L, device):
        # strict lower-triangular: positions can only see themselves + the past
        return nn.Transformer.generate_square_subsequent_mask(L, device=device)

    def forward(self, x_num, x_cat, **_unused):
        x_embed = self.cat_embed(x_cat)
        x = self.input_proj(x_num, x_embed)
        x = self.pos_dropout(self.pos_enc(x))
        mask = self._causal_mask(x.size(1), x.device)
        h = self.encoder(x, mask=mask, is_causal=True)
        return self.head(h)
