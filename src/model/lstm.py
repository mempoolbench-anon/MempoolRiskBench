"""causal LSTM sequence baseline. ignores delta_t."""

import torch.nn as nn

from src.model.components.embeddings import CategoricalEmbedding, InputProjection
from src.model.components.heads import MultiTaskHead


class LSTMBaseline(nn.Module):
    def __init__(self, d_model=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.cat_embed = CategoricalEmbedding()
        self.input_proj = InputProjection(d_model=d_model)
        # nn.LSTM applies `dropout` between layers only; no-op if n_layers=1.
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model, num_layers=n_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = MultiTaskHead(d_model)

    def forward(self, x_num, x_cat, **_unused):
        # x_num (B,L,11), x_cat (B,L,3) -> {'revert','mev','drop'} (B,L,1)
        x_embed = self.cat_embed(x_cat)
        x = self.input_proj(x_num, x_embed)
        h, _ = self.lstm(x)
        return self.head(h)
