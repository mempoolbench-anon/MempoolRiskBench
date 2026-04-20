"""3 binary heads (rev / mev / drop). produces logits, not probabilities."""

import torch.nn as nn


class MultiTaskHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.revert = nn.Linear(d_model, 1)
        self.mev = nn.Linear(d_model, 1)
        self.drop = nn.Linear(d_model, 1)

    def forward(self, h):
        # h: (..., d_model)  ->  dict of (..., 1) logits
        return {
            "revert": self.revert(h),
            "mev":    self.mev(h),
            "drop":   self.drop(h),
        }
