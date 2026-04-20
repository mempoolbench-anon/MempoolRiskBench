"""physical-time delta  adapter (96 params, per-layer × per-head).

separate from the mamba backbone so the corruption eval can swap it in
and out cleanly.

    delta  = softplus(a_{l,h} · log1p(clip(delta _sec, 1e-3, 300)) + b_{l,h})

a init = 1.0; b is calibrated post-init so the median delta  matches the
mamba-3 default. shape contract: in (B, L), out (B, L, nheads) — the
mamba forward path then does its own rearrange to (B, nheads, L).
the const-delta  variant uses the exact same module but feeds a constant
scalar instead of delta _sec.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.schema import DELTA_CLIP_MAX, DELTA_CLIP_MIN


class PhysicalTimeDeltaAdapter(nn.Module):
    def __init__(self, n_layers=6, n_heads=8, clip_min=DELTA_CLIP_MIN, clip_max=DELTA_CLIP_MAX):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.clip_min = clip_min
        self.clip_max = clip_max

        # a, b are per (layer, head). 6 * 8 * 2 = 96 trainable params total
        self.a = nn.Parameter(torch.ones(n_layers, n_heads))
        self.b = nn.Parameter(torch.zeros(n_layers, n_heads))

    def calibrate_b(self, median_delta, dt_bias_median=None):
        """pick b s.t. softplus(a·log1p(median_delta) + b) ≈ mamba-3 default delta ."""
        log1p_med = math.log1p(max(median_delta, self.clip_min))

        # mamba-3 default delta  range is [1e-3, 1e-1] -> geometric-mean target ≈ 1e-2
        dt_target = math.sqrt(0.001 * 0.1) if dt_bias_median is None else dt_bias_median

        # softplus_inv(y) = log(exp(y) - 1); shortcut at large y to avoid overflow
        sp_inv = dt_target if dt_target > 20 else math.log(math.exp(dt_target) - 1)

        # with a == 1, this is just b = sp_inv - log1p_med
        with torch.no_grad():
            self.b.fill_(sp_inv - log1p_med)

    def forward(self, delta_sec, layer_idx):
        # delta_sec: (B, L)  ->  delta : (B, L, nheads)
        delta = delta_sec.clamp(min=self.clip_min, max=self.clip_max)
        log_delta = torch.log1p(delta)

        a_l = self.a[layer_idx]   # (nheads,)
        b_l = self.b[layer_idx]
        pre = log_delta.unsqueeze(-1) * a_l + b_l   # broadcast -> (B, L, nheads)
        return F.softplus(pre)
