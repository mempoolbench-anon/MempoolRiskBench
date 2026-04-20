"""mamba-3 backbone with the physical-time delta  adapter.

CategoricalEmbedding -> InputProjection(51->256) -> 6× Mamba-3 -> MultiTaskHead.

per layer we intercept Mamba-3's delta  computation:
    default: delta  = softplus(dd_dt + dt_bias)        (from in_proj slice)
    here   : delta  = softplus(a · log1p(clip(delta )) + b) (from the adapter)

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.modules.mamba3 import Mamba3

from src.model.components.embeddings import CategoricalEmbedding, InputProjection
from src.model.components.heads import MultiTaskHead
from src.model.delta_adapter import PhysicalTimeDeltaAdapter


class AdaptedMamba3Layer(nn.Module):
    """Wraps a single Mamba3 layer with DT interception via adapter.

    Instead of using dd_dt from in_proj, we compute DT from the adapter
    using physical inter-arrival times.
    """

    def __init__(self, d_model, layer_idx, adapter, **mamba_kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.adapter = adapter
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba3(d_model=d_model, layer_idx=layer_idx, **mamba_kwargs)

    def forward(self, u, delta_t=None):
        # u: (B, L, d_model). delta_t: (B, L) or None
        residual = u
        u_n = self.norm(u)
        if delta_t is not None and self.adapter is not None:
            out = self._forward_with_adapter(u_n, delta_t)
        else:
            out = self.mamba(u_n)
        return out + residual

    def _forward_with_adapter(self, u, delta_t):
        """re-do mamba's forward but feed delta  from our adapter, not from in_proj."""
        mamba = self.mamba
        batch, seqlen, dim = u.shape

        zxBCdtAtrap = mamba.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                mamba.d_inner, mamba.d_inner,
                mamba.d_state * mamba.num_bc_heads * mamba.mimo_rank,
                mamba.d_state * mamba.num_bc_heads * mamba.mimo_rank,
                mamba.nheads, mamba.nheads, mamba.nheads,
                mamba.num_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=mamba.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=mamba.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=mamba.mimo_rank, g=mamba.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=mamba.mimo_rank, g=mamba.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        _A = -F.softplus(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-mamba.A_floor)

        # *** the only thing we changed vs vanilla mamba-3 forward ***
        # default: DT = softplus(dd_dt + mamba.dt_bias)
        # ours   : DT comes from the per-head adapter on physical delta t
        DT = self.adapter(delta_t, self.layer_idx)   # (B, L, nheads)

        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        # angles broadcast to per-head
        angles = angles.unsqueeze(-2).expand(-1, -1, mamba.nheads, -1)

        # rms norm on B / C as in mamba-3
        B = mamba.B_norm(B)
        C = mamba.C_norm(C)

        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

        y = mamba3_siso_combined(
            Q=C.squeeze(2),
            K=B.squeeze(2),
            V=x,
            ADT=ADT,
            DT=DT,
            Trap=trap,
            Q_bias=mamba.C_bias.squeeze(1),
            K_bias=mamba.B_bias.squeeze(1),
            Angles=angles,
            D=mamba.D,
            Z=z if not mamba.is_outproj_norm else None,
            chunk_size=mamba.chunk_size,
            Input_States=None,
            return_final_states=False,
            cu_seqlens=None,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        if mamba.is_outproj_norm:
            z_flat = rearrange(z, "b l h p -> b l (h p)")
            y = mamba.norm(y, z_flat)

        out = mamba.out_proj(y.to(x.dtype))
        return out


class MempoolMamba(nn.Module):
    """mamba-3 backbone + per-head physical-delta  adapter."""

    def __init__(
        self,
        d_model=256,
        n_layers=6,
        n_heads=8,
        use_physical_delta=True,
        constant_delta=None,
        d_state=128,
        expand=2,
        headdim=64,
        drop_categorical=False,
    ):
        super().__init__()
        self.use_physical_delta = use_physical_delta
        self.constant_delta = constant_delta
        self.d_model = d_model

        # mamba's actual head count = d_inner // headdim (NOT the n_heads arg)
        d_inner = int(expand * d_model)
        actual_nheads = d_inner // headdim

        self.cat_embed = CategoricalEmbedding(drop_categorical=drop_categorical)
        self.input_proj = InputProjection(d_model=d_model)
        self.adapter = PhysicalTimeDeltaAdapter(n_layers=n_layers, n_heads=actual_nheads)

        mamba_kwargs = dict(d_state=d_state, expand=expand, headdim=headdim)
        self.layers = nn.ModuleList([
            AdaptedMamba3Layer(
                d_model=d_model, layer_idx=i, adapter=self.adapter, **mamba_kwargs,
            )
            for i in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.head = MultiTaskHead(d_model)

    def forward(self, x_num, x_cat, delta_t):
        # x_num (B, L, 11), x_cat (B, L, 3) long, delta_t (B, L)
        # returns {'revert','mev','drop'} with (B, L, 1) logits
        x_embed = self.cat_embed(x_cat)
        h = self.input_proj(x_num, x_embed)

        # const-delta  variant: just overwrite the input
        if not self.use_physical_delta and self.constant_delta is not None:
            delta_t = torch.full_like(delta_t, self.constant_delta)

        for layer in self.layers:
            h = layer(h, delta_t=delta_t)

        return self.head(self.final_norm(h))
