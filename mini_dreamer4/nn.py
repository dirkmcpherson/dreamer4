"""Efficient block-causal transformer pieces (Dreamer 4, Sec. 3.4 "Efficient Transformer").

Pre-norm RMSNorm, SwiGLU, QKNorm, attention-logit soft capping, RoPE (1D over time,
2D axial over the patch grid), and axial space / time attention where only every
``time_every``-th block carries a causal temporal attention layer.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        hidden = max(8, (hidden + 7) // 8 * 8)
        self.w_in = nn.Linear(dim, 2 * hidden)
        self.w_out = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        a, g = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(a * F.silu(g))


# ---------------------------------------------------------------- rotary embeddings

def rope_angles_1d(positions: Tensor, dim_half: int, base: float = 10000.0) -> Tensor:
    """positions (N,) -> angles (N, dim_half)."""
    inv = base ** (-torch.arange(0, dim_half, device=positions.device, dtype=torch.float32) / dim_half)
    return positions.float()[:, None] * inv[None, :]


def rope_angles_2d(pos_rc: Tensor, valid: Tensor, dim_half: int, base: float = 10000.0) -> Tensor:
    """Axial 2D RoPE: half of the rotary pairs encode the row, half the column.

    pos_rc (N, 2) float, valid (N,) bool. Tokens with valid == False (latents, registers,
    other non-spatial tokens) get a zero angle, i.e. the identity rotation.
    """
    quarter = dim_half // 2
    ang_r = rope_angles_1d(pos_rc[:, 0], quarter, base)
    ang_c = rope_angles_1d(pos_rc[:, 1], dim_half - quarter, base)
    angles = torch.cat((ang_r, ang_c), dim=-1)
    return angles * valid.float()[:, None]


def apply_rotary(x: Tensor, angles: Tensor) -> Tensor:
    """x (..., N, D) with D even; angles (N, D/2) -> rotated x."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


# ---------------------------------------------------------------- attention

class Attention(nn.Module):
    """Multi-head self attention with QKNorm, logit soft capping and an optional bool mask."""

    def __init__(self, dim: int, heads: int, dim_head: int, qk_norm: bool = True, softcap: float | None = 50.0):
        super().__init__()
        assert dim_head % 4 == 0, "dim_head must be divisible by 4 for axial 2D RoPE"
        inner = heads * dim_head
        self.heads, self.dim_head = heads, dim_head
        self.softcap = softcap
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)
        self.q_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()

    def forward(self, x: Tensor, angles: Tensor | None = None, mask: Tensor | None = None, causal: bool = False) -> Tensor:
        n, length, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(n, length, self.heads, self.dim_head).transpose(1, 2) for t in (q, k, v))
        q, k = self.q_norm(q), self.k_norm(k)
        if angles is not None:
            q, k = apply_rotary(q, angles), apply_rotary(k, angles)

        allowed = None
        if mask is not None:
            allowed = mask
        if causal:
            causal_mask = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
            allowed = causal_mask if allowed is None else (allowed & causal_mask)

        if self.softcap is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        else:
            logits = (q @ k.transpose(-1, -2)) * (self.dim_head ** -0.5)
            logits = torch.tanh(logits / self.softcap) * self.softcap
            if allowed is not None:
                logits = logits.masked_fill(~allowed, float("-inf"))
            out = logits.softmax(dim=-1) @ v
        return self.to_out(out.transpose(1, 2).reshape(n, length, -1))


class AxialBlock(nn.Module):
    """Space attention (within a time step) + optional causal time attention + SwiGLU."""

    def __init__(self, dim: int, heads: int, dim_head: int, has_time: bool, ff_mult: float = 4.0,
                 qk_norm: bool = True, softcap: float | None = 50.0):
        super().__init__()
        self.has_time = has_time
        self.norm_space = RMSNorm(dim)
        self.space = Attention(dim, heads, dim_head, qk_norm, softcap)
        if has_time:
            self.norm_time = RMSNorm(dim)
            self.time = Attention(dim, heads, dim_head, qk_norm, softcap)
        self.norm_ff = RMSNorm(dim)
        self.ff = SwiGLU(dim, ff_mult)

    def forward(self, x: Tensor, space_mask: Tensor | None, space_angles: Tensor | None,
                time_angles: Tensor, time_token_mask: Tensor | None) -> Tensor:
        b, t, s, d = x.shape
        h = self.space(self.norm_space(x).reshape(b * t, s, d), angles=space_angles, mask=space_mask)
        x = x + h.view(b, t, s, d)
        if self.has_time:
            xt = self.norm_time(x).permute(0, 2, 1, 3).reshape(b * s, t, d)
            h = self.time(xt, angles=time_angles, causal=True).view(b, s, t, d).permute(0, 2, 1, 3)
            if time_token_mask is not None:  # restrict temporal mixing to a subset of tokens
                h = h * time_token_mask.to(h.dtype)[None, None, :, None]
            x = x + h
        return x + self.ff(self.norm_ff(x))


class BlockCausalTransformer(nn.Module):
    """Stack of AxialBlocks. Input / output shape (B, T, S, D).

    ``space_pos`` (S, 2) and ``space_valid`` (S,) give patch-grid coordinates for the
    2D RoPE in the spatial layers; non-spatial tokens use ``space_valid=False``.
    ``space_mask`` (S, S) bool restricts which tokens may attend to which (True = allowed).
    ``time_token_mask`` (S,) bool optionally restricts temporal attention to a token subset.
    """

    def __init__(self, dim: int, depth: int, heads: int = 4, dim_head: int = 64, time_every: int = 4,
                 ff_mult: float = 4.0, qk_norm: bool = True, softcap: float | None = 50.0,
                 final_norm: bool = True):
        super().__init__()
        self.dim_head = dim_head
        self.blocks = nn.ModuleList([
            AxialBlock(dim, heads, dim_head, has_time=((i + 1) % time_every == 0), ff_mult=ff_mult,
                       qk_norm=qk_norm, softcap=softcap)
            for i in range(depth)
        ])
        if not any(b.has_time for b in self.blocks):
            self.blocks[-1] = AxialBlock(dim, heads, dim_head, has_time=True, ff_mult=ff_mult,
                                         qk_norm=qk_norm, softcap=softcap)
        self.final_norm = RMSNorm(dim) if final_norm else nn.Identity()

    def forward(self, x: Tensor, space_mask: Tensor | None = None, space_pos: Tensor | None = None,
                space_valid: Tensor | None = None, time_token_mask: Tensor | None = None,
                time_offset: int = 0) -> Tensor:
        b, t, s, d = x.shape
        half = self.dim_head // 2
        time_angles = rope_angles_1d(torch.arange(t, device=x.device) + time_offset, half)
        space_angles = None
        if space_pos is not None:
            if space_valid is None:
                space_valid = torch.ones(s, dtype=torch.bool, device=x.device)
            space_angles = rope_angles_2d(space_pos.to(x.device), space_valid.to(x.device), half)
        for block in self.blocks:
            x = block(x, space_mask, space_angles, time_angles, time_token_mask)
        return self.final_norm(x)


# ---------------------------------------------------------------- loss normalization

class LossNormalizer(nn.Module):
    """Divide a loss by a running estimate of its root-mean-square (paper, end of Sec. 3)."""

    def __init__(self, beta: float = 0.99, eps: float = 1e-8):
        super().__init__()
        self.beta, self.eps = beta, eps
        self.register_buffer("ema_sq", torch.tensor(1.0))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, loss: Tensor) -> Tensor:
        if self.training:
            sq = loss.detach().float().pow(2)
            if not bool(self.initialized):
                self.ema_sq.copy_(sq)
                self.initialized.fill_(True)
            else:
                self.ema_sq.mul_(self.beta).add_((1 - self.beta) * sq)
        return loss / self.ema_sq.sqrt().clamp_min(self.eps)
