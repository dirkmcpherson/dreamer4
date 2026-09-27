"""Interactive dynamics model with shortcut forcing (Dreamer 4, Sec. 3.2, Eq. 4-6).

Per time step the transformer sees: one token for (signal level, step size), one action
token, ``num_registers`` register tokens and ``S_z`` spatial tokens made by packing
``pack`` consecutive tokenizer latents into one. Every token attends to every other token
within the step; temporal attention is causal.

The network predicts clean latents (x-prediction). Flow-matching loss at the finest step
size, bootstrap loss at coarser step sizes, both weighted by the ramp w(sigma) = 0.9 sigma + 0.1
and normalized by running RMS.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .nn import BlockCausalTransformer, LossNormalizer, RMSNorm


def shift_actions(actions: Tensor | None, batch: int, time: int, device) -> tuple[Tensor | None, Tensor]:
    """Convert actions "taken at frame t" into actions "that produced frame t" (shift right by one).

    Returns (prev_actions, valid) where valid[:, 0] is False (no action produced the first frame).
    """
    valid = torch.ones(batch, time, dtype=torch.bool, device=device)
    valid[:, 0] = False
    if actions is None:
        return None, torch.zeros(batch, time, dtype=torch.bool, device=device)
    assert actions.shape[1] >= time - 1, f"need at least {time - 1} actions, got {actions.shape[1]}"
    prev = actions.new_zeros(batch, time, *actions.shape[2:])
    prev[:, 1:] = actions[:, : time - 1]
    return prev, valid


class ShortcutDynamics(nn.Module):
    def __init__(
        self,
        num_latents: int,
        latent_dim: int,
        pack: int = 2,
        dim: int = 256,
        depth: int = 6,
        heads: int = 4,
        dim_head: int = 64,
        time_every: int = 4,
        num_registers: int = 4,
        k_max: int = 8,
        action_dim: int = 0,              # continuous action components (linear projection)
        num_discrete_actions: int = 0,    # categorical action (embedding lookup)
        loss_norm: bool = True,
        bootstrap_warmup: int = 0,        # steps during which only the flow loss is used
        bootstrap_fraction: float | None = 0.25,  # fraction of (b, t) rows trained with the bootstrap target;
                                          # None samples the step size uniformly over powers of two as in the paper
        ctx_noise: float | None = None,   # noise fraction mixed into context latents at inference (default 1 / k_max, the finest trained level)
        softcap: float | None = 50.0,
        x_skip: bool = False,             # parameterize x_hat = sigma * z_noised + net(...) (EDM-style skip)
    ):
        super().__init__()
        assert num_latents % pack == 0
        assert k_max > 1 and (k_max & (k_max - 1)) == 0, "k_max must be a power of two"
        self.num_latents, self.latent_dim, self.pack = num_latents, latent_dim, pack
        self.n_spatial, self.d_spatial = num_latents // pack, latent_dim * pack
        self.k_max, self.max_exp = k_max, int(math.log2(k_max))
        self.num_registers = num_registers
        self.action_dim, self.num_discrete_actions = action_dim, num_discrete_actions
        self.bootstrap_warmup = bootstrap_warmup
        self.bootstrap_fraction = bootstrap_fraction
        self.ctx_noise = (1.0 / k_max) if ctx_noise is None else ctx_noise
        self.x_skip = x_skip

        self.in_proj = nn.Linear(self.d_spatial, dim)
        self.spatial_pos = nn.Parameter(torch.randn(self.n_spatial, dim) * 0.02)
        self.registers = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.signal_embed = nn.Embedding(k_max + 1, dim // 2)   # index k_max == clean context
        self.step_embed = nn.Embedding(self.max_exp + 1, dim // 2)
        self.action_base = nn.Parameter(torch.randn(dim) * 0.02)
        self.action_proj = nn.Linear(action_dim, dim) if action_dim > 0 else None
        self.action_table = nn.Embedding(num_discrete_actions, dim) if num_discrete_actions > 0 else None

        self.transformer = BlockCausalTransformer(dim, depth, heads, dim_head, time_every, softcap=softcap, final_norm=False)
        self.out_norm = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, self.d_spatial)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.loss_norm = LossNormalizer() if loss_norm else None
        self.register_buffer("train_steps", torch.tensor(0), persistent=True)

    # ------------------------------------------------------------------ network
    def _action_tokens(self, actions: Tensor | None, valid: Tensor, b: int, t: int) -> Tensor:
        tok = self.action_base.expand(b, t, -1)
        if actions is None:
            return tok
        if self.action_table is not None and actions.dtype in (torch.int64, torch.int32):
            emb = self.action_table(actions.long().squeeze(-1) if actions.ndim == 3 else actions.long())
        else:
            assert self.action_proj is not None, "model built without continuous actions"
            emb = self.action_proj(actions.float())
        return tok + emb * valid.to(emb.dtype)[..., None]

    def predict(self, z_noised: Tensor, signal_idx: Tensor, step_idx: Tensor,
                prev_actions: Tensor | None = None, action_valid: Tensor | None = None) -> Tensor:
        """x-prediction of the clean latents. z_noised (B, T, N_l, d_b); signal_idx, step_idx (B, T) long."""
        b, t = z_noised.shape[:2]
        if action_valid is None:
            action_valid = torch.zeros(b, t, dtype=torch.bool, device=z_noised.device)
        sp = self.in_proj(z_noised.reshape(b, t, self.n_spatial, self.d_spatial)) + self.spatial_pos
        flow = torch.cat((self.signal_embed(signal_idx), self.step_embed(step_idx)), dim=-1)[:, :, None]
        act = self._action_tokens(prev_actions, action_valid, b, t)[:, :, None]
        reg = self.registers.expand(b, t, -1, -1)
        h = self.transformer(torch.cat((flow, act, reg, sp), dim=2))
        out = self.out_proj(self.out_norm(h[:, :, 2 + self.num_registers:]))
        out = out.reshape(b, t, self.num_latents, self.latent_dim)
        if self.x_skip:
            sigma = signal_idx.float() / self.k_max
            out = out + sigma[..., None, None] * z_noised
        return out

    # ------------------------------------------------------------------ shortcut forcing loss
    def forward(self, z1: Tensor, actions: Tensor | None = None) -> tuple[Tensor, dict]:
        """z1 (B, T, N_l, d_b) clean latents; actions (B, T, A) taken at frame t (or None for unlabeled)."""
        b, t, device = z1.shape[0], z1.shape[1], z1.device
        e_max, k_max = self.max_exp, self.k_max
        prev_actions, valid = shift_actions(actions, b, t, device)

        # step size d = 2^-e sampled uniformly over powers of two; sigma uniform on the grid reachable by d
        if self.bootstrap_fraction is None:
            e = torch.randint(0, e_max + 1, (b, t), device=device)
        else:
            coarse = torch.randint(0, e_max, (b, t), device=device)
            is_boot = torch.rand(b, t, device=device) < self.bootstrap_fraction
            e = torch.where(is_boot, coarse, torch.full_like(coarse, e_max))
        # during warmup every row uses the flow target (clean latents) whatever its step size, so the
        # step-size embeddings are trained before they are used to build bootstrap targets
        warmup = self.training and int(self.train_steps) < self.bootstrap_warmup
        n_grid = 2 ** e
        j = (torch.rand(b, t, device=device) * n_grid).floor().long().clamp(max=n_grid - 1)
        sigma = j.float() / n_grid.float()
        signal_idx = (sigma * k_max).round().long()

        z0 = torch.randn_like(z1)
        s4 = sigma[..., None, None]
        zt = (1 - s4) * z0 + s4 * z1
        x_hat = self.predict(zt, signal_idx, e, prev_actions, valid)

        is_flow = (e == e_max) | warmup
        flow_pt = (x_hat - z1).pow(2).mean(dim=(2, 3))

        boot_pt = torch.zeros_like(flow_pt)
        if bool((~is_flow).any()):
            with torch.no_grad():
                e_half = (e + 1).clamp(max=e_max)
                d_half = 0.5 ** (e + 1).float()
                x1 = self.predict(zt, signal_idx, e_half, prev_actions, valid)
                b1 = (x1 - zt) / (1 - s4)
                z_prime = zt + b1 * d_half[..., None, None]
                sigma2 = sigma + d_half
                sig2_idx = (sigma2 * k_max).round().long().clamp(max=k_max)
                x2 = self.predict(z_prime, sig2_idx, e_half, prev_actions, valid)
                b2 = (x2 - z_prime) / (1 - sigma2[..., None, None]).clamp_min(1e-4)
                target = (b1 + b2) / 2
            v_hat = (x_hat - zt) / (1 - s4)
            boot_pt = (1 - sigma) ** 2 * (v_hat - target).pow(2).mean(dim=(2, 3))

        # both terms are in x-space units (the (1 - sigma)^2 factor), so they share one ramp
        # weight and one RMS normalizer, as in Eq. 6 of the paper
        ramp = 0.9 * sigma + 0.1
        per_row = torch.where(is_flow, flow_pt, boot_pt) * ramp
        loss = per_row.mean()
        if self.loss_norm is not None:
            loss = self.loss_norm(loss)
        n_flow, n_boot = is_flow.sum().clamp_min(1), (~is_flow).sum().clamp_min(1)
        if self.training:
            self.train_steps += 1
        stats = {"flow_mse": (flow_pt * is_flow).sum() / n_flow, "boot_mse": (boot_pt * ~is_flow).sum() / n_boot,
                 "frac_flow": is_flow.float().mean()}
        return loss, {k: v.detach() for k, v in stats.items()}

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, z_ctx: Tensor, actions: Tensor | None, horizon: int, num_steps: int = 4,
               ctx_noise: float | None = None) -> Tensor:
        """Autoregressively generate ``horizon`` frames after ``z_ctx`` (B, T0, N_l, d_b).

        ``actions`` (B, T0 + horizon, A) are the actions taken at each frame (the action at the
        last generated frame is unused). Returns (B, T0 + horizon, N_l, d_b).
        """
        assert self.k_max % num_steps == 0 and num_steps & (num_steps - 1) == 0
        k_max, e_max = self.k_max, self.max_exp
        ctx_noise = self.ctx_noise if ctx_noise is None else ctx_noise
        e_gen = int(math.log2(num_steps))
        b, device = z_ctx.shape[0], z_ctx.device
        total = z_ctx.shape[1] + horizon
        prev_actions, valid = shift_actions(actions, b, total, device)
        ctx_sig = min(k_max - 1, int(round((1 - ctx_noise) * k_max)))
        z = z_ctx.clone()
        for _ in range(horizon):
            t0 = z.shape[1]
            ctx = z if ctx_noise == 0 else (1 - ctx_noise) * z + ctx_noise * torch.randn_like(z)
            zn = torch.randn(b, 1, self.num_latents, self.latent_dim, device=device)
            pa = prev_actions[:, : t0 + 1] if prev_actions is not None else None
            va = valid[:, : t0 + 1]
            for k in range(num_steps):
                sigma = k / num_steps
                sig = torch.full((b, t0 + 1), ctx_sig, dtype=torch.long, device=device)
                sig[:, -1] = int(round(sigma * k_max))
                step = torch.full((b, t0 + 1), e_max, dtype=torch.long, device=device)
                step[:, -1] = e_gen
                x_hat = self.predict(torch.cat((ctx, zn), dim=1), sig, step, pa, va)[:, -1:]
                zn = zn + (x_hat - zn) / (1 - sigma) / num_steps
            z = torch.cat((z, zn.clamp(-1, 1)), dim=1)
        return z
