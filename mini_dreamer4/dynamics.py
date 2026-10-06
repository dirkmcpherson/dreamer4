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
        ramp_weight: bool = True,         # paper: w(sigma) = 0.9 sigma + 0.1; False weights all signal levels equally
        clean_context_prob: float = 0.0,  # fraction of sequences whose first frames are near-clean context (as at inference)
                                          # and excluded from the loss; 0 is the paper's independent per-frame noise
        regress_only: bool = False,       # diagnostic: one target frame per sequence, predicted in one step from pure
                                          # noise given clean history, i.e. plain next-frame regression
        agent: bool = False,              # add an agent token per time step with policy and reward heads (paper, Sec. 3.3)
        action_bins: int = 128,           # policy head: categorical over bins per action dimension
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
        self.ramp_weight, self.clean_context_prob, self.regress_only = ramp_weight, clean_context_prob, regress_only

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

        # ---- agent: one extra token per time step that reads everything in its step (and its own past) while no
        # other token attends to it, so the world model's predictions are unchanged by its presence
        self.agent, self.action_bins = agent, action_bins
        if agent:
            assert action_dim == 2, "the policy head is written for 2-D continuous actions"
            head = lambda out: nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, out))
            self.agent_token = nn.Parameter(torch.randn(dim) * 0.02)
            self.agent_norm = RMSNorm(dim)
            self.pi_first = head(action_bins)                 # p(a_x | h)
            self.bin_embed = nn.Embedding(action_bins, dim)
            self.pi_second = head(action_bins)                # p(a_y | h, a_x): the two dimensions are not independent
            self.reward_head = head(1)                        # logit of "this frame is a success"
            self.value_head = head(1)                         # expected discounted success from this frame (imagination training)
            s = 2 + num_registers + self.n_spatial + 1
            allowed = torch.ones(s, s, dtype=torch.bool)
            allowed[:-1, -1] = False
            self.register_buffer("agent_mask", allowed, persistent=False)
            self.bc_norm, self.rew_norm = LossNormalizer(), LossNormalizer()

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
                prev_actions: Tensor | None = None, action_valid: Tensor | None = None, return_agent: bool = False):
        """x-prediction of the clean latents. z_noised (B, T, N_l, d_b); signal_idx, step_idx (B, T) long.
        With ``return_agent`` also returns the agent token features (B, T, dim)."""
        b, t = z_noised.shape[:2]
        if action_valid is None:
            action_valid = torch.zeros(b, t, dtype=torch.bool, device=z_noised.device)
        sp = self.in_proj(z_noised.reshape(b, t, self.n_spatial, self.d_spatial)) + self.spatial_pos
        flow = torch.cat((self.signal_embed(signal_idx), self.step_embed(step_idx)), dim=-1)[:, :, None]
        act = self._action_tokens(prev_actions, action_valid, b, t)[:, :, None]
        reg = self.registers.expand(b, t, -1, -1)
        tokens, mask = [flow, act, reg, sp], None
        if self.agent:
            tokens.append(self.agent_token.expand(b, t, 1, -1))
            mask = self.agent_mask
        h = self.transformer(torch.cat(tokens, dim=2), space_mask=mask)
        first = 2 + self.num_registers
        out = self.out_proj(self.out_norm(h[:, :, first:first + self.n_spatial]))
        out = out.reshape(b, t, self.num_latents, self.latent_dim)
        if self.x_skip:
            sigma = signal_idx.float() / self.k_max
            out = out + sigma[..., None, None] * z_noised
        if return_agent:
            return out, self.agent_norm(h[:, :, -1])
        return out

    # ------------------------------------------------------------------ agent
    def to_bins(self, a: Tensor) -> Tensor:
        return ((a.clamp(-1, 1) + 1) / 2 * self.action_bins).long().clamp(max=self.action_bins - 1)

    def from_bins(self, idx: Tensor) -> Tensor:
        return (idx.float() + 0.5) / self.action_bins * 2 - 1

    def agent_features(self, z: Tensor, actions: Tensor) -> Tensor:
        """Agent token features (B, T, dim) from CLEAN latents, as at inference. The feature at frame t sees
        frames <= t and the actions that produced them, never the action taken at frame t."""
        b, t = z.shape[:2]
        prev, valid = shift_actions(actions, b, t, z.device)
        sig = torch.full((b, t), self.k_max - 1, dtype=torch.long, device=z.device)
        step = torch.full((b, t), self.max_exp, dtype=torch.long, device=z.device)
        return self.predict(z, sig, step, prev, valid, return_agent=True)[1]

    def agent_loss(self, z: Tensor, actions: Tensor, action_mask: Tensor, success: Tensor | None = None) -> tuple[Tensor, Tensor, dict]:
        """Behaviour cloning (and success prediction) from the agent token. actions (B, T, 2) taken at frame t;
        action_mask (B, T) False where no action was taken (last frame of an episode); success (B, T) in {0, 1}."""
        h = self.agent_features(z, actions)
        target = self.to_bins(actions)
        logit_x = self.pi_first(h)
        logit_y = self.pi_second(h + self.bin_embed(target[..., 0]))
        ce = F.cross_entropy(logit_x.flatten(0, 1), target[..., 0].flatten(), reduction="none") \
            + F.cross_entropy(logit_y.flatten(0, 1), target[..., 1].flatten(), reduction="none")
        m = action_mask.flatten().float()
        bc = (ce * m).sum() / m.sum().clamp_min(1)
        with torch.no_grad():                                  # greedy action as at inference: argmax x, then y given that x
            ax = logit_x.argmax(-1)
            ay = self.pi_second(h + self.bin_embed(ax)).argmax(-1)
            err = (self.from_bins(torch.stack((ax, ay), -1)) - actions).abs().sum(-1).flatten()
            stats = {"bc_ce": bc.detach(), "bc_l1": (err * m).sum() / m.sum().clamp_min(1),
                     "bc_within_2_bins": ((((torch.stack((ax, ay), -1) - target).abs() <= 2).all(-1).flatten().float()) * m).sum() / m.sum().clamp_min(1)}
        rew = torch.zeros((), device=z.device)
        if success is not None:
            rl = self.reward_head(h).squeeze(-1)
            rew = F.binary_cross_entropy_with_logits(rl, success.float())
            stats["success_acc"] = ((rl > 0) == (success > 0.5)).float().mean()
        return bc, rew, stats

    # ------------------------------------------------------------------ policy distribution utilities (imagination training)
    def policy_heads(self):
        return [self.pi_first, self.bin_embed, self.pi_second]

    def policy_logits(self, h: Tensor, ax: Tensor) -> tuple[Tensor, Tensor]:
        """Logits of p(a_x | h) and p(a_y | h, a_x) for given first-dimension bins ax."""
        return self.pi_first(h), self.pi_second(h + self.bin_embed(ax))

    def policy_log_prob(self, h: Tensor, bins: Tensor) -> Tensor:
        """log pi(a | h) for actions given as bins (..., 2)."""
        lx, ly = self.policy_logits(h, bins[..., 0])
        return lx.log_softmax(-1).gather(-1, bins[..., :1]).squeeze(-1) + ly.log_softmax(-1).gather(-1, bins[..., 1:]).squeeze(-1)

    def policy_entropy(self, h: Tensor, bins: Tensor) -> Tensor:
        lx, ly = self.policy_logits(h, bins[..., 0])
        ent = lambda l: -(l.log_softmax(-1) * l.softmax(-1)).sum(-1)
        return ent(lx) + ent(ly)

    def _refine(self, logits: Tensor, idx: Tensor, radius: int = 2) -> Tensor:
        """Sub-bin value: probability-weighted mean of the bin centres within ``radius`` of the chosen bin
        (stays inside the chosen mode, removes most of the bin quantisation)."""
        offs = torch.arange(-radius, radius + 1, device=logits.device)
        near = (idx[..., None] + offs).clamp(0, self.action_bins - 1)
        p = logits.softmax(-1).gather(-1, near)
        return (self.from_bins(near) * p).sum(-1) / p.sum(-1).clamp_min(1e-9)

    @torch.no_grad()
    def act(self, z: Tensor, actions: Tensor, sample: bool = False, temperature: float = 1.0, refine: bool = False) -> Tensor:
        """Action for the LAST frame of z (B, T, N_l, d_b); actions (B, T, 2) with the last entry ignored."""
        h = self.agent_features(z, actions)[:, -1]
        pick = (lambda l: torch.distributions.Categorical(logits=l / temperature).sample()) if sample else (lambda l: l.argmax(-1))
        lx = self.pi_first(h); ax = pick(lx)
        ly = self.pi_second(h + self.bin_embed(ax)); ay = pick(ly)
        if refine:
            return torch.stack((self._refine(lx, ax), self._refine(ly, ay)), -1)
        return self.from_bins(torch.stack((ax, ay), -1))

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

        # optionally make the first frames of a sequence near-clean context, the situation met at inference
        in_loss = torch.ones(b, t, dtype=torch.bool, device=device)
        if self.regress_only or self.clean_context_prob > 0:
            p_ctx = 1.0 if self.regress_only else self.clean_context_prob
            split = torch.randint(1, t, (b, 1), device=device)
            frame = torch.arange(t, device=device)[None]
            is_ctx = (torch.rand(b, 1, device=device) < p_ctx) & (frame < split)
            e = torch.where(is_ctx, torch.full_like(e, e_max), e)
            sigma = torch.where(is_ctx, torch.full_like(sigma, (k_max - 1) / k_max), sigma)
            in_loss = ~is_ctx
            if self.regress_only:
                e = torch.where(is_ctx, e, torch.zeros_like(e))
                sigma = torch.where(is_ctx, sigma, torch.zeros_like(sigma))
                in_loss = frame == split
        signal_idx = (sigma * k_max).round().long()

        z0 = torch.randn_like(z1)
        s4 = sigma[..., None, None]
        zt = (1 - s4) * z0 + s4 * z1
        x_hat = self.predict(zt, signal_idx, e, prev_actions, valid)

        is_flow = (e == e_max) | warmup | self.regress_only
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
        ramp = 0.9 * sigma + 0.1 if self.ramp_weight else torch.ones_like(sigma)
        per_row = torch.where(is_flow, flow_pt, boot_pt) * ramp
        loss = (per_row * in_loss).sum() / in_loss.sum().clamp_min(1)
        if self.loss_norm is not None:
            loss = self.loss_norm(loss)
        is_flow, is_boot = is_flow & in_loss, ~is_flow & in_loss
        n_flow, n_boot = is_flow.sum().clamp_min(1), is_boot.sum().clamp_min(1)
        if self.training:
            self.train_steps += 1
        stats = {"flow_mse": (flow_pt * is_flow).sum() / n_flow, "boot_mse": (boot_pt * is_boot).sum() / n_boot,
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
