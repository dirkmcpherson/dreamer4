"""Causal tokenizer (Dreamer 4, Sec. 3.1).

Encoder: per frame, patch tokens + learned latent tokens through the block-causal
transformer; latents attend to everything, patches only to patches. The latents are read
out with a linear projection to a small channel dimension followed by tanh.

Decoder: latents are projected back up, concatenated with learned patch query tokens;
patch queries attend to patches and latents, latents only to latents. Patch queries are
read out linearly into pixels.

Training: masked autoencoding with per-image patch dropout p ~ U(0, 0.9), full-frame MSE
plus optional 0.2 * LPIPS, each term normalized by its running RMS.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .nn import BlockCausalTransformer, LossNormalizer, RMSNorm


def patchify(video: Tensor, p: int) -> Tensor:
    """(B, T, C, H, W) -> (B, T, (H/p)(W/p), C p p)."""
    b, t, c, h, w = video.shape
    x = video.reshape(b, t, c, h // p, p, w // p, p)
    return x.permute(0, 1, 3, 5, 2, 4, 6).reshape(b, t, (h // p) * (w // p), c * p * p)


def unpatchify(patches: Tensor, p: int, h: int, w: int, c: int) -> Tensor:
    b, t, n, _ = patches.shape
    x = patches.reshape(b, t, h // p, w // p, c, p, p)
    return x.permute(0, 1, 4, 2, 5, 3, 6).reshape(b, t, c, h, w)


def patch_grid_positions(h_patches: int, w_patches: int) -> Tensor:
    rows = torch.arange(h_patches).repeat_interleave(w_patches)
    cols = torch.arange(w_patches).repeat(h_patches)
    return torch.stack((rows, cols), dim=-1).float()


class CausalTokenizer(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        channels: int = 3,
        dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        dim_head: int = 64,
        num_latents: int = 16,
        latent_dim: int = 32,
        time_every: int = 4,
        time_tokens: str = "all",            # "all" (paper) or "latents" (Hansen / Hu simplification)
        mask_prob: tuple[float, float] = (0.0, 0.9),
        lpips_weight: float = 0.0,           # paper uses 0.2 (requires the `lpips` package)
        loss_norm: bool = True,
        softcap: float | None = 50.0,
        center_patches: bool = True,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.center_patches = center_patches
        self.image_size, self.patch_size, self.channels = image_size, patch_size, channels
        self.num_latents, self.latent_dim = num_latents, latent_dim
        self.mask_prob = mask_prob
        self.time_tokens = time_tokens
        self.lpips_weight = lpips_weight

        hp = wp = image_size // patch_size
        self.num_patches = hp * wp
        patch_dim = channels * patch_size ** 2
        s = num_latents + self.num_patches

        # ---- encoder
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.enc_pos = nn.Parameter(torch.randn(self.num_patches, dim) * 0.02)
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.encoder = BlockCausalTransformer(dim, depth, heads, dim_head, time_every, softcap=softcap)
        self.to_latent = nn.Linear(dim, latent_dim)

        # ---- decoder
        self.from_latent = nn.Linear(latent_dim, dim)
        self.patch_queries = nn.Parameter(torch.randn(self.num_patches, dim) * 0.02)
        self.decoder = BlockCausalTransformer(dim, depth, heads, dim_head, time_every, softcap=softcap)
        self.to_pixels = nn.Linear(dim, patch_dim)

        # ---- token layout: [latents | patches]
        is_latent = torch.zeros(s, dtype=torch.bool)
        is_latent[:num_latents] = True
        enc_mask = torch.zeros(s, s, dtype=torch.bool)
        enc_mask[is_latent] = True                                   # latents attend to everything
        enc_mask[~is_latent] = ~is_latent[None, :]                   # patches attend to patches only
        dec_mask = torch.zeros(s, s, dtype=torch.bool)
        dec_mask[is_latent] = is_latent[None, :]                     # latents attend to latents only
        dec_mask[~is_latent] = True                                  # patch queries attend to all
        pos = torch.cat((torch.zeros(num_latents, 2), patch_grid_positions(hp, wp)), dim=0)
        self.register_buffer("is_latent", is_latent, persistent=False)
        self.register_buffer("enc_mask", enc_mask, persistent=False)
        self.register_buffer("dec_mask", dec_mask, persistent=False)
        self.register_buffer("space_pos", pos, persistent=False)
        self.register_buffer("space_valid", ~is_latent, persistent=False)

        self.recon_norm = LossNormalizer() if loss_norm else None
        self.lpips_norm = LossNormalizer() if (loss_norm and lpips_weight > 0) else None
        self._lpips = None
        if lpips_weight > 0:
            try:
                import lpips  # noqa: F401
            except ImportError as e:
                raise ImportError("lpips_weight > 0 requires `pip install lpips`") from e

    # ------------------------------------------------------------------ helpers
    @property
    def time_token_mask(self) -> Tensor | None:
        return self.is_latent if self.time_tokens == "latents" else None

    def _lpips_fn(self):
        # kept in __dict__ rather than registered as a submodule: its frozen weights must not end up in
        # checkpoints (load_state_dict would reject them) and tok.train() must not switch on its dropout
        if self.__dict__.get("_lpips") is None:
            import lpips
            self.__dict__["_lpips"] = lpips.LPIPS(net="alex", verbose=False).eval().requires_grad_(False)
        return self.__dict__["_lpips"].to(self.enc_pos.device)

    # ------------------------------------------------------------------ encode / decode
    def encode(self, video: Tensor, mask_patches: bool = False) -> Tensor:
        """video (B, T, C, H, W) in [0, 1] -> latents (B, T, num_latents, latent_dim) in (-1, 1)."""
        b, t = video.shape[:2]
        x = self.patch_embed(patchify(video, self.patch_size))
        if self.center_patches:
            # remove the per-frame mean over patches: with mostly-uniform backgrounds the shared component
            # otherwise dominates the latent tokens and training sits on the mean-image plateau for a long time
            x = x - x.mean(dim=2, keepdim=True)
        if mask_patches:
            lo, hi = self.mask_prob
            p = torch.empty(b, t, 1, device=x.device).uniform_(lo, hi)
            drop = torch.rand(b, t, self.num_patches, device=x.device) < p
            x = torch.where(drop[..., None], self.mask_token.to(x.dtype), x)
        x = x + self.enc_pos
        lat = self.latents.expand(b, t, -1, -1)
        tokens = torch.cat((lat, x), dim=2)
        h = self.encoder(tokens, self.enc_mask, self.space_pos, self.space_valid, self.time_token_mask)
        return torch.tanh(self.to_latent(h[:, :, : self.num_latents]))

    def decode(self, z: Tensor) -> Tensor:
        b, t = z.shape[:2]
        lat = self.from_latent(z)
        q = self.patch_queries.expand(b, t, -1, -1)
        h = self.decoder(torch.cat((lat, q), dim=2), self.dec_mask, self.space_pos, self.space_valid, self.time_token_mask)
        pix = self.to_pixels(h[:, :, self.num_latents:])
        return unpatchify(pix, self.patch_size, self.image_size, self.image_size, self.channels)

    # ------------------------------------------------------------------ training
    def forward(self, video: Tensor) -> tuple[Tensor, dict]:
        z = self.encode(video, mask_patches=self.training)
        recon = self.decode(z)
        mse = F.mse_loss(recon, video)
        loss = self.recon_norm(mse) if self.recon_norm is not None else mse
        stats = {"mse": mse.detach(), "latent_std": z.detach().std()}
        if self.lpips_weight > 0:
            b, t = video.shape[:2]
            lp = self._lpips_fn()(recon.flatten(0, 1) * 2 - 1, video.flatten(0, 1) * 2 - 1).mean()
            stats["lpips"] = lp.detach()
            loss = loss + self.lpips_weight * (self.lpips_norm(lp) if self.lpips_norm is not None else lp)
        return loss, stats
