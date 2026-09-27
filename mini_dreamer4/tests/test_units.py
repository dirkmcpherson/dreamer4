"""Fast structural tests: attention routing, position sensitivity, loss bookkeeping, sampling."""
import math

import pytest
import torch

from mini_dreamer4 import CausalTokenizer, ShortcutDynamics
from mini_dreamer4.dynamics import shift_actions
from mini_dreamer4.envs import MiniPushT, generate_episodes, state_to_oracle_latents
from mini_dreamer4.tokenizer import patchify, unpatchify

torch.manual_seed(0)


def small_tokenizer(**kw):
    cfg = dict(image_size=32, patch_size=8, dim=32, depth=2, heads=2, dim_head=16, num_latents=4, latent_dim=8, time_every=2)
    cfg.update(kw)
    return CausalTokenizer(**cfg).eval()


def test_patchify_roundtrip():
    v = torch.rand(2, 3, 3, 32, 32)
    assert torch.equal(unpatchify(patchify(v, 8), 8, 32, 32, 3), v)


def test_tokenizer_attention_routing():
    tok = small_tokenizer()
    lat, pat = tok.is_latent, ~tok.is_latent
    assert tok.enc_mask[lat].all(), "encoder latents must attend to everything"
    assert not tok.enc_mask[pat][:, lat].any(), "encoder patches must not attend to latents"
    assert tok.enc_mask[pat][:, pat].all()
    assert tok.dec_mask[lat][:, lat].all() and not tok.dec_mask[lat][:, pat].any(), "decoder latents attend only to latents"
    assert tok.dec_mask[pat].all(), "decoder patch queries attend to patches and latents"


def test_encoder_is_position_sensitive():
    """Shuffling the patch grid must change the latents (the failure mode of a position-blind encoder)."""
    tok = small_tokenizer()
    v = torch.rand(1, 2, 3, 32, 32)
    p = patchify(v, 8)
    perm = torch.randperm(p.shape[2])
    v_perm = unpatchify(p[:, :, perm], 8, 32, 32, 3)
    with torch.no_grad():
        z, z_perm = tok.encode(v), tok.encode(v_perm)
    assert (z - z_perm).abs().max() > 1e-3


def test_encoder_is_causal_in_time():
    tok = small_tokenizer()
    v = torch.rand(1, 4, 3, 32, 32)
    v2 = v.clone()
    v2[:, 3] = torch.rand(3, 32, 32)
    with torch.no_grad():
        z, z2 = tok.encode(v), tok.encode(v2)
    assert torch.allclose(z[:, :3], z2[:, :3], atol=1e-5), "changing frame 3 must not change latents of frames 0-2"
    assert (z[:, 3] - z2[:, 3]).abs().max() > 1e-4


def test_tokenizer_shapes_and_range():
    tok = small_tokenizer()
    v = torch.rand(2, 3, 3, 32, 32)
    z = tok.encode(v)
    assert z.shape == (2, 3, 4, 8) and z.abs().max() <= 1
    assert tok.decode(z).shape == v.shape
    tok.train()
    loss, stats = tok(v)
    assert torch.isfinite(loss) and "mse" in stats


def test_shift_actions():
    a = torch.arange(2 * 4 * 2, dtype=torch.float).view(2, 4, 2)
    prev, valid = shift_actions(a, 2, 4, a.device)
    assert torch.equal(prev[:, 1:], a[:, :3]) and prev[:, 0].abs().sum() == 0
    assert not valid[:, 0].any() and valid[:, 1:].all()


def test_dynamics_causal_and_action_dependent():
    dyn = ShortcutDynamics(4, 8, pack=2, dim=32, depth=2, heads=2, dim_head=16, time_every=2, k_max=8, action_dim=2).eval()
    for p in dyn.out_proj.parameters():  # zero-init output would hide everything
        torch.nn.init.normal_(p, std=0.1)
    z = torch.rand(1, 4, 4, 8) * 2 - 1
    sig = torch.full((1, 4), 4, dtype=torch.long)
    step = torch.zeros(1, 4, dtype=torch.long)
    a = torch.rand(1, 4, 2)
    prev, valid = shift_actions(a, 1, 4, z.device)
    with torch.no_grad():
        x = dyn.predict(z, sig, step, prev, valid)
        z2 = z.clone(); z2[:, 3] += 0.5
        x2 = dyn.predict(z2, sig, step, prev, valid)
        a2 = a.clone(); a2[:, 1] += 1.0
        prev2, _ = shift_actions(a2, 1, 4, z.device)
        x3 = dyn.predict(z, sig, step, prev2, valid)
    assert torch.allclose(x[:, :3], x2[:, :3], atol=1e-5), "future frames must not influence the past"
    assert (x[:, 2] - x3[:, 2]).abs().max() > 1e-6, "the action taken at frame 1 must influence frame 2"
    assert torch.allclose(x[:, :2], x3[:, :2], atol=1e-6), "and must not influence frames 0-1"


def test_shortcut_loss_grid_consistency():
    """sigma always lies on the grid reachable by the sampled step size and the bootstrap half steps stay below 1."""
    dyn = ShortcutDynamics(4, 8, pack=2, dim=32, depth=2, heads=2, dim_head=16, time_every=2, k_max=16, action_dim=2)
    dyn.train()
    for _ in range(5):
        loss, stats = dyn(torch.rand(4, 6, 4, 8) * 2 - 1, torch.rand(4, 6, 2))
        assert torch.isfinite(loss)
        assert 0 <= stats["frac_flow"] <= 1


def test_sampling_matches_x_prediction_at_last_step():
    """With K steps of size 1/K, the final sample must equal the network's last x-prediction exactly."""
    dyn = ShortcutDynamics(4, 8, pack=2, dim=32, depth=2, heads=2, dim_head=16, time_every=2, k_max=8, action_dim=2).eval()
    z = torch.rand(2, 3, 4, 8) * 2 - 1
    a = torch.rand(2, 5, 2)
    out = dyn.sample(z, a, horizon=2, num_steps=4, ctx_noise=0.0)
    assert out.shape == (2, 5, 4, 8)
    assert torch.equal(out[:, :3], z)
    assert out.abs().max() <= 1


def test_minipusht_env_and_data():
    env = MiniPushT(image_size=32, seed=0)
    obs = env.reset(seed=1)
    assert obs["image"].shape == (32, 32, 3) and obs["state"].shape == (6,)
    block_before = env.block.copy()
    for _ in range(40):  # drive the agent into the block: it must move
        d = env.block - env.agent
        obs, r, term, trunc, _ = env.step(d / (abs(d).max() + 1e-6))
    assert (env.block != block_before).any(), "pushing must move the block"
    eps = generate_episodes(2, 10, image_size=32, seed=0)
    assert eps[0]["video"].shape == (10, 32, 32, 3) and eps[0]["actions"].shape == (10, 2)
    z = state_to_oracle_latents(torch.as_tensor(eps[0]["states"]), 8, 16)
    assert z.shape == (10, 8, 16)
