"""Policy-relevant checks of a (tokenizer, dynamics) pair on the real PushT demonstrations.

A probe trained on TRUE latents of the training episodes is applied to IMAGINED latents of held-out
episodes and compared with the recorded simulator state (agent xy, block xy, block angle).

    python -m mini_dreamer4.tools.policy_checks_pusht <pusht.zarr> <tokenizer.pt> <dynamics.pt> [sampling steps]
"""
import sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from mini_dreamer4.data import load_pusht_zarr
from mini_dreamer4.train import load_tokenizer, load_dynamics, psnr

dev, CTX, WIN, H = "cuda", 4, 16, 44
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
tok, dyn = load_tokenizer(sys.argv[2], dev), load_dynamics(sys.argv[3], dev)
torch.manual_seed(0)

episodes = load_pusht_zarr(sys.argv[1])
n_val = max(1, len(episodes) // 20)                       # same split as mini_dreamer4.train
train_eps, val_eps = episodes[n_val:], episodes[:n_val]

def targets(states):                                      # (T, 5) pixels / radians -> (T, 6) like MiniPushT
    s = torch.as_tensor(states)
    return torch.cat((s[:, :4] / 512, torch.cos(s[:, 4:5]), torch.sin(s[:, 4:5])), dim=-1)

def windows(eps, length, stride):
    v, a, s = [], [], []
    for e in eps:
        for i in range(0, len(e["video"]) - length + 1, stride):
            v.append(torch.as_tensor(e["video"][i:i + length])); a.append(torch.as_tensor(e["actions"][i:i + length])); s.append(targets(e["states"][i:i + length]))
    return torch.stack(v), torch.stack(a).float().to(dev), torch.stack(s).to(dev)

@torch.no_grad()
def encode(video_u8, batch=32):                           # (N, T, H, W, 3) uint8 -> latents, in training-length chunks
    out = []
    for i in range(0, len(video_u8), batch):
        v = (video_u8[i:i + batch].float() / 255).movedim(-1, -3).to(dev)
        out.append(torch.cat([tok.encode(v[:, j:j + WIN]) for j in range(0, v.shape[1], WIN)], dim=1))
    return torch.cat(out)

@torch.no_grad()
def rollout_sliding(z_ctx, actions, horizon):
    z = z_ctx.clone()
    for _ in range(horizon):
        t0 = z.shape[1]; lo = max(0, t0 + 1 - WIN)
        z = torch.cat((z, dyn.sample(z[:, lo:], actions[:, lo:t0 + 1], horizon=1, num_steps=NS)[:, -1:]), dim=1)
    return z

def state_errors(pred, true):
    ang = lambda s: torch.atan2(s[..., 5], s[..., 4])
    d = (ang(pred) - ang(true) + np.pi) % (2 * np.pi) - np.pi
    return dict(agent=(pred[..., :2] - true[..., :2]).norm(dim=-1), block=(pred[..., 2:4] - true[..., 2:4]).norm(dim=-1), angle_deg=d.abs() * 180 / np.pi)

# ------------------------------------------------------------------ probe
tv, _, ts = windows(train_eps, WIN, WIN)
vv, va, vs = windows(val_eps, CTX + H, 8)
ztr, zva = encode(tv), encode(vv)
print("train: %d episodes, %d frames | held-out: %d episodes, %d rollout windows of %d frames | sampling steps K=%d"
      % (len(train_eps), sum(len(e["video"]) for e in train_eps), len(val_eps), len(vv), CTX + H, NS))
probe = nn.Sequential(nn.Flatten(-2), nn.Linear(ztr.shape[-2] * ztr.shape[-1], 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 6)).to(dev)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
X, Y = ztr.flatten(0, 1), ts.flatten(0, 1)
for step in range(4000):
    idx = torch.randint(0, len(X), (256,), device=dev)
    loss = F.mse_loss(probe(X[idx] + 0.02 * torch.randn_like(X[idx])), Y[idx])
    opt.zero_grad(); loss.backward(); opt.step()
probe.eval()
with torch.no_grad():
    e = state_errors(probe(zva), vs)
    et = state_errors(probe(ztr), ts)
print("probe on its own training latents: agent %.3f  block %.3f  angle %.1f deg" % (et["agent"].mean(), et["block"].mean(), et["angle_deg"].mean()))
if len(sys.argv) > 5 and sys.argv[5] == "probe-only":
    print("probe on TRUE held-out latents: agent %.3f  block %.3f  angle %.1f deg" % (e["agent"].mean(), e["block"].mean(), e["angle_deg"].mean())); sys.exit()
print("probe on TRUE held-out latents (ceiling): agent %.3f  block %.3f  angle %.1f deg   [positions as a fraction of the 512 px workspace; agent radius ~0.03]"
      % (e["agent"].mean(), e["block"].mean(), e["angle_deg"].mean()))
d1 = (vs[:, 1:] - vs[:, :-1])
print("real per-frame motion: agent %.4f, block %.4f; block moving in %.0f%% of frames"
      % (d1[..., :2].norm(dim=-1).mean(), d1[..., 2:4].norm(dim=-1).mean(), 100 * (d1[..., 2:4].norm(dim=-1) > 1e-4).float().mean()))

# ------------------------------------------------------------------ rollouts
with torch.no_grad():
    gen = rollout_sliding(zva[:, :CTX], va, H)
    gen_wrong = rollout_sliding(zva[:, :CTX], va.roll(len(va) // 2, dims=0), H)
    sp, sw, sc = probe(gen), probe(gen_wrong), probe(zva)
print("  h | latent mse: model  wrong-act  copy-last | agent err: model  wrong-act  copy-last | block err: model  wrong-act  copy-last | angle: model  copy-last")
for h in (1, 2, 4, 8, 12, 16, 24, 32, 44):
    t = CTX + h - 1
    lm = lambda a: F.mse_loss(a[:, t], zva[:, t]).item()
    em, ew, ec = state_errors(sp[:, t], vs[:, t]), state_errors(sw[:, t], vs[:, t]), state_errors(sc[:, CTX - 1], vs[:, t])
    print(" %2d |            %.3f   %.3f      %.3f     |            %.3f   %.3f      %.3f     |            %.3f   %.3f      %.3f     |        %4.1f   %4.1f"
          % (h, lm(gen), lm(gen_wrong), lm(zva[:, CTX - 1:CTX].expand_as(zva)), em["agent"].mean(), ew["agent"].mean(), ec["agent"].mean(),
             em["block"].mean(), ew["block"].mean(), ec["block"].mean(), em["angle_deg"].mean(), ec["angle_deg"].mean()))

def corr(a, b, rank=False):
    if rank: a, b = a.argsort().argsort().float(), b.argsort().argsort().float()
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-9)).item()
goal = torch.tensor([256.0, 256.0], device=dev) / 512
for h in (4, 12):
    t = CTX + h - 1
    disp = lambda s: (s[:, t, 2:4] - s[:, CTX - 1, 2:4]).norm(dim=-1)
    real, im = disp(vs), (sp[:, t, 2:4] - sc[:, CTX - 1, 2:4]).norm(dim=-1)
    moved = real > 0.02
    print("block displacement over %2d frames: imagined vs real Spearman %.2f | moved in %.0f%% of windows: imagined %.3f vs real %.3f; when still: imagined %.3f"
          % (h, corr(im, real, rank=True), 100 * moved.float().mean(), im[moved].mean(), real[moved].mean(), im[~moved].mean() if (~moved).any() else float("nan")))
    # direction of the block's motion: does the imagined displacement point the right way?
    dv_r, dv_i = (vs[:, t, 2:4] - vs[:, CTX - 1, 2:4])[moved], (sp[:, t, 2:4] - sc[:, CTX - 1, 2:4])[moved]
    print("   direction agreement (cosine) of imagined vs real block displacement, when it moved: %.2f" % F.cosine_similarity(dv_i, dv_r, dim=-1).mean())
    g = lambda s: (s[:, t, 2:4] - goal).norm(dim=-1)
    print("   change in block-to-goal distance, imagined vs real: Pearson %.2f" % corr(g(sp) - (sc[:, CTX - 1, 2:4] - goal).norm(dim=-1), g(vs) - (vs[:, CTX - 1, 2:4] - goal).norm(dim=-1)))
sat = (gen.abs() > 0.999).float().mean(dim=(0, 2, 3))
print("fraction of imagined latent values clamped at +-1, h=1 / 12 / 44: %.3f / %.3f / %.3f   (true latents: %.3f)" % (sat[CTX], sat[CTX + 11], sat[CTX + H - 1], (zva.abs() > 0.999).float().mean()))
