"""Policy-relevant checks of a (tokenizer, dynamics) pair on MiniPushT, using the real simulator as ground truth.

A. per-horizon rollout quality on held-out episodes, up to 44 frames (training window: 16)
B. state decodability: a probe trained on TRUE latents, applied to IMAGINED latents
C. counterfactuals: same start state, K different action sequences, imagined vs real outcomes

    python mini_dreamer4/tools/policy_checks.py <tokenizer.pt | oracle> <dynamics.pt> [sampling steps]

With "oracle" the latents are the fixed projection of the true state used by tools/oracle_dynamics.py.
"""
import copy, os, sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from mini_dreamer4.train import load_tokenizer, load_dynamics, psnr
from mini_dreamer4.envs import generate_episodes, MiniPushT, state_to_oracle_latents
from mini_dreamer4.envs.pusht import _pusher_policy

dev, IMG, CTX, WIN = "cuda", 64, 4, 16
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
print("sampling steps K =", NS)
ORACLE = sys.argv[1] == "oracle"
tok, dyn = (None if ORACLE else load_tokenizer(sys.argv[1], dev)), load_dynamics(sys.argv[2], dev)
torch.manual_seed(0)

def to_video(frames_u8):                       # (..., H, W, 3) uint8 -> (..., 3, H, W) float
    v = torch.as_tensor(np.asarray(frames_u8)).float() / 255
    return v.movedim(-1, -3)

@torch.no_grad()
def encode(video, states=None, chunk=WIN, batch=64):   # (B, T, 3, H, W) -> latents, encoded in training-length chunks
    if ORACLE:
        return state_to_oracle_latents(torch.as_tensor(np.asarray(states)), dyn.num_latents, dyn.latent_dim).to(dev)
    return torch.cat([torch.cat([tok.encode(video[b:b + batch, i:i + chunk].to(dev)) for i in range(0, video.shape[1], chunk)], dim=1)
                      for b in range(0, len(video), batch)])

@torch.no_grad()
def rollout_sliding(z_ctx, actions, horizon, num_steps=None):
    num_steps = NS if num_steps is None else num_steps
    """Autoregressive rollout that never shows the model more than WIN frames (its training length)."""
    z = z_ctx.clone()
    for _ in range(horizon):
        t0 = z.shape[1]
        lo = max(0, t0 + 1 - WIN)
        z = torch.cat((z, dyn.sample(z[:, lo:], actions[:, lo:t0 + 1], horizon=1, num_steps=num_steps)[:, -1:]), dim=1)
    return z

def state_errors(pred, true):                  # (..., 6): agent xy, block xy, cos, sin
    ang = lambda s: torch.atan2(s[..., 5], s[..., 4])
    d = (ang(pred) - ang(true) + np.pi) % (2 * np.pi) - np.pi
    return dict(agent=(pred[..., :2] - true[..., :2]).norm(dim=-1), block=(pred[..., 2:4] - true[..., 2:4]).norm(dim=-1),
                angle_deg=d.abs() * 180 / np.pi)

# ------------------------------------------------------------------ probe: true latents -> state, reward
PROBE_EPS, PROBE_STEPS = int(os.environ.get("PROBE_EPS", 200)), int(os.environ.get("PROBE_STEPS", 4000))
train = generate_episodes(PROBE_EPS, 48, image_size=IMG, seed=0)   # a prefix of the training distribution (same seed as training)
print("probe trained on %d episodes for %d steps" % (PROBE_EPS, PROBE_STEPS))
val = generate_episodes(40, 48, image_size=IMG, seed=1)        # the held-out episodes used in training logs
stack = lambda eps, k: torch.as_tensor(np.stack([e[k] for e in eps]))
ztr, zva = encode(to_video(stack(train, "video")), stack(train, "states")), encode(to_video(stack(val, "video")), stack(val, "states"))
ytr = torch.cat((stack(train, "states"), stack(train, "rewards")[..., None]), dim=-1).to(dev)
yva = torch.cat((stack(val, "states"), stack(val, "rewards")[..., None]), dim=-1).to(dev)
probe = nn.Sequential(nn.Flatten(-2), nn.Linear(ztr.shape[-2] * ztr.shape[-1], 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 7)).to(dev)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
X, Y = ztr.flatten(0, 1), ytr.flatten(0, 1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PROBE_STEPS)
for step in range(PROBE_STEPS):
    idx = torch.randint(0, len(X), (512,), device=dev)
    loss = F.mse_loss(probe(X[idx] + 0.02 * torch.randn_like(X[idx])), Y[idx])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
probe.eval()
with torch.no_grad():
    e = state_errors(probe(zva)[..., :6], yva[..., :6])
    print("B. probe on TRUE held-out latents (ceiling): agent %.3f  block %.3f  angle %.1f deg  | reward mae %.3f   [positions in frame widths; agent radius 0.045]"
          % (e["agent"].mean(), e["block"].mean(), e["angle_deg"].mean(), (probe(zva)[..., 6] - yva[..., 6]).abs().mean()))

# ------------------------------------------------------------------ A + B: long rollouts on held-out episodes
va_a = stack(val, "actions").to(dev)
H = 44 if NS <= 8 else 12
with torch.no_grad():
    gen = rollout_sliding(zva[:, :CTX], va_a, H)
    gen_wrong = rollout_sliding(zva[:, :CTX], va_a.roll(1, dims=0), H)
    if not ORACLE:
        pix = tok.decode(gen[:, :WIN]).clamp(0, 1)          # decode the first window for pixel metrics
        rec = tok.decode(zva[:, :WIN]).clamp(0, 1)
    sp, st = probe(gen)[..., :6], yva[..., :6]
video = to_video(stack(val, "video")).to(dev)
print("\nA/B. held-out rollouts from %d context frames, K=4, sliding %d-frame window, %d episodes" % (CTX, WIN, len(val)))
print("  h | latent mse: model  wrong-act  copy-last | agent err: model  copy-last | block err: model  copy-last | angle: model  copy-last")
for h in [h for h in (1, 2, 4, 8, 12, 16, 24, 32, 44) if h <= H]:
    t = CTX + h - 1
    lm = lambda a: F.mse_loss(a[:, t], zva[:, t]).item()
    em, ec = state_errors(sp[:, t], st[:, t]), state_errors(sp[:, CTX - 1], st[:, t])
    print(" %2d |            %.3f   %.3f      %.3f     |            %.3f   %.3f     |            %.3f   %.3f     |        %4.1f   %4.1f"
          % (h, lm(gen), lm(gen_wrong), lm(zva[:, CTX - 1:CTX].expand_as(zva)), em["agent"].mean(), ec["agent"].mean(),
             em["block"].mean(), ec["block"].mean(), em["angle_deg"].mean(), ec["angle_deg"].mean()))
if not ORACLE: print("  pixel psnr by horizon (model / repeat-last-frame / tokenizer recon):",
      "  ".join("h%d: %.1f/%.1f/%.1f" % (h, psnr(F.mse_loss(pix[:, CTX + h - 1], video[:, CTX + h - 1]).item()),
                                          psnr(F.mse_loss(video[:, CTX - 1], video[:, CTX + h - 1]).item()),
                                          psnr(F.mse_loss(rec[:, CTX + h - 1], video[:, CTX + h - 1]).item())) for h in (1, 2, 4, 8, 12)))
sat = (gen.abs() > 0.999).float().mean(dim=(0, 2, 3))
print("  fraction of imagined latent values clamped at +-1, h=1 / 12 / 44: %.3f / %.3f / %.3f   (true latents: %.3f)"
      % (sat[CTX], sat[CTX + 11], sat[CTX + H - 1], (zva.abs() > 0.999).float().mean()))

# ------------------------------------------------------------------ C: counterfactual branches from a shared start
N, K, HB = 32, 8, 12
rng = np.random.default_rng(123)
frames, states, actions, rewards = [], [], [], []
for n in range(N):
    env = MiniPushT(image_size=IMG, max_steps=1000, seed=int(rng.integers(1 << 30)))
    obs = env.reset(); hold = [0, None]
    f, s, a, r = [obs["image"]], [obs["state"]], [], [0.0]
    for _ in range(CTX - 1):
        act = _pusher_policy(env, rng, hold); obs, rew, *_ = env.step(act)
        f.append(obs["image"]); s.append(obs["state"]); a.append(act); r.append(rew)
    for k in range(K):
        e2, fk, sk, ak, rk, hold = copy.deepcopy(env), list(f), list(s), list(a), list(r), [0, None]
        brng = np.random.default_rng(int(rng.integers(1 << 30)))
        for _ in range(HB):
            act = _pusher_policy(e2, brng, hold); obs, rew, *_ = e2.step(act)
            fk.append(obs["image"]); sk.append(obs["state"]); ak.append(act); rk.append(rew)
        ak.append(np.zeros(2, dtype=np.float32))
        frames.append((np.stack(fk) * 255).round().astype(np.uint8)); states.append(np.stack(sk)); actions.append(np.stack(ak)); rewards.append(np.asarray(rk, dtype=np.float32))
vid = to_video(np.stack(frames)); st = torch.as_tensor(np.stack(states)).to(dev); ac = torch.as_tensor(np.stack(actions)).float().to(dev)
rw = torch.as_tensor(np.stack(rewards)).to(dev)
zt = encode(vid, np.stack(states))
with torch.no_grad():
    g = dyn.sample(zt[:, :CTX], ac, horizon=HB, num_steps=NS)
    out = probe(g)
T = CTX + HB - 1
zt_, g_ = zt.view(N, K, *zt.shape[1:]), g.view(N, K, *g.shape[1:])
# which real branch is each imagined branch closest to (final frame, latent space)?  chance = 1 / K
d = (g_[:, :, None, T] - zt_[:, None, :, T]).pow(2).mean(dim=(-1, -2))          # (N, K imagined, K real)
acc = (d.argmin(dim=-1) == torch.arange(K, device=dev)).float().mean().item()
sp, s_ = out[..., :6].view(N, K, T + 1, 6), st.view(N, K, T + 1, 6)
e = state_errors(sp[:, :, T], s_[:, :, T]); ec = state_errors(sp[:, :, CTX - 1], s_[:, :, T])
spread = (s_[:, :, T, :2] - s_[:, :, T, :2].mean(dim=1, keepdim=True)).norm(dim=-1).mean().item()
print("\nC. counterfactuals: %d start states x %d action sequences, %d imagined frames" % (N, K, HB))
print("  imagined branch matched to the correct real branch (final frame): %.2f   (chance %.2f)" % (acc, 1 / K))
print("  final agent position error %.3f  (copy-last %.3f, spread between real branches %.3f)" % (e["agent"].mean(), ec["agent"].mean(), spread))
print("  final block position error %.3f  (copy-last %.3f) | angle %.1f deg (copy-last %.1f)" % (e["block"].mean(), ec["block"].mean(), e["angle_deg"].mean(), ec["angle_deg"].mean()))

def spearman(a, b):                                                           # rows: start states, cols: branches
    ra, rb = a.argsort(dim=1).argsort(dim=1).float(), b.argsort(dim=1).argsort(dim=1).float()
    ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
    den = ra.norm(dim=1) * rb.norm(dim=1)
    ok = (a.std(dim=1) > 1e-4) & (den > 0)
    return ((ra * rb).sum(1) / den.clamp_min(1e-9))[ok].mean().item(), int(ok.sum())
goal = torch.tensor([0.5, 0.5], device=dev)
quantities = {
    "block displacement":        lambda s: (s[:, :, T, 2:4] - s[:, :, CTX - 1, 2:4]).norm(dim=-1),
    "agent-to-block distance":   lambda s: (s[:, :, T, :2] - s[:, :, T, 2:4]).norm(dim=-1),
    "block-to-goal distance":    lambda s: (s[:, :, T, 2:4] - goal).norm(dim=-1),
}
for name, fn in quantities.items():
    rho, n = spearman(fn(sp), fn(s_))
    print("  ranking of action sequences by %-24s imagined vs real, Spearman %.2f  (%d start states)" % (name + ":", rho, n))
ret_im, ret_re = out[..., 6].view(N, K, T + 1)[:, :, CTX:].sum(-1), rw.view(N, K, T + 1)[:, :, CTX:].sum(-1)
rho, n = spearman(ret_im, ret_re)
print("  ranking by return (coverage reward, probe on imagined latents):   Spearman %.2f  (%d start states where real return varies)" % (rho, n))
moved = quantities["block displacement"](s_) > 0.02
pm = quantities["block displacement"](sp)
print("  contact: block really moved in %.0f%% of branches; imagined displacement when it moved %.3f (real %.3f), when it did not %.3f"
      % (100 * moved.float().mean(), pm[moved].mean(), quantities["block displacement"](s_)[moved].mean(), pm[~moved].mean()))
