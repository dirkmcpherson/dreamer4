"""Where does the imagined-state error come from?  (oracle latents, so the state can be read back almost exactly)

1. single network call: x-prediction of the next frame from pure noise, with the TRUE history as context
2. full sampling of one frame (K steps) from the true history
3. autoregressive rollouts with different amounts of context noise

    python -m mini_dreamer4.tools.oracle_error_sources <dynamics.pt>
"""
import math, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from mini_dreamer4.dynamics import shift_actions
from mini_dreamer4.envs import generate_episodes, state_to_oracle_latents
from mini_dreamer4.train import load_dynamics

dev, CTX, WIN = "cuda", 4, 16
dyn = load_dynamics(sys.argv[1], dev)
torch.manual_seed(0)
stack = lambda eps, k: torch.as_tensor(np.stack([e[k] for e in eps]))
lat = lambda s: state_to_oracle_latents(s, dyn.num_latents, dyn.latent_dim).to(dev)
train, val = generate_episodes(2000, 48, image_size=16, seed=0), generate_episodes(40, 48, image_size=16, seed=1)
ztr, ytr = lat(stack(train, "states")), stack(train, "states").to(dev)
zva, yva, ava = lat(stack(val, "states")), stack(val, "states").to(dev), stack(val, "actions").to(dev)

probe = nn.Sequential(nn.Flatten(-2), nn.Linear(ztr.shape[-2] * ztr.shape[-1], 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 6)).to(dev)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 15000)
X, Y = ztr.flatten(0, 1), ytr.flatten(0, 1)
for _ in range(15000):
    i = torch.randint(0, len(X), (512,), device=dev)
    loss = F.mse_loss(probe(X[i] + 0.02 * torch.randn_like(X[i])), Y[i])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
probe.eval()

def err(z, s):
    p = probe(z)
    return (p[..., :2] - s[..., :2]).norm(dim=-1).mean().item(), (p[..., 2:4] - s[..., 2:4]).norm(dim=-1).mean().item()

# held-out 16-frame windows; predict the last frame from the 15 before it
w = torch.stack([zva[:, i:i + WIN] for i in range(0, 48 - WIN + 1, 8)]).flatten(0, 1)
ws = torch.stack([yva[:, i:i + WIN] for i in range(0, 48 - WIN + 1, 8)]).flatten(0, 1)
wa = torch.stack([ava[:, i:i + WIN] for i in range(0, 48 - WIN + 1, 8)]).flatten(0, 1)
b, t = w.shape[:2]
pa, va = shift_actions(wa, b, t, dev)
moved = (ws[:, -1, 2:4] - ws[:, -2, 2:4]).norm(dim=-1) > 1e-4
print("%d held-out windows; block moves in the predicted step in %.0f%% of them" % (b, 100 * moved.float().mean()))
print("reference for the last frame: probe on true latent  agent %.4f block %.4f | copy previous frame  agent %.4f block %.4f" % (*err(w[:, -1], ws[:, -1]), *err(w[:, -2], ws[:, -1])))
print("   real motion in that step: agent %.4f, block %.4f (%.4f when it moves)" % ((ws[:, -1, :2] - ws[:, -2, :2]).norm(dim=-1).mean(), (ws[:, -1, 2:4] - ws[:, -2, 2:4]).norm(dim=-1).mean(), (ws[:, -1, 2:4] - ws[:, -2, 2:4]).norm(dim=-1)[moved].mean()))

@torch.no_grad()
def one_call(ctx_noise, e_target, ctx_label=None):
    k, e_max = dyn.k_max, dyn.max_exp
    ctx = (1 - ctx_noise) * w + ctx_noise * torch.randn_like(w)
    z = ctx.clone(); z[:, -1] = torch.randn_like(z[:, -1])
    sig = torch.full((b, t), k - 1 if ctx_label is None else ctx_label, dtype=torch.long, device=dev); sig[:, -1] = 0
    step = torch.full((b, t), e_max, dtype=torch.long, device=dev); step[:, -1] = e_target
    return dyn.predict(z, sig, step, pa, va)[:, -1]

print("\n1. single network call from pure noise, true history as context")
for name, cn, e in (("clean context, step size 1/4 (first of K=4)", 0.0, 2), ("clean context, step size 1 (one-shot)", 0.0, 0), ("clean context, step size 1/64", 0.0, dyn.max_exp),
                    ("context noise 1/64, step size 1/4", 1 / 64, 2), ("context noise 0.1, step size 1/4", 0.1, 2)):
    x = one_call(cn, e, ctx_label=int(round((1 - cn) * dyn.k_max)) if cn > 1 / 64 else None)
    a, bl = err(x, ws[:, -1])
    bs = (probe(x)[~moved][:, 2:4] - ws[~moved][:, -1, 2:4]).norm(dim=-1).mean().item()
    print("   %-46s latent mse %.5f | agent %.4f | block %.4f (%.4f when the block is still)" % (name, F.mse_loss(x, w[:, -1]).item(), a, bl, bs))

print("\n2. full sampling of one frame from the true history")
with torch.no_grad():
    for K in (1, 4, 64):
        for cn in (0.0, 1 / 64):
            g = dyn.sample(w[:, :-1], wa, horizon=1, num_steps=K, ctx_noise=cn)[:, -1]
            a, bl = err(g, ws[:, -1])
            print("   K=%-2d context noise %.4f: latent mse %.5f | agent %.4f | block %.4f" % (K, cn, F.mse_loss(g, w[:, -1]).item(), a, bl))

print("\n3. autoregressive rollouts from %d true frames (sliding %d-frame window), K=4" % (CTX, WIN))
@torch.no_grad()
def rollout(cn, horizon=12, K=4):
    z = zva[:, :CTX].clone()
    for _ in range(horizon):
        t0 = z.shape[1]; lo = max(0, t0 + 1 - WIN)
        z = torch.cat((z, dyn.sample(z[:, lo:], ava[:, lo:t0 + 1], horizon=1, num_steps=K, ctx_noise=cn)[:, -1:]), dim=1)
    return z
for cn in (0.0, 1 / 64, 0.1):
    g = rollout(cn)
    print("   context noise %.4f: " % cn + " | ".join("h=%d agent %.3f block %.3f" % (h, *err(g[:, CTX + h - 1], yva[:, CTX + h - 1])) for h in (1, 4, 12)))
print("   copy-last:            " + " | ".join("h=%d agent %.3f block %.3f" % (h, *err(zva[:, CTX - 1], yva[:, CTX + h - 1])) for h in (1, 4, 12)))
