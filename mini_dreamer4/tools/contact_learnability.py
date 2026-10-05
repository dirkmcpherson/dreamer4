"""Is MiniPushT's contact physics learnable from the offline data at all?

A plain MLP on the RAW simulator state: (state_t, action_t) -> state_{t+1}. No latents, no transformer,
no diffusion. If this cannot predict the block's motion either, the limit is the data / physics, not Dreamer.

    python -m mini_dreamer4.tools.contact_learnability [episodes ...]
"""
import sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from mini_dreamer4.envs import generate_episodes

dev = "cuda" if torch.cuda.is_available() else "cpu"

def pairs(eps):
    s = torch.as_tensor(np.stack([e["states"] for e in eps])); a = torch.as_tensor(np.stack([e["actions"] for e in eps]))
    return torch.cat((s[:, :-1], a[:, :-1]), dim=-1).flatten(0, 1).to(dev), s[:, :-1].flatten(0, 1).to(dev), s[:, 1:].flatten(0, 1).to(dev)

def report(name, pred, cur, nxt):
    moved = (nxt[:, 2:4] - cur[:, 2:4]).norm(dim=-1) > 1e-4
    e = (pred[:, 2:4] - nxt[:, 2:4]).norm(dim=-1)
    ang = lambda s: torch.atan2(s[:, 5], s[:, 4])
    da = ((ang(pred) - ang(nxt) + np.pi) % (2 * np.pi) - np.pi).abs() * 180 / np.pi
    print("   %-34s agent %.4f | block: all %.4f, when it moves %.4f, when still %.4f | angle when it moves %.2f deg"
          % (name, (pred[:, :2] - nxt[:, :2]).norm(dim=-1).mean(), e.mean(), e[moved].mean(), e[~moved].mean(), da[moved].mean()))

t0 = time.time()
vx, vc, vn = pairs(generate_episodes(200, 48, image_size=16, seed=1))
moved = (vn[:, 2:4] - vc[:, 2:4]).norm(dim=-1) > 1e-4
print("held-out transitions: %d; block moves in %.0f%%; real block motion when it moves %.4f, angle change %.2f deg"
      % (len(vx), 100 * moved.float().mean(), (vn[:, 2:4] - vc[:, 2:4]).norm(dim=-1)[moved].mean(),
         (((torch.atan2(vn[:, 5], vn[:, 4]) - torch.atan2(vc[:, 5], vc[:, 4]) + np.pi) % (2 * np.pi) - np.pi).abs() * 180 / np.pi)[moved].mean()))
report("copy the current state", vc, vc, vn)
for n_eps in [int(x) for x in sys.argv[1:]] or [400, 4000]:
    x, c, n = pairs(generate_episodes(n_eps, 48, image_size=16, seed=0))
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 6)).to(dev)
    steps = 30000
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(0, len(x), (1024,), device=dev)
        loss = F.mse_loss(net(x[i]), (n[i] - c[i]) * 20)          # predict the scaled change of state
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        i = torch.randint(0, len(x), (20000,), device=dev)
        report("MLP, %d episodes (train set)" % n_eps, c[i] + net(x[i]) / 20, c[i], n[i])
        report("MLP, %d episodes (held-out)" % n_eps, vc + net(vx) / 20, vc, vn)
    print("   (%d transitions, %.0fs elapsed)" % (len(x), time.time() - t0), flush=True)
