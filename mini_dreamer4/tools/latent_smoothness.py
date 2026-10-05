import sys, torch, torch.nn.functional as F
from mini_dreamer4.train import load_tokenizer
from mini_dreamer4.data import EpisodeWindowDataset, collate
from mini_dreamer4.envs import generate_episodes
dev = "cuda"
tok = load_tokenizer(sys.argv[1], dev)
eps = generate_episodes(40, 48, image_size=tok.image_size, seed=1)
ds = EpisodeWindowDataset(eps, 16, tok.image_size, samples_per_epoch=64, seed=1)
b = collate([ds[i] for i in range(64)])
v, st = b["video"].to(dev), b["states"].to(dev)
with torch.no_grad():
    z = tok.encode(v)
f = z.flatten(0, 1)                      # (frames, N, D)
std = f.std(dim=0)
print("per-dim std: mean %.3f  median %.3f  max %.3f | mean var %.3f | 2*mean var %.3f" % (std.mean(), std.median(), std.max(), std.pow(2).mean(), 2 * std.pow(2).mean()))
perm = torch.randperm(z.shape[0], device=dev)
print("unrelated clips mse %.4f (fixed points in perm: %d)" % (F.mse_loss(z, z[perm]).item(), (perm == torch.arange(len(perm), device=dev)).sum()))
print("copy_last %.4f  -> fraction of unrelated-frame distance: %.2f" % (F.mse_loss(z[:, 1:], z[:, :-1]).item(), F.mse_loss(z[:, 1:], z[:, :-1]).item() / (2 * std.pow(2).mean().item())))
pv = v.flatten(0, 1).var(dim=0).mean().item()
print("pixels: copy_last %.5f  2*var %.5f -> fraction %.2f" % (F.mse_loss(v[:, 1:], v[:, :-1]).item(), 2 * pv, F.mse_loss(v[:, 1:], v[:, :-1]).item() / (2 * pv)))
# how far does the scene actually move per frame?  state = agent xy, block xy, cos, sin
d = (st[:, 1:] - st[:, :-1])
print("per-frame motion: agent %.3f of frame width (radius 0.045), block %.4f, frames with block motion %.2f" % (
    d[..., :2].norm(dim=-1).mean(), d[..., 2:4].norm(dim=-1).mean(), (d[..., 2:].abs().sum(-1) > 1e-6).float().mean()))
# is the latent a smooth function of the state?  linear probe latents -> state, and k-NN consistency
X = torch.cat((f.flatten(1), torch.ones(len(f), 1, device=dev)), dim=1); Y = st.flatten(0, 1)
n = len(X) * 3 // 4
w = torch.linalg.lstsq(X[:n].T @ X[:n] + 1e-2 * torch.eye(X.shape[1], device=dev), X[:n].T @ Y[:n]).solution
err = (X[n:] @ w - Y[n:]).abs().mean(dim=0)
print("linear probe latent -> state, mean abs error (agent x,y | block x,y | cos,sin):", " ".join(f"{e:.3f}" for e in err.tolist()))
# latent distance vs agent displacement for frame pairs where the block did not move
dz = (z[:, 1:] - z[:, :-1]).pow(2).mean(dim=(2, 3)).flatten()
da = d[..., :2].norm(dim=-1).flatten(); still = (d[..., 2:].abs().sum(-1) < 1e-6).flatten()
for lo, hi in ((0, .01), (.01, .03), (.03, .05), (.05, .1)):
    m = still & (da >= lo) & (da < hi)
    if m.any(): print(f"block still, agent moved {lo:.2f}-{hi:.2f}: latent mse {dz[m].mean():.4f}  (n={int(m.sum())})")
print(f"block moved: latent mse {dz[~still].mean():.4f} (n={int((~still).sum())})")
