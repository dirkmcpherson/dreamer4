"""Block and agent tracking on real PushT measured in PIXELS, with no learned probe.

Imagined latents are decoded to frames and the block / agent are located by colour; the same
segmentation of the true frame is the reference. Baselines: repeating the last context frame,
rollouts with another clip's actions, and the tokenizer's reconstruction of the true frame (ceiling).

    python -m mini_dreamer4.tools.pixel_checks_pusht <pusht.zarr> <tokenizer.pt> [<dynamics.pt> ...]

With no dynamics checkpoints only the tokenizer's reconstruction is measured. BACKGROUND=texture in the
environment loads the textured variant of the data (the tokenizer must have been trained on it).
"""
import os, sys
import torch, torch.nn.functional as F
from mini_dreamer4.data import load_pusht_zarr
from mini_dreamer4.train import load_tokenizer, load_dynamics

dev, CTX, WIN, H, K = "cuda", 4, 16, 44, 4
BLOCK = torch.tensor([[143, 163, 184], [119, 136, 153]], device=dev).float() / 255
AGENT = torch.tensor([[78, 126, 255], [65, 105, 225]], device=dev).float() / 255
tok = load_tokenizer(sys.argv[2], dev)
episodes = load_pusht_zarr(sys.argv[1], background=os.environ.get("BACKGROUND"))
val_eps = episodes[:max(1, len(episodes) // 20)]          # same held-out split as mini_dreamer4.train

v, a = [], []
for e in val_eps:
    for i in range(0, len(e["video"]) - (CTX + H) + 1, 8):
        v.append(torch.as_tensor(e["video"][i:i + CTX + H])); a.append(torch.as_tensor(e["actions"][i:i + CTX + H]))
video = (torch.stack(v).float() / 255).movedim(-1, -3)       # (N, T, 3, H, W) on cpu
actions = torch.stack(a).float().to(dev)
N, T = video.shape[:2]

def chunks(fn, x, batch=16):
    return torch.cat([torch.cat([fn(x[b:b + batch, i:i + WIN].to(dev)) for i in range(0, x.shape[1], WIN)], dim=1) for b in range(0, len(x), batch)])

def mask(frames, ref, thr):                                # frames (..., 3, H, W) -> bool (..., H, W)
    d = (frames.unsqueeze(-4) - ref[:, :, None, None]).abs().sum(dim=-3)
    return d.min(dim=-3).values < thr

def centroid(m):                                           # (..., H, W) -> (..., 2) in frame widths, and validity
    h, w = m.shape[-2:]
    ys = (torch.arange(h, device=m.device).float() + 0.5) / h; xs = (torch.arange(w, device=m.device).float() + 0.5) / w
    n = m.flatten(-2).sum(-1).clamp_min(1)
    return torch.stack(((m * xs).flatten(-2).sum(-1) / n, (m * ys[:, None]).flatten(-2).sum(-1) / n), dim=-1), m.flatten(-2).sum(-1) >= 8

def measure(frames, truth):
    """frames, truth (N, 3, H, W) on dev -> block centroid error, block IoU, agent centroid error, fraction with a visible block"""
    mb, tb = mask(frames, BLOCK, 0.2), mask(truth, BLOCK, 0.2)
    (cb, okb), (ct, okt) = centroid(mb), centroid(tb)
    ok = okb & okt
    iou = ((mb & tb).flatten(-2).sum(-1).float() / (mb | tb).flatten(-2).sum(-1).clamp_min(1))[okt]
    (ca, oka), (cta, okta) = centroid(mask(frames, AGENT, 0.35)), centroid(mask(truth, AGENT, 0.35))
    oa = (mask(frames, AGENT, 0.35).flatten(-2).sum(-1) >= 3) & (mask(truth, AGENT, 0.35).flatten(-2).sum(-1) >= 3)
    return (cb - ct).norm(dim=-1)[ok].mean().item(), iou.mean().item(), (ca - cta).norm(dim=-1)[oa].mean().item(), (okb[okt]).float().mean().item(), cb, ok

@torch.no_grad()
def rollout(dyn, z_ctx, act):
    z = z_ctx.clone()
    for _ in range(H):
        t0 = z.shape[1]; lo = max(0, t0 + 1 - WIN)
        z = torch.cat((z, dyn.sample(z[:, lo:], act[:, lo:t0 + 1], horizon=1, num_steps=K)[:, -1:]), dim=1)
    return z

with torch.no_grad():
    z_true = chunks(tok.encode, video)
    recon = chunks(lambda z: tok.decode(z).clamp(0, 1), z_true.cpu()).cpu()
HS = (1, 2, 4, 8, 12, 16, 24, 32, 44)
truth_at = lambda h: video[:, CTX + h - 1].to(dev)
last = video[:, CTX - 1].to(dev)
d_true = (truth_at(1)[:, :1] * 0)
tb_prev, _ = centroid(mask(video[:, CTX - 1:CTX + H - 1].to(dev), BLOCK, 0.2)); tb_next, _ = centroid(mask(video[:, CTX:CTX + H].to(dev), BLOCK, 0.2))
print("%d held-out windows of %d frames from %d episodes; real per-frame block centroid motion %.4f; K=%d, sliding %d-frame window; background=%s"
      % (N, T, len(val_eps), (tb_next - tb_prev).norm(dim=-1).mean(), K, WIN, os.environ.get("BACKGROUND", "plain")))
with torch.no_grad():
    v_all = video.flatten(0, 1); r_all = recon.flatten(0, 1)
    e, iou, ea, vis, _, _ = measure(r_all.to(dev), v_all.to(dev))
    print("tokenizer reconstruction over all %d held-out frames: psnr %.2f dB | block centroid err %.4f | block IoU %.3f | agent centroid err %.4f"
          % (len(v_all), 10 * torch.log10(1 / F.mse_loss(r_all, v_all)).item(), e, iou, ea))

def table(name, frames_at):
    print("\n%s\n   h | block centroid err | block IoU | agent centroid err | block visible" % name)
    for h in HS:
        e, iou, ea, vis, _, _ = measure(frames_at(h), truth_at(h))
        print("  %2d |       %.4f       |   %.3f   |       %.4f       |     %.2f" % (h, e, iou, ea, vis))

table("tokenizer reconstruction of the true frame (ceiling)", lambda h: recon[:, CTX + h - 1].to(dev))
table("repeat the last context frame (copy-last)", lambda h: last)
c_last, ok_last = centroid(mask(last, BLOCK, 0.2))
for path in sys.argv[3:]:
    dyn = load_dynamics(path, dev)
    torch.manual_seed(0)
    with torch.no_grad():
        gen = torch.cat([rollout(dyn, z_true[b:b + 32, :CTX], actions[b:b + 32]) for b in range(0, N, 32)])
        wrong = torch.cat([rollout(dyn, z_true[b:b + 32, :CTX], actions.roll(N // 2, dims=0)[b:b + 32]) for b in range(0, N, 32)])
        pix = chunks(lambda z: tok.decode(z).clamp(0, 1), gen.cpu()).cpu()
        pix_w = chunks(lambda z: tok.decode(z).clamp(0, 1), wrong.cpu()).cpu()
    name = path.split("/")[-1]
    table("MODEL %s, true actions" % name, lambda h: pix[:, CTX + h - 1].to(dev))
    table("MODEL %s, another clip's actions" % name, lambda h: pix_w[:, CTX + h - 1].to(dev))
    for h in (12, 44):
        ct, okt = centroid(mask(truth_at(h), BLOCK, 0.2)); cm, okm = centroid(mask(pix[:, CTX + h - 1].to(dev), BLOCK, 0.2))
        ok = okt & okm & ok_last
        dr, dm = (ct - c_last)[ok], (cm - c_last)[ok]
        moved = dr.norm(dim=-1) > 0.02
        rank = lambda x: x.argsort().argsort().float()
        ra, rb = rank(dm.norm(dim=-1)), rank(dr.norm(dim=-1)); ra, rb = ra - ra.mean(), rb - rb.mean()
        print("   block displacement over %2d frames: moved in %.0f%% of windows; imagined %.3f vs real %.3f when it moved, imagined %.3f when still; "
              "direction cosine %.2f; rank correlation %.2f" % (h, 100 * moved.float().mean(), dm.norm(dim=-1)[moved].mean(), dr.norm(dim=-1)[moved].mean(),
                                                               dm.norm(dim=-1)[~moved].mean(), F.cosine_similarity(dm[moved], dr[moved], dim=-1).mean(),
                                                               (ra * rb).sum() / (ra.norm() * rb.norm())))
