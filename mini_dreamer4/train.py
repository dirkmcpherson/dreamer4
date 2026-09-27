"""Training entry points.

    python -m mini_dreamer4.train tokenizer --data synthetic --out runs/tok
    python -m mini_dreamer4.train dynamics  --data synthetic --tokenizer runs/tok/tokenizer.pt --out runs/dyn
    python -m mini_dreamer4.train rollout   --tokenizer runs/tok/tokenizer.pt --dynamics runs/dyn/dynamics.pt

``--data`` is either ``synthetic`` (the built-in MiniPushT environment) or a path to a PushT
zarr dataset (``data/img``, ``data/action``, ``meta/episode_ends``).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import EpisodeWindowDataset, collate, load_pusht_zarr
from .dynamics import ShortcutDynamics
from .envs import generate_episodes
from .tokenizer import CausalTokenizer


def load_episodes(args) -> tuple[list[dict], list[dict]]:
    if args.data == "synthetic":
        train = generate_episodes(args.synthetic_episodes, args.synthetic_length, image_size=args.image_size, seed=args.seed)
        val = generate_episodes(max(4, args.synthetic_episodes // 10), args.synthetic_length, image_size=args.image_size, seed=args.seed + 1)
        return train, val
    episodes = load_pusht_zarr(args.data)
    n_val = max(1, len(episodes) // 20)
    return episodes[n_val:], episodes[:n_val]


def save(model: torch.nn.Module, cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state_dict": model.state_dict()}, path)


def load_tokenizer(path: str, device) -> CausalTokenizer:
    ckpt = torch.load(path, map_location=device)
    tok = CausalTokenizer(**ckpt["config"]).to(device)
    tok.load_state_dict(ckpt["state_dict"])
    return tok.eval()


def load_dynamics(path: str, device) -> ShortcutDynamics:
    ckpt = torch.load(path, map_location=device)
    dyn = ShortcutDynamics(**ckpt["config"]).to(device)
    dyn.load_state_dict(ckpt["state_dict"])
    return dyn.eval()


def psnr(mse: float) -> float:
    return 10 * math.log10(1.0 / max(mse, 1e-10))


# ---------------------------------------------------------------------------------- tokenizer
def train_tokenizer(args):
    device = torch.device(args.device)
    train_eps, val_eps = load_episodes(args)
    ds = EpisodeWindowDataset(train_eps, args.seq_len, args.image_size, samples_per_epoch=args.steps * args.batch_size, seed=args.seed)
    dl = iter(DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers))
    val = collate([EpisodeWindowDataset(val_eps, args.seq_len, args.image_size, samples_per_epoch=32, seed=1)[i] for i in range(32)])["video"].to(device)

    cfg = dict(image_size=args.image_size, patch_size=args.patch_size, dim=args.dim, depth=args.depth, heads=args.heads,
               dim_head=args.dim_head, num_latents=args.num_latents, latent_dim=args.latent_dim, time_every=args.time_every,
               lpips_weight=args.lpips_weight)
    tok = CausalTokenizer(**cfg).to(device)
    opt = torch.optim.AdamW(tok.parameters(), lr=args.lr, weight_decay=0.01)
    out = Path(args.out)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = next(dl)
        loss, stats = tok(batch["video"].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps:
            tok.eval()
            with torch.no_grad():
                mse = F.mse_loss(tok.decode(tok.encode(val)).clamp(0, 1), val).item()
            tok.train()
            print(f"step {step} train_mse={stats['mse']:.5f} latent_std={stats['latent_std']:.3f} val_psnr={psnr(mse):.2f}dB ({time.time() - t0:.0f}s)", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save(tok, cfg, out / "tokenizer.pt")
    print(f"saved {out / 'tokenizer.pt'}")


# ---------------------------------------------------------------------------------- dynamics
def train_dynamics(args):
    device = torch.device(args.device)
    tok = load_tokenizer(args.tokenizer, device)
    train_eps, val_eps = load_episodes(args)
    ds = EpisodeWindowDataset(train_eps, args.seq_len, tok.image_size, samples_per_epoch=args.steps * args.batch_size, seed=args.seed)
    dl = iter(DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers))
    val = collate([EpisodeWindowDataset(val_eps, args.seq_len, tok.image_size, samples_per_epoch=16, seed=1)[i] for i in range(16)])
    with torch.no_grad():
        val_z = tok.encode(val["video"].to(device))
    val_a = val["actions"].to(device)
    action_dim = val_a.shape[-1]

    cfg = dict(num_latents=tok.num_latents, latent_dim=tok.latent_dim, pack=args.pack, dim=args.dim, depth=args.depth, heads=args.heads,
               dim_head=args.dim_head, time_every=args.time_every, num_registers=args.registers, k_max=args.k_max,
               action_dim=action_dim, bootstrap_warmup=args.bootstrap_warmup)
    dyn = ShortcutDynamics(**cfg).to(device)
    opt = torch.optim.AdamW(dyn.parameters(), lr=args.lr, weight_decay=0.01)
    out = Path(args.out)
    ctx = max(1, args.seq_len // 4)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = next(dl)
        with torch.no_grad():
            z = tok.encode(batch["video"].to(device))
        loss, stats = dyn(z, batch["actions"].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps:
            dyn.eval()
            with torch.no_grad():
                torch.manual_seed(0); l_true, _ = dyn(val_z, val_a)
                torch.manual_seed(0); l_shuf, _ = dyn(val_z, val_a[torch.randperm(val_a.shape[0])])
                gen = dyn.sample(val_z[:, :ctx], val_a, horizon=args.seq_len - ctx, num_steps=4)
                pred = tok.decode(gen[:, ctx:]).clamp(0, 1)
                target = val["video"][:, ctx:].to(device)
                floor = val["video"][:, ctx - 1:ctx].to(device).expand_as(target)
                gain = psnr(F.mse_loss(pred, target).item()) - psnr(F.mse_loss(floor, target).item())
            dyn.train()
            print(f"step {step} flow_mse={stats['flow_mse']:.4f} boot_mse={stats['boot_mse']:.4f} action_shuffle_ratio={l_shuf / l_true:.2f} "
                  f"rollout_psnr_gain_over_floor={gain:+.2f}dB ({time.time() - t0:.0f}s)", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save(dyn, cfg, out / "dynamics.pt")
    print(f"saved {out / 'dynamics.pt'}")


# ---------------------------------------------------------------------------------- rollout
def rollout(args):
    device = torch.device(args.device)
    tok, dyn = load_tokenizer(args.tokenizer, device), load_dynamics(args.dynamics, device)
    _, val_eps = load_episodes(args)
    val = collate([EpisodeWindowDataset(val_eps, args.seq_len, tok.image_size, samples_per_epoch=8, seed=2)[i] for i in range(8)])
    ctx = max(1, args.seq_len // 4)
    with torch.no_grad():
        z = tok.encode(val["video"][:, :ctx].to(device))
        gen = dyn.sample(z, val["actions"].to(device), horizon=args.seq_len - ctx, num_steps=args.num_steps)
        video = tok.decode(gen).clamp(0, 1).cpu()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"generated": video, "ground_truth": val["video"], "context_frames": ctx}, out / "rollout.pt")
    target, pred = val["video"][:, ctx:], video[:, ctx:]
    print(f"rollout PSNR {psnr(F.mse_loss(pred, target).item()):.2f}dB over {args.seq_len - ctx} frames; saved {out / 'rollout.pt'}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default="synthetic")
    common.add_argument("--synthetic-episodes", type=int, default=400)
    common.add_argument("--synthetic-length", type=int, default=48)
    common.add_argument("--image-size", type=int, default=64)
    common.add_argument("--seq-len", type=int, default=8)
    common.add_argument("--batch-size", type=int, default=16)
    common.add_argument("--steps", type=int, default=20000)
    common.add_argument("--lr", type=float, default=3e-4)
    common.add_argument("--log-every", type=int, default=200)
    common.add_argument("--save-every", type=int, default=1000)
    common.add_argument("--workers", type=int, default=0)
    common.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--dim", type=int, default=256)
    common.add_argument("--depth", type=int, default=4)
    common.add_argument("--heads", type=int, default=4)
    common.add_argument("--dim-head", type=int, default=64)
    common.add_argument("--time-every", type=int, default=4)
    common.add_argument("--out", default="runs/mini_dreamer4")

    t = sub.add_parser("tokenizer", parents=[common])
    t.add_argument("--patch-size", type=int, default=8)
    t.add_argument("--num-latents", type=int, default=16)
    t.add_argument("--latent-dim", type=int, default=32)
    t.add_argument("--lpips-weight", type=float, default=0.0)

    d = sub.add_parser("dynamics", parents=[common])
    d.add_argument("--tokenizer", required=True)
    d.add_argument("--pack", type=int, default=2)
    d.add_argument("--registers", type=int, default=4)
    d.add_argument("--k-max", type=int, default=64)
    d.add_argument("--bootstrap-warmup", type=int, default=2000)

    r = sub.add_parser("rollout", parents=[common])
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--dynamics", required=True)
    r.add_argument("--num-steps", type=int, default=4)

    args = p.parse_args()
    {"tokenizer": train_tokenizer, "dynamics": train_dynamics, "rollout": rollout}[args.cmd](args)


if __name__ == "__main__":
    main()
