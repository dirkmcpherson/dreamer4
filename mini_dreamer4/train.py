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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import EpisodeWindowDataset, collate, load_pusht
from .dynamics import ShortcutDynamics
from .envs import generate_episodes
from .tokenizer import CausalTokenizer


def load_episodes(args) -> tuple[list[dict], list[dict]]:
    if args.data == "synthetic":
        train = generate_episodes(args.synthetic_episodes, args.synthetic_length, image_size=args.image_size, seed=args.seed)
        val = generate_episodes(max(4, args.synthetic_episodes // 10), args.synthetic_length, image_size=args.image_size, seed=args.seed + 1)
        return train, val
    episodes = load_pusht(args.data, background=args.background)
    # every 20th episode is held out, so each dataset (and each start state) contributes to validation
    return [e for i, e in enumerate(episodes) if i % 20], [e for i, e in enumerate(episodes) if not i % 20]


def save(model: torch.nn.Module, cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state_dict": model.state_dict()}, path)


def load_tokenizer(path: str, device) -> CausalTokenizer:
    ckpt = torch.load(path, map_location=device)
    # the perceptual loss is only needed for training: load without it so inference does not require `lpips`
    tok = CausalTokenizer(**{**ckpt["config"], "lpips_weight": 0.0}).to(device)
    tok.load_state_dict({k: v for k, v in ckpt["state_dict"].items() if not k.startswith("lpips_norm.")})
    return tok.eval()


def load_dynamics(path: str, device) -> ShortcutDynamics:
    ckpt = torch.load(path, map_location=device)
    dyn = ShortcutDynamics(**ckpt["config"]).to(device)
    dyn.load_state_dict(ckpt["state_dict"])
    return dyn.eval()


def psnr(mse: float) -> float:
    return 10 * math.log10(1.0 / max(mse, 1e-10))


def val_windows(episodes: list[dict], seq_len: int, image_size: int, n: int, seed: int = 1) -> dict:
    """n distinct held-out windows (one dataset object, so its RNG advances between draws)."""
    ds = EpisodeWindowDataset(episodes, seq_len, image_size, samples_per_epoch=n, seed=seed)
    return collate([ds[i] for i in range(n)])


@torch.no_grad()
def evaluate_dynamics(tok, dyn, video: torch.Tensor, z: torch.Tensor, actions: torch.Tensor, ctx: int,
                      num_steps: int = 4) -> tuple[dict, torch.Tensor]:
    """Held-out diagnostics for a dynamics model; returns (metrics, decoded rollout (B, T, 3, H, W)).

    "Wrong" actions are the batch rolled by one, so every clip is paired with another clip's actions.
    Paired passes share their noise; the global RNG state is restored afterwards so that evaluating
    does not change (or periodically repeat) the noise seen in training.
    """
    devices = [torch.cuda.current_device()] if z.is_cuda else []
    rng_state = torch.random.get_rng_state(), [torch.cuda.get_rng_state(d) for d in devices]
    was_training = dyn.training
    dyn.eval()
    try:
        wrong, horizon = actions.roll(1, dims=0), z.shape[1] - ctx
        torch.manual_seed(0); l_true, _ = dyn(z, actions)
        torch.manual_seed(0); l_wrong, _ = dyn(z, wrong)
        torch.manual_seed(0); gen = dyn.sample(z[:, :ctx], actions, horizon=horizon, num_steps=num_steps)
        torch.manual_seed(0); gen_wrong = dyn.sample(z[:, :ctx], wrong, horizon=horizon, num_steps=num_steps)
    finally:
        dyn.train(was_training)
        torch.random.set_rng_state(rng_state[0])
        for d, state in zip(devices, rng_state[1]):
            torch.cuda.set_rng_state(state, d)
    full = tok.decode(gen).clamp(0, 1)
    pred, target = full[:, ctx:], video[:, ctx:]
    floor = video[:, ctx - 1:ctx].expand_as(target)                  # repeat the last context frame
    ceiling = tok.decode(z)[:, ctx:].clamp(0, 1)                     # tokenizer reconstruction of the true future
    roll = F.mse_loss(gen[:, ctx:], z[:, ctx:]).item()
    metrics = dict(
        action_shuffle_ratio=(l_wrong / l_true).item(),              # training loss, wrong / true actions
        rollout_shuffle_ratio=F.mse_loss(gen_wrong[:, ctx:], z[:, ctx:]).item() / max(roll, 1e-12),
        rollout_psnr=psnr(F.mse_loss(pred, target).item()),
        rollout_psnr_gain_over_floor=psnr(F.mse_loss(pred, target).item()) - psnr(F.mse_loss(floor, target).item()),
        recon_psnr_ceiling=psnr(F.mse_loss(ceiling, target).item()),
        rollout_latent_mse=roll,
        copy_last_latent_mse=F.mse_loss(z[:, ctx - 1:ctx].expand_as(z[:, ctx:]), z[:, ctx:]).item(),
    )
    return metrics, full


class Logger:
    """Prints scalars and, with --wandb, mirrors them (and eval videos) to Weights & Biases."""

    def __init__(self, args, cfg: dict):
        self.run, self.t0 = None, time.time()
        if args.wandb:
            import wandb
            self.run = wandb.init(project=args.wandb_project, name=args.run_name, dir=args.out,
                                  config={**vars(args), "model": cfg})

    def scalars(self, step: int, **values):
        body = " ".join(f"{k}={v:.5g}" for k, v in values.items())
        print(f"step {step} {body} ({time.time() - self.t0:.0f}s)", flush=True)
        if self.run is not None:
            self.run.log(values, step=step)

    def video(self, step: int, name: str, truth: torch.Tensor, pred: torch.Tensor, fps: int = 8, n: int = 4):
        """truth, pred (B, T, 3, H, W) in [0, 1]; logged as rows of [truth | prediction]."""
        if self.run is None:
            return
        import wandb
        pair = torch.cat((truth[:n], pred[:n]), dim=-1)                       # side by side
        grid = torch.cat(list(pair), dim=-2)                                  # samples stacked vertically -> (T, 3, nH, 2W)
        grid = F.interpolate(grid, scale_factor=2, mode="nearest")
        frames = (grid.clamp(0, 1) * 255).round().byte().cpu().numpy()
        try:
            self.run.log({name: wandb.Video(frames, fps=fps, format="gif")}, step=step)
        except Exception as e:  # no video encoder in the environment: fall back to a strip of frames
            print(f"video logging failed ({e!r}); logging frames as an image", flush=True)
            strip = frames[:: max(1, len(frames) // 8)].transpose(0, 2, 3, 1)
            self.run.log({name: wandb.Image(np.concatenate(list(strip), axis=1))}, step=step)

    def close(self):
        if self.run is not None:
            self.run.finish()


# ---------------------------------------------------------------------------------- tokenizer
def train_tokenizer(args):
    device = torch.device(args.device)
    train_eps, val_eps = load_episodes(args)
    ds = EpisodeWindowDataset(train_eps, args.seq_len, args.image_size, samples_per_epoch=args.steps * args.batch_size, seed=args.seed,
                              shift=args.shift, shift_actions=args.data != "synthetic")
    dl = iter(DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers))
    val = val_windows(val_eps, args.seq_len, args.image_size, args.val_windows)["video"].to(device)

    cfg = dict(image_size=args.image_size, patch_size=args.patch_size, dim=args.dim, depth=args.depth, heads=args.heads,
               dim_head=args.dim_head, num_latents=args.num_latents, latent_dim=args.latent_dim, time_every=args.time_every,
               lpips_weight=args.lpips_weight, mask_prob=(0.0, args.mask_max), center_patches=not args.no_center_patches)
    tok = CausalTokenizer(**cfg).to(device)
    log = Logger(args, cfg)
    opt = torch.optim.AdamW(tok.parameters(), lr=args.lr, weight_decay=0.01)
    out = Path(args.out)
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
                z = tok.encode(val)
                rec = tok.decode(z).clamp(0, 1)
            tok.train()
            # latent_input_std: spread of each latent dim across frames (near 0 = latents ignore the input);
            # latent_copy_last: MSE between consecutive-frame latents (temporal smoothness, see GPU_HANDOFF.md)
            log.scalars(step, train_mse=stats["mse"].item(), val_psnr=psnr(F.mse_loss(rec, val).item()),
                        latent_std=z.std().item(), latent_input_std=z.flatten(0, 1).std(dim=0).mean().item(),
                        latent_copy_last=F.mse_loss(z[:, 1:], z[:, :-1]).item(),
                        **{k: v.item() for k, v in stats.items() if k == "lpips"})
            if step % args.video_every == 0 or step == args.steps:
                log.video(step, "reconstruction", val, rec)
        if step % args.save_every == 0 or step == args.steps:
            save(tok, cfg, out / "tokenizer.pt")
    print(f"saved {out / 'tokenizer.pt'}")
    log.close()


# ---------------------------------------------------------------------------------- dynamics
def train_dynamics(args):
    device = torch.device(args.device)
    tok = load_tokenizer(args.tokenizer, device)
    train_eps, val_eps = load_episodes(args)
    ds = EpisodeWindowDataset(train_eps, args.seq_len, tok.image_size, samples_per_epoch=args.steps * args.batch_size, seed=args.seed,
                              shift=args.shift, shift_actions=args.data != "synthetic")
    dl = iter(DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers))
    val = val_windows(val_eps, args.seq_len, tok.image_size, args.val_windows)
    with torch.no_grad():
        val_z = tok.encode(val["video"].to(device))
    val_a = val["actions"].to(device)
    action_dim = val_a.shape[-1]

    cfg = dict(num_latents=tok.num_latents, latent_dim=tok.latent_dim, pack=args.pack, dim=args.dim, depth=args.depth, heads=args.heads,
               dim_head=args.dim_head, time_every=args.time_every, num_registers=args.registers, k_max=args.k_max,
               action_dim=action_dim, bootstrap_warmup=args.bootstrap_warmup,
               ramp_weight=not args.no_ramp, clean_context_prob=args.clean_context_prob)
    dyn = ShortcutDynamics(**cfg).to(device)
    log = Logger(args, cfg)
    opt = torch.optim.AdamW(dyn.parameters(), lr=args.lr, weight_decay=0.01)
    lr_at = lambda s: min(1.0, (s + 1) / 1000) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))) * 0.98 + 0.02)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at if args.cosine else (lambda s: 1.0))
    out = Path(args.out)
    ctx = max(1, args.seq_len // 4)
    val_video = val["video"].to(device)
    print(f"val latents: copy_last={F.mse_loss(val_z[:, 1:], val_z[:, :-1]).item():.5f} "
          f"input_std={val_z.flatten(0, 1).std(dim=0).mean().item():.3f}", flush=True)
    for step in range(1, args.steps + 1):
        batch = next(dl)
        with torch.no_grad():
            z = tok.encode(batch["video"].to(device))
        loss, stats = dyn(z, batch["actions"].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps:
            metrics, full = evaluate_dynamics(tok, dyn, val_video, val_z, val_a, ctx)
            log.scalars(step, flow_mse=stats["flow_mse"].item(), boot_mse=stats["boot_mse"].item(), **metrics)
            if step % args.video_every == 0 or step == args.steps:
                log.video(step, "rollout", val_video, full)
        if step % args.save_every == 0 or step == args.steps:
            save(dyn, cfg, out / "dynamics.pt")
    print(f"saved {out / 'dynamics.pt'}")
    log.close()


# ---------------------------------------------------------------------------------- agent
def train_agent(args):
    """Joint training of the dynamics model and an agent token with policy / success heads (behaviour cloning).

    Starts from a trained dynamics checkpoint. Every step optimises the shortcut-forcing loss and, through a second
    pass on clean latents, the behaviour-cloning and success-prediction losses; all three update the shared transformer.
    """
    device = torch.device(args.device)
    tok = load_tokenizer(args.tokenizer, device)
    train_eps, val_eps = load_episodes(args)
    assert all("action_mask" in e for e in train_eps + val_eps), "agent training needs datasets with action masks (rendered tapes, .npz)"
    ds = EpisodeWindowDataset(train_eps, args.seq_len, tok.image_size, samples_per_epoch=args.steps * args.batch_size, seed=args.seed,
                              shift=args.shift, shift_actions=args.data != "synthetic")
    dl = iter(DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers))
    val = val_windows(val_eps, args.seq_len, tok.image_size, args.val_windows)
    with torch.no_grad():
        val_z = tok.encode(val["video"].to(device))
    val_a, val_m, val_s = val["actions"].to(device), val["action_mask"].to(device) > 0.5, (val["coverage"].to(device) > args.success_threshold).float()
    val_video = val["video"].to(device)

    if args.dynamics == "none":                       # control: same network and heads, no dynamics pretraining
        cfg = dict(num_latents=tok.num_latents, latent_dim=tok.latent_dim, pack=args.pack, dim=args.dim, depth=args.depth, heads=args.heads,
                   dim_head=args.dim_head, time_every=args.time_every, num_registers=args.registers, k_max=args.k_max, action_dim=2,
                   bootstrap_warmup=0, ramp_weight=not args.no_ramp, clean_context_prob=args.clean_context_prob, agent=True, action_bins=args.action_bins)
        dyn = ShortcutDynamics(**cfg).to(device)
        print("control run: randomly initialised transformer", flush=True)
    else:
        init = torch.load(args.dynamics, map_location=device)
        cfg = {**init["config"], "agent": True, "action_bins": args.action_bins, "bootstrap_warmup": 0}
        dyn = ShortcutDynamics(**cfg).to(device)
        missing = dyn.load_state_dict(init["state_dict"], strict=False)
        print(f"initialised from {args.dynamics}; new parameter groups: {sorted({k.split('.')[0] for k in missing.missing_keys})}", flush=True)
    log = Logger(args, cfg)
    opt = torch.optim.AdamW(dyn.parameters(), lr=args.lr, weight_decay=0.01)
    lr_at = lambda s: min(1.0, (s + 1) / 500) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))) * 0.98 + 0.02)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    out, ctx = Path(args.out), max(2, args.seq_len // 4)
    for step in range(1, args.steps + 1):
        batch = next(dl)
        actions = batch["actions"].to(device)
        with torch.no_grad():
            z = tok.encode(batch["video"].to(device))
        dyn_loss, stats = dyn(z, actions)
        if args.dyn_weight == 0:
            dyn_loss = dyn_loss.detach() * 0           # control: no dynamics gradient at all
        bc, rew, astats = dyn.agent_loss(z, actions, batch["action_mask"].to(device) > 0.5,
                                         (batch["coverage"].to(device) > args.success_threshold).float())
        loss = dyn_loss + args.bc_weight * dyn.bc_norm(bc) + args.reward_weight * dyn.rew_norm(rew)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps:
            dyn.eval()
            with torch.no_grad():
                _, _, vstats = dyn.agent_loss(val_z, val_a, val_m, val_s)
            dyn.train()
            metrics, full = evaluate_dynamics(tok, dyn, val_video, val_z, val_a, ctx)
            log.scalars(step, flow_mse=stats["flow_mse"].item(), train_bc_ce=astats["bc_ce"].item(),
                        **{"val_" + k: v.item() for k, v in vstats.items()},
                        val_bc_l1_px=vstats["bc_l1"].item() * tok.image_size / 2 * 512 / tok.image_size,
                        **{k: metrics[k] for k in ("rollout_psnr_gain_over_floor", "rollout_shuffle_ratio", "rollout_latent_mse")})
            if step % args.video_every == 0 or step == args.steps:
                log.video(step, "rollout", val_video, full)
        if step % args.save_every == 0 or step == args.steps:
            save(dyn, cfg, out / "agent.pt")
    print(f"saved {out / 'agent.pt'}")
    log.close()


# ---------------------------------------------------------------------------------- rollout
def rollout(args):
    device = torch.device(args.device)
    tok, dyn = load_tokenizer(args.tokenizer, device), load_dynamics(args.dynamics, device)
    _, val_eps = load_episodes(args)
    val = val_windows(val_eps, args.seq_len, tok.image_size, args.val_windows, seed=2)
    ctx = max(1, args.seq_len // 4)
    video = val["video"].to(device)
    with torch.no_grad():
        z = tok.encode(video)
    metrics, full = evaluate_dynamics(tok, dyn, video, z, val["actions"].to(device), ctx, num_steps=args.num_steps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # uint8 copies: saving a slice of a float tensor would store the whole underlying batch
    as_u8 = lambda x: (x[:8] * 255).round().to(torch.uint8).cpu().clone()
    torch.save({"generated": as_u8(full), "ground_truth": as_u8(val["video"]), "context_frames": ctx}, out / "rollout.pt")
    (out / "rollout_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"rollout over {args.seq_len - ctx} frames, {args.val_windows} held-out windows, K={args.num_steps}: "
          + " ".join(f"{k}={v:.5g}" for k, v in metrics.items()), flush=True)
    print(f"saved {out / 'rollout.pt'}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default="synthetic", help="synthetic, or comma-separated PushT datasets (.zarr human demos, .npz rendered tapes)")
    common.add_argument("--shift", type=int, default=0, help="training augmentation: random clip-consistent translation of up to N pixels (PushT actions are shifted too)")
    common.add_argument("--background", default=None, choices=[None, "texture"], help="PushT only: replace the white background with a static texture")
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
    common.add_argument("--wandb", action="store_true")
    common.add_argument("--wandb-project", default="mini_dreamer4")
    common.add_argument("--run-name", default=None)
    common.add_argument("--video-every", type=int, default=2000)
    common.add_argument("--val-windows", type=int, default=64, help="number of distinct held-out windows used for evaluation")

    t = sub.add_parser("tokenizer", parents=[common])
    t.add_argument("--patch-size", type=int, default=8)
    t.add_argument("--num-latents", type=int, default=16)
    t.add_argument("--latent-dim", type=int, default=32)
    t.add_argument("--lpips-weight", type=float, default=0.0)
    t.add_argument("--mask-max", type=float, default=0.9, help="upper end of the per-frame patch masking probability")
    t.add_argument("--no-center-patches", action="store_true", help="paper behaviour: do not subtract the per-frame patch mean")

    d = sub.add_parser("dynamics", parents=[common])
    d.add_argument("--tokenizer", required=True)
    d.add_argument("--pack", type=int, default=2)
    d.add_argument("--registers", type=int, default=4)
    d.add_argument("--k-max", type=int, default=64)
    d.add_argument("--bootstrap-warmup", type=int, default=2000)
    d.add_argument("--clean-context-prob", type=float, default=0.0,
                   help="fraction of sequences whose first frames are near-clean context, as at inference (0 = paper)")
    d.add_argument("--no-ramp", action="store_true", help="weight all signal levels equally instead of 0.9 sigma + 0.1")
    d.add_argument("--cosine", action="store_true", help="cosine learning-rate decay with 1000 warmup steps (default: constant)")

    a = sub.add_parser("agent", parents=[common])
    a.add_argument("--tokenizer", required=True)
    a.add_argument("--dynamics", required=True, help="trained dynamics checkpoint to start from, or 'none' for a randomly initialised control")
    a.add_argument("--dyn-weight", type=float, default=1.0, help="0 disables the dynamics loss (control)")
    a.add_argument("--pack", type=int, default=2); a.add_argument("--registers", type=int, default=4); a.add_argument("--k-max", type=int, default=64)
    a.add_argument("--clean-context-prob", type=float, default=0.0); a.add_argument("--no-ramp", action="store_true")
    a.add_argument("--action-bins", type=int, default=128)
    a.add_argument("--bc-weight", type=float, default=1.0)
    a.add_argument("--reward-weight", type=float, default=0.3)
    a.add_argument("--success-threshold", type=float, default=0.95, help="a frame is a success when its coverage exceeds this")

    r = sub.add_parser("rollout", parents=[common])
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--dynamics", required=True)
    r.add_argument("--num-steps", type=int, default=4)

    args = p.parse_args()
    {"tokenizer": train_tokenizer, "dynamics": train_dynamics, "agent": train_agent, "rollout": rollout}[args.cmd](args)


if __name__ == "__main__":
    main()
