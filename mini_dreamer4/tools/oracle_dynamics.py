"""Train the dynamics model on ORACLE latents (a fixed smooth projection of the true simulator state).

This removes the learned tokenizer from the picture: if the dynamics model cannot predict the block
even here, the problem is in the dynamics recipe, not in the tokenizer's latents.

    python -m mini_dreamer4.tools.oracle_dynamics --out runs/oracle/dyn_deep --depth 8 --time-every 2
"""
import argparse, math, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_dreamer4.data import EpisodeWindowDataset, collate
from mini_dreamer4.dynamics import ShortcutDynamics
from mini_dreamer4.envs import generate_episodes, state_to_oracle_latents
from mini_dreamer4.train import save, val_windows

NL, DL = 16, 32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--time-every", type=int, default=2)
    p.add_argument("--k-max", type=int, default=64)
    p.add_argument("--bootstrap-warmup", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--image-size", type=int, default=16, help="frames are rendered but unused; keep them small")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/oracle/dyn")
    p.add_argument("--cosine", action="store_true", help="cosine learning-rate decay with 1000 warmup steps (default: constant)")
    p.add_argument("--no-ramp", action="store_true", help="weight all signal levels equally")
    p.add_argument("--clean-context-prob", type=float, default=0.0)
    p.add_argument("--regress-only", action="store_true", help="diagnostic: plain next-frame regression with the same network")
    p.add_argument("--eval-steps", type=int, default=4, help="sampling steps used for the logged rollouts")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    dev = torch.device(args.device)

    train = generate_episodes(args.episodes, 48, image_size=args.image_size, seed=0)
    val = val_windows(generate_episodes(40, 48, image_size=args.image_size, seed=1), args.seq_len, args.image_size, 64)
    enc = lambda b: state_to_oracle_latents(b["states"], NL, DL).to(dev)
    val_z, val_a = enc(val), val["actions"].to(dev)
    ds = EpisodeWindowDataset(train, args.seq_len, args.image_size, samples_per_epoch=args.steps * args.batch_size, seed=0)
    dl = iter(DataLoader(ds, batch_size=args.batch_size, collate_fn=collate))

    cfg = dict(num_latents=NL, latent_dim=DL, pack=2, dim=args.dim, depth=args.depth, heads=4, dim_head=64, time_every=args.time_every,
               num_registers=4, k_max=args.k_max, action_dim=2, bootstrap_warmup=args.bootstrap_warmup,
               ramp_weight=not args.no_ramp, clean_context_prob=args.clean_context_prob, regress_only=args.regress_only)
    dyn = ShortcutDynamics(**cfg).to(dev)
    opt = torch.optim.AdamW(dyn.parameters(), lr=args.lr, weight_decay=0.01)
    lr_at = lambda s: min(1.0, (s + 1) / 1000) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))) * 0.98 + 0.02)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at if args.cosine else (lambda s: 1.0))
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="mini_dreamer4", name=args.run_name, dir=args.out, config={**vars(args), "model": cfg, "latents": "oracle"})
    ctx, t0 = args.seq_len // 4, time.time()
    print(f"oracle latents: copy_last={F.mse_loss(val_z[:, 1:], val_z[:, :-1]).item():.5f}", flush=True)
    for step in range(1, args.steps + 1):
        batch = next(dl)
        loss, stats = dyn(enc(batch), batch["actions"].to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps:
            state = torch.random.get_rng_state(), torch.cuda.get_rng_state() if dev.type == "cuda" else None
            dyn.eval()
            with torch.no_grad():
                torch.manual_seed(0); gen = dyn.sample(val_z[:, :ctx], val_a, horizon=args.seq_len - ctx, num_steps=args.eval_steps)
                torch.manual_seed(0); wrong = dyn.sample(val_z[:, :ctx], val_a.roll(1, dims=0), horizon=args.seq_len - ctx, num_steps=args.eval_steps)
            dyn.train()
            torch.random.set_rng_state(state[0])
            if state[1] is not None:
                torch.cuda.set_rng_state(state[1])
            roll = F.mse_loss(gen[:, ctx:], val_z[:, ctx:]).item()
            m = dict(flow_mse=stats["flow_mse"].item(), boot_mse=stats["boot_mse"].item(), rollout_latent_mse=roll,
                     copy_last_latent_mse=F.mse_loss(val_z[:, ctx - 1:ctx].expand_as(val_z[:, ctx:]), val_z[:, ctx:]).item(),
                     rollout_shuffle_ratio=F.mse_loss(wrong[:, ctx:], val_z[:, ctx:]).item() / max(roll, 1e-12))
            print(f"step {step} " + " ".join(f"{k}={v:.5g}" for k, v in m.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
            if run is not None:
                run.log(m, step=step)
            save(dyn, cfg, Path(args.out) / "dynamics.pt")
    print(f"saved {Path(args.out) / 'dynamics.pt'}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
