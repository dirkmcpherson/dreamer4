"""Imagination training (Dreamer 4, phase 3): improve the policy on rollouts imagined by the frozen world model.

Rollouts start from real frames (a random real prefix of 1-4 frames from the replay data, or the fixed start
states), continue for ``--horizon`` steps with actions sampled from the current policy and next latents sampled
from the dynamics model (K=4), and are rewarded by the success head. The world model and agent features are
frozen; only the policy heads and the value head are trained, with lambda-returns, a value regression loss, the
paper's PMPO policy update and a KL penalty towards the behaviour-cloned policy (the prior).

    SDL_VIDEODRIVER=dummy python -m mini_dreamer4.tools.imagine_train <tokenizer.pt> <agent.pt> --data tapes.npz,onpolicy.npz \
        --starts ic10.json --iters 1000 --out runs/imag
"""
import argparse, copy, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mini_dreamer4.data import EpisodeWindowDataset, collate, load_pusht
from mini_dreamer4.train import load_tokenizer, load_dynamics, save

W = 10


@torch.no_grad()
def imagine(tok, agent, z_ctx, a_ctx, horizon, temperature):
    """z_ctx (B, L, N_l, d) real latents, a_ctx (B, L-1, 2) actions taken at the first L-1 context frames.
    Returns features h (B, H, dim) of the agent token at the frames where actions are chosen, the chosen
    action bins (B, H, 2), success probabilities p (B, H) of the frames reached, and the first-frame baseline."""
    b, dev = z_ctx.shape[0], z_ctx.device
    z, acts = z_ctx, a_ctx
    feats, bins, probs = [], [], []
    for t in range(horizon):
        zw = z[:, -W:]
        aw = torch.cat((acts[:, acts.shape[1] - (zw.shape[1] - 1):] if zw.shape[1] > 1 else acts[:, :0], torch.zeros(b, 1, 2, device=dev)), dim=1)
        h = agent.agent_features(zw, aw)[:, -1]
        lx = agent.pi_first(h); ax = torch.distributions.Categorical(logits=lx / temperature).sample()
        ly = agent.pi_second(h + agent.bin_embed(ax)); ay = torch.distributions.Categorical(logits=ly / temperature).sample()
        a_bins = torch.stack((ax, ay), -1); a = agent.from_bins(a_bins)
        acts = torch.cat((acts, a[:, None]), dim=1)
        ctx = z[:, -(W - 1):]
        a_in = torch.cat((acts[:, acts.shape[1] - ctx.shape[1]:], torch.zeros(b, 1, 2, device=dev)), dim=1)
        nxt = agent.sample(ctx, a_in, horizon=1, num_steps=4)[:, -1:]
        z = torch.cat((z, nxt), dim=1)
        zw = z[:, -W:]
        aw = torch.cat((acts[:, acts.shape[1] - (zw.shape[1] - 1):], torch.zeros(b, 1, 2, device=dev)), dim=1)
        h_next = agent.agent_features(zw, aw)[:, -1]
        probs.append(torch.sigmoid(agent.reward_head(h_next).squeeze(-1)))
        feats.append(h); bins.append(a_bins)
    return torch.stack(feats, 1), torch.stack(bins, 1), torch.stack(probs, 1), h_next


def lambda_returns(reward, cont, value_next, gamma, lam):
    """reward, cont, value_next (B, H): reward/continuation of the frame reached by step t and the value of that frame."""
    H = reward.shape[1]
    R = torch.zeros_like(reward)
    nxt = value_next[:, -1]
    for t in reversed(range(H)):
        nxt = reward[:, t] + gamma * cont[:, t] * ((1 - lam) * value_next[:, t] + lam * nxt)
        R[:, t] = nxt
    return R


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer"); p.add_argument("agent")
    p.add_argument("--data", required=True, help="replay data for start contexts (comma-separated npz)")
    p.add_argument("--starts", default=None, help="manifest json: a share of rollouts start from these first frames")
    p.add_argument("--start-share", type=float, default=0.25)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--gamma", type=float, default=0.97); p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--alpha", type=float, default=0.5, help="PMPO balance between positive and negative advantages")
    p.add_argument("--kl", type=float, default=1.0, help="weight of KL(pi || pi_bc)")
    p.add_argument("--entropy", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", required=True)
    p.add_argument("--wandb", action="store_true"); p.add_argument("--run-name", default=None)
    args = p.parse_args()
    dev = "cuda"
    tok, agent = load_tokenizer(args.tokenizer, dev), load_dynamics(args.agent, dev)
    agent.eval()                                                  # frozen trunk; dropout-free anyway
    prior = copy.deepcopy(agent).eval()
    for q in prior.parameters():
        q.requires_grad_(False)
    heads = list(agent.policy_heads()) + [agent.value_head]
    params = [q for m in heads for q in m.parameters()]
    for q in agent.parameters():
        q.requires_grad_(False)
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=args.lr)

    episodes = load_pusht(args.data)
    ds = EpisodeWindowDataset(episodes, 4, 96, samples_per_epoch=10 ** 7, seed=0)   # 4-frame real prefixes
    start_frames = None
    if args.starts:
        from mini_dreamer4.tools.plan_pusht import make_env, set_state_exact
        env = make_env(); env.reset(seed=0)
        fr = []
        for ic in json.load(open(args.starts))["initial_conditions"]:
            set_state_exact(env, np.array(ic["raw_state_after_reset"]))
            fr.append(torch.as_tensor(env.unwrapped._render()).float().div(255).movedim(-1, -3))
        with torch.no_grad():
            start_frames = tok.encode(torch.stack(fr).to(dev)[:, None])          # (10, 1, N_l, d)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="mini_dreamer4", name=args.run_name, dir=args.out, config=vars(args))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        # ---- start contexts: real prefixes of random length 1-4 (and the fixed start frames)
        n_start = int(args.batch * args.start_share) if start_frames is not None else 0
        batch = collate([ds[i] for i in range(args.batch - n_start)])
        with torch.no_grad():
            z_real = tok.encode(batch["video"].to(dev))
        L = int(np.random.randint(1, 5))
        z_ctx, a_ctx = z_real[:, :L], batch["actions"].to(dev)[:, : L - 1]
        if n_start:
            idx = torch.randint(0, start_frames.shape[0], (n_start,))
            zs = start_frames[idx]
            if L > 1:                                            # pad start frames to the prefix length by repeating; no actions taken
                zs = zs.expand(-1, L, -1, -1)
            z_ctx = torch.cat((z_ctx, zs)); a_ctx = torch.cat((a_ctx, torch.zeros(n_start, L - 1, 2, device=dev)))
        # ---- imagine with the current policy (no gradients)
        h, bins, p_succ, h_last = imagine(tok, agent, z_ctx, a_ctx, args.horizon, args.temperature)
        success = (p_succ > 0.5).float()
        alive = torch.cumprod(torch.cat((torch.ones_like(success[:, :1]), 1 - success[:, :-1]), 1), 1)   # 1 until the step after success
        reward = p_succ * alive
        cont = (1 - success) * alive
        with torch.no_grad():
            v_next = torch.cat((agent.value_head(h[:, 1:]).squeeze(-1), agent.value_head(h_last).squeeze(-1)[:, None]), 1)
        R = lambda_returns(reward, cont, v_next, args.gamma, args.lam)
        # ---- losses on the stored (frozen) features
        v = agent.value_head(h).squeeze(-1)
        value_loss = ((v - R.detach()) ** 2 * alive).sum() / alive.sum()
        adv = (R - v).detach()
        logp = agent.policy_log_prob(h, bins)
        pos, neg = ((adv > 0) & (alive > 0)).float(), ((adv < 0) & (alive > 0)).float()
        pmpo = -(1 - args.alpha) * (logp * pos).sum() / pos.sum().clamp_min(1) + args.alpha * (logp * neg).sum() / neg.sum().clamp_min(1)
        with torch.no_grad():
            lx0, ly0 = prior.policy_logits(h, bins[..., 0])
        lx, ly = agent.policy_logits(h, bins[..., 0])
        kl = (F.kl_div(lx0.log_softmax(-1), lx.log_softmax(-1), log_target=True, reduction="none").sum(-1)
              + F.kl_div(ly0.log_softmax(-1), ly.log_softmax(-1), log_target=True, reduction="none").sum(-1))   # KL(pi || prior)
        kl = (kl * alive).sum() / alive.sum()
        ent = (agent.policy_entropy(h, bins) * alive).sum() / alive.sum()
        loss = pmpo + value_loss + args.kl * kl - args.entropy * ent
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if it % args.log_every == 0 or it == args.iters:
            m = dict(imagined_success=success.max(1).values.mean().item(), mean_return=R[:, 0].mean().item(), value_loss=value_loss.item(),
                     pmpo=pmpo.item(), kl_to_prior=kl.item(), entropy=ent.item(), frac_pos_adv=(pos.sum() / alive.sum()).item())
            print(f"iter {it} " + " ".join(f"{k}={v:.4g}" for k, v in m.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
            if run is not None:
                run.log(m, step=it)
            save(agent, {**torch.load(args.agent, map_location="cpu")["config"]}, out / "agent.pt")
    print(f"saved {out / 'agent.pt'}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
