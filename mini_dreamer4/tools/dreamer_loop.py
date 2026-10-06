"""DreamerV3-style loop: world model and actor-critic trained together, interleaved with real collection.

One process alternates between
  1. collecting real rollouts with the current policy in gym-pusht (successes and failures) into the replay buffer,
  2. a training chunk in which EVERY step updates the world model (shortcut-forcing dynamics loss, success head,
     and a behaviour-cloning anchor on the demonstration tapes) on a replay batch, and every ``--imag-every``
     steps updates the policy and value heads on rollouts imagined from that batch's real states (lambda-returns,
     PMPO, KL to the frozen behaviour-cloned prior). Actor gradients stop at the features, as in DreamerV3.
  3. evaluating the policy in the real simulator on the fixed start states.

    python -m mini_dreamer4.tools.dreamer_loop <tokenizer.pt> <agent.pt> --tapes tapes.npz --starts ic10.json \
        --rounds 4 --model-steps 5000 --out runs/dreamer
"""
import argparse, copy, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_dreamer4.data import EpisodeWindowDataset, collate, load_pusht
from mini_dreamer4.train import load_tokenizer, load_dynamics, save, val_windows, evaluate_dynamics
from mini_dreamer4.tools.imagine_train import imagine, lambda_returns
from mini_dreamer4.tools.plan_pusht import make_env, set_state_exact, to_env

W = 10


# ------------------------------------------------------------------------------------------ real environment
def run_episode(env, tok, agent, s0, dev, steps, sample, temperature, seed):
    env.reset(seed=0)
    set_state_exact(env, s0)
    u = env.unwrapped
    frames, acts = [u._render()], []
    states, cov = [np.array([*u.agent.position, *u.block.position, u.block.angle])], [float(u._get_coverage())]
    torch.manual_seed(seed)
    for t in range(steps):
        w = [torch.as_tensor(f).float().div(255).movedim(-1, -3) for f in frames[-W:]]
        hist = acts[-(len(w) - 1):] if len(w) > 1 else []
        a_in = torch.as_tensor(np.array(hist + [np.zeros(2, np.float32)]), device=dev).float()[None]
        with torch.no_grad():
            a = agent.act(tok.encode(torch.stack(w).to(dev)[None]), a_in, sample=sample, temperature=temperature, refine=True)[0].cpu().numpy()
        obs, _, _, _, info = env.step(to_env(a))
        frames.append(obs["pixels"]); acts.append(a.astype(np.float32))
        states.append(np.array([*u.agent.position, *u.block.position, u.block.angle])); cov.append(float(info["coverage"]))
        if info["coverage"] > 0.95:
            break
    return dict(video=np.stack(frames), actions=np.concatenate((np.stack(acts), np.zeros((1, 2), np.float32))).astype(np.float32),
                states=np.stack(states).astype(np.float32), coverage=np.array(cov, np.float32),
                action_mask=np.zeros(len(frames), dtype=bool))      # on-policy data is never imitated


def collect(env, tok, agent, starts, dev, per_start, steps, temperature, seed):
    eps = []
    for i, s0 in enumerate(starts):
        for k in range(per_start):
            eps.append(run_episode(env, tok, agent, s0, dev, steps, sample=k > 0, temperature=temperature, seed=seed * 100000 + i * 1000 + k))
    return eps


def evaluate(env, tok, agent, starts, dev, n_sampled, steps):
    greedy, sampled = [], []
    for i, s0 in enumerate(starts):
        greedy.append(run_episode(env, tok, agent, s0, dev, steps, False, 1.0, 1)["coverage"][-1] > 0.95)
        sampled += [run_episode(env, tok, agent, s0, dev, steps, True, 1.0, 5000 + i * 100 + k)["coverage"][-1] > 0.95 for k in range(n_sampled)]
    return float(np.mean(greedy)), float(np.mean(sampled)) if sampled else float("nan")


# ------------------------------------------------------------------------------------------ main loop
def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer"); p.add_argument("agent")
    p.add_argument("--tapes", required=True, help="demonstration tapes (npz); the only data used for behaviour cloning")
    p.add_argument("--replay", default="", help="comma-separated earlier on-policy npz files to seed the buffer (never imitated)")
    p.add_argument("--starts", required=True)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--collect-per-start", type=int, default=20)
    p.add_argument("--episode-steps", type=int, default=60)
    p.add_argument("--collect-temperature", type=float, default=0.7)
    p.add_argument("--model-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=10)
    p.add_argument("--shift", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bc-weight", type=float, default=0.3)
    p.add_argument("--reward-weight", type=float, default=0.3)
    p.add_argument("--imag-every", type=int, default=4)
    p.add_argument("--imag-batch", type=int, default=32)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--gamma", type=float, default=0.97); p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--alpha", type=float, default=0.5); p.add_argument("--kl", type=float, default=0.1); p.add_argument("--entropy", type=float, default=1e-3)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--eval-sampled", type=int, default=10)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--wandb", action="store_true"); p.add_argument("--run-name", default=None)
    args = p.parse_args()
    dev = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tok, agent = load_tokenizer(args.tokenizer, dev), load_dynamics(args.agent, dev)
    agent.train()
    prior = copy.deepcopy(agent).eval()
    for q in prior.parameters():
        q.requires_grad_(False)
    actor_params = [q for m in list(agent.policy_heads()) + [agent.value_head] for q in m.parameters()]
    actor_ids = {id(q) for q in actor_params}
    model_params = [q for q in agent.parameters() if id(q) not in actor_ids]
    opt_model = torch.optim.AdamW(model_params, lr=args.lr, weight_decay=0.01)
    opt_actor = torch.optim.Adam(actor_params, lr=args.actor_lr)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="mini_dreamer4", name=args.run_name, dir=args.out, config=vars(args))

    tapes = load_pusht(args.tapes)
    replay = load_pusht(",".join(f"{f}:nobc" for f in args.replay.split(","))) if args.replay else []
    ics = json.load(open(args.starts))["initial_conditions"]
    starts = [np.array(ic["raw_state_after_reset"]) for ic in ics]
    env = make_env(); env.reset(seed=0)
    val = val_windows([e for i, e in enumerate(tapes) if not i % 20], args.seq_len, 96, 32)
    with torch.no_grad():
        val_z = tok.encode(val["video"].to(dev))
    step_global, t0 = 0, time.time()
    log = lambda d, s: (print(f"[{time.time() - t0:.0f}s] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()), flush=True),
                        run.log(d, step=s) if run is not None else None)

    g, smp = evaluate(env, tok, agent, starts, dev, args.eval_sampled, args.episode_steps)
    log(dict(round=0, real_greedy=g, real_sampled=smp), step_global)
    for rnd in range(1, args.rounds + 1):
        # ---- 1. collect with the current policy
        agent.eval()
        new = collect(env, tok, agent, starts, dev, args.collect_per_start, args.episode_steps, args.collect_temperature, args.seed * 10 + rnd)
        agent.train()
        replay += new
        succ = float(np.mean([e["coverage"][-1] > 0.95 for e in new]))
        np.savez_compressed(out / f"onpolicy_round{rnd}.npz", img=np.concatenate([e["video"] for e in new]), action=np.concatenate([e["actions"] for e in new]),
                            state=np.concatenate([e["states"] for e in new]), coverage=np.concatenate([e["coverage"] for e in new]),
                            episode_ends=np.cumsum([len(e["video"]) for e in new]).astype(np.int64))
        log(dict(round=rnd, collected=len(new), collect_success=succ, replay_episodes=len(tapes) + len(replay)), step_global)

        # ---- 2. train world model + actor-critic together
        ds = EpisodeWindowDataset(tapes + replay, args.seq_len, 96, samples_per_epoch=args.model_steps * args.batch_size, seed=args.seed * 100 + rnd,
                                  shift=args.shift, shift_actions=True)
        dl = iter(DataLoader(ds, batch_size=args.batch_size, collate_fn=collate))
        for step in range(1, args.model_steps + 1):
            batch = next(dl)
            actions = batch["actions"].to(dev)
            with torch.no_grad():
                z = tok.encode(batch["video"].to(dev))
            dyn_loss, dstats = agent(z, actions)
            bc, rew, astats = agent.agent_loss(z, actions, batch["action_mask"].to(dev) > 0.5, (batch["coverage"].to(dev) > 0.95).float())
            loss = dyn_loss + args.bc_weight * agent.bc_norm(bc) + args.reward_weight * agent.rew_norm(rew)
            opt_model.zero_grad(set_to_none=True); opt_actor.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params, 1.0); torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
            opt_model.step(); opt_actor.step()                      # the BC anchor also moves the policy heads
            imag = {}
            if step % args.imag_every == 0:
                agent.eval()                                        # imagined rollouts from the real prefixes of this batch
                L = int(np.random.randint(1, 5)); nb = min(args.imag_batch, z.shape[0])
                h, bins, p_succ, h_last = imagine(tok, agent, z[:nb, :L], actions[:nb, :L - 1], args.horizon, 1.0)
                agent.train()
                success = (p_succ > 0.5).float()
                alive = torch.cumprod(torch.cat((torch.ones_like(success[:, :1]), 1 - success[:, :-1]), 1), 1)
                reward, cont = p_succ * alive, (1 - success) * alive
                with torch.no_grad():
                    v_next = torch.cat((agent.value_head(h[:, 1:]).squeeze(-1), agent.value_head(h_last).squeeze(-1)[:, None]), 1)
                R = lambda_returns(reward, cont, v_next, args.gamma, args.lam)
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
                      + F.kl_div(ly0.log_softmax(-1), ly.log_softmax(-1), log_target=True, reduction="none").sum(-1))
                kl = (kl * alive).sum() / alive.sum()
                ent = (agent.policy_entropy(h, bins) * alive).sum() / alive.sum()
                actor_loss = pmpo + value_loss + args.kl * kl - args.entropy * ent
                opt_actor.zero_grad(set_to_none=True); actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, 1.0); opt_actor.step()
                imag = dict(imagined_success=success.max(1).values.mean().item(), value_loss=value_loss.item(), kl_to_prior=kl.item(), entropy=ent.item())
            step_global += 1
            if step % args.log_every == 0:
                agent.eval()
                with torch.no_grad():
                    _, _, vstats = agent.agent_loss(val_z, val["actions"].to(dev), val["action_mask"].to(dev) > 0.5, (val["coverage"].to(dev) > 0.95).float())
                metrics, _ = evaluate_dynamics(tok, agent, val["video"].to(dev), val_z, val["actions"].to(dev), 2)
                agent.train()
                log(dict(round=rnd, step=step, flow_mse=dstats["flow_mse"].item(), tape_bc_l1_px=vstats["bc_l1"].item() * 256,
                         rollout_gain_db=metrics["rollout_psnr_gain_over_floor"], rollout_latent_mse=metrics["rollout_latent_mse"], **imag), step_global)

        # ---- 3. evaluate in the real simulator and checkpoint
        agent.eval()
        g, smp = evaluate(env, tok, agent, starts, dev, args.eval_sampled, args.episode_steps)
        agent.train()
        cfg = torch.load(args.agent, map_location="cpu")["config"]
        save(agent, cfg, out / f"agent_round{rnd}.pt"); save(agent, cfg, out / "agent.pt")
        log(dict(round=rnd, real_greedy=g, real_sampled=smp), step_global)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
