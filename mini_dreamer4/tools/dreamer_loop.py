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
    """Real-simulator evaluation. Besides success (final coverage > 0.95) this tracks how close the policy gets:
    the mean peak coverage of the sampled episodes and the share of failures that peaked within 0.90-0.95."""
    greedy, sampled, peaks = [], [], []
    for i, s0 in enumerate(starts):
        greedy.append(run_episode(env, tok, agent, s0, dev, steps, False, 1.0, 1)["coverage"][-1] > 0.95)
        for k in range(n_sampled):
            cov = run_episode(env, tok, agent, s0, dev, steps, True, 1.0, 5000 + i * 100 + k)["coverage"]
            sampled.append(cov[-1] > 0.95); peaks.append(float(cov.max()))
    peaks = np.array(peaks) if peaks else np.array([float("nan")])
    fails = peaks[~np.array(sampled, dtype=bool)] if sampled else peaks
    return dict(real_greedy=float(np.mean(greedy)), real_sampled=float(np.mean(sampled)) if sampled else float("nan"),
                real_peak_coverage=float(peaks.mean()), real_near_miss=float(((fails >= 0.90) & (fails <= 0.95)).mean()) if len(fails) else 0.0,
                real_fail_peak_coverage=float(fails.mean()) if len(fails) else float("nan"))


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
    p.add_argument("--frozen-actor-trunk", action="store_true",
                   help="policy/value heads read features from a frozen copy of the initial trunk; the live world model only supplies imagined transitions and the success reward")
    p.add_argument("--adv-min", type=float, default=0.0, help="PMPO ignores steps whose |advantage| is below this (noise gate)")
    p.add_argument("--anchor-best", action="store_true", help="move the KL anchor to the policy with the best real sampled success so far instead of the initial prior")
    p.add_argument("--regime", choices=["d4", "v3"], default="d4",
                   help="d4: Dreamer 4 imagination phase (PMPO, KL to the BC prior, actor every --imag-every steps). "
                        "v3: DreamerV3 regime (live trunk, actor+critic every step from several prefixes of every batch, "
                        "REINFORCE with percentile-normalised returns + entropy, EMA critic target, no KL, no cloning)")
    p.add_argument("--imag-prefixes", type=int, default=2, help="v3: number of distinct prefix lengths per update; each gives batch_size imagined rollouts")
    p.add_argument("--imag-steps", type=int, default=4, help="shortcut sampling steps per imagined frame (1 is cheapest)")
    p.add_argument("--critic-ema", type=float, default=0.02, help="v3: EMA rate of the critic target")
    p.add_argument("--cov-reward", type=float, default=0.0,
                   help="imagination only: weight of a potential-based shaping reward on the coverage head's prediction between --cov-floor and 0.95 (the real task stays sparse)")
    p.add_argument("--cov-floor", type=float, default=0.9)
    p.add_argument("--reset-heads", action="store_true", help="re-initialise the policy and value heads (no behaviour cloning at all; demos only in replay)")
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
    if args.reset_heads:
        for m in list(agent.policy_heads()) + [agent.value_head]:
            for sub in m.modules():
                if hasattr(sub, "reset_parameters"):
                    sub.reset_parameters()
    if args.regime == "v3":                                  # DreamerV3: actor reads the live trunk and trains every step
        args.frozen_actor_trunk, args.imag_every = False, 1
    agent.train()
    prior = copy.deepcopy(agent).eval()
    for q in prior.parameters():
        q.requires_grad_(False)
    # The actor is a copy whose trunk stays frozen (as in Dreamer 4's imagination phase): only its policy and
    # value heads train. Without a cloning term, letting the policy read the live, continuously trained trunk
    # destroyed the policy through feature drift alone (the heads themselves barely moved).
    actor = copy.deepcopy(agent).eval() if args.frozen_actor_trunk else agent
    head_params = lambda m: [q for mod in list(m.policy_heads()) + [m.value_head] for q in mod.parameters()]
    if args.frozen_actor_trunk:
        for q in actor.parameters():
            q.requires_grad_(False)
        for q in head_params(actor):
            q.requires_grad_(True)
    actor_params = head_params(actor)
    head_ids = {id(q) for q in head_params(agent)}
    model_params = [q for q in agent.parameters() if id(q) not in head_ids]
    opt_model = torch.optim.AdamW(model_params, lr=args.lr, weight_decay=0.01)
    opt_actor = torch.optim.Adam(actor_params, lr=args.actor_lr)
    critic_target = copy.deepcopy(actor.value_head).eval()      # v3: slow critic for bootstrapping
    for q in critic_target.parameters():
        q.requires_grad_(False)
    ret_scale = None                                             # v3: EMA of the 5-95 percentile range of returns
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

    ev = evaluate(env, tok, actor, starts, dev, args.eval_sampled, args.episode_steps)
    log(dict(round=0, **ev), step_global)
    best_smp, best_round = ev["real_sampled"], 0
    for rnd in range(1, args.rounds + 1):
        # ---- 1. collect with the current policy
        actor.eval()
        new = collect(env, tok, actor, starts, dev, args.collect_per_start, args.episode_steps, args.collect_temperature, args.seed * 10 + rnd)
        agent.train()
        replay += new
        succ = float(np.mean([e["coverage"][-1] > 0.95 for e in new]))
        collect_peak = float(np.mean([e["coverage"].max() for e in new]))
        np.savez_compressed(out / f"onpolicy_round{rnd}.npz", img=np.concatenate([e["video"] for e in new]), action=np.concatenate([e["actions"] for e in new]),
                            state=np.concatenate([e["states"] for e in new]), coverage=np.concatenate([e["coverage"] for e in new]),
                            episode_ends=np.cumsum([len(e["video"]) for e in new]).astype(np.int64))
        log(dict(round=rnd, collected=len(new), collect_success=succ, collect_peak_coverage=collect_peak, replay_episodes=len(tapes) + len(replay)), step_global)

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
            bc, rew, astats = agent.agent_loss(z, actions, batch["action_mask"].to(dev) > 0.5, (batch["coverage"].to(dev) > 0.95).float(), batch["coverage"].to(dev))
            loss = dyn_loss + args.bc_weight * agent.bc_norm(bc) + args.reward_weight * agent.rew_norm(rew)
            opt_model.zero_grad(set_to_none=True); opt_actor.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params, 1.0); torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
            opt_model.step()
            if args.bc_weight > 0:
                opt_actor.step()                                    # the BC anchor also moves the policy heads (with weight 0 the heads' grads are zero and an Adam step would only replay momentum)
            imag = {}
            if step % args.imag_every == 0:
                agent.eval()                                        # imagined rollouts from real prefixes of this batch
                T = z.shape[1]
                if args.regime == "v3":                             # several prefix lengths -> many distinct start states
                    outs = [imagine(tok, agent, z[:, :L], actions[:, :L - 1], args.horizon, 1.0, actor=actor, num_steps=args.imag_steps)
                            for L in np.random.choice(np.arange(1, T + 1), size=min(args.imag_prefixes, T), replace=False)]
                    h, bins, p_succ, p_cov, h_last, c0 = (torch.cat(x, 0) for x in zip(*outs))
                else:
                    L = int(np.random.randint(1, 5))
                    reps = -(-args.imag_batch // z.shape[0])        # several imagined rollouts per real prefix when imag_batch > batch_size
                    z_ctx, a_ctx = z[:, :L].repeat(reps, 1, 1, 1)[:args.imag_batch], actions[:, :L - 1].repeat(reps, 1, 1)[:args.imag_batch]
                    h, bins, p_succ, p_cov, h_last, c0 = imagine(tok, agent, z_ctx, a_ctx, args.horizon, 1.0, actor=actor, num_steps=args.imag_steps)
                agent.train()
                success = (p_succ > 0.5).float()
                alive = torch.cumprod(torch.cat((torch.ones_like(success[:, :1]), 1 - success[:, :-1]), 1), 1)
                reward, cont = p_succ * alive, (1 - success) * alive
                shaping = torch.zeros(())
                if args.cov_reward > 0:                             # potential-based shaping on predicted coverage inside [cov_floor, 0.95]:
                    phi = lambda c: args.cov_reward * (c - args.cov_floor).clamp(0, 0.95 - args.cov_floor) / (0.95 - args.cov_floor)
                    shaping = (phi(p_cov) - phi(torch.cat((c0[:, None], p_cov[:, :-1]), 1))) * alive   # sums to cov_reward from below the floor to the threshold; hovering earns nothing
                    reward = reward + shaping
                vsrc = critic_target if args.regime == "v3" else actor.value_head
                with torch.no_grad():
                    v_next = torch.cat((vsrc(h[:, 1:]).squeeze(-1), vsrc(h_last).squeeze(-1)[:, None]), 1)
                R = lambda_returns(reward, cont, v_next, args.gamma, args.lam)
                v = actor.value_head(h).squeeze(-1)
                value_loss = ((v - R.detach()) ** 2 * alive).sum() / alive.sum()
                adv = (R - v).detach()
                logp = actor.policy_log_prob(h, bins)
                with torch.no_grad():
                    lx0, ly0 = prior.policy_logits(h, bins[..., 0])
                lx, ly = actor.policy_logits(h, bins[..., 0])
                kl = (F.kl_div(lx0.log_softmax(-1), lx.log_softmax(-1), log_target=True, reduction="none").sum(-1)
                      + F.kl_div(ly0.log_softmax(-1), ly.log_softmax(-1), log_target=True, reduction="none").sum(-1))
                kl = (kl * alive).sum() / alive.sum()
                ent = (actor.policy_entropy(h, bins) * alive).sum() / alive.sum()
                if args.regime == "v3":                             # REINFORCE with percentile-normalised returns, entropy bonus, no prior
                    live = R.detach()[alive > 0]
                    rng = (torch.quantile(live, 0.95) - torch.quantile(live, 0.05)).item() if live.numel() > 1 else 0.0
                    ret_scale = rng if ret_scale is None else 0.99 * ret_scale + 0.01 * rng
                    adv_n = adv / max(1.0, ret_scale)
                    pg = -(logp * adv_n * alive).sum() / alive.sum()
                    actor_loss = pg + value_loss - args.entropy * ent
                else:                                               # Dreamer 4: PMPO + KL to the cloned prior
                    pos, neg = ((adv > args.adv_min) & (alive > 0)).float(), ((adv < -args.adv_min) & (alive > 0)).float()
                    pg = -(1 - args.alpha) * (logp * pos).sum() / pos.sum().clamp_min(1) + args.alpha * (logp * neg).sum() / neg.sum().clamp_min(1)
                    actor_loss = pg + value_loss + args.kl * kl - args.entropy * ent
                opt_actor.zero_grad(set_to_none=True); actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, 1.0); opt_actor.step()
                if args.regime == "v3":
                    with torch.no_grad():
                        for qt, q in zip(critic_target.parameters(), actor.value_head.parameters()):
                            qt.lerp_(q, args.critic_ema)
                peak = p_cov.max(1).values                          # how close imagined rollouts get, by the diagnostic coverage head
                imag = dict(imagined_success=success.max(1).values.mean().item(), imagined_peak_coverage=peak.mean().item(),
                            imagined_near_miss=((peak >= 0.90) & (peak <= 0.95)).float().mean().item(), imagined_return=R.mean().item(),
                            value_loss=value_loss.item(), kl_to_prior=kl.item(), entropy=ent.item(), policy_loss=pg.item(),
                            shaping_reward=shaping.sum(-1).mean().item() if shaping.dim() else 0.0,
                            **({"return_scale": ret_scale} if ret_scale is not None else {}))
            step_global += 1
            if step % args.log_every == 0:
                agent.eval()
                with torch.no_grad():
                    _, _, vstats = actor.agent_loss(val_z, val["actions"].to(dev), val["action_mask"].to(dev) > 0.5, (val["coverage"].to(dev) > 0.95).float())
                    _, _, wstats = agent.agent_loss(val_z, val["actions"].to(dev), val["action_mask"].to(dev) > 0.5, (val["coverage"].to(dev) > 0.95).float(), val["coverage"].to(dev))
                metrics, _ = evaluate_dynamics(tok, agent, val["video"].to(dev), val_z, val["actions"].to(dev), 2)
                agent.train()
                log(dict(round=rnd, step=step, flow_mse=dstats["flow_mse"].item(), tape_bc_l1_px=vstats["bc_l1"].item() * 256, coverage_mae=wstats["coverage_mae"].item(),
                         rollout_gain_db=metrics["rollout_psnr_gain_over_floor"], rollout_latent_mse=metrics["rollout_latent_mse"], **imag), step_global)

        # ---- 3. evaluate in the real simulator and checkpoint
        actor.eval()
        ev = evaluate(env, tok, actor, starts, dev, args.eval_sampled, args.episode_steps)
        smp = ev["real_sampled"]
        agent.train()
        cfg = torch.load(args.agent, map_location="cpu")["config"]
        save(actor, cfg, out / f"agent_round{rnd}.pt"); save(actor, cfg, out / "agent.pt")      # the acting policy
        if actor is not agent:
            save(agent, cfg, out / "world_model.pt")
        if smp > best_smp:                                                                         # best by real evaluation
            best_smp, best_round = smp, rnd
            save(actor, cfg, out / "best.pt")
            if args.anchor_best:
                prior = copy.deepcopy(actor).eval()
                for q in prior.parameters():
                    q.requires_grad_(False)
        log(dict(round=rnd, **ev, best_sampled=best_smp, best_round=best_round), step_global)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
