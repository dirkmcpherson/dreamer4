"""Model-predictive control inside the world model, evaluated in the real gym-pusht simulator.

At every environment step a CEM planner imagines N candidate action sequences with the dynamics
model, scores them with a reward computed from the DECODED frames (block/goal overlap plus a
block-to-goal distance shaping term), and executes the best first action in the real simulator.

Baselines: the same planner driven by the true simulator (oracle model, the planner's ceiling) and
a random-walk controller. Periodically the chosen open-loop plan is also executed in a cloned
simulator so imagined and real outcomes of planner-chosen (off-distribution) actions can be compared.

    SDL_VIDEODRIVER=dummy python -m mini_dreamer4.tools.plan_pusht <tokenizer.pt> <dynamics.pt> \
        --episodes 5 --steps 100 --mode model|oracle|random --out runs/plan
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mini_dreamer4.train import load_tokenizer, load_dynamics

BLOCK = torch.tensor([[143, 163, 184], [119, 136, 153]]).float() / 255
GOAL = torch.tensor([[144, 238, 144]]).float() / 255
CTX, K = 4, 4


def make_env(obs_type="pixels_agent_pos"):
    import gymnasium as gym, gym_pusht  # noqa: F401
    return gym.make("gym_pusht/PushT-v0", obs_type=obs_type, render_mode="rgb_array")


def sim_state(env) -> np.ndarray:
    u = env.unwrapped
    return np.array([*u.agent.position, *u.block.position, u.block.angle], dtype=np.float64)


def set_state_exact(env, s) -> None:
    """Put the simulator in recorded state ``s`` = (agent xy, block xy, block angle), gp_wmfi_gym convention:
    angle before position restores the recorded body origin exactly. The shapes are re-indexed because the
    renderer draws from cached shape geometry; without it the next frame still shows the previous pose."""
    u = env.unwrapped
    u.agent.position = tuple(map(float, s[:2]))
    u.block.angle = float(s[4])
    u.block.position = tuple(map(float, s[2:4]))
    u.agent.velocity = (0, 0)
    u.block.velocity = (0, 0)
    u.block.angular_velocity = 0
    u.space.reindex_shapes_for_body(u.agent)
    u.space.reindex_shapes_for_body(u.block)


def corrected_state(env) -> np.ndarray:
    """State for ``reset(options={"reset_to_state": s})`` that reproduces the live agent/block pose exactly
    (gym-pusht applies the pose about the block's centre of mass, so the raw pose lands elsewhere).
    Body velocities are not reproduced."""
    u = env.unwrapped
    target = np.array([*u.agent.position, *u.block.position, u.block.angle], dtype=np.float64)
    probe = make_env(); probe.reset(seed=0)
    s = target.copy()
    for _ in range(3):
        probe.reset(options={"reset_to_state": s})
        s[2:4] += target[2:4] - np.array(probe.unwrapped.block.position)
    probe.close()
    return s


def clone_env(env):
    c = make_env(); c.reset(seed=0); c.reset(options={"reset_to_state": corrected_state(env)})
    return c


def to_norm(pos):            # env action (pixels 0-512) -> dataset convention [-1, 1]
    return np.asarray(pos, dtype=np.float32) / 512 * 2 - 1


def to_env(a_norm):
    return np.clip((np.asarray(a_norm, dtype=np.float32) + 1) / 2 * 512, 0, 512)


class PixelReward:
    """Reward from frames: fraction of the goal covered by the block, minus a block-to-goal distance term.
    The goal mask is fixed (rendered once with the block moved away) because the block occludes it."""

    def __init__(self, env, dev, dist_weight=1.0):
        s = sim_state(env)
        far = s.copy(); far[2:4] = [60, 60]
        env.reset(options={"reset_to_state": far})
        frame = torch.as_tensor(env.unwrapped._render()).float().div(255).movedim(-1, -3).to(dev)
        self.goal = self.mask(frame[None], GOAL.to(dev), 0.2)[0]
        ys, xs = torch.nonzero(self.goal, as_tuple=True)
        self.goal_c = torch.stack((xs.float().mean() / 96, ys.float().mean() / 96))
        env.reset(options={"reset_to_state": s})
        self.dev, self.dist_weight = dev, dist_weight

    @staticmethod
    def mask(frames, ref, thr):
        return ((frames.unsqueeze(-4) - ref[:, :, None, None]).abs().sum(dim=-3)).min(dim=-3).values < thr

    def __call__(self, frames):       # (..., 3, H, W) -> coverage (...), block centroid (..., 2), reward (...)
        b = self.mask(frames, BLOCK.to(self.dev), 0.2)
        cover = (b & self.goal).flatten(-2).sum(-1).float() / self.goal.sum()
        h, w = b.shape[-2:]
        xs = (torch.arange(w, device=self.dev).float() + 0.5) / w
        ys = (torch.arange(h, device=self.dev).float() + 0.5) / h
        n = b.flatten(-2).sum(-1).clamp_min(1).float()
        c = torch.stack(((b * xs).flatten(-2).sum(-1) / n, (b * ys[:, None]).flatten(-2).sum(-1) / n), dim=-1)
        dist = (c - self.goal_c).norm(dim=-1)
        return cover, c, cover - self.dist_weight * dist


class ModelPlanner:
    def __init__(self, tok, dyn, reward, horizon=12, pop=128, elites=16, iters=3, step_std=0.06, dev="cuda"):
        self.tok, self.dyn, self.reward, self.dev = tok, dyn, reward, dev
        self.H, self.N, self.E, self.iters, self.step_std = horizon, pop, elites, iters, step_std
        self.mean = None

    def sample_actions(self, a0, mean):
        """N sequences of H target positions: a smoothed random walk starting at the agent's position."""
        noise = torch.randn(self.N, self.H, 2, device=self.dev) * self.step_std
        noise = torch.cumsum(noise, dim=1) * 0.5 + noise          # correlated steps
        return (mean[None] + noise).clamp(-1, 1)

    @torch.no_grad()
    def imagine(self, z_ctx, a_ctx, cand):
        """z_ctx (CTX, N_l, d), a_ctx (CTX-1, 2) actions already taken at the context frames; cand (N, H, 2)."""
        n = cand.shape[0]
        acts = torch.cat((a_ctx[None].expand(n, -1, -1), cand, torch.zeros(n, 1, 2, device=self.dev)), dim=1)
        gen = self.dyn.sample(z_ctx[None].expand(n, -1, -1, -1), acts, horizon=self.H, num_steps=K)
        frames = self.tok.decode(gen).clamp(0, 1)[:, CTX:]           # (N, H, 3, 96, 96)
        return frames

    def plan(self, frames_ctx, a_ctx, agent_norm):
        """frames_ctx (CTX, 3, 96, 96) real frames, a_ctx (CTX-1, 2) normalized actions taken at them."""
        with torch.no_grad():
            z_ctx = self.tok.encode(frames_ctx[None])[0]
        a0 = torch.as_tensor(agent_norm, device=self.dev)
        if self.mean is None:
            mean = a0[None].expand(self.H, -1).clone()
        else:
            mean = torch.cat((self.mean[1:], self.mean[-1:]), dim=0)
        std_scale = 1.0
        best = None
        for it in range(self.iters):
            cand = self.sample_actions(a0, mean) if it == 0 else (mean[None] + torch.randn(self.N, self.H, 2, device=self.dev) * std[None]).clamp(-1, 1)
            cand[0] = mean                                           # keep the shifted previous plan in the population
            frames = self.imagine(z_ctx, a_ctx, cand)
            cover, cent, r = self.reward(frames)                     # (N, H)
            score = r[:, -1] + 0.5 * r.mean(dim=1)
            top = score.topk(self.E).indices
            mean, std = cand[top].mean(0), cand[top].std(0).clamp_min(0.01)
            b = top[0].item()
            best = dict(actions=cand[b].cpu().numpy(), frames=frames[b].cpu(), cover=cover[b].cpu().numpy(),
                        centroid=cent[b].cpu().numpy(), score=score[b].item())
        self.mean = torch.as_tensor(best["actions"], device=self.dev)
        return best


class OraclePlanner:
    """Same CEM, but every candidate is rolled out in a cloned simulator (true dynamics, true coverage)."""

    def __init__(self, horizon=12, pop=64, elites=8, iters=2, step_std=0.06, dist_weight=1.0):
        self.H, self.N, self.E, self.iters, self.step_std, self.dist_weight = horizon, pop, elites, iters, step_std, dist_weight
        self.env = make_env("state")
        self.env.reset(seed=0)
        self.mean = None

    def rollout(self, state, seq):
        self.env.reset(options={"reset_to_state": state})
        r, cover = [], []
        goal = np.array([256.0, 256.0])
        for a in seq:
            _, rew, _, _, info = self.env.step(to_env(a))
            c = info["coverage"]; d = np.linalg.norm(np.asarray(info["block_pose"][:2]) - goal) / 512
            r.append(c - self.dist_weight * d); cover.append(c)
        return np.array(r), np.array(cover)

    def plan(self, state, agent_norm):
        a0 = np.asarray(agent_norm, dtype=np.float32)
        mean = np.repeat(a0[None], self.H, 0) if self.mean is None else np.concatenate((self.mean[1:], self.mean[-1:]))
        std = np.full((self.H, 2), self.step_std)
        for it in range(self.iters):
            if it == 0:
                noise = np.random.randn(self.N, self.H, 2) * self.step_std
                cand = np.clip(mean[None] + np.cumsum(noise, 1) * 0.5 + noise, -1, 1)
            else:
                cand = np.clip(mean[None] + np.random.randn(self.N, self.H, 2) * std[None], -1, 1)
            cand[0] = mean
            scores = []
            for seq in cand:
                r, _ = self.rollout(state, seq)
                scores.append(r[-1] + 0.5 * r.mean())
            top = np.argsort(scores)[::-1][:self.E]
            mean, std = cand[top].mean(0), np.maximum(cand[top].std(0), 0.01)
        self.mean = cand[top[0]]
        return dict(actions=cand[top[0]], score=float(scores[top[0]]))


def run_episode(args, ep, tok, dyn, reward, dev, out):
    env = make_env()
    obs, info = env.reset(seed=args.seed + ep)
    if args.starts:                                                # fixed start states from a gp_wmfi_gym manifest
        ics = json.load(open(args.starts))["initial_conditions"]
        s0 = np.array(ics[ep % len(ics)]["raw_state_after_reset"])
        set_state_exact(env, s0)
        obs = env.unwrapped.get_obs()
    frames = [torch.as_tensor(obs["pixels"]).float().div(255).movedim(-1, -3)]
    acts = []                                                      # normalized action taken at each frame
    planner = ModelPlanner(tok, dyn, reward, horizon=args.horizon, pop=args.pop, elites=args.elites, iters=args.iters, step_std=args.step_std, dev=dev) if args.mode == "model" else \
        OraclePlanner(horizon=args.horizon, pop=args.pop // 2, elites=args.elites // 2, iters=2, step_std=args.step_std) if args.mode == "oracle" else None
    log = dict(coverage=[], open_loop=[], one_step=[])
    rw = np.random.default_rng(args.seed + ep)
    agent = to_norm(obs["agent_pos"])
    wander = agent.copy()
    t0 = time.time()
    for t in range(args.steps):
        if args.mode == "random":
            wander = np.clip(wander + rw.normal(0, args.step_std, 2), -1, 1)
            a = wander
        elif args.mode == "oracle":
            a = planner.plan(corrected_state(env), agent)["actions"][0]
        else:
            ctx = torch.stack(frames[-CTX:]).to(dev)
            if len(ctx) < CTX:                                     # pad the first steps by repeating the first frame
                ctx = torch.cat((ctx[:1].expand(CTX - len(ctx), -1, -1, -1), ctx))
            a_ctx = torch.as_tensor(np.array(acts[-(CTX - 1):]) if len(acts) >= CTX - 1 else np.array([agent] * (CTX - 1)), device=dev).float()
            best = planner.plan(ctx, a_ctx, agent)
            a = best["actions"][0]
            if t % args.open_loop_every == 0:                      # execute the whole plan in a cloned simulator
                clone = clone_env(env)
                real_cover, real_cent, real_frames = [], [], []
                for pa in best["actions"]:
                    o2, _, _, _, i2 = clone.step(to_env(pa))
                    f2 = torch.as_tensor(o2["pixels"]).float().div(255).movedim(-1, -3)
                    c, cen, _ = reward(f2.to(dev)[None])
                    real_cover.append(c.item()); real_cent.append(cen[0].cpu().numpy()); real_frames.append(f2)
                clone.close()
                real_cent = np.array(real_cent)
                if t % (4 * args.open_loop_every) == 0:                 # imagined (top row) vs real (bottom row) plan frames
                    from PIL import Image
                    rows = torch.cat((torch.cat(list(best["frames"][::2]), dim=-1), torch.cat(list(torch.stack(real_frames)[::2]), dim=-1)), dim=-2)
                    Image.fromarray((rows.movedim(0, -1).numpy() * 255).round().astype(np.uint8)).resize((rows.shape[-1] * 2, rows.shape[-2] * 2), Image.NEAREST).save(out / f"{args.mode}_ep{ep}_t{t}_plan.png")
                log["open_loop"].append(dict(t=t, imagined_cover=best["cover"].tolist(), real_cover=real_cover,
                                             block_err=np.linalg.norm(best["centroid"] - real_cent, axis=-1).tolist(),
                                             imagined_disp=float(np.linalg.norm(best["centroid"][-1] - best["centroid"][0])),
                                             real_disp=float(np.linalg.norm(real_cent[-1] - real_cent[0]))))
            log["one_step"].append(float(best["cover"][0]))
        obs, r, term, trunc, info = env.step(to_env(a))
        acts.append(np.asarray(a, dtype=np.float32))
        frames.append(torch.as_tensor(obs["pixels"]).float().div(255).movedim(-1, -3))
        agent = to_norm(obs["agent_pos"])
        log["coverage"].append(float(info["coverage"]))
        if term:
            break
    res = dict(mode=args.mode, episode=ep, steps=t + 1, success=bool(info["is_success"]), final_coverage=log["coverage"][-1],
               max_coverage=max(log["coverage"]), seconds=round(time.time() - t0), coverage=log["coverage"], open_loop=log["open_loop"])
    if args.mode == "model":
        real_next = np.array(log["coverage"][: len(log["one_step"])])
        res["one_step_cover_mae"] = float(np.abs(np.array(log["one_step"]) - real_next).mean())
    print(f"[{args.mode}] episode {ep}: steps {res['steps']} success {res['success']} final coverage {res['final_coverage']:.3f} "
          f"max {res['max_coverage']:.3f} ({res['seconds']}s)", flush=True)
    for o in res["open_loop"][:3]:
        print(f"    open-loop plan at t={o['t']}: imagined coverage {o['imagined_cover'][-1]:.3f} vs real {o['real_cover'][-1]:.3f} after {args.horizon} steps; "
              f"block centroid err by step {' '.join(f'{e:.3f}' for e in o['block_err'][::3])}; block moved imagined {o['imagined_disp']:.3f} vs real {o['real_disp']:.3f}", flush=True)
    vid = (torch.stack(frames) * 255).round().byte()
    torch.save(dict(frames=vid, actions=np.array(acts), result=res), out / f"{args.mode}_ep{ep}.pt")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer"); p.add_argument("dynamics")
    p.add_argument("--mode", default="model", choices=["model", "oracle", "random"])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--elites", type=int, default=16)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--open-loop-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--starts", default=None, help="gp_wmfi_gym manifest json: episodes cycle over its initial_conditions")
    p.add_argument("--step-std", type=float, default=0.06, help="per-step std of sampled target moves in [-1, 1] units (0.06 ~ human demos, 0.3 ~ RLPD tapes)")
    p.add_argument("--out", default="runs/plan")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tok, dyn = load_tokenizer(args.tokenizer, dev), load_dynamics(args.dynamics, dev)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    env = make_env(); env.reset(seed=0)
    reward = PixelReward(env, dev)
    results = [run_episode(args, ep, tok, dyn, reward, dev, out) for ep in range(args.episodes)]
    summary = dict(mode=args.mode, episodes=len(results), success_rate=float(np.mean([r["success"] for r in results])),
                   successes=[int(r["success"]) for r in results], steps=[r["steps"] for r in results],
                   mean_final_coverage=float(np.mean([r["final_coverage"] for r in results])),
                   mean_max_coverage=float(np.mean([r["max_coverage"] for r in results])))
    if args.mode == "model":
        ol = [o for r in results for o in r["open_loop"]]
        summary.update(open_loop_plans=len(ol),
                       open_loop_cover_mae=float(np.mean([abs(o["imagined_cover"][-1] - o["real_cover"][-1]) for o in ol])),
                       open_loop_block_err_final=float(np.mean([o["block_err"][-1] for o in ol])),
                       open_loop_imagined_disp=float(np.mean([o["imagined_disp"] for o in ol])),
                       open_loop_real_disp=float(np.mean([o["real_disp"] for o in ol])),
                       one_step_cover_mae=float(np.mean([r["one_step_cover_mae"] for r in results])))
    print(json.dumps(summary, indent=2))
    (out / f"summary_{args.mode}.json").write_text(json.dumps(dict(summary=summary, episodes=results), indent=1))


if __name__ == "__main__":
    main()
