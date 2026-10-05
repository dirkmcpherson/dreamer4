"""Closed-loop evaluation of a mini_dreamer4 agent in the real gym-pusht simulator, from pixels.

Each step the last ``--window`` frames are encoded with the tokenizer, passed (clean) through the dynamics
transformer together with the actions taken so far, and the policy head on the agent token picks the next action.
Protocol of gp_wmfi_gym: the manifest's fixed start states, success when geometric coverage > 0.95, 300-step limit.

    SDL_VIDEODRIVER=dummy python -m mini_dreamer4.tools.eval_policy_pusht <tokenizer.pt> <agent.pt> --starts ic10.json
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from mini_dreamer4.train import load_tokenizer, load_dynamics
from mini_dreamer4.tools.plan_pusht import make_env, to_env, to_norm


def run(env, tok, agent, s0, args, dev, sample, rng_seed):
    u = env.unwrapped
    env.reset(seed=0)
    u.agent.position = tuple(map(float, s0[:2])); u.block.angle = float(s0[4]); u.block.position = tuple(map(float, s0[2:4]))
    u.agent.velocity = (0, 0); u.block.velocity = (0, 0); u.block.angular_velocity = 0
    frames = [torch.as_tensor(u._render()).float().div(255).movedim(-1, -3)]
    actions, cov = [], [float(u._get_coverage())]
    torch.manual_seed(rng_seed)
    for t in range(args.steps):
        w = frames[-args.window:]
        a_hist = actions[-(len(w) - 1):] if len(w) > 1 else []
        a_in = torch.as_tensor(np.array(a_hist + [np.zeros(2, np.float32)]), device=dev).float()[None]
        with torch.no_grad():
            z = tok.encode(torch.stack(w).to(dev)[None])
            a = agent.act(z, a_in, sample=sample, temperature=args.temperature)[0].cpu().numpy()
        obs, r, term, trunc, info = env.step(to_env(a))
        actions.append(a.astype(np.float32)); frames.append(torch.as_tensor(obs["pixels"]).float().div(255).movedim(-1, -3))
        cov.append(float(info["coverage"]))
        if info["coverage"] > args.success_threshold:
            return dict(success=True, steps=t + 1, max_coverage=max(cov), final_coverage=cov[-1]), frames
    return dict(success=False, steps=args.steps, max_coverage=max(cov), final_coverage=cov[-1]), frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer"); p.add_argument("agent")
    p.add_argument("--starts", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--sampled", type=int, default=5, help="additional episodes per start with sampled actions")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--success-threshold", type=float, default=0.95)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, agent = load_tokenizer(args.tokenizer, dev), load_dynamics(args.agent, dev)
    ics = json.load(open(args.starts))["initial_conditions"]
    env = make_env()
    rows, t0 = [], time.time()
    for i, ic in enumerate(ics):
        s0 = np.array(ic["raw_state_after_reset"])
        greedy, frames = run(env, tok, agent, s0, args, dev, sample=False, rng_seed=0)
        sampled = [run(env, tok, agent, s0, args, dev, sample=True, rng_seed=1000 + k)[0] for k in range(args.sampled)]
        rows.append(dict(seed=ic["reset_seed"], greedy=greedy, sampled=sampled))
        print("start %d: greedy %s in %3d steps (max coverage %.3f) | sampled %d/%d succeed, steps %s"
              % (ic["reset_seed"], "SUCCESS" if greedy["success"] else "fail   ", greedy["steps"], greedy["max_coverage"],
                 sum(s["success"] for s in sampled), len(sampled), [s["steps"] for s in sampled]), flush=True)
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            torch.save((torch.stack(frames) * 255).round().byte(), Path(args.out) / f"greedy_{ic['reset_seed']}.pt")
    g = [r["greedy"]["success"] for r in rows]; s = [x["success"] for r in rows for x in r["sampled"]]
    summary = dict(starts=len(rows), greedy_success_rate=float(np.mean(g)), sampled_success_rate=float(np.mean(s)) if s else None,
                   greedy_mean_steps_when_successful=float(np.mean([r["greedy"]["steps"] for r in rows if r["greedy"]["success"]])) if any(g) else None,
                   mean_max_coverage_greedy=float(np.mean([r["greedy"]["max_coverage"] for r in rows])), seconds=round(time.time() - t0))
    print(json.dumps(summary, indent=2))
    if args.out:
        (Path(args.out) / "summary.json").write_text(json.dumps(dict(summary=summary, starts=rows), indent=1))


if __name__ == "__main__":
    main()
