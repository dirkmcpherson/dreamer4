"""Extrinsic check of the dynamics model: does it evaluate the policy correctly?

The same stochastic policy is rolled out (a) in the real gym-pusht simulator and (b) entirely inside the world
model, starting from the same real first frame. Per start state we compare the success rate and the coverage
reached. In imagination, success is judged two ways: by the agent's success head and by measuring block / goal
overlap on the decoded frames (independent of any learned head).

    SDL_VIDEODRIVER=dummy python -m mini_dreamer4.tools.imagined_vs_real_policy <tokenizer.pt> <agent.pt> --starts ic10.json
"""
import argparse, json
import numpy as np
import torch

from mini_dreamer4.train import load_tokenizer, load_dynamics
from mini_dreamer4.tools.plan_pusht import make_env, set_state_exact, to_env, PixelReward


@torch.no_grad()
def imagine(tok, agent, z0, steps, window, temperature, reward):
    """z0 (B, 1, N_l, d) clean latents of the real first frame. Returns per-rollout dicts of tensors (B,)."""
    b, dev = z0.shape[0], z0.device
    z, acts = z0, torch.zeros(b, 0, 2, device=dev)
    done = torch.zeros(b, dtype=torch.bool, device=dev)
    head_success = torch.zeros(b, dtype=torch.bool, device=dev)
    pix_success = torch.zeros(b, dtype=torch.bool, device=dev)
    max_cov = torch.zeros(b, device=dev)
    first_head = torch.full((b,), steps + 1, device=dev)
    for t in range(steps):
        zw = z[:, -window:]
        aw = torch.cat((acts[:, acts.shape[1] - (zw.shape[1] - 1):], torch.zeros(b, 1, 2, device=dev)), dim=1)
        a = agent.act(zw, aw, sample=True, temperature=temperature, refine=True)
        acts = torch.cat((acts, a[:, None]), dim=1)
        ctx = z[:, -(window - 1):]
        a_in = torch.cat((acts[:, acts.shape[1] - ctx.shape[1]:], torch.zeros(b, 1, 2, device=dev)), dim=1)
        nxt = agent.sample(ctx, a_in, horizon=1, num_steps=4)[:, -1:]
        z = torch.cat((z, nxt), dim=1)
        # judge the new frame
        zw = z[:, -window:]
        aw = torch.cat((acts[:, acts.shape[1] - (zw.shape[1] - 1):], torch.zeros(b, 1, 2, device=dev)), dim=1)
        p = torch.sigmoid(agent.reward_head(agent.agent_features(zw, aw)[:, -1]).squeeze(-1))
        frame = tok.decode(zw)[:, -1].clamp(0, 1)
        cov = reward(frame)[0]
        max_cov = torch.where(done, max_cov, torch.maximum(max_cov, cov))
        hs, ps = (p > 0.5) & ~done, (cov > 0.95) & ~done
        first_head = torch.where(hs & (first_head > steps), torch.full_like(first_head, t + 1), first_head)
        head_success |= hs
        pix_success |= ps
        done |= hs                      # an episode ends when the agent's own success head says so, as it would in imagination training
    return dict(head_success=head_success, pix_success=pix_success, max_cov=max_cov, steps=first_head)


def real(env, tok, agent, s0, steps, window, temperature, dev, seed):
    set_state_exact(env, s0)
    u = env.unwrapped
    frames = [torch.as_tensor(u._render()).float().div(255).movedim(-1, -3)]
    acts, best = [], float(u._get_coverage())
    torch.manual_seed(seed)
    for t in range(steps):
        w = frames[-window:]
        hist = acts[-(len(w) - 1):] if len(w) > 1 else []
        a_in = torch.as_tensor(np.array(hist + [np.zeros(2, np.float32)]), device=dev).float()[None]
        with torch.no_grad():
            a = agent.act(tok.encode(torch.stack(w).to(dev)[None]), a_in, sample=True, temperature=temperature, refine=True)[0].cpu().numpy()
        obs, _, _, _, info = env.step(to_env(a))
        acts.append(a.astype(np.float32)); frames.append(torch.as_tensor(obs["pixels"]).float().div(255).movedim(-1, -3))
        best = max(best, float(info["coverage"]))
        if info["coverage"] > 0.95:
            return True, t + 1, best
    return False, steps, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer"); p.add_argument("agent")
    p.add_argument("--starts", required=True)
    p.add_argument("--n", type=int, default=20, help="rollouts per start, in each of the two worlds")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()
    dev = "cuda"
    tok, agent = load_tokenizer(args.tokenizer, dev), load_dynamics(args.agent, dev)
    ics = json.load(open(args.starts))["initial_conditions"]
    env = make_env(); env.reset(seed=0)
    reward = PixelReward(env, dev)
    rows = []
    print("start    | REAL: success  mean steps  mean max cov | IMAGINED: success (head)  success (pixels)  mean steps  mean max cov")
    for ic in ics:
        s0 = np.array(ic["raw_state_after_reset"])
        r = [real(env, tok, agent, s0, args.steps, args.window, args.temperature, dev, 1000 + k) for k in range(args.n)]
        set_state_exact(env, s0)
        f0 = torch.as_tensor(env.unwrapped._render()).float().div(255).movedim(-1, -3).to(dev)
        torch.manual_seed(7)
        with torch.no_grad():
            z0 = tok.encode(f0[None, None]).expand(args.n, -1, -1, -1).contiguous()
        im = imagine(tok, agent, z0, args.steps, args.window, args.temperature, reward)
        rs = np.mean([x[0] for x in r]); rsteps = np.mean([x[1] for x in r if x[0]]) if rs > 0 else float("nan")
        ih, ip = im["head_success"].float().mean().item(), im["pix_success"].float().mean().item()
        isteps = im["steps"][im["head_success"]].float().mean().item() if ih > 0 else float("nan")
        rows.append((rs, ih, ip, np.mean([x[2] for x in r]), im["max_cov"].mean().item()))
        print("%d |       %4.2f      %5.1f        %.3f   |            %4.2f              %4.2f           %5.1f        %.3f"
              % (ic["reset_seed"], rs, rsteps, rows[-1][3], ih, ip, isteps, rows[-1][4]), flush=True)
    rows = np.array(rows)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    print("\noverall success: real %.2f | imagined, success head %.2f | imagined, pixel coverage %.2f" % tuple(rows[:, :3].mean(0)))
    print("per-start correlation with real success: success head %.2f, pixel coverage %.2f | mean |imagined - real| success: head %.2f, pixels %.2f"
          % (corr(rows[:, 0], rows[:, 1]), corr(rows[:, 0], rows[:, 2]), np.abs(rows[:, 0] - rows[:, 1]).mean(), np.abs(rows[:, 0] - rows[:, 2]).mean()))
    print("mean max coverage: real %.3f | imagined %.3f; per-start correlation %.2f" % (rows[:, 3].mean(), rows[:, 4].mean(), corr(rows[:, 3], rows[:, 4])))


if __name__ == "__main__":
    main()
