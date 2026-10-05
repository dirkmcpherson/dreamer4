"""Collect on-policy rollouts in the real gym-pusht simulator (successes AND failures) for the online loop.

Output layout matches ``load_pusht_npz``. Failed episodes matter: a world model trained only on successful
demonstrations imagines success almost regardless of what the policy does.

    SDL_VIDEODRIVER=dummy python -m mini_dreamer4.tools.collect_policy_rollouts <tokenizer.pt> <agent.pt> \
        --starts ic10.json --per-start 30 --out onpolicy_r1.npz
"""
import argparse, json, time
import numpy as np
import torch

from mini_dreamer4.train import load_tokenizer, load_dynamics
from mini_dreamer4.tools.plan_pusht import make_env, set_state_exact, to_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer"); p.add_argument("agent")
    p.add_argument("--starts", required=True)
    p.add_argument("--per-start", type=int, default=30)
    p.add_argument("--steps", type=int, default=60, help="episodes are cut here if they have not succeeded")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    dev = "cuda"
    tok, agent = load_tokenizer(args.tokenizer, dev), load_dynamics(args.agent, dev)
    ics = json.load(open(args.starts))["initial_conditions"]
    env = make_env(); env.reset(seed=0); u = env.unwrapped
    imgs, acts, states, covs, ends, succ = [], [], [], [], [], []
    t0 = time.time()
    for ic in ics:
        s0, n_ok = np.array(ic["raw_state_after_reset"]), 0
        for k in range(args.per_start):
            set_state_exact(env, s0)
            torch.manual_seed(args.seed * 100000 + k)
            frames, a_list = [u._render()], []
            st, cv = [np.array([*u.agent.position, *u.block.position, u.block.angle])], [float(u._get_coverage())]
            ok = False
            for t in range(args.steps):
                w = [torch.as_tensor(f).float().div(255).movedim(-1, -3) for f in frames[-args.window:]]
                hist = a_list[-(len(w) - 1):] if len(w) > 1 else []
                a_in = torch.as_tensor(np.array(hist + [np.zeros(2, np.float32)]), device=dev).float()[None]
                with torch.no_grad():
                    a = agent.act(tok.encode(torch.stack(w).to(dev)[None]), a_in, sample=k > 0, temperature=args.temperature, refine=True)[0].cpu().numpy()
                obs, _, _, _, info = env.step(to_env(a))
                frames.append(obs["pixels"]); a_list.append(a.astype(np.float32))
                st.append(np.array([*u.agent.position, *u.block.position, u.block.angle])); cv.append(float(info["coverage"]))
                if info["coverage"] > 0.95:
                    ok = True
                    break
            n_ok += ok; succ.append(ok)
            imgs.append(np.stack(frames)); states.append(np.stack(st)); covs.append(np.array(cv, np.float32))
            acts.append(np.concatenate((np.stack(a_list), np.zeros((1, 2), np.float32))))
            ends.append(sum(len(x) for x in imgs))
        print("start %d: %d/%d succeeded (%.0fs)" % (ic["reset_seed"], n_ok, args.per_start, time.time() - t0), flush=True)
    np.savez_compressed(args.out, img=np.concatenate(imgs), action=np.concatenate(acts).astype(np.float32),
                        state=np.concatenate(states).astype(np.float32), coverage=np.concatenate(covs), episode_ends=np.array(ends, np.int64))
    print("saved %s: %d episodes (%d successful), %d frames" % (args.out, len(ends), sum(succ), ends[-1]))


if __name__ == "__main__":
    main()
