"""Real vs imagined, paired on the SAME executed actions, open-loop from the real first frame.

For each chosen start: run the policy (sampled) in the real simulator, then replay exactly those actions inside
the world model starting from the real first frame only. Saves a PNG: top row real frames, bottom row decoded
imagined frames, with measured block/goal coverage and the success head's probability under each column.

    SDL_VIDEODRIVER=dummy python -m mini_dreamer4.tools.side_by_side_pusht <tokenizer.pt> <agent.pt> --starts ic10.json --seeds 2000060 2000173 2000081 --out dir
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw

from mini_dreamer4.train import load_tokenizer, load_dynamics
from mini_dreamer4.tools.plan_pusht import make_env, set_state_exact, to_env, PixelReward

W = 10


def real_episode(env, tok, agent, s0, steps, dev, seed, temperature):
    set_state_exact(env, s0); u = env.unwrapped
    frames = [torch.as_tensor(u._render()).float().div(255).movedim(-1, -3)]
    acts, cov = [], [float(u._get_coverage())]
    torch.manual_seed(seed)
    for t in range(steps):
        w = frames[-W:]; hist = acts[-(len(w) - 1):] if len(w) > 1 else []
        a_in = torch.as_tensor(np.array(hist + [np.zeros(2, np.float32)]), device=dev).float()[None]
        with torch.no_grad():
            a = agent.act(tok.encode(torch.stack(w).to(dev)[None]), a_in, sample=True, temperature=temperature, refine=True)[0].cpu().numpy()
        obs, _, _, _, info = env.step(to_env(a))
        acts.append(a.astype(np.float32)); frames.append(torch.as_tensor(obs["pixels"]).float().div(255).movedim(-1, -3)); cov.append(float(info["coverage"]))
        if info["coverage"] > 0.95:
            break
    return torch.stack(frames), np.array(acts), np.array(cov)


@torch.no_grad()
def imagine_with_actions(tok, agent, f0, acts, dev):
    z = tok.encode(f0.to(dev)[None, None])
    a = torch.as_tensor(acts, device=dev).float()[None]
    probs, frames = [], [tok.decode(z)[0, 0].clamp(0, 1).cpu()]
    for t in range(len(acts)):
        ctx = z[:, -(W - 1):]
        a_in = torch.cat((a[:, t + 1 - ctx.shape[1]: t + 1], torch.zeros(1, 1, 2, device=dev)), dim=1) if t + 1 >= ctx.shape[1] else \
            torch.cat((a[:, : t + 1], torch.zeros(1, 1, 2, device=dev)), dim=1)
        nxt = agent.sample(ctx, a_in, horizon=1, num_steps=4)[:, -1:]
        z = torch.cat((z, nxt), dim=1)
        zw = z[:, -W:]
        aw = torch.cat((a[:, t + 2 - zw.shape[1]: t + 1], torch.zeros(1, 1, 2, device=dev)), dim=1) if t + 1 >= zw.shape[1] - 1 else torch.cat((a[:, : t + 1], torch.zeros(1, 1, 2, device=dev)), dim=1)
        aw = aw[:, -zw.shape[1]:]
        p = torch.sigmoid(agent.reward_head(agent.agent_features(zw, aw)[:, -1]).squeeze(-1)).item()
        probs.append(p); frames.append(tok.decode(zw)[0, -1].clamp(0, 1).cpu())
    return torch.stack(frames), np.array([float("nan")] + probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer"); p.add_argument("agent")
    p.add_argument("--starts", required=True); p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--steps", type=int, default=60); p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=3, help="real episodes tried per start; the first failure and the first success are shown")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tok, agent = load_tokenizer(args.tokenizer, dev), load_dynamics(args.agent, dev)
    ics = {ic["reset_seed"]: np.array(ic["raw_state_after_reset"]) for ic in json.load(open(args.starts))["initial_conditions"]}
    env = make_env(); env.reset(seed=0); reward = PixelReward(env, dev)
    for seed in args.seeds:
        shown = set()
        for k in range(args.episodes):
            frames, acts, cov = real_episode(env, tok, agent, ics[seed], args.steps, dev, 500 + k, args.temperature)
            ok = cov[-1] > 0.95
            if ok in shown: continue
            shown.add(ok)
            im_frames, probs = imagine_with_actions(tok, agent, frames[0], acts, dev)
            im_cov = np.array([reward(f.to(dev)[None])[0].item() for f in im_frames])
            first_head = next((t for t, q in enumerate(probs) if q > 0.5), None)
            T = len(frames); cols = sorted(set(list(range(0, T, max(1, (T - 1) // 7))) + [T - 1]))[:9]
            S = 2; fh = 96 * S; pad = 34
            canvas = Image.new("RGB", (len(cols) * fh, 2 * fh + 2 * pad + 18), "white"); d = ImageDraw.Draw(canvas)
            for i, t in enumerate(cols):
                for row, (fr, c) in enumerate(((frames[t], cov[t]), (im_frames[t], im_cov[t]))):
                    img = Image.fromarray((fr.movedim(0, -1).numpy() * 255).round().astype(np.uint8)).resize((fh, fh), Image.NEAREST)
                    canvas.paste(img, (i * fh, 18 + row * (fh + pad)))
                d.text((i * fh + 3, 18 + fh + 2), "t=%d real cov %.2f" % (t, cov[t]), fill="black")
                d.text((i * fh + 3, 18 + fh + 14), "imag cov %.2f" % im_cov[t], fill=(150, 0, 0))
                d.text((i * fh + 3, 18 + 2 * fh + pad + 2), "p(success)=%s" % ("-" if np.isnan(probs[t]) else "%.2f" % probs[t]), fill=(0, 0, 150))
            title = "start %d, %s in the real sim after %d steps (coverage %.3f). Imagined open-loop with the SAME actions: success head first fires at t=%s, imagined coverage max %.2f" % (
                seed, "SUCCESS" if ok else "FAILURE", T - 1, cov[-1], first_head, np.nanmax(im_cov))
            d.text((4, 3), title, fill="black")
            path = out / ("%d_%s.png" % (seed, "success" if ok else "failure")); canvas.save(path)
            print(title); print("   saved", path)
            print("   per-step real coverage:", " ".join("%.2f" % c for c in cov[::max(1, (T - 1) // 15)]))
            print("   imagined coverage:     ", " ".join("%.2f" % c for c in im_cov[::max(1, (T - 1) // 15)]))
            print("   success head p:        ", " ".join("%.2f" % q if not np.isnan(q) else " -  " for q in probs[::max(1, (T - 1) // 15)]))


if __name__ == "__main__":
    main()
