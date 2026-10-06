"""Record the policy acting greedily from each start state in the real simulator and render GIFs.

    SDL_VIDEODRIVER=dummy python -m mini_dreamer4.tools.demo_policy_pusht <tokenizer.pt> <agent.pt> --starts ic10.json --out demo/
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw

from mini_dreamer4.train import load_tokenizer, load_dynamics
from mini_dreamer4.tools.plan_pusht import make_env, set_state_exact, to_env

W = 10


def episode(env, tok, agent, s0, steps, dev):
    set_state_exact(env, s0); u = env.unwrapped
    frames, acts, cov = [u._render()], [], [float(u._get_coverage())]
    for t in range(steps):
        w = [torch.as_tensor(f).float().div(255).movedim(-1, -3) for f in frames[-W:]]
        hist = acts[-(len(w) - 1):] if len(w) > 1 else []
        a_in = torch.as_tensor(np.array(hist + [np.zeros(2, np.float32)]), device=dev).float()[None]
        with torch.no_grad():
            a = agent.act(tok.encode(torch.stack(w).to(dev)[None]), a_in, refine=True)[0].cpu().numpy()
        obs, _, _, _, info = env.step(to_env(a))
        frames.append(obs["pixels"]); acts.append(a.astype(np.float32)); cov.append(float(info["coverage"]))
        if info["coverage"] > 0.95:
            break
    return frames, acts, cov


def label(frame, text, color, scale=3):
    img = Image.fromarray(frame).resize((96 * scale, 96 * scale), Image.NEAREST)
    d = ImageDraw.Draw(img); d.rectangle((0, 0, 96 * scale, 14), fill="white"); d.text((3, 1), text, fill=color)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer"); p.add_argument("agent")
    p.add_argument("--starts", required=True); p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", required=True); p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()
    dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tok, agent = load_tokenizer(args.tokenizer, dev), load_dynamics(args.agent, dev)
    ics = json.load(open(args.starts))["initial_conditions"]
    env = make_env(); env.reset(seed=0)
    runs = []
    for ic in ics:
        frames, acts, cov = episode(env, tok, agent, np.array(ic["raw_state_after_reset"]), args.steps, dev)
        ok = cov[-1] > 0.95
        runs.append((ic["reset_seed"], frames, cov, ok))
        print("start %d: %s after %d steps, final coverage %.3f" % (ic["reset_seed"], "SUCCESS" if ok else "fail", len(frames) - 1, cov[-1]), flush=True)
        clip = [label(f, "start %d  t=%d  cov %.2f%s" % (ic["reset_seed"], t, c, "  SUCCESS" if ok and t == len(frames) - 1 else ""), (0, 110, 0) if c > 0.95 else "black")
                for t, (f, c) in enumerate(zip(frames, cov))]
        clip += [clip[-1]] * args.fps                    # hold the last frame for a second
        clip[0].save(out / f"start_{ic['reset_seed']}.gif", save_all=True, append_images=clip[1:], duration=1000 // args.fps, loop=0)
    # tiled gif: 5 x 2, every episode plays from t=0 and holds its last frame; show at most 60 steps
    T = min(60, max(len(r[1]) for r in runs)) + 1
    s = 2; tiles = []
    for t in range(T + args.fps):
        canvas = Image.new("RGB", (5 * 96 * s, 2 * (96 * s + 14)), "white")
        for i, (seed, frames, cov, ok) in enumerate(runs):
            k = min(t, len(frames) - 1)
            img = Image.fromarray(frames[k]).resize((96 * s, 96 * s), Image.NEAREST)
            x, y = (i % 5) * 96 * s, (i // 5) * (96 * s + 14)
            canvas.paste(img, (x, y + 14))
            d = ImageDraw.Draw(canvas)
            txt = "%d t=%d cov %.2f" % (seed, k, cov[k]) + ("  OK" if ok and k == len(frames) - 1 else ("  fail" if not ok and k == len(frames) - 1 else ""))
            d.text((x + 3, y + 1), txt, fill=(0, 110, 0) if cov[k] > 0.95 else "black")
        tiles.append(canvas)
    tiles[0].save(out / "all_starts.gif", save_all=True, append_images=tiles[1:], duration=1000 // args.fps, loop=0)
    # contact sheet: first, middle and last frame per start
    sheet = Image.new("RGB", (3 * 96 * s, len(runs) * (96 * s + 14)), "white")
    for i, (seed, frames, cov, ok) in enumerate(runs):
        for j, k in enumerate((0, (len(frames) - 1) // 2, len(frames) - 1)):
            sheet.paste(Image.fromarray(frames[k]).resize((96 * s, 96 * s), Image.NEAREST), (j * 96 * s, i * (96 * s + 14) + 14))
            ImageDraw.Draw(sheet).text((j * 96 * s + 3, i * (96 * s + 14) + 1), "%d  t=%d  cov %.2f" % (seed, k, cov[k]), fill=(0, 110, 0) if cov[k] > 0.95 else "black")
    sheet.save(out / "contact_sheet.png")
    print("greedy success %d/%d; saved %s" % (sum(r[3] for r in runs), len(runs), out))


if __name__ == "__main__":
    main()
