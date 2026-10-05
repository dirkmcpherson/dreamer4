"""Render the state-only RLPD PushT demonstrations (gp_wmfi_gym npz tapes) into image episodes.

Each tape is replayed in gym-pusht from its recorded start state; the physics is deterministic, so the
replayed states match the recorded ones exactly. Output: one npz with ``img`` (N, 96, 96, 3) uint8,
``action`` (N, 2) normalized to [-1, 1] (the action taken at that frame; zero for the last frame of an
episode), ``state`` (N, 5), ``coverage`` (N,) and ``episode_ends`` (E,), the layout ``load_pusht_npz`` reads.

    SDL_VIDEODRIVER=dummy python -m mini_dreamer4.tools.render_rlpd_demos \
        --tapes ic10.npz:ic10.json transitions.npz:manifest.json --out rlpd_pusht.npz
"""
import argparse, json, time
import numpy as np

from mini_dreamer4.tools.plan_pusht import make_env


def set_state(u, s):
    """Their convention: angle before position, which restores the recorded body origin exactly."""
    u.agent.position = tuple(map(float, s[:2]))
    u.block.angle = float(s[4])
    u.block.position = tuple(map(float, s[2:4]))
    u.agent.velocity = (0, 0)
    u.block.velocity = (0, 0)
    u.block.angular_velocity = 0


def decode_state(obs):             # inverse of their encode_state: (x - 256) / 256, sin, cos
    s = np.asarray(obs, dtype=np.float64)
    return np.concatenate((s[:4] * 256 + 256, [np.arctan2(s[4], s[5])]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tapes", nargs="+", required=True, help="npz[:json] pairs; the json supplies raw reset states per ic_seed when present")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    env = make_env()
    env.reset(seed=0)
    u = env.unwrapped
    imgs, acts, states, covs, ends = [], [], [], [], []
    n_mismatch, t0 = 0, time.time()
    for spec in args.tapes:
        npz_path, json_path = (spec.split(":") + [None])[:2]
        d = np.load(npz_path)
        starts = {}
        if json_path:
            m = json.load(open(json_path))
            starts = {ic["reset_seed"]: np.array(ic["raw_state_after_reset"]) for ic in m.get("initial_conditions", [])}
        for e in np.unique(d["episode"]):
            idx = np.nonzero(d["episode"] == e)[0]
            seed = int(d["ic_seed"][idx[0]]) if "ic_seed" in d.files else None
            s0 = starts.get(seed, decode_state(d["obs"][idx[0]]))
            set_state(u, s0)
            frames, ep_states, ep_cov = [u._render()], [np.array([*u.agent.position, *u.block.position, u.block.angle])], [u._get_coverage()]
            for i in idx:
                obs, r, term, trunc, info = env.step(((d["action"][i] + 1) / 2 * 512).astype(np.float32))
                frames.append(obs["pixels"])
                ep_states.append(np.array([*u.agent.position, *u.block.position, u.block.angle]))
                ep_cov.append(info["coverage"])
            # exactness check against the recorded next state (position error in pixels)
            err = np.abs(np.array(ep_states[-1][:4]) - (d["next_obs"][idx[-1]][:4] * 256 + 256)).max()
            n_mismatch += err > 0.5
            imgs.append(np.stack(frames)); states.append(np.stack(ep_states)); covs.append(np.array(ep_cov, dtype=np.float32))
            acts.append(np.concatenate((d["action"][idx], np.zeros((1, 2), np.float32))).astype(np.float32))
            ends.append(sum(len(x) for x in imgs))
        print(f"{npz_path}: {len(np.unique(d['episode']))} episodes, {len(d['episode'])} transitions ({time.time() - t0:.0f}s)", flush=True)
    img = np.concatenate(imgs)
    np.savez_compressed(args.out, img=img, action=np.concatenate(acts), state=np.concatenate(states).astype(np.float32),
                        coverage=np.concatenate(covs), episode_ends=np.array(ends, dtype=np.int64))
    print(f"saved {args.out}: {len(ends)} episodes, {len(img)} frames, {n_mismatch} episodes whose replay drifted from the recording")


if __name__ == "__main__":
    main()
