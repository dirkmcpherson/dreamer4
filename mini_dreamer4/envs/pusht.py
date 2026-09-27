"""A small, dependency-free PushT-style environment for spatial world-model tests.

A round agent is driven by a 2D continuous action (displacement command in [-1, 1]^2) and
pushes a T-shaped block around a unit workspace; the goal is a fixed T pose drawn in green,
as in the original PushT task. Observations are RGB frames; the reward is the fraction of
the block that covers the goal. The physics are a simple penetration-based push with a
torque term, which is enough to give the same kind of visual, contact-driven dynamics that
make PushT a useful world-model test, at a tiny fraction of the cost.
"""
from __future__ import annotations

import numpy as np
import torch


# T shape in the block frame: a bar on top and a stem below, total extent ~0.3 x 0.3
_T_RECTS = np.array([
    # cx, cy, half_w, half_h
    [0.0, 0.105, 0.150, 0.045],
    [0.0, -0.060, 0.045, 0.100],
], dtype=np.float32)


def _rotate(p: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return p @ np.array([[c, -s], [s, c]], dtype=np.float32).T


def _nearest_on_rect(p: np.ndarray, rect: np.ndarray) -> tuple[np.ndarray, float]:
    cx, cy, hw, hh = rect
    d = p - np.array([cx, cy], dtype=np.float32)
    q = np.abs(d) - np.array([hw, hh], dtype=np.float32)
    if (q <= 0).all():  # inside: project onto the nearest edge
        axis = int(np.argmax(q))
        n = np.zeros(2, dtype=np.float32)
        n[axis] = np.sign(d[axis]) if d[axis] != 0 else 1.0
        nearest = d.copy()
        nearest[axis] = n[axis] * (hw if axis == 0 else hh)
        return nearest + np.array([cx, cy], dtype=np.float32), float(q[axis])
    clamped = np.clip(d, [-hw, -hh], [hw, hh]).astype(np.float32)
    nearest = clamped + np.array([cx, cy], dtype=np.float32)
    return nearest, float(np.linalg.norm(p - nearest))


def _t_mask(xs: np.ndarray, ys: np.ndarray, pos: np.ndarray, angle: float) -> np.ndarray:
    pts = np.stack((xs - pos[0], ys - pos[1]), axis=-1).reshape(-1, 2)
    local = _rotate(pts, -angle)
    inside = np.zeros(local.shape[0], dtype=bool)
    for cx, cy, hw, hh in _T_RECTS:
        inside |= (np.abs(local[:, 0] - cx) <= hw) & (np.abs(local[:, 1] - cy) <= hh)
    return inside.reshape(xs.shape)


class MiniPushT:
    agent_radius = 0.045
    max_step = 0.06

    def __init__(self, image_size: int = 64, max_steps: int = 60, seed: int | None = None,
                 goal_pose: tuple[float, float, float] = (0.5, 0.5, np.pi / 4)):
        self.image_size, self.max_steps = image_size, max_steps
        self.goal = np.array(goal_pose, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        lin = (np.arange(image_size) + 0.5) / image_size
        self._xs, self._ys = np.meshgrid(lin, lin)
        self._goal_mask = _t_mask(self._xs, self._ys, self.goal[:2], float(self.goal[2]))
        self.reset()

    # ------------------------------------------------------------------ gym-like API
    @property
    def state(self) -> np.ndarray:
        return np.array([*self.agent, *self.block, np.cos(self.angle), np.sin(self.angle)], dtype=np.float32)

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.block = self.rng.uniform(0.25, 0.75, size=2).astype(np.float32)
        self.angle = float(self.rng.uniform(-np.pi, np.pi))
        while True:
            self.agent = self.rng.uniform(0.1, 0.9, size=2).astype(np.float32)
            if np.linalg.norm(self.agent - self.block) > 0.25:
                break
        return self._obs()

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1, 1)
        self.steps += 1
        r = self.agent_radius
        self.agent = np.clip(self.agent + a * self.max_step, r, 1 - r).astype(np.float32)

        local_agent = _rotate((self.agent - self.block)[None], -self.angle)[0]
        best = None
        for rect in _T_RECTS:
            nearest, dist = _nearest_on_rect(local_agent, rect)
            if best is None or dist < best[1]:
                best = (nearest, dist)
        nearest_local, dist = best
        if dist < r:
            n_local = local_agent - nearest_local
            norm = np.linalg.norm(n_local)
            n_local = n_local / norm if norm > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
            push_local = -n_local * (r - dist)
            lever = nearest_local  # relative to block centre (origin of block frame)
            torque = float(lever[0] * push_local[1] - lever[1] * push_local[0])
            self.block = np.clip(self.block + _rotate(push_local[None], self.angle)[0], 0.05, 0.95).astype(np.float32)
            self.angle = float((self.angle + 6.0 * torque + np.pi) % (2 * np.pi) - np.pi)

        reward = self.coverage()
        terminated = reward > 0.95
        truncated = self.steps >= self.max_steps
        return self._obs(), float(reward), bool(terminated), bool(truncated), {}

    def coverage(self) -> float:
        block = _t_mask(self._xs, self._ys, self.block, self.angle)
        return float((block & self._goal_mask).sum() / max(1, block.sum()))

    # ------------------------------------------------------------------ rendering
    def render(self) -> np.ndarray:
        """(H, W, 3) float32 in [0, 1]."""
        img = np.ones((self.image_size, self.image_size, 3), dtype=np.float32)
        img[self._goal_mask] = (0.55, 0.9, 0.55)
        block = _t_mask(self._xs, self._ys, self.block, self.angle)
        img[block] = (0.5, 0.5, 0.5)
        agent = (self._xs - self.agent[0]) ** 2 + (self._ys - self.agent[1]) ** 2 <= self.agent_radius ** 2
        img[agent] = (0.2, 0.4, 1.0)
        return img

    def _obs(self) -> dict:
        return {"image": self.render(), "state": self.state}


# ---------------------------------------------------------------------- data collection

def _pusher_policy(env: MiniPushT, rng: np.random.Generator, hold: list) -> np.ndarray:
    """Noisy heuristic: mostly head for a point around the block, sometimes wander."""
    if hold[0] <= 0:
        if rng.random() < 0.7:
            hold[1] = env.block + rng.normal(0, 0.12, size=2)
        else:
            hold[1] = rng.uniform(0.1, 0.9, size=2)
        hold[0] = int(rng.integers(3, 10))
    hold[0] -= 1
    direction = hold[1] - env.agent
    norm = np.linalg.norm(direction)
    a = direction / (norm + 1e-6) * min(1.0, norm / env.max_step)
    return np.clip(a + rng.normal(0, 0.25, size=2), -1, 1).astype(np.float32)


def generate_episodes(num_episodes: int, length: int, image_size: int = 64, seed: int = 0) -> list[dict]:
    """Each episode: video uint8 (T, H, W, 3), states (T, 6), actions float32 (T, 2) taken at frame t
    (the last action is a zero placeholder), rewards (T,)."""
    rng = np.random.default_rng(seed)
    env = MiniPushT(image_size=image_size, max_steps=length, seed=int(rng.integers(1 << 30)))
    episodes = []
    for _ in range(num_episodes):
        obs = env.reset(seed=int(rng.integers(1 << 30)))
        frames, states, actions, rewards = [obs["image"]], [obs["state"]], [], [0.0]
        hold = [0, None]
        for _ in range(length - 1):
            a = _pusher_policy(env, rng, hold)
            obs, r, term, trunc, _ = env.step(a)
            frames.append(obs["image"]); states.append(obs["state"]); actions.append(a); rewards.append(r)
        actions.append(np.zeros(2, dtype=np.float32))
        episodes.append({
            "video": (np.stack(frames) * 255).round().astype(np.uint8),
            "states": np.stack(states).astype(np.float32),
            "actions": np.stack(actions).astype(np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
        })
    return episodes


def state_to_oracle_latents(states: torch.Tensor, num_latents: int, latent_dim: int, seed: int = 0) -> torch.Tensor:
    """Deterministic 'oracle tokenizer': a fixed random tanh projection of the 6-d state into
    (num_latents, latent_dim). Lets the dynamics model be tested independently of a learned tokenizer."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(6, num_latents * latent_dim, generator=g) * 1.5
    b = torch.randn(num_latents * latent_dim, generator=g) * 0.3
    centered = states.float() - torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
    z = torch.tanh(centered @ w.to(states.device) + b.to(states.device))
    return z.reshape(*states.shape[:-1], num_latents, latent_dim)
