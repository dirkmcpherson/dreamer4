"""Episode datasets: fixed-length windows over trajectories, plus a loader for real PushT data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class EpisodeWindowDataset(Dataset):
    """Random fixed-length windows from a list of episodes.

    Each episode is a dict with ``video`` (T, H, W, 3) uint8 or float, ``actions`` (T, A) float32
    taken at frame t, and optionally ``states`` (T, S) and ``rewards`` (T,).
    Returns dict(video (L, 3, H, W) float in [0, 1], actions (L, A), states?, rewards?).

    ``shift`` > 0 applies a random translation of up to that many pixels (replicate padding), the same for
    every frame of a window. With ``shift_actions`` the first two action components are treated as absolute
    image positions normalized to [-1, 1] and are translated by the same amount (PushT); leave it False for
    displacement actions (MiniPushT).
    """

    def __init__(self, episodes: list[dict], seq_len: int, image_size: int | None = None,
                 samples_per_epoch: int | None = None, seed: int = 0, shift: int = 0, shift_actions: bool = False):
        self.episodes = [ep for ep in episodes if ep["video"].shape[0] >= seq_len]
        assert self.episodes, "no episode is long enough for seq_len"
        self.seq_len, self.image_size = seq_len, image_size
        self.shift, self.shift_actions = shift, shift_actions
        self.rng = np.random.default_rng(seed)
        self.starts = [ep["video"].shape[0] - seq_len + 1 for ep in self.episodes]
        self.samples_per_epoch = samples_per_epoch or sum(self.starts)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        i = int(self.rng.integers(len(self.episodes)))
        ep = self.episodes[i]
        s = int(self.rng.integers(self.starts[i]))
        sl = slice(s, s + self.seq_len)
        video = torch.as_tensor(ep["video"][sl])
        if video.dtype == torch.uint8:
            video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)
        if self.image_size is not None and video.shape[-1] != self.image_size:
            video = F.interpolate(video, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        actions = torch.as_tensor(ep["actions"][sl]).float()
        if self.shift > 0:
            dx, dy = (int(v) for v in self.rng.integers(-self.shift, self.shift + 1, size=2))
            h, w = video.shape[-2:]
            padded = F.pad(video, (self.shift,) * 4, mode="replicate")
            video = padded[..., self.shift - dy: self.shift - dy + h, self.shift - dx: self.shift - dx + w]
            if self.shift_actions:  # positions move with the image: +dx pixels is +2 dx / w in [-1, 1] units
                actions = actions.clone()
                actions[:, 0] += 2 * dx / w
                actions[:, 1] += 2 * dy / h
        out = {"video": video, "actions": actions}
        for k in ("states", "rewards", "coverage", "action_mask"):
            if k in ep:
                out[k] = torch.as_tensor(ep[k][sl]).float()
        return out


def static_texture(size: int, seed: int = 0) -> np.ndarray:
    """A fixed wood-like texture (H, W, 3) uint8: low-frequency grain plus fine noise, in warm tones that stay far
    from the PushT block (blue-grey), agent (blue) and goal (green) colours so they remain separable by colour."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    g = np.zeros((size, size))
    for _ in range(6):
        fx, fy, ph = rng.uniform(1, 9), rng.uniform(0.2, 2.5), rng.uniform(0, 2 * np.pi)
        g += np.sin(2 * np.pi * (fx * x + fy * y) + ph) / 6
    g = 0.5 + 0.5 * np.tanh(2 * g) + rng.normal(0, 0.06, (size, size))
    g = np.clip(g, 0, 1)
    rgb = np.stack((0.45 + 0.35 * g, 0.30 + 0.25 * g, 0.15 + 0.15 * g), axis=-1)   # dark to light brown
    return (rgb * 255).round().astype(np.uint8)


def replace_white_background(video: np.ndarray, texture: np.ndarray, threshold: int = 240) -> np.ndarray:
    """Composite ``texture`` (H, W, 3) under the near-white pixels of ``video`` (T, H, W, 3) uint8."""
    white = video.min(axis=-1, keepdims=True) > threshold
    return np.where(white, texture[None], video)


def load_pusht_zarr(path: str | Path, action_range: tuple[float, float] = (0.0, 512.0),
                    background: str | None = None) -> list[dict]:
    """Load the PushT image dataset (diffusion-policy / lerobot ``pusht_cchi_v7_replay.zarr`` layout):
    ``data/img`` (N, 96, 96, 3) uint8, ``data/action`` (N, 2), ``data/state``, ``meta/episode_ends``.
    Actions (target agent positions in pixels) are rescaled to [-1, 1].
    ``background="texture"`` replaces the white background with a fixed texture (see ``static_texture``)."""
    import zarr  # optional dependency

    root = zarr.open(str(path), mode="r")
    img, act = root["data"]["img"], root["data"]["action"]
    state = root["data"]["state"] if "state" in root["data"] else None
    ends = np.asarray(root["meta"]["episode_ends"])
    lo, hi = action_range
    episodes, start = [], 0
    for end in ends:
        a = (np.asarray(act[start:end], dtype=np.float32) - lo) / (hi - lo) * 2 - 1
        frames = np.asarray(img[start:end], dtype=np.uint8)
        if background == "texture":
            frames = replace_white_background(frames, static_texture(frames.shape[1]))
        elif background is not None:
            raise ValueError(f"unknown background {background!r}")
        ep = {"video": frames, "actions": a}
        if state is not None:
            ep["states"] = np.asarray(state[start:end], dtype=np.float32)
        episodes.append(ep)
        start = int(end)
    return episodes


def load_pusht_npz(path: str | Path, background: str | None = None) -> list[dict]:
    """Episodes rendered by ``tools/render_rlpd_demos.py``: ``img`` uint8, ``action`` already in [-1, 1],
    ``state``, ``coverage``, ``episode_ends``."""
    # Read each array ONCE: indexing an NpzFile decompresses the whole array on every access, and a slice of
    # that result keeps the full array alive, so per-episode d["img"][a:b] holds one full copy per episode.
    with np.load(path) as d:
        img, action, state, coverage, ends = d["img"], d["action"].astype(np.float32), d["state"], d["coverage"], d["episode_ends"]
    episodes, start = [], 0
    for end in ends:
        frames = img[start:end]
        if background == "texture":
            frames = replace_white_background(frames, static_texture(frames.shape[1]))
        # "coverage" (not "rewards") so that batches mix cleanly with the zarr episodes, which carry no reward
        n = int(end) - start
        episodes.append({"video": frames, "actions": action[start:end], "states": state[start:end], "coverage": coverage[start:end],
                         "action_mask": np.arange(n) < n - 1})         # no action is taken at an episode's last frame
        start = int(end)
    return episodes


def load_pusht(paths: str | Path, background: str | None = None) -> list[dict]:
    """One or several (comma-separated) PushT datasets: ``.zarr`` (human demos) or ``.npz`` (rendered tapes)."""
    episodes = []
    for path in str(paths).split(","):
        loader = load_pusht_npz if path.endswith(".npz") else load_pusht_zarr
        episodes += loader(path, background=background)
    return episodes


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
