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
    """

    def __init__(self, episodes: list[dict], seq_len: int, image_size: int | None = None,
                 samples_per_epoch: int | None = None, seed: int = 0):
        self.episodes = [ep for ep in episodes if ep["video"].shape[0] >= seq_len]
        assert self.episodes, "no episode is long enough for seq_len"
        self.seq_len, self.image_size = seq_len, image_size
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
        out = {"video": video, "actions": torch.as_tensor(ep["actions"][sl]).float()}
        for k in ("states", "rewards"):
            if k in ep:
                out[k] = torch.as_tensor(ep[k][sl]).float()
        return out


def load_pusht_zarr(path: str | Path, action_range: tuple[float, float] = (0.0, 512.0)) -> list[dict]:
    """Load the PushT image dataset (diffusion-policy / lerobot ``pusht_cchi_v7_replay.zarr`` layout):
    ``data/img`` (N, 96, 96, 3) uint8, ``data/action`` (N, 2), ``data/state``, ``meta/episode_ends``.
    Actions (target agent positions in pixels) are rescaled to [-1, 1]."""
    import zarr  # optional dependency

    root = zarr.open(str(path), mode="r")
    img, act = root["data"]["img"], root["data"]["action"]
    state = root["data"]["state"] if "state" in root["data"] else None
    ends = np.asarray(root["meta"]["episode_ends"])
    lo, hi = action_range
    episodes, start = [], 0
    for end in ends:
        a = (np.asarray(act[start:end], dtype=np.float32) - lo) / (hi - lo) * 2 - 1
        ep = {"video": np.asarray(img[start:end], dtype=np.uint8), "actions": a}
        if state is not None:
            ep["states"] = np.asarray(state[start:end], dtype=np.float32)
        episodes.append(ep)
        start = int(end)
    return episodes


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
