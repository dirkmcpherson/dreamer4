"""PyTorch dataset over LIBERO demonstration HDF5 files (robomimic format).

Each .hdf5 file (one per task) has:
    f['data'][demo_i]/
        actions        (T, 7)      6-DoF EEF delta pose + 1 gripper, ~[-1, 1]
        dones          (T,)
        rewards         (T,)        optional (sparse success); falls back to dones
        obs/
            agentview_rgb   (T, H, W, 3) uint8   third-person cam (stored vertically flipped)
            eye_in_hand_rgb (T, H, W, 3) uint8   wrist cam
            joint_states    (T, 7)
            gripper_states  (T, 2)

mode='tokenizer' -> returns a video tensor (C, T, H, W) in [0, 1]
mode='dynamics'  -> returns dict(video=(C,T,H,W), continuous_actions=(T,7),
                                 rewards=(T,)[, proprio=(T, dp)])

The dict keys are exactly the kwargs `DynamicsWorldModel.forward` /
`BehaviorCloneTrainer` expect, so the dataset plugs straight into the trainers.

Notes
-----
* LIBERO/robosuite render frames upside-down; `flip_images=True` corrects them.
* HDF5 handles are opened lazily per worker (fork-safe).
* Use `--stride` < num_frames for overlapping windows (more, correlated samples).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import h5py

LIBERO_ACTION_DIM = 7  # 6-DoF delta pose + gripper


def exists(v):
    return v is not None


class LiberoDataset(Dataset):
    def __init__(
        self,
        data_dir,                          # dir containing *.hdf5 (a suite folder); searched recursively
        num_frames = 16,
        mode = 'dynamics',                 # 'tokenizer' | 'dynamics'
        image_size = 128,                  # resize if != stored resolution
        camera = 'agentview_rgb',
        stride = None,                     # window stride (default = num_frames, non-overlapping)
        flip_images = True,                # correct LIBERO/robosuite vertical flip
        clamp_actions = True,              # clamp actions to [-1, 1]
        with_rewards = True,               # dynamics: include reward channel
        proprio_keys = None,               # e.g. ('joint_states','gripper_states'); None disables proprio
        max_files = None,                  # cap #hdf5 files (debugging)
        min_demo_len = None,               # skip demos shorter than this (default num_frames)
    ):
        super().__init__()
        assert mode in ('tokenizer', 'dynamics')
        self.mode = mode
        self.num_frames = num_frames
        self.image_size = image_size
        self.camera = camera
        self.stride = stride if exists(stride) else num_frames
        self.flip_images = flip_images
        self.clamp_actions = clamp_actions
        self.with_rewards = with_rewards
        self.proprio_keys = tuple(proprio_keys) if exists(proprio_keys) else None
        min_demo_len = min_demo_len if exists(min_demo_len) else num_frames

        files = sorted(Path(data_dir).rglob('*.hdf5'))
        if exists(max_files):
            files = files[:max_files]
        assert files, f'no .hdf5 files found under {data_dir}'
        self.files = [str(p) for p in files]

        # build a flat index of (file_idx, demo_key, start_frame) windows (metadata only)
        self.index: list[tuple[int, str, int]] = []
        for fi, path in enumerate(self.files):
            with h5py.File(path, 'r') as f:
                demos = f['data']
                for dk in demos.keys():
                    T = demos[dk]['actions'].shape[0]
                    if T < min_demo_len:
                        continue
                    for s in range(0, T - num_frames + 1, self.stride):
                        self.index.append((fi, dk, s))
        assert self.index, f'no windows of length {num_frames} found under {data_dir}'

        # infer proprio dim from the first window
        self.dim_proprio = None
        if self.proprio_keys is not None:
            with h5py.File(self.files[0], 'r') as f:
                dk0 = next(iter(f['data'].keys()))
                self.dim_proprio = int(sum(f['data'][dk0]['obs'][k].shape[-1] for k in self.proprio_keys))

        self.action_dim = LIBERO_ACTION_DIM
        self._open: dict[int, h5py.File] = {}

    # lazy, fork-safe file handles (one set per worker process)
    def _file(self, fi):
        h = self._open.get(fi)
        if h is None:
            h = h5py.File(self.files[fi], 'r')
            self._open[fi] = h
        return h

    def __len__(self):
        return len(self.index)

    def _load_video(self, demo, s, e):
        frames = demo['obs'][self.camera][s:e]              # (T, H, W, 3) uint8
        if self.flip_images:
            frames = frames[:, ::-1, :, :]                  # vertical flip
        frames = np.ascontiguousarray(frames)
        video = torch.from_numpy(frames).float().div_(255.) # (T, H, W, 3) in [0,1]
        video = video.permute(0, 3, 1, 2)                   # (T, 3, H, W)
        if video.shape[-1] != self.image_size or video.shape[-2] != self.image_size:
            video = F.interpolate(video, size=(self.image_size, self.image_size),
                                  mode='bilinear', align_corners=False)
        return video.permute(1, 0, 2, 3).contiguous()       # (3, T, H, W)

    def __getitem__(self, idx):
        fi, dk, s = self.index[idx]
        e = s + self.num_frames
        demo = self._file(fi)['data'][dk]

        video = self._load_video(demo, s, e)
        if self.mode == 'tokenizer':
            return video

        actions = torch.from_numpy(demo['actions'][s:e]).float()   # (T, 7)
        if self.clamp_actions:
            actions = actions.clamp(-1., 1.)

        out = dict(video=video, continuous_actions=actions)

        if self.with_rewards:
            if 'rewards' in demo:
                rewards = torch.from_numpy(demo['rewards'][s:e]).float()
            else:
                rewards = torch.from_numpy(demo['dones'][s:e]).float()  # sparse success
            out['rewards'] = rewards

        if self.proprio_keys is not None:
            parts = [torch.from_numpy(demo['obs'][k][s:e]).float() for k in self.proprio_keys]
            out['proprio'] = torch.cat(parts, dim=-1)                  # (T, dp)

        return out

    def __del__(self):
        for h in getattr(self, '_open', {}).values():
            try:
                h.close()
            except Exception:
                pass


if __name__ == '__main__':
    # quick inspector: python dataset_libero.py /path/to/libero_object
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else './data/libero/libero_object'
    ds = LiberoDataset(d, num_frames=16, mode='dynamics',
                       proprio_keys=('joint_states', 'gripper_states'))
    print(f'files={len(ds.files)} windows={len(ds)} dim_proprio={ds.dim_proprio}')
    sample = ds[0]
    for k, v in sample.items():
        print(f'  {k:20s} {tuple(v.shape)} {v.dtype}  range[{v.min():.3f},{v.max():.3f}]')
