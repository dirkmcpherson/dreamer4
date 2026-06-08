# /// script
# dependencies = [
#   "torch", "numpy", "fire", "tqdm", "h5py",
#   "robosuite", "mujoco", "bddl", "easydict", "dreamer4",
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///
"""Closed-loop evaluation of a trained dreamer4 world model in the LIBERO simulator.

Run headless on a cluster node:
    MUJOCO_GL=egl python eval_libero.py \
        --suite libero_object \
        --dynamics_checkpoint ./logs_libero_dynamics/checkpoints/dynamics-300000.pt \
        --tokenizer_checkpoint ./logs_libero_tokenizer/checkpoints \
        --episodes_per_task 10

================================ READ ME ========================================
This script is a *scaffold validated for structure, not yet on a live cluster*.
Two things MUST be confirmed against your LIBERO install before trusting numbers:

  1. Live-env obs key names. We read `agentview_image` and (for proprio)
     `robot0_joint_pos` + `robot0_gripper_qpos`. Older/newer robosuite may differ;
     print `obs.keys()` once and adjust `IMAGE_KEY` / `PROPRIO_KEYS` below.

  2. The ACTING path. `DynamicsWorldModel.interact_with_env` samples actions from
     the model's policy head (the RL/agent path used by DreamTrainer/SimTrainer).
     A model trained ONLY with BehaviorCloneTrainer may have an untrained policy
     head -> poor actions. If so, either (a) fine-tune the policy head, or
     (b) drive actions from the BC action readout. Sanity-check by watching the
     forward-prediction videos and the first few eval rollouts.

LIBERO eval also historically wants robosuite==1.4.1; if env creation fails on
1.5.x, use the pinned eval env from infra/update_cluster_env.sh.
=================================================================================
"""
import os

import fire
import numpy as np
import torch
from tqdm import tqdm

IMAGE_KEY = 'agentview_image'                          # live robosuite obs key
PROPRIO_KEYS = ('robot0_joint_pos', 'robot0_gripper_qpos')  # used only if model.has_proprio


def exists(v):
    return v is not None


class LiberoDreamerEnv:
    """Adapts a LIBERO OffScreenRenderEnv to the env API dreamer4 expects:
    reset()/step() return image observations (and optional proprio)."""

    def __init__(self, suite_name, task_id, image_size=128, with_proprio=False, device='cpu'):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[suite_name]()
        self.task = suite.get_task(task_id)
        self.init_states = suite.get_task_init_states(task_id)

        bddl = os.path.join(get_libero_path('bddl_files'),
                            self.task.problem_folder, self.task.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=bddl,
                                      camera_heights=image_size, camera_widths=image_size)
        self.image_size = image_size
        self.with_proprio = with_proprio
        self.device = device
        self._init_idx = 0

    def _obs_to_dict(self, obs):
        img = np.ascontiguousarray(obs[IMAGE_KEY][::-1])              # flip to match training
        img = torch.from_numpy(img).float().div_(255.).permute(2, 0, 1).to(self.device)  # (3,H,W)
        out = {'image': img}
        if self.with_proprio:
            parts = [torch.from_numpy(np.asarray(obs[k]).reshape(-1)).float() for k in PROPRIO_KEYS]
            out['proprio'] = torch.cat(parts, dim=-1).to(self.device)
        return out

    def reset(self, seed=None):
        if exists(seed):
            self.env.seed(seed)
        self.env.reset()
        obs = self.env.set_init_state(self.init_states[self._init_idx % len(self.init_states)])
        self._init_idx += 1
        self._success = False
        return self._obs_to_dict(obs)

    def step(self, action):
        # dreamer4 may pass a (discrete, continuous) tuple; take the continuous part
        if isinstance(action, tuple):
            action = next(a for a in reversed(action) if exists(a))
        a = action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        a = a.reshape(-1)[:7]                                          # LIBERO is 7-DoF
        obs, reward, done, info = self.env.step(a)
        self._success = self._success or bool(reward >= 1.0) or bool(info.get('success', False))
        return self._obs_to_dict(obs), float(reward), bool(done or self._success)

    def close(self):
        self.env.close()


@torch.no_grad()
def main(
    dynamics_checkpoint: str,
    tokenizer_checkpoint: str,
    suite: str = 'libero_object',
    image_size: int = 128,
    episodes_per_task: int = 10,
    max_steps: int = 256,
    num_tasks: int = None,                 # default: all tasks in the suite
    use_proprio: bool = False,
    device: str = 'cuda',
):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from pathlib import Path
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel
    from libero.libero import benchmark

    device = device if torch.cuda.is_available() else 'cpu'

    # tokenizer
    tok_path = Path(tokenizer_checkpoint)
    if tok_path.is_dir():
        ema = list(tok_path.glob('tokenizer-*-ema.pt'))
        assert ema, f'no EMA tokenizer in {tok_path}'
        tok_path = max(ema, key=lambda p: int(p.stem.split('-')[1]))
    tokenizer = VideoTokenizer.init_and_load(str(tok_path), strict=False).to(device).eval()

    # dynamics model -- prefer init_and_load if config travelled with the checkpoint
    try:
        model = DynamicsWorldModel.init_and_load(dynamics_checkpoint, strict=False)
    except Exception as e:
        raise RuntimeError(
            'Could not auto-construct DynamicsWorldModel from checkpoint config. '
            'DynamicsWorldModel lacks @save_load (see infra/h200_readiness.md P3). '
            'Reconstruct the model with the same args used in train_libero_dynamics.py '
            'and load the state dict manually.'
        ) from e
    model = model.to(device).eval()
    model.video_tokenizer = tokenizer

    suite_obj = benchmark.get_benchmark_dict()[suite]()
    n = num_tasks if exists(num_tasks) else suite_obj.n_tasks

    per_task = []
    for task_id in range(n):
        env = LiberoDreamerEnv(suite, task_id, image_size=image_size,
                               with_proprio=use_proprio, device=device)
        successes = 0
        for ep in tqdm(range(episodes_per_task), desc=f'{suite} task {task_id}', leave=False):
            exp = model.interact_with_env(env, max_timesteps=max_steps, use_time_cache=True,
                                          store_agent_embed=False, store_old_action_unembeds=False)
            r = exp.rewards
            successes += int(r is not None and float(r.max()) >= 1.0)
        env.close()
        sr = successes / episodes_per_task
        per_task.append(sr)
        print(f'  task {task_id:3d}: success {successes}/{episodes_per_task} = {sr:.2%}')

    print(f'\n[{suite}] mean success rate over {n} tasks: {np.mean(per_task):.2%}')
    return np.mean(per_task)


if __name__ == '__main__':
    fire.Fire(main)
