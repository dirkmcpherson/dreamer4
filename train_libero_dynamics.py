# /// script
# dependencies = [
#   "torch", "torchvision", "fire", "tqdm", "numpy", "einops",
#   "h5py", "moviepy", "imageio", "accelerate", "adam-atan2-pytorch",
#   "torch-einops-utils", "wandb", "tensorboard", "dreamer4",
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///
"""Train the dreamer4 DynamicsWorldModel on LIBERO demos (action-conditioned BC).

Requires a tokenizer checkpoint from train_libero_tokenizer.py (frozen here).

Single GPU:
    python train_libero_dynamics.py \
        --data_dir ./data/libero/libero_90 \
        --tokenizer_checkpoint_path ./logs_libero_tokenizer/checkpoints

Multi-GPU (cluster): launch via `accelerate launch` -- see infra/train_dynamics.sbatch.

Periodic eval video: BehaviorCloneTrainer logs `samples` -- a FORWARD-PREDICTION
rollout generated from `sample_prompt_frames` real frames + the demo's continuous
actions -- to wandb/tensorboard every `log_video_every` steps, plus a real-vs-
generated GIF under `<log_dir>/results`.
"""
from pathlib import Path

import fire
import torch

from dataset_libero import LiberoDataset, LIBERO_ACTION_DIM
from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel
from dreamer4.trainers import BehaviorCloneTrainer


def exists(v):
    return v is not None


def main(
    data_dir: str = './data/libero/libero_90',
    tokenizer_checkpoint_path: str = './logs_libero_tokenizer/checkpoints',
    checkpoint_path: str = None,
    # data / window
    num_frames = 16,
    image_size = 128,
    camera = 'agentview_rgb',
    stride = None,
    use_proprio = False,               # if True, conditions on joint+gripper proprio
    # model
    dim = 512,
    depth = 8,
    attn_dim_head = 64,
    attn_heads = 8,
    multi_token_pred_len = 1,
    shortcut_loss_weight = 5e-2,
    # optim
    batch_size = 16,
    grad_accum_every = 1,
    lr = 3e-4,
    num_train_steps = 300_000,
    use_ema = True,
    mixed_precision = 'bf16',          # 'bf16' | 'fp16' | 'no'
    num_workers = 8,
    # sampling / forward-prediction video
    sample_prompt_frames = 2,
    sample_autoregressive_actions = False,
    # logging / checkpoints
    logger = 'wandb',                  # 'wandb' | 'tensorboard'
    log_dir = './logs_libero_dynamics',
    log_video_every = 2500,
    video_fps = 10,
    checkpoint_every = 5000,
    checkpoint_folder = './logs_libero_dynamics/checkpoints',
    experiment_name = 'dreamer4-libero-dynamics',
):
    torch.set_float32_matmul_precision('high')  # H200/Hopper TF32 fast path

    assert logger in ('wandb', 'tensorboard'), "logger must be 'wandb' or 'tensorboard'"
    use_wandb = logger == 'wandb'
    use_tensorboard = logger == 'tensorboard'

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # resume dynamics from latest checkpoint if present
    latest_checkpoint = Path(checkpoint_path) if exists(checkpoint_path) else None
    ckpt_folder = Path(checkpoint_folder)
    if not latest_checkpoint and ckpt_folder.exists():
        ckpts = [p for p in ckpt_folder.glob('dynamics-*.pt') if 'ema' not in p.name]
        if ckpts:
            latest_checkpoint = max(ckpts, key=lambda p: int(p.stem.split('-')[1]))

    # ---- frozen tokenizer ----
    tok_path = Path(tokenizer_checkpoint_path)
    if tok_path.is_dir():
        ema = list(tok_path.glob('tokenizer-*-ema.pt'))
        assert ema, f'no EMA tokenizer checkpoints in {tok_path}'
        tok_path = max(ema, key=lambda p: int(p.stem.split('-')[1]))
    assert tok_path.exists(), f'tokenizer checkpoint missing: {tok_path}'
    print(f'[libero-dynamics] loading frozen tokenizer: {tok_path}')
    tokenizer = VideoTokenizer.init_and_load(str(tok_path), strict=False)
    tokenizer.eval().requires_grad_(False)

    # ---- dataset ----
    proprio_keys = ('joint_states', 'gripper_states') if use_proprio else None
    dataset = LiberoDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        mode='dynamics',
        image_size=image_size,
        camera=camera,
        stride=stride,
        with_rewards=True,
        proprio_keys=proprio_keys,
    )
    print(f'[libero-dynamics] {len(dataset.files)} files, {len(dataset)} windows; '
          f'action_dim={dataset.action_dim} dim_proprio={dataset.dim_proprio}')

    # ---- world model ----
    model = DynamicsWorldModel(
        video_tokenizer=tokenizer,
        dim_latent=tokenizer.dim_latent,
        dim=dim,
        depth=depth,
        attn_dim_head=attn_dim_head,
        attn_heads=attn_heads,
        num_continuous_actions=LIBERO_ACTION_DIM,
        num_discrete_actions=0,
        dim_proprio=dataset.dim_proprio,        # None when use_proprio=False
        multi_token_pred_len=multi_token_pred_len,
        shortcut_loss_weight=shortcut_loss_weight,
    )

    trainer = BehaviorCloneTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        grad_accum_every=grad_accum_every,
        learning_rate=lr,
        num_train_steps=num_train_steps,
        num_workers=num_workers,
        pin_memory=True,
        use_ema=use_ema,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        log_dir=log_dir,
        log_video=True,
        video_fps=video_fps,
        log_video_every=log_video_every,
        sample_prompt_frames=sample_prompt_frames,
        sample_autoregressive_actions=sample_autoregressive_actions,
        sample_filename_prefix='libero-forward-pred',
        checkpoint_every=checkpoint_every,
        checkpoint_folder=checkpoint_folder,
        project_name=experiment_name,
        accelerate_kwargs=dict(mixed_precision=mixed_precision),
    )

    if exists(latest_checkpoint):
        trainer.load(latest_checkpoint)

    trainer()


if __name__ == '__main__':
    fire.Fire(main)
