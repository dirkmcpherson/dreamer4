# /// script
# dependencies = [
#   "torch", "torchvision", "fire", "tqdm", "numpy", "einops",
#   "h5py", "moviepy", "imageio", "accelerate", "adam-atan2-pytorch",
#   "wandb", "tensorboard", "dreamer4",
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///
"""Train the dreamer4 VideoTokenizer on LIBERO demonstration frames.

Single GPU:
    python train_libero_tokenizer.py --data_dir ./data/libero/libero_90

Multi-GPU (cluster): launch via `accelerate launch` -- see infra/train_tokenizer.sbatch.
"""
from pathlib import Path

import fire
import torch

from dataset_libero import LiberoDataset
from dreamer4.dreamer4 import VideoTokenizer
from dreamer4.trainers import VideoTokenizerTrainer


def exists(v):
    return v is not None


def main(
    data_dir: str = './data/libero/libero_90',
    # data / window
    num_frames = 16,
    image_size = 128,
    camera = 'agentview_rgb',
    stride = None,
    # model
    dim = 512,
    dim_latent = 32,
    patch_size = 16,
    num_latents = 64,
    encoder_depth = 6,
    decoder_depth = 6,
    time_block_every = 4,
    attn_dim_head = 64,
    attn_heads = 8,
    lpips_loss_weight = 0.,
    decoder_flow_steps = 4,
    # optim
    batch_size = 16,
    grad_accum_every = 1,
    lr = 3e-4,
    num_train_steps = 200_000,
    use_ema = True,
    ema_decay = 0.999,
    mixed_precision = 'bf16',          # 'bf16' | 'fp16' | 'no'
    num_workers = 8,
    # logging / checkpoints
    logger = 'wandb',                  # 'wandb' | 'tensorboard'
    log_dir = './logs_libero_tokenizer',
    log_video_every = 2500,
    video_fps = 10,
    checkpoint_every = 5000,
    checkpoint_folder = './logs_libero_tokenizer/checkpoints',
    checkpoint_path = None,
    experiment_name = 'dreamer4-libero-tokenizer',
    run_name = None,
):
    # H200 / Hopper: enable TF32 fast path for the fp32 ops
    torch.set_float32_matmul_precision('high')

    assert logger in ('wandb', 'tensorboard'), "logger must be 'wandb' or 'tensorboard'"
    use_wandb = logger == 'wandb'
    use_tensorboard = logger == 'tensorboard'

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # resume from latest checkpoint if present
    latest_checkpoint = Path(checkpoint_path) if exists(checkpoint_path) else None
    ckpt_folder = Path(checkpoint_folder)
    if not latest_checkpoint and ckpt_folder.exists():
        ckpts = [p for p in ckpt_folder.glob('tokenizer-*.pt') if 'ema' not in p.name]
        if ckpts:
            latest_checkpoint = max(ckpts, key=lambda p: int(p.stem.split('-')[1]))

    dataset = LiberoDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        mode='tokenizer',
        image_size=image_size,
        camera=camera,
        stride=stride,
    )
    print(f'[libero-tokenizer] {len(dataset.files)} files, {len(dataset)} windows of {num_frames} frames @ {image_size}px')

    tokenizer = VideoTokenizer(
        dim=dim,
        dim_latent=dim_latent,
        patch_size=patch_size,
        num_latent_tokens=num_latents,
        channels=3,
        image_height=image_size,
        image_width=image_size,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        time_block_every=time_block_every,
        attn_dim_head=attn_dim_head,
        attn_heads=attn_heads,
        lpips_loss_weight=lpips_loss_weight,
        decoder_flow_steps=decoder_flow_steps,
    )

    trainer = VideoTokenizerTrainer(
        model=tokenizer,
        dataset=dataset,
        checkpoint_path=latest_checkpoint,
        batch_size=batch_size,
        grad_accum_every=grad_accum_every,
        learning_rate=lr,
        num_train_steps=num_train_steps,
        num_workers=num_workers,
        pin_memory=True,
        use_ema=use_ema,
        ema_decay=ema_decay,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        log_dir=log_dir,
        log_video=True,
        video_fps=video_fps,
        log_video_every=log_video_every,
        checkpoint_every=checkpoint_every,
        checkpoint_folder=checkpoint_folder,
        project_name=experiment_name,
        run_name=run_name,
        accelerate_kwargs=dict(mixed_precision=mixed_precision),
    )

    if exists(latest_checkpoint):
        trainer.load(latest_checkpoint)

    trainer()


if __name__ == '__main__':
    fire.Fire(main)
