# Deploying dreamer4 → LIBERO on the H200 cluster

End-to-end recipe to train the dreamer4 video tokenizer + dynamics world model on
the LIBERO multi-task robot dataset and evaluate closed-loop in the LIBERO sim.

See `h200_readiness.md` for the full readiness audit and the remaining punch-list.

## 0. Files
| File | Purpose |
|---|---|
| `update_cluster_env.sh` | add dreamer4 + LIBERO/robosuite/MuJoCo deps to the `dreamer4` conda env |
| `download_libero.py` | resumable HuggingFace download of LIBERO HDF5 demos (suite-selectable) |
| `accelerate_config.yaml` | single-node multi-GPU (DDP, bf16) launch config |
| `train_tokenizer.sbatch` / `train_dynamics.sbatch` | SLURM jobs (1×8 H200) |
| `../dataset_libero.py` | LIBERO HDF5 → dreamer4 `Dataset` (tokenizer & dynamics modes) |
| `../train_libero_tokenizer.py` / `../train_libero_dynamics.py` | training entry points |
| `../eval_libero.py` | closed-loop eval harness (see caveats in the file header) |

## 1. Update the conda env
```bash
bash infra/update_cluster_env.sh dreamer4            # ENV_NAME defaults to dreamer4
# training-only (no simulator):  SKIP_SIM=1 bash infra/update_cluster_env.sh dreamer4
```
Leaves `torch` untouched; installs robosuite/MuJoCo (pins numpy<2, fine for dreamer4)
and clones+installs LIBERO. Verifies a headless EGL render at the end.

## 2. Download the dataset (~81 GB for the recommended multi-task set)
```bash
python infra/download_libero.py \
    --out /scratch/$USER/data/libero \
    --suites libero_90 libero_10          # 90-task train + long-horizon eval
# sizes: libero_90 ~67GB, libero_10 ~13GB, each 10-task suite ~7.5GB, all ~104GB
# quick POC:  --suites libero_object  (~7.8 GB)
```
Put it on scratch/fast storage; HDF5 random-access feeds the dataloader workers.

## 3. Train the tokenizer (1×8 H200)
```bash
DATA_DIR=/scratch/$USER/data/libero SUITE=libero_90 \
  sbatch infra/train_tokenizer.sbatch
```
Logs reconstruction videos (`original_video` / `reconstructed_video`) to wandb every
`--log_video_every` steps. Checkpoints (incl. `-ema.pt`) land in
`logs_libero_tokenizer/checkpoints`.

## 4. Train the dynamics world model
```bash
DATA_DIR=/scratch/$USER/data/libero SUITE=libero_90 \
  TOKENIZER_CKPT=./logs_libero_tokenizer/checkpoints \
  sbatch infra/train_dynamics.sbatch
```
Logs **forward-prediction** rollouts (`samples`) — generated from real prompt frames +
demo actions — to wandb, plus real-vs-generated GIFs under `logs_libero_dynamics/results`.

## 5. Closed-loop eval (headless)
```bash
MUJOCO_GL=egl python eval_libero.py \
    --suite libero_object \
    --dynamics_checkpoint ./logs_libero_dynamics/checkpoints/dynamics-300000.pt \
    --tokenizer_checkpoint ./logs_libero_tokenizer/checkpoints \
    --episodes_per_task 10
```
⚠️ Validate the eval acting path first — see the header of `eval_libero.py`
(BC-trained models may have an untrained policy head; live-env obs keys may differ).

## Knobs that matter on H200
- **Precision:** scripts set `torch.set_float32_matmul_precision('high')` (TF32) and pass
  `--mixed_precision bf16`. The accelerate config also requests bf16.
- **Dataloader:** `--num_workers` (default 8) + `pin_memory` are now wired through the
  trainers. Tune workers to `--cpus-per-task`.
- **Batch/throughput:** raise `--batch_size` to exploit 141 GB HBM3e once bf16 is stable;
  use `--grad_accum_every` if a single step won't fit.
- **wandb:** export `WANDB_API_KEY` (or `~/.netrc`); `moviepy` (installed by the env
  script) is required for mp4 video logging. Use `--logger tensorboard` to log locally.

## Multi-node (next step, not yet scripted)
`accelerate_config.yaml` is single-node. For N nodes, launch one task per node with
`--num_machines N --machine_rank $SLURM_NODEID --main_process_ip <rank0> --main_process_port 29500`
(or a `torchrun` rdzv wrapper), set `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA` for your fabric,
and keep checkpoints on a shared filesystem. FSDP for very large models needs the
checkpoint/EMA/Muon changes noted in `h200_readiness.md` (P2).

## Known gaps (tracked in h200_readiness.md)
- Checkpoints don't yet save optimizer/EMA/RNG/dataloader state → not preemption-safe (P1).
- `DynamicsWorldModel` lacks `@save_load`, so `eval_libero.py` may need the model
  reconstructed with the same constructor args before loading weights (P3).
