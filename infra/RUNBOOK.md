# dreamer4 → LIBERO: Cluster Runbook

Step-by-step to update the conda env, smoke-test the full pipeline, and train on
an H200 cluster. Run commands from the repo root with the `dreamer4` conda env.
Companion docs: `README.md` (overview), `h200_readiness.md` (audit/punch-list).

> **Pre-flight**
> - pip installs + dataset download need **internet**; the env render-check and all
>   training need a **GPU + EGL**. If compute nodes lack internet, do Steps 1–2 on the
>   login node and the GPU steps in an interactive allocation:
>   ```bash
>   srun --gres=gpu:1 --cpus-per-task=8 --pty bash
>   ```
> - For smoke tests, `WANDB_MODE=offline` avoids a wandb login while still exercising
>   the video-logging path. Use real wandb (`wandb login`) for full runs.

---

## Step 1 — Update the conda env
```bash
bash infra/update_cluster_env.sh dreamer4          # ENV_NAME defaults to dreamer4
# training-only (skip simulator/LIBERO):  SKIP_SIM=1 bash infra/update_cluster_env.sh dreamer4
```
Brings dreamer4 deps to required versions (torch untouched), installs
robosuite/MuJoCo + LIBERO, verifies a headless render.

**Success:**
```
dreamer4 import OK
robosuite 1.5.2 | mujoco 3.9.0
EGL offscreen render OK: (128, 128, 3)
```
**EGL line fails →** you're on a node without GPU/EGL. Rerun in the `srun` shell, or use
`SKIP_SIM=1` on the login node and install the sim separately on a GPU node.

---

## Step 2 — Download a small suite first (smoke), big suite later
```bash
python infra/download_libero.py --out /scratch/$USER/data/libero --suites libero_object   # ~7.8 GB
```
Hold off on the ~81 GB `libero_90 libero_10` pull until the smoke passes. Put data on
fast scratch storage.

---

## Step 3 — Smoke-test the DATA (catches the #1 risk: HDF5 key names)
```bash
python dataset_libero.py /scratch/$USER/data/libero/libero_object
```
**Success:**
```
files=10 windows=NNNN dim_proprio=9
  video                (3, 16, 128, 128) torch.float32  range[0.000,1.000]
  continuous_actions   (16, 7) torch.float32 ...
  rewards              (16,)   torch.float32 ...
  proprio              (16, 9) torch.float32 ...
```
**`KeyError` on an obs key →** the HDF5 layout differs from the assumed robomimic keys;
adjust the key names in `dataset_libero.py` (camera / proprio_keys).

---

## Step 4 — Smoke-test TOKENIZER training (1 GPU, ~1 min)
```bash
WANDB_MODE=offline python train_libero_tokenizer.py \
  --data_dir /scratch/$USER/data/libero/libero_object \
  --num_train_steps 60 --batch_size 4 --num_workers 2 \
  --log_video_every 30 --checkpoint_every 30 \
  --log_dir ./smoke_tok --checkpoint_folder ./smoke_tok/checkpoints
```
**Watch for:** per-step loss **trending down**; a recon video at step 30; and
**`smoke_tok/checkpoints/tokenizer-30-ema.pt` written** (needed by Step 6).
**Loss `nan` on step 1 →** lower `--lr 1e-4` and recheck Step 3's `range[0,1]`.

---

## Step 5 — Smoke-test the MULTI-GPU launch (2 GPUs, ~1 min)
Confirm DDP before committing an 8-GPU sbatch:
```bash
WANDB_MODE=offline accelerate launch \
  --config_file infra/accelerate_config.yaml --num_processes 2 \
  train_libero_tokenizer.py \
  --data_dir /scratch/$USER/data/libero/libero_object \
  --num_train_steps 40 --batch_size 4 --num_workers 2 \
  --log_video_every 0 --checkpoint_every 0 --log_dir ./smoke_ddp
```
(`--log_video_every 0` / `--checkpoint_every 0` disable those during the smoke.)
**Success:** two processes start, single rank-0 progress bar, no NCCL hang.
**Hang →** set `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` for your fabric (see the sbatch files).

---

## Step 6 — Smoke-test DYNAMICS training (needs Step 4's tokenizer checkpoint)
```bash
WANDB_MODE=offline python train_libero_dynamics.py \
  --data_dir /scratch/$USER/data/libero/libero_object \
  --tokenizer_checkpoint_path ./smoke_tok/checkpoints \
  --num_train_steps 60 --batch_size 4 --num_workers 2 \
  --log_video_every 30 --checkpoint_every 30 \
  --log_dir ./smoke_dyn --checkpoint_folder ./smoke_dyn/checkpoints
```
**Success:** "loading frozen tokenizer…"; flow/action losses printed; a `samples`
**forward-prediction** video + real-vs-generated GIF under `smoke_dyn/results`; a
`dynamics-30.pt` checkpoint.

If Steps 1–6 pass, the pipeline is sound — delete the `smoke_*` dirs and go full-scale.

---

## Step 7 — Full training (8×H200 via SLURM)
```bash
# real data + wandb
python infra/download_libero.py --out /scratch/$USER/data/libero --suites libero_90 libero_10
wandb login                          # or export WANDB_API_KEY=...

# tokenizer first
DATA_DIR=/scratch/$USER/data/libero SUITE=libero_90 sbatch infra/train_tokenizer.sbatch
squeue -u $USER ; tail -f logs_libero_tokenizer/slurm-*.out

# then dynamics (after the tokenizer has checkpointed)
DATA_DIR=/scratch/$USER/data/libero SUITE=libero_90 \
  TOKENIZER_CKPT=./logs_libero_tokenizer/checkpoints \
  sbatch infra/train_dynamics.sbatch
```
Monitor reconstruction quality (tokenizer) and forward-prediction rollouts (dynamics)
in wandb. Tune `--batch_size` up to exploit 141 GB HBM3e; `--grad_accum_every` if needed.

---

## Step 8 — Closed-loop eval (after a dynamics checkpoint exists)
```bash
MUJOCO_GL=egl python eval_libero.py \
  --suite libero_object \
  --dynamics_checkpoint ./logs_libero_dynamics/checkpoints/dynamics-XXXX.pt \
  --tokenizer_checkpoint ./logs_libero_tokenizer/checkpoints \
  --episodes_per_task 2 --num_tasks 1
```
**Validate, don't trust the number yet** (see `eval_libero.py` header): confirm the env
builds, print `obs.keys()` if keys mismatch, and eyeball whether the agent acts sensibly —
a BC-trained model may have an untrained policy head.

---

## Troubleshooting map
| Symptom | Likely cause | Fix |
|---|---|---|
| Step 1 EGL render fails | no GPU/EGL on node | run in `srun` GPU shell |
| Step 3 `KeyError` on obs | HDF5 keys differ | adjust `camera`/`proprio_keys` in `dataset_libero.py` |
| Step 4 loss `nan` at step 1 | lr / normalization | `--lr 1e-4`, recheck `range[0,1]` |
| Step 5 NCCL hang | wrong network iface | set `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` |
| Step 6 "no EMA tokenizer" | smoke too short | `--checkpoint_every` ≤ `--num_train_steps`, `use_ema` on |
| Step 8 env build fails | LIBERO wants robosuite 1.4.1 | pinned eval env (see `update_cluster_env.sh` notes) |

## Known gaps (see h200_readiness.md)
- Checkpoints don't save optimizer/EMA/RNG/dataloader state → not preemption-safe (P1).
- `DynamicsWorldModel` lacks `@save_load`; `eval_libero.py` may need the model
  reconstructed with the same constructor args before loading weights (P3).
- Single-node launch config only; multi-node notes in `README.md`.
