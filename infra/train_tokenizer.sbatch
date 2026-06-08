#!/usr/bin/env bash
#SBATCH --job-name=d4-libero-tok
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=logs_libero_tokenizer/slurm-%j.out
# Train the LIBERO video tokenizer on one 8xH200 node (DDP, bf16).
# Submit:  sbatch infra/train_tokenizer.sbatch
set -euo pipefail

# ------- config (edit or pass via --export) -------
ENV_NAME="${ENV_NAME:-dreamer4}"
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
DATA_DIR="${DATA_DIR:-/scratch/$USER/data/libero}"
SUITE="${SUITE:-libero_90}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
# WANDB_API_KEY should be exported in your environment / ~/.netrc

cd "$REPO_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# NCCL / IB (adjust interface to your cluster fabric)
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_HCA=mlx5

GPUS_PER_NODE=$(echo "${SLURM_GPUS_ON_NODE:-8}")

accelerate launch \
  --config_file infra/accelerate_config.yaml \
  --num_processes "$GPUS_PER_NODE" \
  train_libero_tokenizer.py \
    --data_dir "$DATA_DIR/$SUITE" \
    --num_train_steps "$NUM_TRAIN_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --logger wandb \
    --mixed_precision bf16 \
    --run_name "tok-$SUITE-$SLURM_JOB_ID"
