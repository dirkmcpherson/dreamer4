#!/usr/bin/env bash
# Generic wrapper: sbatch [--job-name/--time/--output ...] job.sh <command...>
#SBATCH --partition=gpu,preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --requeue
set -euo pipefail
root=/cluster/tufts/shortlab/jstale02/mini_dreamer4_gpu_2026-09-27_v1
export PY=/cluster/tufts/shortlab/jstale02/condaenv/dreamer4/bin/python
export PYTHONPATH="$root/source:$root/deps"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME=/cluster/tufts/shortlab/jstale02/.cache/torch
export WANDB_DIR=/cluster/tufts/shortlab/jstale02/wandb_cache WANDB_CACHE_DIR=/cluster/tufts/shortlab/jstale02/wandb_cache WANDB_DATA_DIR=/cluster/tufts/shortlab/jstale02/wandb_cache
export PUSHT=/cluster/tufts/shortlab/jstale02/gym-pusht/demonstrations/pusht/pusht_cchi_v7_replay.zarr
cd "$root"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash -c "$*"
