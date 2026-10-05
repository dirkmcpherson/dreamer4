#!/usr/bin/env bash
# Submits tokenizer -> {baseline, deep} dynamics -> rollout chains for synthetic MiniPushT and real PushT.
# Run on the pax login node from the run directory (next to job.sh): 6 jobs, 1 GPU each.
set -euo pipefail
root=/cluster/tufts/shortlab/jstale02/mini_dreamer4_gpu_2026-09-27_v1
cd "$root"
sub() { sbatch --parsable --output="$root/logs/%x_%j.out" "$@"; }
TOK="--patch-size 8 --num-latents 16 --latent-dim 32 --dim 256 --depth 4 --steps 30000 --batch-size 32 --lpips-weight 0.2 --log-every 500 --video-every 5000 --wandb"
DYN="--seq-len 16 --steps 30000 --batch-size 32 --k-max 64 --bootstrap-warmup 2000 --log-every 500 --video-every 5000 --wandb"

for ds in syn pusht; do
  if [ $ds = syn ]; then data="--data synthetic --image-size 64"; ttime=03:00:00; dtime=03:00:00
  else data="--data \$PUSHT --image-size 96"; ttime=06:00:00; dtime=05:00:00; fi
  tok=$(sub --job-name=md4_tok_$ds --time=$ttime job.sh \
    "\$PY -m mini_dreamer4.train tokenizer $data $TOK --run-name tok_$ds --out runs/$ds/tok")
  echo "tok_$ds $tok"
  for variant in base deep; do
    arch=""; [ $variant = deep ] && arch="--depth 8 --time-every 2"
    dyn=$(sub --job-name=md4_dyn_${ds}_$variant --time=$dtime --dependency=afterok:$tok job.sh \
      "\$PY -m mini_dreamer4.train dynamics $data --tokenizer runs/$ds/tok/tokenizer.pt $DYN $arch --run-name dyn_${ds}_$variant --out runs/$ds/dyn_$variant && \
       \$PY -m mini_dreamer4.train rollout $data --seq-len 16 --tokenizer runs/$ds/tok/tokenizer.pt --dynamics runs/$ds/dyn_$variant/dynamics.pt --out runs/$ds/dyn_$variant")
    echo "dyn_${ds}_$variant $dyn"
  done
done
