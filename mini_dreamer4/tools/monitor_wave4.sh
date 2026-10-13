#!/bin/bash
# Monitor the wave-4 coverage-shaped imagination runs (2026-10-13) from any machine with the `pax` ssh alias (VPN up).
#   bash mini_dreamer4/tools/monitor_wave4.sh            # one readout
#   watch -n 600 bash mini_dreamer4/tools/monitor_wave4.sh
# Jobs: 5201387 shape_r2b_s0, 5201388 shape_r2b_s1 (k=0.5), 5201389 shape_r2b_k2 (k=2.0), 5201390 shape_weak_s0 (weak start, k=0.5).
# Baselines to beat (mean sampled success over rounds): frozen_long s0/s1 71% / 75%; weak control 33%.
R=/cluster/tufts/shortlab/jstale02/mini_dreamer4_gpu_2026-09-27_v1
ssh -o BatchMode=yes -o ConnectTimeout=20 pax "cd $R || exit 1
echo '=== queue'; squeue -u jstale02 -h -o '%i %j %T %M' | grep md4 || echo '(no md4 jobs running)'
for f in logs/md4_shape_*_52013*.out; do
  echo; echo \"=== \$f\"
  grep -E 'Traceback|Error' \$f | grep -v Warning | tail -2
  echo 'round greedy sampled peak_cov near_miss best(round)'
  grep -E 'real_greedy' \$f | grep 'round=' | sed -E 's/.*round=([0-9]+).*real_greedy=([0-9.]+).*real_sampled=([0-9.]+).*real_peak_coverage=([0-9.na]+).*real_near_miss=([0-9.]+).*/\1 \2 \3 \4 \5/' | tail -30 | column -t
  grep -E 'real_greedy' \$f | grep -oE 'best_sampled=[0-9.]+ best_round=[0-9]+' | tail -1
  echo -n 'latest imagination: '; grep -oE 'imagined_success=[0-9.]+|imagined_peak_coverage=[0-9.]+|shaping_reward=[0-9.e-]+|kl_to_prior=[0-9.]+|coverage_mae=[0-9.]+' \$f | tail -5 | tr '\n' ' '; echo
done
echo; echo '=== mean sampled success over rounds 13-24 (the retention number)'
for f in logs/md4_shape_*_52013*.out; do
  n=\$(basename \$f .out); grep -E 'real_greedy' \$f | grep 'round=' | awk -v n=\$n '{for(i=1;i<=NF;i++){if(\$i~/^round=/)r=substr(\$i,7); if(\$i~/^real_sampled=/)s=substr(\$i,14)} if(r>=13){sum+=s;c++}} END{if(c) printf \"%s  rounds 13-%d  mean %.1f%%\n\", n, 12+c, 100*sum/c; else printf \"%s  not yet at round 13\n\", n}'
done"
