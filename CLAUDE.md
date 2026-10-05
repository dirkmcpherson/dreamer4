# dreamer4

## Agent Status

- **Status:** 🟡 paused
- **Last session:** 2026-10-02 (morning EDT)
- **Current branch:** `claude/adoring-lovelace-rfe11h` (uncommitted changes in `mini_dreamer4/`)
- **What happened:** GPU-tested `mini_dreamer4` on pax (run dir `mini_dreamer4_gpu_2026-09-27_v1`, 17 jobs). Breakthrough was shift-4 augmentation (`--shift 4`) on real PushT: tokenizer 30.1 -> 34.3 dB held-out with no train/held-out gap; dynamics (`runs/pusht/dyn_shift`, on `runs/pusht/tok_shift`) now tracks the block: centroid err 0.011 at 12 frames / 0.028 at 44 vs copy-last 0.031 / 0.099, displacement direction cosine 0.91-0.97, rank correlation 0.78-0.90, rollout PSNR gain +4.6 dB, wrong-action rollouts fall to copy-last level. White background + augmentation is good enough; the static texture is not needed.
- **What's next:** The dynamics model is now plausibly usable for policy learning on PushT. Next: counterfactual imagined-vs-real return test with the real gym-pusht simulator (on the cluster, `/cluster/tufts/shortlab/jstale02/gym-pusht`), then a minimal online agent stage (reward/policy/value heads + imagination training) with the tokenizer frozen. Nothing is committed yet (consider committing `mini_dreamer4/` changes).
- **Blocked on:** user decision on next step
- **Key decisions made:** Cluster runs use a source snapshot + `deps/` pip overlay (zarr 2.18.3) instead of modifying the `dreamer4` conda env; `--qos=preempt` on L40S per existing convention; dynamics tested in two variants because the default `--depth 4 --time-every 4` has a single temporal layer. World models are judged by policy-relevant checks (state probe on imagined latents, counterfactual branches vs the real simulator), not PSNR.
