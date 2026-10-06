# dreamer4

## Agent Status

- **Status:** 🟢 active
- **Last session:** 2026-10-07 (morning EDT)
- **Current branch:** `claude/adoring-lovelace-rfe11h` (committed 6f95c35)
- **What happened:** Pixel policy on a Dreamer 4 tokenizer + dynamics model for PushT: BC on an agent token trained jointly with the dynamics. Best so far `runs/pusht/agent_r2b/agent.pt`: greedy 9/10 of the user's starts (~24 steps), sampled 78%. Online loop: on-policy rollouts (failures included) train the dynamics + success head; the policy is cloned from the tapes only (`:nobc` flag) — cloning noisy on-policy successes had broken two precision starts. Extrinsic check now agrees with reality: imagined vs real success per start correlates 0.90 (mean gap 0.12; was 0.25-0.31).
- **What's next:** User asked for the DreamerV3 recipe (world model + policy trained together). Built `tools/dreamer_loop.py`: per round collect real rollouts -> training chunk where every step updates the world model (dyn + success head + BC anchor on tapes) and every 4th step updates policy/value heads on imagined rollouts (PMPO, KL 0.1 to the BC prior, actor grads stop at features) -> real evaluation. gym-pusht now in the cluster overlay `deps2` (job.sh points there). Job 4976140 `md4_dreamer` -> `runs/pusht/dreamer_v1/` (4 rounds, 5000 model steps each, 20 collected/start, eval 10 sampled/start). Baseline r2b: greedy 9/10, sampled 72-78%.
- **Blocked on:** nothing (waiting on cluster job, ~2-3 h)
- **Key decisions made:** Cluster runs use a source snapshot + `deps/` pip overlay (zarr 2.18.3) instead of modifying the `dreamer4` conda env; `--qos=preempt` on L40S per existing convention; dynamics tested in two variants because the default `--depth 4 --time-every 4` has a single temporal layer. World models are judged by policy-relevant checks (state probe on imagined latents, counterfactual branches vs the real simulator), not PSNR. Keep local artifacts outside /tmp and cap memory on local runs (an npz loader bug of mine exhausted RAM on 2026-10-05).
