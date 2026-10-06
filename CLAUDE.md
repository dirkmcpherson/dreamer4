# dreamer4

## Agent Status

- **Status:** 🟢 active
- **Last session:** 2026-10-06 (midday EDT)
- **Current branch:** `claude/adoring-lovelace-rfe11h` (committed 6f95c35)
- **What happened:** Pixel policy on a Dreamer 4 tokenizer + dynamics model for PushT: BC on an agent token trained jointly with the dynamics. Best so far `runs/pusht/agent_r2b/agent.pt`: greedy 9/10 of the user's starts (~24 steps), sampled 78%. Online loop: on-policy rollouts (failures included) train the dynamics + success head; the policy is cloned from the tapes only (`:nobc` flag) — cloning noisy on-policy successes had broken two precision starts. Extrinsic check now agrees with reality: imagined vs real success per start correlates 0.90 (mean gap 0.12; was 0.25-0.31).
- **What's next:** Round 3 submitted (`agent_r3`, from `agent_r2b`, + `onpolicy_r3.npz:nobc`). The missing arrow is model -> policy: imagination training (value head + policy improvement on imagined rollouts, paper phase 3), now justified by the 0.90 correlation. Local no-dynamics control run in `local_artifacts/runs/agent_control/` (compare greedy/sampled success with `agent_rlpd_v2`: 8/10, 76%).
- **Blocked on:** nothing (user decision on building imagination training)
- **Key decisions made:** Cluster runs use a source snapshot + `deps/` pip overlay (zarr 2.18.3) instead of modifying the `dreamer4` conda env; `--qos=preempt` on L40S per existing convention; dynamics tested in two variants because the default `--depth 4 --time-every 4` has a single temporal layer. World models are judged by policy-relevant checks (state probe on imagined latents, counterfactual branches vs the real simulator), not PSNR. Keep local artifacts outside /tmp and cap memory on local runs (an npz loader bug of mine exhausted RAM on 2026-10-05).
