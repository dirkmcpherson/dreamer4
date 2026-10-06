# dreamer4

## Agent Status

- **Status:** 🟢 active
- **Last session:** 2026-10-06 (night EDT)
- **Current branch:** `claude/adoring-lovelace-rfe11h` (committed 6f95c35)
- **What happened:** Pixel policy on a Dreamer 4 tokenizer + dynamics model for PushT: BC on an agent token trained jointly with the dynamics. Best so far `runs/pusht/agent_r2b/agent.pt`: greedy 9/10 of the user's starts (~24 steps), sampled 78%. Online loop: on-policy rollouts (failures included) train the dynamics + success head; the policy is cloned from the tapes only (`:nobc` flag) — cloning noisy on-policy successes had broken two precision starts. Extrinsic check now agrees with reality: imagined vs real success per start correlates 0.90 (mean gap 0.12; was 0.25-0.31).
- **What's next:** Imagination training results: kl=1.0 (`local_artifacts/runs/imag_r2b_kl1`) leaves the policy essentially unchanged (KL to prior 0.008; greedy 9/10, sampled 78%, same as r2b). kl=0.1 variant running on the cluster (job 4966569 -> `runs/pusht/imag_r2b_kl01/agent.pt`); evaluate with `eval_policy_pusht --refine` and `imagined_vs_real_policy`. If still no change, lower KL further / raise LR / more iterations; watch for exploitation (imagined success up, real not). Then interleave imagine-train -> collect -> model retrain.
- **Blocked on:** nothing
- **Key decisions made:** Cluster runs use a source snapshot + `deps/` pip overlay (zarr 2.18.3) instead of modifying the `dreamer4` conda env; `--qos=preempt` on L40S per existing convention; dynamics tested in two variants because the default `--depth 4 --time-every 4` has a single temporal layer. World models are judged by policy-relevant checks (state probe on imagined latents, counterfactual branches vs the real simulator), not PSNR. Keep local artifacts outside /tmp and cap memory on local runs (an npz loader bug of mine exhausted RAM on 2026-10-05).
