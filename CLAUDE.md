# dreamer4

## Agent Status

- **Status:** 🟢 active
- **Last session:** 2026-10-06 (night EDT)
- **Current branch:** `claude/adoring-lovelace-rfe11h` (committed 6f95c35)
- **What happened:** Pixel policy on a Dreamer 4 tokenizer + dynamics model for PushT: BC on an agent token trained jointly with the dynamics. Best so far `runs/pusht/agent_r2b/agent.pt`: greedy 9/10 of the user's starts (~24 steps), sampled 78%. Online loop: on-policy rollouts (failures included) train the dynamics + success head; the policy is cloned from the tapes only (`:nobc` flag) — cloning noisy on-policy successes had broken two precision starts. Extrinsic check now agrees with reality: imagined vs real success per start correlates 0.90 (mean gap 0.12; was 0.25-0.31).
- **What's next:** Imagination training built (`tools/imagine_train.py`: frozen world model, policy + value heads trained on imagined rollouts with PMPO + KL to the BC prior). First run in progress locally: `local_artifacts/runs/imag_r2b_kl1/agent.pt` from `agent_r2b` (1000 iters, batch 64, horizon 16, kl 1.0). Then `eval_policy_pusht --refine` and `imagined_vs_real_policy` vs r2b (9/10 greedy, 78% sampled n=5, 72% n=20). Demo of r2b in `local_artifacts/demo_r2b/`. If imagination helps, interleave: imagine-train -> collect -> model retrain (`:nobc`) -> repeat.
- **Blocked on:** nothing
- **Key decisions made:** Cluster runs use a source snapshot + `deps/` pip overlay (zarr 2.18.3) instead of modifying the `dreamer4` conda env; `--qos=preempt` on L40S per existing convention; dynamics tested in two variants because the default `--depth 4 --time-every 4` has a single temporal layer. World models are judged by policy-relevant checks (state probe on imagined latents, counterfactual branches vs the real simulator), not PSNR. Keep local artifacts outside /tmp and cap memory on local runs (an npz loader bug of mine exhausted RAM on 2026-10-05).
