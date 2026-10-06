# dreamer4

## Agent Status

- **Status:** 🟢 active
- **Last session:** 2026-10-06 (evening EDT)
- **Current branch:** `claude/adoring-lovelace-rfe11h` (committed 6f95c35)
- **What happened:** Pixel policy on a Dreamer 4 tokenizer + dynamics model for PushT: BC on an agent token trained jointly with the dynamics. Best so far `runs/pusht/agent_r2b/agent.pt`: greedy 9/10 of the user's starts (~24 steps), sampled 78%. Online loop: on-policy rollouts (failures included) train the dynamics + success head; the policy is cloned from the tapes only (`:nobc` flag) — cloning noisy on-policy successes had broken two precision starts. Extrinsic check now agrees with reality: imagined vs real success per start correlates 0.90 (mean gap 0.12; was 0.25-0.31).
- **What's next:** Round 3 (`agent_r3`) regressed to greedy 7/10 (2000060 0/5 again), sampled 68%; imagined-vs-real correlation 0.41. Across rounds the sampled real success is flat (71/67/67/72/66% at n=20): the loop improves the model's coverage of each policy's failures but does NOT improve the policy; greedy counts swing by 2 starts between rounds. Best checkpoint remains `agent_r2b` (9/10, 78%). Awaiting user: build imagination training (model -> policy arrow) or stop here. Control: no-dynamics policy 7/10 greedy, 62% sampled vs 8/10 / 76% (one seed).
- **Blocked on:** user decision (imagination training vs. stop/iterate)
- **Key decisions made:** Cluster runs use a source snapshot + `deps/` pip overlay (zarr 2.18.3) instead of modifying the `dreamer4` conda env; `--qos=preempt` on L40S per existing convention; dynamics tested in two variants because the default `--depth 4 --time-every 4` has a single temporal layer. World models are judged by policy-relevant checks (state probe on imagined latents, counterfactual branches vs the real simulator), not PSNR. Keep local artifacts outside /tmp and cap memory on local runs (an npz loader bug of mine exhausted RAM on 2026-10-05).
