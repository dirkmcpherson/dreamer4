# dreamer4

## Agent Status

- **Status:** 🟡 paused
- **Last session:** 2026-10-07 (early EDT)
- **Current branch:** `claude/adoring-lovelace-rfe11h` (committed 6f95c35)
- **What happened:** Pixel policy on a Dreamer 4 tokenizer + dynamics model for PushT: BC on an agent token trained jointly with the dynamics. Best so far `runs/pusht/agent_r2b/agent.pt`: greedy 9/10 of the user's starts (~24 steps), sampled 78%. Online loop: on-policy rollouts (failures included) train the dynamics + success head; the policy is cloned from the tapes only (`:nobc` flag) — cloning noisy on-policy successes had broken two precision starts. Extrinsic check now agrees with reality: imagined vs real success per start correlates 0.90 (mean gap 0.12; was 0.25-0.31).
- **What's next:** Imagination training (frozen model, heads only, 1000 iters) does not improve real success: kl=1.0 leaves the policy unchanged (9/10, 76% n=20); kl=0.1 moves it (KL 0.09) and reshuffles the precision starts (2000060 -> 100%, 2000173 -> 0%): greedy 8/10, 69% n=20; imagined 86%. Awaiting user: run the full interleaved Dreamer loop (imagine -> collect -> model retrain, 2-3 rounds, ~1.5 h each), try a dense pixel-coverage reward in imagination, or stop with `agent_r2b`.
- **Blocked on:** user decision
- **Key decisions made:** Cluster runs use a source snapshot + `deps/` pip overlay (zarr 2.18.3) instead of modifying the `dreamer4` conda env; `--qos=preempt` on L40S per existing convention; dynamics tested in two variants because the default `--depth 4 --time-every 4` has a single temporal layer. World models are judged by policy-relevant checks (state probe on imagined latents, counterfactual branches vs the real simulator), not PSNR. Keep local artifacts outside /tmp and cap memory on local runs (an npz loader bug of mine exhausted RAM on 2026-10-05).
