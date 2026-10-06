# dreamer4

## Agent Status

- **Status:** 🟢 active
- **Last session:** 2026-10-06 (morning EDT)
- **Current branch:** `claude/adoring-lovelace-rfe11h` (committed 6f95c35)
- **What happened:** World model (tokenizer + dynamics) works on PushT after shift-4 augmentation; dynamics retrained on the user's RLPD tapes (`runs/pusht/dyn_rlpd`, `dyn_mixed`: rollout +9.7 dB over copy-last, within ~1.3 dB of the tokenizer ceiling). User clarified the goal: shortest path to a policy that solves PushT using a Dreamer 4 tokenizer + dynamics model, ideally trained together with the dynamics. Built the paper-style agent stage: an agent token in the dynamics transformer (reads everything, nothing attends to it) with a categorical policy head (128 bins, y conditioned on x; demos are multi-modal) and a success head, trained jointly with the shortcut-forcing loss (`python -m mini_dreamer4.train agent`).
- **What's next:** Round 2 (`agent_r2`): greedy 8/10 with the same failures as round 1 (2000060, 2000173, both unimodal starts the original clone solved), sampled 64%. Hypothesis: imitating noisy on-policy successes degrades precision at those starts. Variant `agent_r2b` (job 4945503): same data, but on-policy rollouts marked `:nobc` (dynamics + success head only; policy cloned from tapes only), from `agent_rlpd_v2`. Local control run (no dynamics pretraining/loss) training on the RTX 3060 -> `local_artifacts/runs/agent_control/agent.pt`. Evaluate both with `eval_policy_pusht --refine` and `imagined_vs_real_policy`.
- **Blocked on:** nothing (cluster job + local control run in progress)
- **Key decisions made:** Cluster runs use a source snapshot + `deps/` pip overlay (zarr 2.18.3) instead of modifying the `dreamer4` conda env; `--qos=preempt` on L40S per existing convention; dynamics tested in two variants because the default `--depth 4 --time-every 4` has a single temporal layer. World models are judged by policy-relevant checks (state probe on imagined latents, counterfactual branches vs the real simulator), not PSNR. Keep local artifacts outside /tmp and cap memory on local runs (an npz loader bug of mine exhausted RAM on 2026-10-05).
