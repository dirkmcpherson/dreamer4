# dreamer4

## Agent Status

- **Status:** 🟢 active
- **Last session:** 2026-10-05 15:15 EDT
- **Current branch:** `claude/adoring-lovelace-rfe11h` (committed 6f95c35)
- **What happened:** World model (tokenizer + dynamics) works on PushT after shift-4 augmentation; dynamics retrained on the user's RLPD tapes (`runs/pusht/dyn_rlpd`, `dyn_mixed`: rollout +9.7 dB over copy-last, within ~1.3 dB of the tokenizer ceiling). User clarified the goal: shortest path to a policy that solves PushT using a Dreamer 4 tokenizer + dynamics model, ideally trained together with the dynamics. Built the paper-style agent stage: an agent token in the dynamics transformer (reads everything, nothing attends to it) with a categorical policy head (128 bins, y conditioned on x; demos are multi-modal) and a success head, trained jointly with the shortcut-forcing loss (`python -m mini_dreamer4.train agent`).
- **What's next:** Job 4933363 `agent_rlpd` (30k steps, 10-frame windows because many tapes are 11-15 frames) writes `runs/pusht/agent_rlpd/agent.pt`. Fetch it and run closed-loop from pixels: `python -m mini_dreamer4.tools.eval_policy_pusht <tok_shift> <agent.pt> --starts ~/workspace/gp_wmfi_gym/experiments/pusht_visual_ic_2026-10-01/data/ic10.json` (success = coverage > 0.95 within 300 steps, 10 fixed starts; local, memory-capped). If BC succeeds, next is imagination training (value head + policy improvement inside the model) and/or an online loop.
- **Blocked on:** nothing (waiting on cluster job)
- **Key decisions made:** Cluster runs use a source snapshot + `deps/` pip overlay (zarr 2.18.3) instead of modifying the `dreamer4` conda env; `--qos=preempt` on L40S per existing convention; dynamics tested in two variants because the default `--depth 4 --time-every 4` has a single temporal layer. World models are judged by policy-relevant checks (state probe on imagined latents, counterfactual branches vs the real simulator), not PSNR. Keep local artifacts outside /tmp and cap memory on local runs (an npz loader bug of mine exhausted RAM on 2026-10-05).
