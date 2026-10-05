# dreamer4

## Agent Status

- **Status:** 🟡 paused
- **Last session:** 2026-10-05 17:45 EDT
- **Current branch:** `claude/adoring-lovelace-rfe11h` (committed 6f95c35)
- **What happened:** World model (tokenizer + dynamics) works on PushT after shift-4 augmentation; dynamics retrained on the user's RLPD tapes (`runs/pusht/dyn_rlpd`, `dyn_mixed`: rollout +9.7 dB over copy-last, within ~1.3 dB of the tokenizer ceiling). User clarified the goal: shortest path to a policy that solves PushT using a Dreamer 4 tokenizer + dynamics model, ideally trained together with the dynamics. Built the paper-style agent stage: an agent token in the dynamics transformer (reads everything, nothing attends to it) with a categorical policy head (128 bins, y conditioned on x; demos are multi-modal) and a success head, trained jointly with the shortcut-forcing loss (`python -m mini_dreamer4.train agent`).
- **What's next:** RESULT: the behaviour-cloned agent `runs/pusht/agent_rlpd_v2/agent.pt` (local `local_artifacts/ckpt/agent_rlpd_v2.pt`, tokenizer `tok_shift`) solves 8/10 of the user's PushT starts greedily from pixels with `--refine` (7/10 without; sampled 76%), ~22 steps. Failures are threshold near-misses (coverage 0.950 and 0.922 vs > 0.95) with no recovery behaviour, since all demos end at success. It is BC only: the dynamics model is the shared trunk, not used for imagination or planning at decision time. Options awaiting user: (a) online loop (roll out, add data, retrain dynamics + policy together), (b) imagination training with a value head, (c) more seeds / evaluate ic3, ic1.
- **Blocked on:** user decision on next step
- **Key decisions made:** Cluster runs use a source snapshot + `deps/` pip overlay (zarr 2.18.3) instead of modifying the `dreamer4` conda env; `--qos=preempt` on L40S per existing convention; dynamics tested in two variants because the default `--depth 4 --time-every 4` has a single temporal layer. World models are judged by policy-relevant checks (state probe on imagined latents, counterfactual branches vs the real simulator), not PSNR. Keep local artifacts outside /tmp and cap memory on local runs (an npz loader bug of mine exhausted RAM on 2026-10-05).
