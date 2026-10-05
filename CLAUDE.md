# dreamer4

## Agent Status

- **Status:** 🟢 active
- **Last session:** 2026-10-05 16:20 EDT
- **Current branch:** `claude/adoring-lovelace-rfe11h` (committed 6f95c35)
- **What happened:** World model (tokenizer + dynamics) works on PushT after shift-4 augmentation; dynamics retrained on the user's RLPD tapes (`runs/pusht/dyn_rlpd`, `dyn_mixed`: rollout +9.7 dB over copy-last, within ~1.3 dB of the tokenizer ceiling). User clarified the goal: shortest path to a policy that solves PushT using a Dreamer 4 tokenizer + dynamics model, ideally trained together with the dynamics. Built the paper-style agent stage: an agent token in the dynamics transformer (reads everything, nothing attends to it) with a categorical policy head (128 bins, y conditioned on x; demos are multi-modal) and a success head, trained jointly with the shortcut-forcing loss (`python -m mini_dreamer4.train agent`).
- **What's next:** First BC agent (job 4933363) solved 1/10 starts: its first action was wrong by 150-300 px everywhere because my renderer drew each episode's first frame from stale simulator geometry (fixed: `plan_pusht.set_state_exact` re-indexes shapes; steps >= 1 were already within 2-8 px of the tapes). Corrected dataset `data/rlpd_pusht_v2.npz`; retraining as job 4935929 `agent_rlpd_v2` -> `runs/pusht/agent_rlpd_v2/agent.pt`. Then: `python -m mini_dreamer4.tools.eval_policy_pusht <tok_shift> <agent.pt> --starts .../ic10.json` (local, memory-capped). `local_artifacts/debug_policy.py` shows per-step on-tape action error and where closed loop leaves the tapes.
- **Blocked on:** nothing (waiting on cluster job)
- **Key decisions made:** Cluster runs use a source snapshot + `deps/` pip overlay (zarr 2.18.3) instead of modifying the `dreamer4` conda env; `--qos=preempt` on L40S per existing convention; dynamics tested in two variants because the default `--depth 4 --time-every 4` has a single temporal layer. World models are judged by policy-relevant checks (state probe on imagined latents, counterfactual branches vs the real simulator), not PSNR. Keep local artifacts outside /tmp and cap memory on local runs (an npz loader bug of mine exhausted RAM on 2026-10-05).
