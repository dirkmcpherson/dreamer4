# mini_dreamer4

A compact, paper-faithful PyTorch implementation of the Dreamer 4 world model
(Hafner, Yan, Lillicrap, *Training Agents Inside of Scalable World Models*, 2025),
written to be trainable and testable on a single task from pixels. It covers the
causal tokenizer and the shortcut-forcing dynamics model; the agent / imagination
stage is not included.

## What is implemented

| Paper component | Here |
|---|---|
| Efficient transformer: pre-norm RMSNorm, SwiGLU, QKNorm, logit soft capping, RoPE, axial space / time attention with temporal attention every 4th layer | `nn.py` (2D axial RoPE over the patch grid, 1D RoPE over time) |
| Causal tokenizer: patch + latent tokens, latents attend to all / patches to patches, linear + tanh bottleneck, decoder with learned patch queries, MAE with p ~ U(0, 0.9), MSE + 0.2 LPIPS, loss normalization | `tokenizer.py` (LPIPS optional, needs `pip install lpips`) |
| Dynamics: packed latents into spatial tokens, register tokens, (signal, step) token, action token with learned base embedding, diffusion forcing, shortcut forcing with x-prediction, bootstrap loss in v-space with (1-σ)² weight, ramp weight 0.9σ + 0.1, K = 4 sampling with context noise | `dynamics.py` |
| Continuous action input (e.g. PushT 2-D commands) or categorical actions | `ShortcutDynamics(action_dim=...)` / `num_discrete_actions=...` |

Deliberately not included: agent tokens, reward / policy / value heads, PMPO
imagination training, KV caching, alternating batch lengths, image-only batches.

### Deviations from the paper, and why

These were forced by experiments on the synthetic PushT task at CPU scale (see the tests);
each is a flag so the paper's behaviour can be restored.

* **Patch tokens are centered per frame** (`center_patches=True`): the per-frame mean over
  patch embeddings is subtracted before the encoder. With a mostly uniform background the
  shared component dominates the latent tokens (input-dependent variation was ~2% of their
  norm) and training sits on the mean-image plateau for thousands of steps; centering removed
  the plateau (foreground error fell 5x in 600 steps on an overfit probe).
* **Bootstrap fraction 25%** (`bootstrap_fraction=0.25`, `None` restores the paper's uniform
  sampling of the step size, i.e. ~6/7 of rows on bootstrap targets). With the paper's ratio a
  small model collapsed to self-consistent garbage as soon as the bootstrap loss switched on;
  25% is also what Hansen's and Hu's reimplementations and the shortcut-models paper use.
* **Bootstrap warmup** (`bootstrap_warmup`): during warmup every row is trained on the flow
  target *at its sampled step size*, so the step-size embeddings are trained before they are
  used to build bootstrap targets.
* **Context noise at inference defaults to 1 / K_max** (the finest trained level) rather than
  the paper's 0.1. With slow dynamics the noise injected into the context otherwise exceeds the
  per-step latent change; use `K_max = 64` (paper) rather than 8 for the same reason.

## Data

* `envs/pusht.py` — `MiniPushT`, a dependency-free PushT-style environment (round agent
  pushes a T block to a goal pose, RGB observations, 2-D continuous actions, coverage
  reward) plus `generate_episodes` for offline data and an *oracle tokenizer*
  (`state_to_oracle_latents`) used to test the dynamics model on its own.
* `data.py` — `EpisodeWindowDataset` (fixed-length windows) and `load_pusht_zarr`
  for the real PushT image dataset (`data/img`, `data/action`, `meta/episode_ends`).

Action convention: `actions[t]` is the action taken at frame `t`. The dynamics model
shifts them internally so that the token at time `t` carries the action that produced
frame `t`; the first frame gets a learned "no action" embedding.

## Training

```bash
python -m mini_dreamer4.train tokenizer --data synthetic --out runs/tok --steps 20000
python -m mini_dreamer4.train dynamics  --data synthetic --tokenizer runs/tok/tokenizer.pt --out runs/dyn --steps 20000
python -m mini_dreamer4.train rollout   --tokenizer runs/tok/tokenizer.pt --dynamics runs/dyn/dynamics.pt --out runs/dyn
```

Replace `--data synthetic` with the path to a PushT zarr to train on real PushT.
The dynamics log reports the two diagnostics that matter most for a single task: the
action-shuffle loss ratio (should rise clearly above 1) and the rollout PSNR gain over
a repeat-last-frame baseline.

## Tests

```bash
pytest mini_dreamer4/tests -q
```

`test_units.py` checks masks, position sensitivity, shortcut-loss bookkeeping and
sampling. `test_learning.py` trains small models on CPU and asserts that they learn:
tokenizer alone (reconstruction of the moving objects, not just the background),
dynamics alone on oracle latents (beats the copy-last-frame baseline and depends on
actions), and both together (action-conditioned rollouts decoded to pixels beat the
baseline).
