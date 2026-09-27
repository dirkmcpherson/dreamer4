# GPU handoff for mini_dreamer4

State when this was written: everything in this package was developed and tested on a 4-core CPU
VM with no GPU (torch 2.14, CPU). Structural tests and the two single-stage learning tests pass;
the joint tokenizer+dynamics test is marked expected-fail. This document says exactly what to run
on a GPU, what to measure, and how to decide what to change.

## Setup

```bash
pip install torch torchvision lpips zarr pytest
pytest mini_dreamer4/tests/test_units.py -q                     # ~30 s, must pass
MINI_DREAMER4_TEST_SCALE=1 pytest mini_dreamer4/tests/test_learning.py -q -s   # ~15 min CPU, ~2 min GPU
```

The learning tests print their metrics with `-s`. `MINI_DREAMER4_TEST_SCALE=4` multiplies every
training budget by 4.

## Experiment 1: tokenizer at a realistic budget (the open question)

The joint test fails because latents from a briefly trained tokenizer are not temporally smooth:
consecutive-frame latent MSE ("copy-last") was 0.05-0.09, against 0.002 for the oracle latents,
so the action-driven part of a transition is buried. Training the tokenizer longer at CPU scale
(5000 steps) made copy-last *worse* (0.087), so more steps alone may not fix it.

Run, on synthetic data first:

```bash
python -m mini_dreamer4.train tokenizer --data synthetic --image-size 64 --patch-size 8 \
    --num-latents 16 --latent-dim 32 --dim 256 --depth 4 --steps 30000 --batch-size 32 \
    --lpips-weight 0.2 --out runs/tok64
```

Then measure, on held-out clips, before training any dynamics:

* `val_psnr` from the log (Hansen reaches 33-35 dB on 128x128 DMControl; expect > 30 dB here).
* latent copy-last MSE: `F.mse_loss(z[:, 1:], z[:, :-1])` with `z = tok.encode(video)`.
  Target: well below 0.01. If it stays around 0.05 with a good PSNR, the latents carry
  frame-specific nuisance variation and the dynamics stage will struggle.

Decision if copy-last stays high (both are small changes and both are outside the paper):
1. Lower the masking ceiling: `CausalTokenizer(mask_prob=(0.0, 0.5))`.
2. Add a latent temporal-smoothness penalty, e.g. `0.1 * F.mse_loss(z[:, 1:], z[:, :-1])` in
   `CausalTokenizer.forward`, or Hansen-style: nothing, and rely on much longer training.

## Experiment 2: dynamics on the learned latents

```bash
python -m mini_dreamer4.train dynamics --data synthetic --tokenizer runs/tok64/tokenizer.pt \
    --seq-len 16 --steps 30000 --batch-size 32 --k-max 64 --bootstrap-warmup 2000 --out runs/dyn64
```

The log prints the two diagnostics that matter:

* `action_shuffle_ratio`: loss with shuffled actions divided by loss with true actions on held-out
  data. Hansen's runs sit at 1.5-2.5. Below 1.2 means the model is ignoring actions.
* `rollout_psnr_gain_over_floor`: decoded 4-step rollouts vs repeating the last context frame.
  Hansen's runs show +1 to +2 dB at a 16-frame horizon. Negative means not yet a world model.

Then `python -m mini_dreamer4.train rollout ...` saves `rollout.pt` with generated and true frames
for visual inspection.

## Experiment 3: real PushT

`mini_dreamer4.data.load_pusht_zarr` reads the diffusion-policy / lerobot layout
(`data/img` uint8 (N,96,96,3), `data/action` (N,2) in pixel units 0-512, `meta/episode_ends`).
It has not been run against a real file; check shapes and the action range first. Pass
`--data path/to/pusht_cchi_v7_replay.zarr --image-size 96 --patch-size 8` (144 patches).

## Things not implemented

Agent tokens, reward / policy / value heads, PMPO imagination training, KV caching, alternating
batch lengths, image-only batches, GQA. See README.md for the deviations that were forced by
the CPU experiments (patch centering, 25% bootstrap fraction, warmup, context noise level).
