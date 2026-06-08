# Dreamer4 — H200 Cluster Readiness Assessment

Status: living document. Last updated 2026-06-08. Source: read-only audit of
`/home/james/workspace/dreamer4` at commit `be59556`. No code was changed.

This assesses what it takes to run dreamer4 training (`VideoTokenizer` +
`DynamicsWorldModel`) on an NVIDIA H200 (Hopper, sm_90, 141 GB HBM3e) cluster,
first multi-GPU single-node, then multi-node.

---

## TL;DR — top blockers

1. **Environment** — *RESOLVED on this dev machine (2026-06-08).* The repo `.venv`
   (shared robotics env, Python 3.10, torch 2.9.1+cu128 / accelerate 1.12.0) was
   originally **missing required deps** and had **below-minimum pins**, so
   `import dreamer4` failed. Fixed in place by installing the missing packages and
   upgrading the low ones (torch left untouched) — see "Environment fix applied"
   below. `import dreamer4` + a tiny tokenizer forward/backward now succeed, CUDA
   available. **Still open for the cluster:** reproduce this as a clean,
   fully-pinned env from `pyproject.toml` with a lockfile (don't rely on the
   shared dev `.venv`).
2. **PoPE flash kernel (`flash_attn_with_pope`) and compiled `flex_attention`
   are unvalidated on sm_90.** These are the two attention fast-paths. They must
   be smoke-tested on an actual H200 before any large run; otherwise fall back to
   the naive SDPA path.
3. **No preemption-safe resume.** Checkpoints save only `model + dehydrated
   config + step`. Optimizer state, EMA weights' optimizer, RNG, dataloader
   position, and accelerate state are **not** saved. A preempted job restarts
   the optimizer from scratch — unacceptable on a shared/spot cluster.
4. **Data pipeline does not scale.** All `DataLoader`s use `num_workers=0`, no
   `pin_memory`, no prefetch, no sharding/streaming/webdataset. Fine for
   in-memory Moving-MNIST; a hard bottleneck for real video at H200 throughput.
5. **No multi-GPU launch story committed.** DDP is wired through `accelerate`,
   but there is no `accelerate` config, no launch script, no FSDP plugin. Large
   models that exceed one GPU need FSDP, which the EMA/checkpoint code does not
   yet support.

---

## 1. Entry points & config system

- **Core package:** `dreamer4/dreamer4.py` (~6557 lines) defines `VideoTokenizer`,
  `DynamicsWorldModel`, `AxialSpaceTimeTransformer`, `Attention`, `SelfFlow`,
  `Experience`, etc. `dreamer4/trainers.py` defines `VideoTokenizerTrainer`,
  `BehaviorCloneTrainer`, `DreamTrainer`, `SimTrainer`.
- **Entry points are standalone `fire` scripts**, not a unified CLI:
  - `train_moving_mnist_tokenizer.py`, `train_moving_mnist_dynamics.py`,
    `train_cartpole_with_dynamics_rl.py`, `test_toy_action_bc.py`.
  - Each carries **PEP-723 inline script metadata** (`# /// script ... ///`) with
    `[tool.uv.sources] dreamer4 = { path = "." }` — designed to be run with
    `uv run`. Config is plain `fire` keyword args (`num_frames`, `image_size`,
    `batch_size`, `lr`, …). There is **no YAML/Hydra/argparse config layer** and
    no central experiment config.
- **`scripts/train_dreamer.py` is broken/aspirational.** It does
  `from dreamer4._core import build_pinpad_datasets_for_trainers, DataLoader` and
  `from dreamer4._tokenizer import SimpleConvTokenizer` — **neither
  `dreamer4._core` nor `dreamer4._tokenizer` exists** in the package. This script
  cannot run. It is also the only place that passes
  `accelerate_kwargs={'mixed_precision':'bf16'}`. Do not use it as a launch
  reference until those modules are provided.
- **No `accelerate` config file, no launch `.sh`, no Dockerfile, no
  requirements/lock file** are present in the repo root.

Implication: for a cluster you will add a thin launch layer (accelerate config +
sbatch/torchrun wrapper). The `fire`-style `main(...)` functions are easy to call
under `accelerate launch`, but you must thread an `accelerate_kwargs` /
`mixed_precision` argument through (most example scripts hard-omit it, so they
default to fp32).

## 2. Device / dtype handling and single-GPU assumptions

- **Device handling is clean and accelerate-driven.** Models are constructed on
  CPU and moved by `accelerator.prepare`. Internally the code consistently uses
  `tensor.device` / `self.device` (= `accelerator.device`) for new tensors; there
  are **no hard-coded `.cuda()` calls**. `self.device` on models resolves through
  module params. Good — no single-GPU device pinning to undo.
- **Dtype:** there is **no manual `autocast`, no `GradScaler`, no
  `set_float32_matmul_precision`, and no `torch.backends.*.allow_tf32`** anywhere
  in `dreamer4/` or the example scripts. Precision is entirely delegated to
  `accelerate`'s `mixed_precision`, which **defaults to `'no'` (fp32)** unless a
  trainer is given `accelerate_kwargs={'mixed_precision':'bf16'}`. Today only the
  broken `scripts/train_dreamer.py` sets that. So the example training paths run
  **fp32 with TF32 disabled** — leaving most H200 throughput on the table.
- **Single-GPU assumptions that matter under multi-GPU:**
  - EMA is created **only on the main process** (`if use_ema and
    is_main_process`) and is **not** passed to `accelerator.prepare`. Fine under
    DDP (EMA is a main-rank bookkeeping copy), but it tracks `self.model`
    (the DDP-wrapped module) and lives only on rank 0.
  - Checkpoint save/load is guarded by `is_main_process` and uses
    `unwrap_model` — correct for DDP, **incorrect for FSDP** (needs a full
    state-dict gather).
  - RL trainers (`DreamTrainer`, `SimTrainer`) call
    `self.unwrapped_model.generate(...)` / `interact_with_env(...)`. Every rank
    would generate/roll out independently and redundantly; env interaction is not
    sharded or coordinated. These trainers are effectively single-process today.

## 3. Distributed readiness

- **DDP is largely wired** via `accelerate`: every trainer uses
  `Accelerator(...)`, `accelerator.prepare(model, dataloader, optim)`,
  `accelerator.backward(loss)`, `accelerator.clip_grad_norm_(...)`,
  `is_main_process`, `wait_for_everyone()`, `unwrap_model`, and tracker logging.
  `accelerate.prepare` auto-shards the `DataLoader` across ranks. So
  **single-node multi-GPU DDP via `accelerate launch` / `torchrun` should work**
  for the supervised trainers (`VideoTokenizerTrainer`, `BehaviorCloneTrainer`)
  once the env is fixed.
- **Multi-node needs:** (a) an `accelerate` config or env vars
  (`MACHINE_RANK`, `MAIN_PROCESS_IP/PORT`, `NUM_MACHINES`, `NUM_PROCESSES`) or an
  equivalent `torchrun --nnodes/--node_rank/--rdzv_*` wrapper, plus NCCL setup
  (`NCCL_SOCKET_IFNAME`, IB/`NCCL_IB_HCA`); (b) a Slurm sbatch (or similar)
  launcher; (c) shared-filesystem checkpoint paths. None of this is in-repo yet.
- **FSDP for large models is not supported by the current code.** To train a
  `DynamicsWorldModel` too large for one H200 you'd enable accelerate's FSDP
  plugin, but:
  - checkpointing uses `unwrap_model(...).state_dict()` on main only → must switch
    to `accelerator.get_state_dict` / `FullStateDictConfig` gather.
  - EMA-on-rank-0-only is incompatible with sharded params.
  - `MuonAdamAtan2` (Muon) does Newton–Schulz orthogonalization on 2-D weight
    matrices; under FSDP params are flattened/sharded, which can break the
    per-matrix Muon update unless the optimizer is FSDP-aware. Validate Muon +
    FSDP carefully, or keep Muon params replicated (DDP) and only FSDP the rest.
  - Models expose `muon_parameters()` and the optimizer splits Muon vs
    AdamAtan2 groups — preserve that grouping through any sharding wrapper.
- **`grad_accum_every` is implemented manually** (loss divided by accum, N
  backward calls) **without `accelerator.accumulate()`/`no_sync()`**, so under DDP
  every micro-step triggers a gradient all-reduce instead of only the last —
  correct results but extra communication. Wrap micro-steps in
  `accelerator.no_sync()` for all but the last to cut DDP comm by `grad_accum_every`x.

## 4. Dependency / CUDA concerns on H200

- **torch 2.9.1 is present** (well above the `>=2.4` floor) and ships CUDA 12.x
  wheels with sm_90 support — good for Hopper. `accelerate 1.12.0` supports FSDP2.
- **The active `.venv` is the wrong/incomplete env** (see TL;DR #1): missing
  `PoPE_pytorch`, `x_transformers`, `h_net_dynamic_chunking`, `wandb`,
  `tensorboard`; several deps below `pyproject` minimums. `import dreamer4` fails
  here because `dreamer4.py` does a top-level `from PoPE_pytorch import PoPE,
  flash_attn_with_pope`. **Action: build a fresh, fully-pinned env from
  `pyproject.toml` on the cluster** (prefer `uv` given the PEP-723 scripts) and
  generate a lockfile for reproducibility.
- **`flex_attention` is `torch.compile`d at import** (`dreamer4.py` ~L94-101) when
  CUDA is available, with `torch._dynamo.config.cache_size_limit = 256`. On
  Hopper + torch 2.9 this generally works, but: (a) first-iteration compile
  latency is significant; (b) each new `(seq_len, kv_len, block_mask)` shape can
  trigger recompiles — the 256 cache limit hints they already hit this. Video
  axial attention varies spatial/temporal seq lengths, so expect recompile churn.
  Consider fixed/bucketed sequence shapes.
- **`flash_attn_with_pope` (from `PoPE_pytorch`) is the other fast attention
  path** (`dreamer4.py` L1758), used whenever rotary/PoPE positions are present;
  it passes `head_dimension_at_first=True` and `causal=True`. This is a custom
  flash kernel whose sm_90 support and numerical behavior in bf16 are **unknown
  and must be smoke-tested on the H200**. If it fails, the code has a naive SDPA
  fallback (`naive_attend` → `F.scaled_dot_product_attention`), but only on the
  non-PoPE branch — verify there is a working path that doesn't require the PoPE
  kernel.
- **`MuonAdamAtan2`** (`adam_atan2_pytorch` 0.2.4): Muon's Newton–Schulz
  iterations run matmuls that are fp32/bf16-friendly on H200; main risks are FSDP
  interaction (above) and any multi-GPU all-gather of Muon momentum (confirm the
  lib distributes or that you keep Muon params replicated).
- **No flash-attn (FA2/FA3) package is installed** and none is in `pyproject`.
  If you want FA3 on Hopper you'd add it explicitly; today attention is
  flex_attention (compiled) + PoPE kernel + SDPA fallback only.

## 5. Data pipeline scalability

- Datasets are **simple in-memory `torch.utils.data.Dataset`s**
  (`dataset_moving_mnist.py` generates frames on the fly; RL uses `TensorDataset`
  over collected experience). Good for toy runs, not for real corpora.
- **Every `DataLoader` uses defaults: `num_workers=0`, no `pin_memory`, no
  `persistent_workers`, no `prefetch_factor`.** Grep confirms none of these
  kwargs appear anywhere. At H200 compute rates a single-process,
  synchronous loader will starve the GPU on any non-trivial decode/augment.
- **No sharding / streaming / `webdataset` / `IterableDataset`**, and no
  `sampler.set_epoch(...)`. Trainers wrap the loader in an infinite
  `cycle(dataloader)` generator, so epoch boundaries (and thus
  `DistributedSampler.set_epoch`) are never signaled — shuffling can repeat the
  same per-rank order across passes. `accelerate.prepare` shards the loader, but
  the lack of `set_epoch` hurts shuffle decorrelation on long runs.
- `drop_last=True, shuffle=True` are set, which is fine for DDP evenness.
- **Action for scale:** move to a sharded streaming format (webdataset / tar
  shards or Mosaic StreamingDataset), set `num_workers` (8–16/GPU),
  `pin_memory=True`, `persistent_workers=True`, `prefetch_factor`, and thread
  `set_epoch`. Keep raw video off the in-memory path.

## 6. Checkpoint / resume / preemption robustness

- **Save (`save_checkpoint` / inline in `VideoTokenizerTrainer`):** writes a
  `.pt` with `dict(model=state_dict, config=pickle(dehydrate_config(...)),
  step=step)`; EMA saved to a sibling `-ema.pt`. Save is gated on
  `is_main_process` and uses `unwrap_model`.
- **Load (`load`)**: `torch.load(..., weights_only=True)`, copies `step`, then
  `load_state_dict(..., strict=False)`. `strict=False` will **silently ignore
  missing/extra keys** — a real risk of partially-loaded models going unnoticed.
- **Not saved → not preemption-safe:**
  - **Optimizer state** (Adam moments, Muon momentum) — lost on resume.
  - **LR schedule / step-conditioned hyperparams** — only the integer `step` is
    restored; there is no scheduler object anyway.
  - **RNG state** (torch/cuda/numpy/python) — no reproducible resume.
  - **Dataloader position** — `cycle()` restarts from a fresh shuffle.
  - **accelerate state** — `Accelerator.save_state/load_state` is not used.
  - **EMA optimizer/decay step count** beyond raw weights.
- **`DynamicsWorldModel` has no `@save_load` decorator** (only `VideoTokenizer`
  at L2922 does). So `getattr(model,'_config',None)` is `None` for the world
  model → checkpoints store `config=None`, and there is **no `init_and_load`**
  for it. You must reconstruct the model with the exact constructor args before
  `load()`. The tokenizer, by contrast, supports `VideoTokenizer.init_and_load`.
- **No auto-resume / "latest" pointer**, no atomic write (write-tmp-then-rename),
  no retention policy. On a spot/preemptible H200 queue this means lost progress.
- **Action:** switch to `accelerator.save_state()/load_state()` (captures model,
  optimizer, RNG, and—if registered—custom objects), register EMA + a `step`
  buffer with `accelerator.register_for_checkpointing`, write atomically, keep a
  `latest` symlink, and add `--resume` auto-detection. Tighten model load to
  `strict=True` (or log diffs).

## 7. H200-specific opportunities

- **Enable TF32 globally:** `torch.set_float32_matmul_precision('high')` (or
  `'medium'`) + `torch.backends.cuda.matmul.allow_tf32 = True` — free ~1.3-2x on
  fp32 matmuls; currently nothing sets these.
- **Default to bf16 mixed precision** across all trainers (pass
  `mixed_precision='bf16'` into every `Accelerator`, not just the broken script).
  Hopper bf16 is the right default for stability vs fp16.
- **`torch.compile` the models, not just `flex_attention`.** Only the attention
  kernel is compiled today; wrapping the transformer forward (or using
  `accelerator` with `dynamo_backend`) can give large gains — but reconcile with
  the variable axial sequence shapes (bucket shapes to limit recompiles).
- **FP8** (transformer-engine or torchao float8) is a Hopper headline feature and
  is **entirely unused**; a candidate for the largest matmuls once bf16 is stable.
- **FA3 / Hopper flash kernels** are not wired; the PoPE kernel and flex_attention
  are the current paths. Validate them first; FA3 is an optimization, not a
  blocker.
- **Activation/gradient checkpointing is absent** (`torch.utils.checkpoint`
  appears nowhere). For deep `AxialSpaceTimeTransformer` stacks at long video
  context this is the main lever to fit larger batches in 141 GB and is a
  prerequisite for very large models even before FSDP.
- **Exploit 141 GB HBM3e** with larger batch / longer context once bf16 +
  checkpointing land; revisit `grad_accum_every` downward as real batch grows.

## 8. Logging / experiment tracking

- Trainers support **tensorboard OR wandb** (mutually exclusive, asserted) via
  `accelerate` trackers; video samples logged to TB/wandb and saved as GIFs on
  the main process. Run naming is supported for wandb.
- **Gaps:** neither `wandb` nor `tensorboard` is installed in the current env.
  No throughput/MFU/GPU-mem/grad-norm metrics are logged from the trainers
  (grad-norm probing exists only in `tests/debug_grad_norm.py`). For cluster runs
  add: tokens|frames/sec, step time, GPU mem, grad-norm, LR, and loss-scale.
- Logging is correctly main-process-gated, so it's multi-node safe as-is.

## 9. Prioritized punch-list (highest impact first)

**P0 — required to launch any multi-GPU H200 job**
1. **Build a correct, pinned environment** from `pyproject.toml` on the cluster
   (uv + lockfile). Verify `import dreamer4` succeeds and `PoPE_pytorch`,
   `x_transformers`, `h_net_dynamic_chunking`, `wandb`/`tensorboard` are present
   at/above the required versions. The repo `.venv` is the wrong env.
2. **Smoke-test the attention fast-paths on a real H200**: compiled
   `flex_attention` and `flash_attn_with_pope` (sm_90, bf16). Confirm a working
   fallback if either misbehaves.
3. **Add an `accelerate` config + launch wrapper** (single-node multi-GPU first):
   `accelerate launch`/`torchrun`, bf16, NCCL env. Use
   `BehaviorCloneTrainer`/`VideoTokenizerTrainer` (DDP-ready) as the first target;
   do NOT start from `scripts/train_dreamer.py` (broken imports).

**P1 — correctness/perf for real runs**
4. **Preemption-safe checkpointing:** move to
   `accelerator.save_state/load_state`, include optimizer + EMA + RNG +
   dataloader/step, atomic write, `latest` pointer, `--resume`. Tighten
   `strict=False` model loads.
5. **Enable TF32 + bf16 everywhere** (`set_float32_matmul_precision('high')`,
   `mixed_precision='bf16'` in all trainers).
6. **Scale the data pipeline:** *partially done (2026-06-08)* — `num_workers`,
   `pin_memory`, `persistent_workers` are now wired through both
   `VideoTokenizerTrainer` and `BehaviorCloneTrainer` (defaults 0/False) and
   exposed in the LIBERO scripts (`--num_workers`, default 8). Remaining:
   `prefetch_factor`, a sharded/streaming dataset for real video, and `set_epoch`.
7. **DDP grad-accum efficiency:** wrap micro-steps in `accelerator.no_sync()` /
   `accelerator.accumulate()` to avoid all-reducing every micro-step.

**P2 — scale-up (multi-node + large models)**
8. **Multi-node launch**: machine rank / rendezvous / NCCL-IB config, sbatch,
   shared-FS checkpoints.
9. **FSDP support**: switch checkpoint save/load to gathered state dicts, make
   EMA FSDP-aware (or keep on a replicated copy), and validate `MuonAdamAtan2`
   under sharding (keep Muon params replicated if needed).
10. **Activation checkpointing + `torch.compile` of the full model** (bucketed
    shapes) to fit larger batch/context in 141 GB.
11. **Add throughput/MFU/GPU-mem/grad-norm logging.**

**P3 — opportunistic**
12. FP8 (transformer-engine/torchao) on the largest matmuls once bf16 is stable.
13. FA3 Hopper kernels as an attention optimization.
14. Give `DynamicsWorldModel` a `@save_load`/`init_and_load` so its config travels
    with the checkpoint like the tokenizer's.
15. Fix or remove `scripts/train_dreamer.py` (missing `dreamer4._core` /
    `dreamer4._tokenizer`).

---

### Environment fix applied (2026-06-08)
Ran in the repo `.venv` (torch deliberately not touched):
```
uv pip install --python .venv/bin/python \
  "PoPE_pytorch>=0.1.1" "x-transformers>=2.19.7" "h-net-dynamic-chunking>=0.5.12" \
  "torch-einops-utils>=0.1.2" "vit-pytorch>=1.21.5" "x-mlps-pytorch>=0.3.1" \
  "discrete-continuous-embed-readout>=0.2.4" wandb tensorboard fire
```
Resulting key versions: torch 2.9.1+cu128 (untouched), accelerate 1.12.0,
pope-pytorch 0.1.4, x-transformers 2.20.1, h-net-dynamic-chunking 0.5.14,
torch-einops-utils 0.1.2, vit-pytorch 1.21.5, x-mlps-pytorch 0.3.4,
discrete-continuous-embed-readout 0.2.7, einops 0.8.2, wandb 0.27.2,
tensorboard 2.20.0, fire 0.7.1. Verified: `import dreamer4` OK; tiny
`VideoTokenizer` forward+backward OK on CPU; `torch.cuda.is_available()` True.
Note: this is still a *shared* env with unrelated robotics packages — for the
cluster, build a dedicated, lockfile-pinned env from `pyproject.toml`.
