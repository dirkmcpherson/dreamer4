# dreamer4 — Project Notes

PyTorch implementation of Dreamer 4 (Hafner et al., arXiv:2509.24527): a scalable
world model with a `VideoTokenizer` + `DynamicsWorldModel`. Core code in
`dreamer4/dreamer4.py` and `dreamer4/trainers.py` (built on HuggingFace `accelerate`).

**Current initiative:** prepare the repo to train on a multi-task robot manipulation
dataset and evaluate closed-loop in a robot simulator, on an H200 GPU cluster.
- Dataset/simulator research → `research/datasets_and_sims.md`
- H200 cluster readiness assessment → `infra/h200_readiness.md`

**Environment:** use the repo `.venv` (`.venv/bin/python`, or `uv run`). It's a shared
Python 3.10 env (torch 2.9.1+cu128); dreamer4's deps were installed/upgraded into it
on 2026-06-08 so `import dreamer4` works.

## Agent Status

- **Status:** 🟢 active
- **Last session:** 2026-06-08
- **Current branch:** main
- **What happened:** Chose **LIBERO + robosuite/MuJoCo** and built the full deploy set:
  `dataset_libero.py` (HDF5→dreamer4 adapter), `train_libero_tokenizer.py`,
  `train_libero_dynamics.py`, `eval_libero.py`, plus `infra/` launch files
  (`accelerate_config.yaml`, two `.sbatch`, `update_cluster_env.sh`, `download_libero.py`,
  `README.md`). Verified locally: dreamer4+robosuite coexist, headless EGL render works,
  and a synthetic-HDF5 integration test confirms dataset→trainer→model shapes line up.
  Also: periodic wandb eval videos for both models (tokenizer recon + dynamics
  forward-prediction), `logger=` flags, fixed a tokenizer eval-mode-restore bug, wired
  `num_workers`/`pin_memory` through both trainers, set TF32/bf16 in the LIBERO scripts.
- **What's next:** On the cluster — run `update_cluster_env.sh`, download LIBERO
  (`libero_90`+`libero_10`, ~81 GB), train tokenizer then dynamics; then validate
  `eval_libero.py`'s acting path (policy head vs BC readout) and live-env obs keys.
- **Blocked on:** nothing. Eval acting-path correctness is the main open risk.
- **Key decisions made:** LIBERO data sizes ~104 GB total (libero_90 ~67 GB). Install LIBERO
  into the existing `dreamer4` conda env (modern robosuite 1.5.x; documented 1.4.1 fallback).
  dreamer4 ingests 7-DoF continuous actions; proprio (joint+gripper, dim 9) optional.
