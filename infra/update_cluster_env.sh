#!/usr/bin/env bash
# Update an existing `dreamer4` conda environment on the cluster so it can both
# (a) train dreamer4 on the LIBERO robot-manipulation demos and
# (b) run closed-loop evaluation in the LIBERO / robosuite / MuJoCo simulator.
#
# Verified locally (2026-06-08, Python 3.10 / torch 2.9.1+cu128): these installs
# leave torch untouched, and dreamer4 + robosuite(EGL headless) coexist. robosuite
# pins numpy<2 (downgrades numpy to 1.26.x) — dreamer4 works fine with that.
#
# Usage:
#   bash infra/update_cluster_env.sh [ENV_NAME] [REPO_DIR]
#     ENV_NAME  conda env to update           (default: dreamer4)
#     REPO_DIR  path to this dreamer4 checkout (default: parent of this script)
#   Env vars:
#     LIBERO_DIR  where to clone LIBERO        (default: $REPO_DIR/third_party/LIBERO)
#     SKIP_SIM=1  install only dreamer4 deps, skip robosuite/LIBERO (training-only)
set -euo pipefail

ENV_NAME="${1:-dreamer4}"
REPO_DIR="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LIBERO_DIR="${LIBERO_DIR:-$REPO_DIR/third_party/LIBERO}"

echo "==> Activating conda env: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> Interpreter / torch sanity"
python -c "import sys; print('python', sys.version.split()[0])"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

PIP="python -m pip install --upgrade"

echo "==> [1/4] dreamer4 runtime deps (idempotent; torch/torchvision NOT pinned here)"
# Brings below-minimum packages up to the versions dreamer4 requires.
$PIP \
  "PoPE_pytorch>=0.1.1" \
  "x-transformers>=2.19.7" \
  "h-net-dynamic-chunking>=0.5.12" \
  "torch-einops-utils>=0.1.2" \
  "vit-pytorch>=1.21.5" \
  "x-mlps-pytorch>=0.3.1" \
  "discrete-continuous-embed-readout>=0.2.4" \
  "adam-atan2-pytorch>=0.2.2" \
  einx einops ema-pytorch assoc-scan hl-gauss-pytorch accelerate tqdm \
  wandb tensorboard fire h5py moviepy imageio

echo "==> [2/4] install dreamer4 itself (editable, --no-deps to protect torch)"
if [ -f "$REPO_DIR/pyproject.toml" ]; then
  python -m pip install -e "$REPO_DIR" --no-deps
fi
python -c "import dreamer4; print('    dreamer4 import OK')"

if [ "${SKIP_SIM:-0}" = "1" ]; then
  echo "==> SKIP_SIM=1 set: skipping simulator/LIBERO install (training-only env ready)."
  exit 0
fi

echo "==> [3/4] simulator stack: robosuite + MuJoCo (+ LIBERO sim deps)"
# robosuite forces numpy<2; this is expected and compatible with dreamer4/torch 2.9.
$PIP robosuite mujoco opencv-python huggingface_hub bddl easydict

echo "==> [4/4] LIBERO package for closed-loop eval (--no-deps to protect torch/numpy)"
if [ ! -d "$LIBERO_DIR/.git" ]; then
  echo "    cloning LIBERO -> $LIBERO_DIR"
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi
python -m pip install -e "$LIBERO_DIR" --no-deps

echo "==> Verifying headless render (MUJOCO_GL=egl)"
MUJOCO_GL=egl python - <<'PY'
import robosuite, mujoco
print("    robosuite", robosuite.__version__, "| mujoco", mujoco.__version__)
env = robosuite.make(env_name="Lift", robots="Panda", has_renderer=False,
                     has_offscreen_renderer=True, use_camera_obs=True,
                     camera_names="agentview", camera_heights=128, camera_widths=128, horizon=2)
obs = env.reset(); print("    EGL offscreen render OK:", obs["agentview_image"].shape); env.close()
PY

cat <<'NOTE'

==> DONE.

  Set MUJOCO_GL=egl for headless rendering on cluster nodes; on multi-GPU nodes
  also pin the render device, e.g. MUJOCO_EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES.

  COMPATIBILITY NOTE: this installs the *modern* robosuite (1.5.x) + mujoco 3.x.
  Canonical LIBERO historically pins robosuite==1.4.1. The standalone robosuite
  render is verified, but confirm LIBERO's task suites load on 1.5.x:
      MUJOCO_GL=egl python -c "from libero.libero import benchmark; \
        benchmark.get_benchmark_dict()['libero_object'](); print('LIBERO suites OK')"
  If that fails, create a SEPARATE eval env pinned to:
      pip install robosuite==1.4.1 'mujoco==2.3.*' 'numpy<1.24'
  (training on the HDF5 demos via h5py does not need the simulator at all).
NOTE
