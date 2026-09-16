#!/usr/bin/env bash
set -euo pipefail

# Run this AFTER:
#   conda env create -f environment.yml
#   conda activate pebble-mujoco

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the conda environment first: conda activate pebble-mujoco"
  exit 1
fi

# -------------------------------------------------------------------
# 1. MuJoCo 2.1 binaries for mujoco-py
# -------------------------------------------------------------------
mkdir -p "$HOME/.mujoco"

if [[ ! -d "$HOME/.mujoco/mujoco210" ]]; then
  wget -q \
    https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
    -O /tmp/mujoco210.tar.gz
  tar -xzf /tmp/mujoco210.tar.gz -C "$HOME/.mujoco"
  rm -f /tmp/mujoco210.tar.gz
fi

# -------------------------------------------------------------------
# 2. Conda activation variables
# -------------------------------------------------------------------
mkdir -p \
  "$CONDA_PREFIX/etc/conda/activate.d" \
  "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/mujoco.sh" <<'EOF'
export _OLD_MUJOCO_GL="${MUJOCO_GL-}"
export _OLD_MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH-}"
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/mujoco.sh" <<'EOF'
if [[ -n "${_OLD_MUJOCO_GL:-}" ]]; then
  export MUJOCO_GL="$_OLD_MUJOCO_GL"
else
  unset MUJOCO_GL
fi

if [[ -n "${_OLD_MUJOCO_PY_MUJOCO_PATH:-}" ]]; then
  export MUJOCO_PY_MUJOCO_PATH="$_OLD_MUJOCO_PY_MUJOCO_PATH"
else
  unset MUJOCO_PY_MUJOCO_PATH
fi

if [[ -n "${_OLD_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
else
  unset LD_LIBRARY_PATH
fi

unset _OLD_MUJOCO_GL
unset _OLD_MUJOCO_PY_MUJOCO_PATH
unset _OLD_LD_LIBRARY_PATH
EOF

# Apply the variables now, without requiring deactivate/activate.
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

# -------------------------------------------------------------------
# 3. mujoco-py
#
# Cython < 3 is deliberately installed first, and build isolation is
# disabled so pip does not create a temporary build env with Cython 3.
# -------------------------------------------------------------------
python -m pip install --upgrade "cython<3"

SRC_ROOT="${HOME}/src"
mkdir -p "$SRC_ROOT"

if [[ ! -d "$SRC_ROOT/mujoco-py/.git" ]]; then
  git clone https://github.com/openai/mujoco-py.git "$SRC_ROOT/mujoco-py"
fi

cd "$SRC_ROOT/mujoco-py"
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir -r requirements.dev.txt
python -m pip install --no-cache-dir --no-build-isolation -e .

# Force the native extension to compile now, so failures happen here.
python -c "import mujoco_py; print('mujoco-py:', mujoco_py.__version__)"

# -------------------------------------------------------------------
# 4. MetaWorld v2.0.0, matching the Dockerfile
# -------------------------------------------------------------------
cd "$SRC_ROOT"
rm -rf Metaworld-2.0.0 metaworld-v2.0.0.tar.gz

wget -q \
  https://github.com/Farama-Foundation/Metaworld/archive/refs/tags/v2.0.0.tar.gz \
  -O metaworld-v2.0.0.tar.gz

tar -xzf metaworld-v2.0.0.tar.gz
cd Metaworld-2.0.0
python -m pip install .

# -------------------------------------------------------------------
# 5. Sanity checks
# -------------------------------------------------------------------
python - <<'PY'
import torch
import mujoco
import gym
import dm_control
import dmc2gym
import mujoco_py
import metaworld
import cv2
import imageio
import matplotlib.pyplot as plt
from PIL import Image

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("mujoco:", mujoco.__version__)
print("gym:", gym.__version__)
print("mujoco_py:", mujoco_py.__version__)
print("OpenCV:", cv2.__version__)
print("matplotlib:", plt.matplotlib.__version__)
print("pillow:", Image.__version__)
print("All core imports succeeded.")
PY
