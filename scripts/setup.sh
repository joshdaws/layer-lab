#!/usr/bin/env bash
# Make Layer Lab's Python environment: CUDA PyTorch + ComfyUI's requirements (for SAM 3.1 in-process) + the app's own.
#   bash scripts/setup.sh                      # default: CUDA 13.0 wheels (DGX Spark / GB10)
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."; . scripts/env.sh
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
command -v uv >/dev/null || { echo "uv is needed: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[ -f "$COMFYUI_DIR/requirements.txt" ] || { echo "ComfyUI not found at $COMFYUI_DIR (set COMFYUI_DIR or PREFIX)"; exit 1; }
mkdir -p "$LAYERLAB_HOME/projects" "$LAYERLAB_HOME/shots"
uv venv -p 3.12 "$LAYERLAB_VENV"
uv pip install -p "$LAYERLAB_VENV/bin/python" torch torchvision torchaudio --index-url "$TORCH_INDEX"
uv pip install -p "$LAYERLAB_VENV/bin/python" -r "$COMFYUI_DIR/requirements.txt"
uv pip install -p "$LAYERLAB_VENV/bin/python" -r requirements.txt
"$LAYERLAB_VENV/bin/python" -c "import torch, cv2, fastapi; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "Environment ready: $LAYERLAB_VENV. Next: bash scripts/install_viggle.sh, then python scripts/check.py"
