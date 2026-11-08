#!/usr/bin/env bash
# Fetch Viggle Remake (int8 merged swap + edit models, its ComfyUI node pack and API workflows) and wire it into the
# Studio's ComfyUI. About 42 GB. Files already there are kept. Restart ComfyUI afterwards (bash spark/comfy.sh stop).
set -euo pipefail
cd "$(dirname "$0")/.."; . scripts/env.sh
HF="${HF:-$(command -v hf || echo "$PREFIX/venv/bin/hf")}"
mkdir -p "$VIGGLE_DIR"
"$HF" download Viggle/Viggle-Remake --local-dir "$VIGGLE_DIR" \
  --include "int8/*" --include "comfyui/ComfyUI-Viggle-Remake/*" --include "comfyui/workflows/*" \
  --include "*.md" --include "LICENSE*" --include "NOTICE"
DM="$SPARK_MODELS/MiniMax-H3-Comfy/diffusion_models"; mkdir -p "$DM"
for m in swap edit; do
  ln -sfn "$VIGGLE_DIR/int8/viggle_remake_${m}_pruned_int8_convrot.safetensors" "$DM/viggle_remake_${m}_pruned_int8_convrot.safetensors"
done
rm -rf "$COMFYUI_DIR/custom_nodes/ComfyUI-Viggle-Remake"
cp -r "$VIGGLE_DIR/comfyui/ComfyUI-Viggle-Remake" "$COMFYUI_DIR/custom_nodes/"
echo "Viggle Remake installed. Restart ComfyUI so it loads the node pack and sees the models."
