# Shared defaults for the scripts. They match engine/config.py and Imagination Studio's layout; export any of them to override.
export LAYERLAB_HOME="${LAYERLAB_HOME:-$HOME/layerlab}"
export PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
export COMFYUI_DIR="${COMFYUI_DIR:-$PREFIX/ComfyUI}"
export SPARK_MODELS="${SPARK_MODELS:-$HOME/spark-studio-models}"
export VIGGLE_DIR="${VIGGLE_DIR:-$SPARK_MODELS/Viggle-Remake}"
export LAYERLAB_VENV="${LAYERLAB_VENV:-$LAYERLAB_HOME/venv}"
# a quoted ~ in an override, as engine/config.py accepts it
for v in LAYERLAB_HOME PREFIX COMFYUI_DIR SPARK_MODELS VIGGLE_DIR LAYERLAB_VENV; do printf -v "$v" "%s" "${!v/#\~/$HOME}"; export "$v"; done
