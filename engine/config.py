"""Where Layer Lab finds things. Every value can be overridden with an environment variable.

The defaults follow Imagination Studio's own layout on a DGX Spark (the same variable names where the Studio
has one: PREFIX, SPARK_MODELS, SPARK_COMFY_PORT, STUDIO_HOME), so on a machine with the Studio installed
nothing needs to be set.
"""
import os
import shutil


def _path(name, default):
    return os.path.expanduser(os.environ.get(name, default))


# Layer Lab's own folder: projects (one per shot) and optional extra clips in shots/
LAYERLAB_HOME = _path("LAYERLAB_HOME", "~/layerlab")
PROJECTS = os.path.join(LAYERLAB_HOME, "projects")
SHOTS = os.path.join(LAYERLAB_HOME, "shots")

# Imagination Studio: its checkout, its install prefix (ComfyUI + Python env), its model folder and its data folder
STUDIO_DIR = _path("STUDIO_DIR", "~/Imagination-Studio-by-keys-Wasserman")
PREFIX = _path("PREFIX", "~/.local/opt/tensorfold-studio")
STUDIO_PY = _path("STUDIO_PY", os.path.join(PREFIX, "venv", "bin", "python"))
COMFYUI_DIR = _path("COMFYUI_DIR", os.path.join(PREFIX, "ComfyUI"))
SPARK_MODELS = _path("SPARK_MODELS", "~/spark-studio-models")
STUDIO_HOME = _path("STUDIO_HOME", "~/ImaginationStudio")
STUDIO_RENDERS = _path("STUDIO_RENDERS", os.path.join(STUDIO_HOME, "renders"))

# ComfyUI, as the Studio runs it (spark/comfy.sh)
COMFY_URL = os.environ.get("COMFY_URL", f"http://127.0.0.1:{os.environ.get('SPARK_COMFY_PORT', '8190')}")
COMFY_INPUT = _path("COMFY_INPUT", os.path.join(COMFYUI_DIR, "input"))
COMFY_OUTPUT = _path("COMFY_OUTPUT", os.path.join(COMFYUI_DIR, "output"))

# Models
SAM_CKPT = _path("SAM_CKPT", os.path.join(SPARK_MODELS, "Workflows-Comfy", "checkpoints", "sam3.1_multiplex_fp16.safetensors"))
VIGGLE_DIR = _path("VIGGLE_DIR", os.path.join(SPARK_MODELS, "Viggle-Remake"))
VIGGLE_TEXT_ENCODER = os.environ.get("VIGGLE_TEXT_ENCODER", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors")

# Claude Code writes the video model's prompt (headless: claude -p)
CLAUDE_BIN = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "opus")

# Before a render: stop the Studio's resident QA server (~30 GB) so H3 fits under the Studio's memory guard.
# Set LAYERLAB_FREE_CMD="" to skip.
FREE_CMD = os.environ.get("LAYERLAB_FREE_CMD", "pkill -f '[t]ensorfold-native serve'")

# Jobs wait while any process matching this runs (other GPU work on the same machine). Empty = never wait.
BUSY_PATTERN = os.environ.get("LAYERLAB_BUSY_PATTERN", "spark_generate.py|spark_swap.py")
