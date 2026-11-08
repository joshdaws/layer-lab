"""Check that everything Layer Lab needs is where engine/config.py expects it. Prints one line per item.

    python scripts/check.py
"""
import os
import shutil
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))
import config  # noqa: E402

ok = True


def item(name, good, detail):
    global ok
    ok &= bool(good)
    print(f"  {'ok ' if good else 'MISSING'}  {name}: {detail}")


print("Layer Lab check")
item("Layer Lab folder", os.path.isdir(config.LAYERLAB_HOME), config.LAYERLAB_HOME)
item("ffmpeg", shutil.which("ffmpeg"), shutil.which("ffmpeg") or "not on PATH")
item("ComfyUI", os.path.isfile(os.path.join(config.COMFYUI_DIR, "main.py")), config.COMFYUI_DIR)
item("ComfyUI SAM 3 nodes", os.path.isfile(os.path.join(config.COMFYUI_DIR, "comfy_extras", "nodes_sam3.py")), "comfy_extras/nodes_sam3.py (update ComfyUI if missing)")
item("SAM 3.1 checkpoint", os.path.isfile(config.SAM_CKPT), config.SAM_CKPT)
item("Imagination Studio", os.path.isfile(os.path.join(config.STUDIO_DIR, "spark", "spark_edit.py")), config.STUDIO_DIR)
item("Studio Python", os.path.isfile(config.STUDIO_PY), config.STUDIO_PY)
wf = os.path.join(config.VIGGLE_DIR, "comfyui", "workflows", "remake_edit_api.json")
item("Viggle Remake workflows", os.path.isfile(wf), wf)
item("Viggle Remake node pack", os.path.isdir(os.path.join(config.COMFYUI_DIR, "custom_nodes", "ComfyUI-Viggle-Remake")), "custom_nodes/ComfyUI-Viggle-Remake")
item("Viggle Remake edit model", os.path.isfile(os.path.join(config.VIGGLE_DIR, "int8", "viggle_remake_edit_pruned_int8_convrot.safetensors")), "int8/viggle_remake_edit_pruned_int8_convrot.safetensors")
item("Claude Code", os.path.isfile(config.CLAUDE_BIN) or shutil.which(config.CLAUDE_BIN), config.CLAUDE_BIN)
try:
    urllib.request.urlopen(config.COMFY_URL + "/system_stats", timeout=3)
    item("ComfyUI server", True, config.COMFY_URL + " (running)")
except Exception:
    print(f"  note     ComfyUI server: not running at {config.COMFY_URL}; Layer Lab starts it with the Studio's spark/comfy.sh")
print("all good" if ok else "fix the MISSING lines above (paths can be changed with environment variables, see README)")
sys.exit(0 if ok else 1)
