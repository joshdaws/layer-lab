"""Run a Viggle Remake edit on the Studio's ComfyUI (config.COMFY_URL): int8 merged model, int8 text encoder."""
import json, os, shutil, subprocess, sys, time, urllib.request, uuid
import config
API = config.COMFY_URL
CIN = config.COMFY_INPUT
COUT = config.COMFY_OUTPUT
STUDIO = config.STUDIO_DIR
def call(path, data=None, timeout=120):
    req = urllib.request.Request(API + path, data=json.dumps(data).encode() if data is not None else None, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)
def free_memory():
    """Before an H3 render: the Studio's Nemotron QA server (~30 GB) must not be resident (memory guard floor)."""
    if config.FREE_CMD:
        subprocess.run(config.FREE_CMD, shell=True); time.sleep(5)
def up():
    free_memory()
    try: call("/system_stats", timeout=3); return True
    except Exception: pass
    subprocess.run(["bash", "spark/comfy.sh", "start"], cwd=STUDIO, capture_output=True)
    for _ in range(100):
        try: call("/system_stats", timeout=3); return True
        except Exception: time.sleep(3)
    raise RuntimeError("ComfyUI did not come up")
def stage(path, tag):
    name = f"ll_{tag}_{uuid.uuid4().hex[:6]}{os.path.splitext(path)[1]}"; shutil.copy(path, os.path.join(CIN, name)); return name
def edit(source_mp4, refs, pins, prompt, out_mp4, frames=124, short_edge=736, seed=0, mode="edit", log=print):
    w = json.load(open(os.path.join(config.VIGGLE_DIR, "comfyui", "workflows", f"remake_{mode}_api.json")))
    w["1"]["inputs"]["unet_name"] = f"viggle_remake_{mode}_pruned_int8_convrot.safetensors"
    del w["2"]; w["3"]["inputs"]["model"] = ["1", 0]
    w["4"]["inputs"]["clip_name"] = config.VIGGLE_TEXT_ENCODER
    w["7"]["inputs"]["file"] = stage(source_mp4, "src")
    n11 = w["11"]["inputs"]
    for k in [k for k in list(n11) if k.startswith("ref_")]: del n11[k]
    for k in ("9", "10"): w.pop(k, None)
    for i, r in enumerate(refs):
        nid = str(30 + i); w[nid] = {"class_type": "LoadImage", "inputs": {"image": stage(r, f"ref{i+1}")}}
        n11[f"ref_{i+1}"] = [nid, 0]
    n11.update(prompt=prompt, pins=", ".join(str(p) for p in pins), frames=frames, short_edge=short_edge)
    w["12"]["inputs"]["seed"] = seed
    pref = f"layerlab/{uuid.uuid4().hex[:8]}"; w["20"]["inputs"]["filename_prefix"] = pref
    up(); pid = call("/prompt", {"prompt": w})["prompt_id"]; log("queued", pid); t0 = time.time()
    while True:
        h = call(f"/history/{pid}")
        if pid in h:
            st = h[pid]["status"]; log("status", st.get("status_str"), round(time.time() - t0), "s")
            if st.get("status_str") != "success": raise RuntimeError(json.dumps(st)[:1500])
            import glob
            hits = sorted(glob.glob(os.path.join(COUT, pref + "_*.mp4")), key=os.path.getmtime)
            if not hits: raise RuntimeError("no video output for " + pref)
            shutil.copy(hits[-1], out_mp4); return out_mp4, round(time.time() - t0)
        time.sleep(5)
