"""Layer Lab find/track: SAM 3.1 (ComfyUI native model) run standalone, kept warm by the app.
detect(frame, text) -> candidates on one frame; track(frames, masks) -> per-frame masks per object."""
import sys, os, time, json
import config
C = config.COMFYUI_DIR
if C not in sys.path: sys.path.insert(0, C)
_cwd = os.getcwd(); os.chdir(C); sys_argv = sys.argv; sys.argv = [sys.argv[0]]
import comfy.options; comfy.options.enable_args_parsing()
import folder_paths, comfy.sd, comfy.model_management, torch, numpy as np
from comfy_extras import nodes_sam3
from comfy.ldm.sam3.tracker import unpack_masks
import torch.nn.functional as F
sys.argv = sys_argv; os.chdir(_cwd)
CK = config.SAM_CKPT
_M = {}
def load():
    if not _M:
        m, c = comfy.sd.load_checkpoint_guess_config(CK, output_vae=False, output_clip=True)[:2]
        _M["model"], _M["clip"] = m, c
    return _M["model"], _M["clip"]
def unload():
    _M.clear(); comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
def cond(text):
    _, clip = load()
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))
def to_tensor(frames):  # list/array of HxWx3 uint8 -> [N,H,W,3] float
    a = np.asarray(frames)
    if a.ndim == 3: a = a[None]
    return torch.from_numpy(a).float().div_(255.0)
def multi(text, n=12):
    import re
    return ", ".join(t if re.search(r":\s*[\d.]+\s*$", t) else f"{t}:{n}" for t in [p.strip() for p in text.split(",") if p.strip()])
def detect(frame, text, threshold=0.4):
    """frame: HxWx3 uint8. Returns list of {mask: HxW bool, box: [x,y,w,h], score}."""
    model, _ = load(); text = multi(text)
    img = to_tensor(frame)
    out = nodes_sam3.SAM3_Detect.execute(model=model, image=img, conditioning=cond(text), threshold=threshold, refine_iterations=2, individual_masks=True).result
    masks, boxes = out[0], out[1]
    masks = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
    if masks.ndim == 2: masks = masks[None]
    blist = boxes[0] if (isinstance(boxes, list) and boxes and isinstance(boxes[0], list)) else boxes
    res = []
    for i in range(masks.shape[0]):
        m = masks[i] > 0.5
        if m.sum() < 16: continue
        b = blist[i] if isinstance(blist, list) and i < len(blist) else None
        if b is None:
            ys, xs = np.nonzero(m); b = {"x": int(xs.min()), "y": int(ys.min()), "width": int(xs.ptp() + 1), "height": int(ys.ptp() + 1)}
        res.append({"mask": m, "box": [int(b["x"]), int(b["y"]), int(b["width"]), int(b["height"])], "score": float(b.get("score", 0) if isinstance(b, dict) else 0)})
    return res
def track(frames, init_masks=None, text=None, max_objects=8, detect_interval=12, threshold=0.5):
    """frames: [N,H,W,3] uint8. init_masks: [K,H,W] bool on frame 0 (one per object). Returns [K,N,H,W] bool."""
    model, _ = load()
    imgs = to_tensor(frames); N, H, W = imgs.shape[:3]
    kw = dict(images=imgs, model=model, detection_threshold=threshold, max_objects=max_objects, detect_interval=detect_interval)
    if init_masks is not None: kw["initial_mask"] = torch.from_numpy(np.asarray(init_masks, dtype=np.float32))
    if text: kw["conditioning"] = cond(text)
    td = nodes_sam3.SAM3_VideoTrack.execute(**kw).result[0]
    packed = td["packed_masks"]
    if packed is None: return np.zeros((0, N, H, W), bool)
    out = []
    for k in range(packed.shape[1]):
        u = unpack_masks(packed[:, k].clone()).unsqueeze(1).float()
        out.append((F.interpolate(u, size=(H, W), mode="bilinear", align_corners=False)[:, 0] > 0.5).cpu().numpy())
    return np.stack(out)
