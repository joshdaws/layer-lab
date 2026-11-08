"""Layer Lab pipeline: find -> track -> delete / move, on a finished shot. All files live in $LAYERLAB_HOME/projects/<pid>/."""
import os, json, time, subprocess, shutil, uuid, numpy as np, cv2
import frames as F
import config
ROOT = config.PROJECTS
STUDIO = config.STUDIO_DIR
SPY = config.STUDIO_PY
def P(pid, *a): return os.path.join(ROOT, pid, *a)
def jload(p, d=None):
    try: return json.load(open(p))
    except Exception: return d
def jsave(p, o): os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(o, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
def valid_len(n): return max(90, min(345, 5 + 17 * ((max(n, 90) - 5 + 16) // 17)))
# ---------- projects ----------
def new_project(src_path, title=None):
    pid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]; os.makedirs(P(pid, "objects")); os.makedirs(P(pid, "edits"))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src_path, "-r", "24", "-t", "14.3", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/32)*32:trunc(ih/32)*32", "-c:a", "aac", "-movflags", "+faststart", P(pid, "source.mp4")], check=True)
    w, h, fps, _ = F.probe(P(pid, "source.mp4")); fr, _ = F.read(P(pid, "source.mp4"))
    F.save_png(P(pid, "poster.png"), fr[0])
    cv2.imwrite(P(pid, "poster.jpg"), cv2.cvtColor(fr[0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    jsave(P(pid, "meta.json"), {"id": pid, "title": title or os.path.splitext(os.path.basename(src_path))[0], "w": w, "h": h, "fps": 24, "n": len(fr), "created": time.time()})
    return pid
def projects():
    out = []
    for pid in sorted(os.listdir(ROOT), reverse=True) if os.path.isdir(ROOT) else []:
        m = jload(P(pid, "meta.json"))
        if m: out.append(m)
    return out
def source(pid):
    return F.read(P(pid, "source.mp4"))[0]
# ---------- find / track ----------
def find(pid, text, frame=0):
    import sam
    fr = source(pid)[frame]
    cands = sam.detect(fr, text)
    keep = []
    for c in sorted(cands, key=lambda c: -c["score"]):   # drop parts of an object already listed
        a = c["mask"].sum()
        if any((c["mask"] & k["mask"]).sum() > 0.85 * a for k in keep): continue
        keep.append(c)
    fid = uuid.uuid4().hex[:8]; d = P(pid, "finds", fid); os.makedirs(d)
    res = []
    for i, c in enumerate(keep[:12]):
        np.save(os.path.join(d, f"{i}.npy"), c["mask"])
        rgba = np.zeros(c["mask"].shape + (4,), np.uint8); rgba[c["mask"]] = (255, 255, 255, 255)
        cv2.imwrite(os.path.join(d, f"{i}.png"), rgba)
        res.append({"i": i, "box": c["box"], "score": round(c["score"], 3), "area": int(c["mask"].sum())})
    jsave(os.path.join(d, "find.json"), {"text": text, "frame": frame, "cands": res})
    return {"fid": fid, "text": text, "cands": res}
def track(pid, fid, picks, label, log=print):
    import sam
    d = P(pid, "finds", fid); m0 = np.zeros_like(np.load(os.path.join(d, f"{picks[0]}.npy")))
    for i in picks: m0 |= np.load(os.path.join(d, f"{i}.npy"))
    fr = source(pid); log("tracking", label)
    tr = sam.track(fr, init_masks=m0[None], max_objects=1, detect_interval=100000)[0]
    oid = uuid.uuid4().hex[:8]; od = P(pid, "objects", oid); os.makedirs(od)
    np.savez_compressed(os.path.join(od, "masks.npz"), masks=np.packbits(tr, axis=-1), shape=tr.shape)
    F.write(os.path.join(od, "track.mp4"), np.stack([F.overlay(f, [tr[i]]) for i, f in enumerate(fr)]), 24)
    ys, xs = np.nonzero(tr[0]); box = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    a = (F.soft(tr[0], 1, 2) * 255).astype(np.uint8)
    cut = np.dstack([fr[0], a])[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
    F.save_png(os.path.join(od, "cutout.png"), cut)
    o = {"id": oid, "label": label, "box": box, "frames_present": int((tr.reshape(len(tr), -1).any(1)).sum()), "created": time.time()}
    jsave(os.path.join(od, "object.json"), o); sam.unload(); return o
def masks(pid, oid):
    d = np.load(P(pid, "objects", oid, "masks.npz")); sh = tuple(d["shape"])
    return np.unpackbits(d["masks"], axis=-1)[..., :sh[-1]].astype(bool).reshape(sh)
def shadow_masks(pid, oid, obj_masks, log=print):
    """Find the object's shadow on the first frame (SAM text 'shadow' touching the object's grown mask) and track it."""
    import sam
    fr = source(pid); near = F.dilate(obj_masks[0], 60); best = None
    for c in sam.detect(fr[0], "shadow", threshold=0.3):
        ov = (c["mask"] & near).sum()
        if ov > 200 and (best is None or ov > best[0]): best = (ov, c["mask"] & ~obj_masks[0])
    if best is None: log("no shadow found"); return np.zeros_like(obj_masks)
    log("finding its shadow"); tr = sam.track(fr, init_masks=best[1][None], max_objects=1, detect_interval=100000)[0]
    sam.unload(); return tr
# ---------- keyframe editing (Qwen-Image 2.1 edit through the Studio) ----------
def qwen_edit(image_png, prompt, out_png, seed=7, log=print):
    log("painting keyframe", os.path.basename(out_png).split("_")[0].lstrip("k"))
    r = subprocess.run([SPY, "spark_edit.py", "--image", image_png, "--prompt", prompt, "--seed", str(seed), "-o", out_png], cwd=os.path.join(STUDIO, "spark"), capture_output=True, text=True)
    if not os.path.exists(out_png): raise RuntimeError("picture edit failed: " + (r.stdout + r.stderr)[-800:])
    return F.load_png(out_png)[..., :3]
def clean_plate(frame, mask, label, work, tag, log=print):
    F.save_png(os.path.join(work, f"{tag}_src.png"), frame)
    raw = qwen_edit(os.path.join(work, f"{tag}_src.png"), f"Remove the {label} completely, including its shadow and anything it holds. Fill where it was with what is behind it, matching the surrounding surfaces, texture, perspective and light. Keep everything else in the picture exactly as it is: every other object and person, the framing, colours and lighting.", os.path.join(work, f"{tag}_clean_raw.png"), log=log)
    if raw.shape != frame.shape: raw = cv2.resize(raw, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    clean = F.blend(frame, raw, F.soft(mask, 22, 16)); F.save_png(os.path.join(work, f"{tag}_clean.png"), clean); return clean
def moved_key(frame, mask, clean, dx, dy, scale, label, work, tag, log=print, shadow=None):
    ys, xs = np.nonzero(mask); cx, cy = (xs.min() + xs.max()) / 2, ys.max()   # scale about the object's base
    a = F.shift(F.soft(mask, 1, 2), dx, dy, scale, cx, cy); obj = F.shift(frame, dx, dy, scale, cx, cy)
    comp = clean.copy()
    if shadow is not None and shadow.sum() > 200:
        # carry the object's real cast shadow along: the darkening it makes on the ground, moved with it
        ratio = np.clip((frame.astype(np.float32) + 4) / (clean.astype(np.float32) + 4), 0.15, 1.0)
        sa = F.soft(shadow, 2, 6)
        dark = (1 - ratio) * sa[..., None]                      # how much the shadow darkens each pixel
        ratio = 1 - F.shift(dark, dx, dy, scale, cx, cy)        # moved with the object; empty edges = no change
        comp = (comp.astype(np.float32) * ratio).clip(0, 255).astype(np.uint8)
    comp = F.blend(comp, obj, a); F.save_png(os.path.join(work, f"{tag}_comp.png"), comp)
    if shadow is not None and shadow.sum() > 200:
        F.save_png(os.path.join(work, f"{tag}_key.png"), comp); return comp, a > 0.5
    harm = qwen_edit(os.path.join(work, f"{tag}_comp.png"), f"The {label} now stands in this new place. Make it sit naturally where it is: add its soft contact shadow and the gentle shadow it casts in this scene's light, matching the other shadows. Do not move, resize or change the {label} or anything else; keep the framing, colours and the rest of the picture exactly as they are.", os.path.join(work, f"{tag}_harm_raw.png"), log=log)
    if harm.shape != frame.shape: harm = cv2.resize(harm, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    new = a > 0.5; band = shadow_band(new)
    key = F.blend(comp, harm, F.soft(band, 0, 24) * (1 - F.soft(new, 0, 3))); F.save_png(os.path.join(work, f"{tag}_key.png"), key)
    return key, new
def shadow_band(m):
    ys, xs = np.nonzero(m); r = np.zeros_like(m)
    if len(ys) == 0: return r
    h, w = m.shape; bh = ys.max() - ys.min(); bw = xs.max() - xs.min()
    r[max(ys.max() - int(0.35 * bh) - 20, 0):min(ys.max() + int(0.12 * bh) + 30, h), max(xs.min() - int(0.6 * bw), 0):min(xs.max() + int(0.6 * bw), w)] = True
    return F.dilate(m, 40) | r
# ---------- the edit jobs ----------
def run_edit(pid, oid, op, dx=0, dy=0, scale=1.0, with_shadow=True, lock_pixels=True, seed=0, log=print):
    import viggle, prompter, lock
    meta = jload(P(pid, "meta.json")); obj = jload(P(pid, "objects", oid, "object.json")); label = obj["label"]
    eid = time.strftime("%H%M%S-") + uuid.uuid4().hex[:4]; work = P(pid, "edits", eid); os.makedirs(work)
    jsave(os.path.join(work, "edit.json"), {"id": eid, "object": oid, "label": label, "op": op, "dx": dx, "dy": dy, "scale": scale, "with_shadow": with_shadow, "lock": lock_pixels, "status": "running", "created": time.time()})
    src = source(pid); N, H, W = src.shape[:3]; om = masks(pid, oid)
    sh = shadow_masks(pid, oid, om, log) if with_shadow else np.zeros_like(om)
    gone = om | sh
    # keyframes at the first and last frame where the object is visible
    vis = [i for i in range(N) if om[i].any()] or [0]
    if op == "move":   # a moved object is cut out of its keyframes, so use frames where it is wholly inside the picture
        e = 3; inside = [i for i in vis if not (om[i][:e].any() or om[i][-e:].any() or om[i][:, :e].any() or om[i][:, -e:].any())]
        vis = inside or [max(vis, key=lambda i: om[i].sum())]
    pins = sorted({vis[0], vis[-1]}) if vis[-1] - vis[0] > 30 else [vis[0]]
    if op == "move" and len(vis) > 60:   # an object that travels needs more anchors, or the render keeps it on its old path
        cxs = [np.nonzero(om[i])[1].mean() for i in (vis[0], vis[-1])]
        if abs(cxs[1] - cxs[0]) > 0.08 * W: pins = sorted({vis[round(t * (len(vis) - 1))] for t in (0, 1 / 3, 2 / 3, 1)})
    log("keyframes at", ", ".join(str(p) for p in pins))
    keys, news = [], []
    for k in pins:
        cl = clean_plate(src[k], gone[k], label, work, f"k{k}", log)
        if op == "delete": keys.append(cl)
        else:
            key, new = moved_key(src[k], om[k], cl, dx, dy, scale, label, work, f"k{k}", log, shadow=sh[k] if with_shadow else None); keys.append(key); news.append(new)
        F.save_png(os.path.join(work, f"key{k}.png"), keys[-1])
    stills = []
    for t in (0, N // 2, N - 1):
        F.save_png(os.path.join(work, f"orig_{t}.png"), src[t]); stills.append((os.path.join(work, f"orig_{t}.png"), t / 24))
    if op == "delete":
        task = f"Delete the {label} from the whole shot" + (", together with its shadow" if with_shadow else "") + ". Where it was, show what is behind it, continuing the surfaces and light naturally. Everything and everyone else stays exactly as in the source."
    else:
        ys, xs = np.nonzero(news[0]); where = f"to the spot shown in the target frame (its new centre is about {int(100*(xs.mean())/W)}% across and {int(100*(ys.mean())/H)}% down the frame)"
        task = f"Move the {label} {where}: in every frame it is where the target frames show it (its whole path is shifted and it keeps its own motion), together with its shadow exactly as the target frames show it. There is only one {label}: nothing of it remains on its old path. Where it used to be is now empty, showing what is behind it. Everything and everyone else stays exactly as in the source."
    log("writing the edit prompt")
    prompt = prompter.write(task, stills, [os.path.join(work, f"key{k}.png") for k in pins], pins, N / 24, out=os.path.join(work, "prompt.txt"))
    # H3 needs 17n+5 frames: pad by holding the last frame
    L = valid_len(N); srcp = src if L == N else np.concatenate([src, np.repeat(src[-1:], L - N, 0)])[:L]
    F.write(os.path.join(work, "src_padded.mp4"), srcp, 24, audio_from=P(pid, "source.mp4"))
    log("rendering the shot (H3)")
    se = 736 if L <= 192 else 480
    viggle.edit(os.path.join(work, "src_padded.mp4"), [os.path.join(work, f"key{k}.png") for k in pins], pins, prompt, os.path.join(work, "gen.mp4"), frames=L, short_edge=se, seed=seed, log=log)
    gen, _ = F.read(os.path.join(work, "gen.mp4")); gen = gen[:N]
    canvas, valid = lock.to_canvas(gen, H, W)
    if lock_pixels:
        log("keeping untouched pixels")
        regions = np.stack([F.dilate(F.hull(om[i]), 16) | F.dilate(sh[i], 14) for i in range(N)])
        if op == "move":   # follow the moved object where the render actually put it (camera moves shift it)
            import sam
            gnew = sam.track(canvas, init_masks=news[0][None], max_objects=1, detect_interval=100000)[0]; sam.unload()
            if pins[0] > 0: gnew[:pins[0]] = gnew[pins[0]]
            regions |= np.stack([shadow_band(F.hull(gnew[i])) if gnew[i].any() else np.zeros_like(gnew[i]) for i in range(N)])
            if sh.any():   # the moved shadow too (same move as the object, about the object's base)
                for i in range(N):
                    if om[i].any() and sh[i].any():
                        ys, xs = np.nonzero(om[i]); regions[i] |= F.dilate(F.shift(sh[i], dx, dy, scale, (xs.min() + xs.max()) / 2, ys.max()), 16)
        final = lock.lock(src, canvas, regions, valid)
    else: final = canvas
    F.write(os.path.join(work, "final.mp4"), final, 24, audio_from=P(pid, "source.mp4"))
    F.write(os.path.join(work, "compare.mp4"), np.concatenate([src, final], 2), 24, audio_from=P(pid, "source.mp4"), crf=18)
    e = jload(os.path.join(work, "edit.json")); e.update(status="done", finished=time.time()); jsave(os.path.join(work, "edit.json"), e)
    return e
