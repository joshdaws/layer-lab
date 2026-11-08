"""Video/frame I/O and mask/compositing helpers (numpy + OpenCV + ffmpeg)."""
import subprocess, json, os, numpy as np, cv2
def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_frames", "-of", "json", path], capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]; n, d = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(n) / float(d), int(s.get("nb_frames") or 0)
def read(path):
    w, h, fps, _ = probe(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), fps
def write(path, frames, fps=24, audio_from=None, crf=14):
    n, h, w = frames.shape[:3]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio_from: cmd += ["-i", audio_from, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)
def save_png(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA if img.shape[-1] == 4 else cv2.COLOR_RGB2BGR))
def load_png(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA if im.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
def dilate(m, px):
    if px <= 0: return m.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0
def soft(m, grow=0, feather=8):
    m = dilate(m, grow).astype(np.float32)
    if feather > 0: m = cv2.GaussianBlur(m, (0, 0), feather / 2.0)
    return np.clip(m, 0, 1)
def shift(m, dx, dy, scale=1.0, cx=None, cy=None):
    """Move a mask/image by (dx,dy) px, optionally scaling about (cx,cy)."""
    h, w = m.shape[:2]
    if cx is None: cx, cy = w / 2, h / 2
    M = np.float32([[scale, 0, (1 - scale) * cx + dx], [0, scale, (1 - scale) * cy + dy]])
    src = m.astype(np.float32) if m.dtype == bool else m
    out = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out > 0.5 if m.dtype == bool else out
def blend(base, top, alpha):
    a = alpha[..., None] if alpha.ndim == base.ndim - 1 else alpha
    return (base.astype(np.float32) * (1 - a) + top.astype(np.float32) * a).round().clip(0, 255).astype(np.uint8)
def overlay(frame, masks, colors=((255, 70, 200), (60, 200, 255), (255, 200, 40), (120, 255, 120)), opacity=0.45):
    out = frame.astype(np.float32)
    for i, m in enumerate(masks):
        c = np.array(colors[i % len(colors)], np.float32)
        a = soft(m, 0, 2) * opacity
        out = out * (1 - a[..., None]) + c * a[..., None]
        edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        out[edge] = c
    return out.clip(0, 255).astype(np.uint8)
def hull(m):
    """Filled convex hull of a mask (covers gaps a tracker drops, e.g. a pot under fronds)."""
    m8 = m.astype(np.uint8)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return m.astype(bool)
    out = np.zeros_like(m8); cv2.fillPoly(out, [cv2.convexHull(np.concatenate(cs))], 1); return out > 0
