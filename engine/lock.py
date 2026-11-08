"""Put a generated clip back on the source canvas and keep source pixels outside the edit region."""
import numpy as np, cv2
import frames as F
def to_canvas(gen, H, W):
    """Viggle/H3 output is the source cover-cropped to its canvas; undo that (centre crop) onto HxW."""
    n, h, w = gen.shape[:3]
    s = max(W / w, H / h) if (W / H) > (w / h) else max(W / w, H / h)
    # the source was scaled by k=min-cover then centre-cropped; invert: scale gen by 1/k... compute directly:
    k = max(w / W, h / H)            # source -> canvas scale used by cover()
    sw, sh = W * k, H * k            # scaled source size before the crop
    ox, oy = (sw - w) / 2, (sh - h) / 2
    M = np.float32([[1 / k, 0, ox / k], [0, 1 / k, oy / k]])  # gen pixel (x,y) -> source pixel ((x+ox)/k, (y+oy)/k)
    out = np.empty((n, H, W, 3), np.uint8); valid = np.zeros((H, W), np.float32)
    for i in range(n): out[i] = cv2.warpAffine(gen[i], M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.ones((h, w), np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    return out, valid
def color_match(gen, src, ring):
    """Per-channel gain/offset so gen matches src on the ring (where both should agree)."""
    if ring.sum() < 500: return gen
    g = gen[ring].astype(np.float32); s = src[ring].astype(np.float32)
    gm, gs, sm, ss = g.mean(0), g.std(0) + 1e-3, s.mean(0), s.std(0) + 1e-3
    gain = np.clip(ss / gs, 0.85, 1.15)
    return np.clip((gen.astype(np.float32) - gm) * gain + sm, 0, 255).astype(np.uint8)
def lock(src, gen_canvas, regions, valid=None, feather=18, match=True):
    """src, gen_canvas: [N,H,W,3]; regions: [N,H,W] bool where the edit may change pixels."""
    out = np.empty_like(src)
    for i in range(len(src)):
        a = F.soft(regions[i], 0, feather)
        if valid is not None: a = a * (valid > 0.99)
        g = gen_canvas[i]
        if match:
            ring = F.dilate(regions[i], 60) & ~F.dilate(regions[i], 8)
            if valid is not None: ring &= valid > 0.99
            g = color_match(g, src[i], ring)
        out[i] = F.blend(src[i], g, a)
    return out
