"""Layer Lab app (FastAPI). Run: uvicorn server:app --host 127.0.0.1 --port 7890 (see scripts/run.sh).
One GPU worker runs track/edit jobs in order; find (SAM) never overlaps a job."""
import os, sys, json, time, threading, queue, traceback, glob, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))
import config
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
import pipeline as PL
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(PL.ROOT, exist_ok=True)
app = FastAPI()
JOBS, Q, LOCK = {}, queue.Queue(), threading.Lock()
GPU = threading.Lock()   # find (SAM) and jobs never overlap
def busy_elsewhere():
    import subprocess
    if not config.BUSY_PATTERN: return []
    r = subprocess.run(["pgrep", "-fa", config.BUSY_PATTERN], capture_output=True, text=True)
    if r.returncode > 1:   # 1 = no match; anything above is a bad pattern
        print(f"LAYERLAB_BUSY_PATTERN is not a valid pattern ({r.stderr.strip()}); not waiting for other GPU jobs", file=sys.stderr)
        return []
    return [l for l in r.stdout.splitlines() if "pgrep" not in l]
def worker():
    while True:
        jid = Q.get(); j = JOBS[jid]
        if j["status"] == "cancelled": continue
        def log(*a):
            line = " ".join(str(x) for x in a); j["log"].append(time.strftime("%H:%M:%S ") + line); j["stage"] = line
        try:
            while busy_elsewhere() and j["status"] != "cancelled":
                j["stage"] = "waiting: another GPU job is running"; time.sleep(20)
            if j["status"] == "cancelled": continue
            j["status"] = "running"; j["started"] = time.time()
            with GPU:
                if j["kind"] == "track": j["result"] = PL.track(j["pid"], j["args"]["fid"], j["args"]["picks"], j["args"]["label"], log=log)
                else: j["result"] = PL.run_edit(j["pid"], log=log, **j["args"])
            j["status"] = "done"; log("done")
        except Exception as e:
            j["status"] = "failed"; j["error"] = str(e)[-1500:]; j["log"].append(traceback.format_exc()[-3000:])
        j["finished"] = time.time()
threading.Thread(target=worker, daemon=True).start()
def submit(kind, pid, args, title):
    jid = f"{int(time.time()*1000)}"; JOBS[jid] = {"id": jid, "kind": kind, "pid": pid, "args": args, "title": title, "status": "queued", "stage": "queued", "log": [], "created": time.time()}
    Q.put(jid); return JOBS[jid]
@app.get("/", response_class=HTMLResponse)
def index(): return open(os.path.join(HERE, "index.html")).read()
@app.get("/api/projects")
def projects(): return PL.projects()
@app.get("/api/studio-renders")
def renders():
    fs = sorted(glob.glob(os.path.join(config.STUDIO_RENDERS, "*.mp4")) + glob.glob(os.path.join(config.SHOTS, "*.mp4")), key=os.path.getmtime, reverse=True)[:40]
    return [{"path": f, "name": os.path.basename(f), "mtime": os.path.getmtime(f)} for f in fs]
@app.post("/api/projects/from-path")
def from_path(body: dict):
    p = os.path.realpath(body.get("path", ""))
    ok = any(p.startswith(os.path.realpath(d) + os.sep) for d in (config.STUDIO_RENDERS, config.SHOTS))
    if not ok or not os.path.isfile(p): raise HTTPException(400, "not a known render")
    return PL.jload(PL.P(PL.new_project(p, body.get("title")), "meta.json"))
@app.post("/api/projects/upload")
async def upload(file: UploadFile = File(...)):
    d = tempfile.mkdtemp(); p = os.path.join(d, os.path.basename(file.filename or "clip.mp4"))
    with open(p, "wb") as f: shutil.copyfileobj(file.file, f)
    pid = PL.new_project(p); shutil.rmtree(d, ignore_errors=True); return PL.jload(PL.P(pid, "meta.json"))
@app.get("/api/projects/{pid}")
def project(pid: str):
    m = PL.jload(PL.P(pid, "meta.json"))
    if not m: raise HTTPException(404)
    objs = [PL.jload(f) for f in glob.glob(PL.P(pid, "objects", "*", "object.json"))]
    eds = [PL.jload(f) for f in glob.glob(PL.P(pid, "edits", "*", "edit.json"))]
    return {"meta": m, "objects": sorted([o for o in objs if o], key=lambda o: o["created"]), "edits": sorted([e for e in eds if e], key=lambda e: -e["created"])}
@app.post("/api/projects/{pid}/find")
def find(pid: str, body: dict):
    text = (body.get("text") or "").strip()
    if not text: raise HTTPException(400, "type what to find")
    if not GPU.acquire(timeout=2): raise HTTPException(409, "the GPU is rendering; find runs when it is free")
    try: return PL.find(pid, text, int(body.get("frame", 0)))
    finally:
        GPU.release()
@app.post("/api/projects/{pid}/track")
def track(pid: str, body: dict):
    return submit("track", pid, {"fid": body["fid"], "picks": [int(i) for i in body["picks"]], "label": body.get("label") or "object"}, f"Layer: {body.get('label')}")
@app.post("/api/projects/{pid}/edit")
def edit(pid: str, body: dict):
    op = body.get("op")
    if op not in ("delete", "move"): raise HTTPException(400, "op is delete or move")
    args = {"oid": body["oid"], "op": op, "dx": float(body.get("dx", 0)), "dy": float(body.get("dy", 0)), "scale": float(body.get("scale", 1)),
            "with_shadow": bool(body.get("with_shadow", True)), "lock_pixels": bool(body.get("lock", True)), "seed": int(body.get("seed", 0))}
    obj = PL.jload(PL.P(pid, "objects", body["oid"], "object.json")) or {}
    return submit("edit", pid, args, f"{op.title()} {obj.get('label', '')}")
@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    j = JOBS.get(jid)
    if not j or j["status"] != "queued": raise HTTPException(400, "only a queued job can be cancelled")
    j["status"] = "cancelled"; j["stage"] = "cancelled"; return j
@app.get("/api/jobs")
def jobs(): return sorted(JOBS.values(), key=lambda j: -j["created"])[:30]
@app.get("/files/{pid}/{path:path}")
def files(pid: str, path: str):
    shot = os.path.realpath(PL.P(pid))
    p = os.path.realpath(os.path.join(shot, path))
    if not shot.startswith(os.path.realpath(PL.ROOT) + os.sep) or not p.startswith(shot + os.sep) or not os.path.isfile(p): raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "no-cache"})
