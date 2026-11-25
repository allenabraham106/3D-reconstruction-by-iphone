#!/usr/bin/env python3
"""Live "tell the robot where to go": a web page + JSON API on the GPU machine, any prompt, any scan.

    python gpu/goto_server.py --scans gpu/goto_scans.json --port 8765
    # open http://<this machine>:8765 , pick a scan, type "take me to the red toolbox"

Each scan in the JSON file names its splat checkpoint, ViPE folder + video stem, robot/ folder, and a
cache folder for the CLIP crop index (built once, ~1.5 min on the GPU). Scans load on first use and
stay in memory. A prompt is found + planned in ~1 s (the page shows the match and the path right away),
then the trip is rendered from the splat (~10-30 s) and the video appears.

API (JSON):
    GET  /api/scans                         [{name, title, loaded}]
    POST /api/goto   {scan, prompt, render}  -> {job}       (render defaults to true)
    GET  /api/job/<job>                      -> {state: queued|finding|rendering|done|error, progress, result, files}
    GET  /api/plan?scan=..&prompt=..         -> result.json without a video (what ros2/pioneer_map's goto node calls)
    GET  /files/<job>/<plan.png|match.jpg|goto.mp4|arrived.jpg|result.json>
"""
import argparse
import json
import mimetypes
import re
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu"))
PAGE = ROOT / "gpu" / "goto_web" / "index.html"


class Hub:
    def __init__(self, scans, work, min_conf):
        self.cfg = {s["name"]: s for s in scans}
        self.work, self.min_conf = Path(work), min_conf
        self.engines, self.jobs = {}, {}
        self.load_lock, self.gpu = threading.Lock(), threading.Lock()  # one GPU job at a time

    def engine(self, name):
        with self.load_lock:
            if name not in self.engines:
                from goto import GotoEngine
                c = self.cfg[name]
                rel = lambda p: p if Path(p).is_absolute() else ROOT / p  # noqa: E731
                self.engines[name] = GotoEngine(rel(c["ckpt"]), rel(c["vipe"]), c["video"], rel(c["robot"]),
                                                rel(c.get("cache", f"out_goto/{name}")))
            return self.engines[name]

    def start(self, scan, prompt, render=True):
        if scan not in self.cfg:
            raise KeyError(f"unknown scan {scan!r}")
        jid = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:6]
        job = dict(id=jid, scan=scan, prompt=prompt, state="queued", progress=0.0, result=None, error=None,
                   dir=self.work / jid, t0=time.time())
        self.jobs[jid] = job
        threading.Thread(target=self._run, args=(job, render), daemon=True).start()
        return jid

    def _run(self, job, render):
        try:
            with self.gpu:
                job["state"] = "finding"
                eng = self.engine(job["scan"])
                res = eng.run(job["prompt"], job["dir"], render=False, min_conf=self.min_conf)
                job["result"] = res
                if render and res.get("found"):
                    job["state"] = "rendering"
                    pl = eng.plan(eng.find(job["prompt"]))
                    t1 = time.time()
                    vs = eng.render(pl, res["target_m"], job["dir"] / "goto.mp4", preset="veryfast",
                                    progress=lambda p: job.__setitem__("progress", round(p, 3)))
                    res.update(video_s=round(vs, 1), render_s=round(time.time() - t1, 1))
                    (job["dir"] / "result.json").write_text(json.dumps(res, indent=1))
            job["state"], job["progress"] = "done", 1.0
        except Exception as e:  # noqa: BLE001 - reported to the page
            traceback.print_exc()
            job["state"], job["error"] = "error", f"{type(e).__name__}: {e}"

    def view(self, job):
        files = [f.name for f in sorted(job["dir"].glob("*")) if f.suffix in (".png", ".jpg", ".mp4", ".json")] \
            if job["dir"].exists() else []
        return {k: job[k] for k in ("id", "scan", "prompt", "state", "progress", "result", "error")} | {
            "files": files, "elapsed_s": round(time.time() - job["t0"], 1)}


def make_handler(hub):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("[http] " + fmt % args + "\n")

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/scans":
                return self._send(200, [{"name": n, "title": c.get("title", n), "loaded": n in hub.engines}
                                        for n, c in hub.cfg.items()])
            m = re.fullmatch(r"/api/job/([\w-]+)", u.path)
            if m:
                job = hub.jobs.get(m.group(1))
                return self._send(200, hub.view(job)) if job else self._send(404, {"error": "no such job"})
            if u.path == "/api/plan":
                q = parse_qs(u.query)
                scan, prompt = q.get("scan", [None])[0], q.get("prompt", [""])[0]
                if scan not in hub.cfg or not prompt:
                    return self._send(400, {"error": "need scan and prompt"})
                with hub.gpu:
                    res = hub.engine(scan).run(prompt, hub.work / ("plan-" + uuid.uuid4().hex[:6]), render=False,
                                               min_conf=hub.min_conf)
                return self._send(200, res)
            m = re.fullmatch(r"/files/([\w-]+)/([\w.]+)", u.path)
            if m and m.group(1) in hub.jobs:
                f = hub.jobs[m.group(1)]["dir"] / m.group(2)
                if f.is_file():
                    return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/api/goto":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                prompt = str(body.get("prompt", "")).strip()[:200]
                if not prompt:
                    return self._send(400, {"error": "type where to go"})
                return self._send(200, {"job": hub.start(body.get("scan"), prompt, bool(body.get("render", True)))})
            except KeyError as e:
                return self._send(400, {"error": str(e)})
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans", type=Path, default=ROOT / "gpu" / "goto_scans.json")
    ap.add_argument("--work", type=Path, default=ROOT / "out_goto" / "live")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other machines on the network")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--min-conf", type=float, default=0.14, help="below this a prompt is reported as not in the scan")
    ap.add_argument("--preload", action="store_true", help="load every scan at start instead of on first use")
    a = ap.parse_args()
    scans = [s for s in json.loads(a.scans.read_text())["scans"]
             if (ROOT / s["ckpt"]).exists() or Path(s["ckpt"]).exists()]
    if not scans:
        raise SystemExit(f"no scan in {a.scans} has its splat checkpoint yet")
    hub = Hub(scans, a.work, a.min_conf)
    if a.preload:
        for s in scans:
            hub.engine(s["name"])
    print(f"[goto_server] {len(scans)} scan(s): {', '.join(s['name'] for s in scans)} -> http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(hub)).serve_forever()


if __name__ == "__main__":
    main()
