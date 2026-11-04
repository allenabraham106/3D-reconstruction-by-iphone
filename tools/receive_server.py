#!/usr/bin/env python3
"""Tiny upload server for PhoneScan. Run it on a laptop, workstation or the robot's computer:

    python3 tools/receive_server.py --port 8765 --dir captures

Then in the app: Settings → Upload server → http://<this-machine-ip>:8765, and tap
"Upload to server" on a capture. Each upload is saved as captures/<name>.zip and extracted to
captures/<name>/. Use --on-upload to kick off processing automatically, e.g.

    python3 tools/receive_server.py --on-upload "python3 tools/fuse_depth.py {dir}"

Standard library only. No auth: run it on a trusted network (home Wi-Fi, Tailscale, robot LAN).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHUNK = 1 << 20
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_extract(zip_path: Path, dest: Path) -> int:
    """Extracts a zip, refusing absolute paths and '..' escapes. Strips a single top-level folder."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]
        tops = {n.split("/", 1)[0] for n in names}
        strip = len(tops) == 1 and all("/" in n for n in names if not n.endswith("/"))
        count = 0
        for info in zf.infolist():
            name = info.filename
            if name.startswith("__MACOSX/") or info.is_dir():
                continue
            rel = name.split("/", 1)[1] if strip else name
            target = (dest / rel).resolve()
            if not str(target).startswith(str(dest) + "/"):
                raise ValueError(f"unsafe path in zip: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            count += 1
    return count


def make_handler(root: Path, on_upload: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PhoneScanReceiver/1.0"

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            captures = sorted(p.name for p in root.iterdir() if p.is_dir())
            self._json(200, {"ok": True, "captures": captures})

        def do_POST(self):
            if self.path.rstrip("/") != "/upload":
                return self._json(404, {"ok": False, "error": "POST to /upload"})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return self._json(411, {"ok": False, "error": "Content-Length required"})
            name = SAFE_NAME.sub("_", self.headers.get("X-Capture-Name") or "capture").strip("._") or "capture"
            zip_path = root / f"{name}.zip"
            remaining = length
            with open(zip_path, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            if remaining:
                return self._json(400, {"ok": False, "error": "upload truncated"})

            out_dir = root / name
            try:
                files = safe_extract(zip_path, out_dir)
            except (zipfile.BadZipFile, ValueError) as e:
                return self._json(400, {"ok": False, "error": str(e)})

            self.log_message("received %s: %d bytes, %d files -> %s", name, length, files, out_dir)
            if on_upload:
                cmd = on_upload.format(dir=shlex.quote(str(out_dir)), name=shlex.quote(name))
                threading.Thread(target=subprocess.run, args=(cmd,), kwargs={"shell": True}, daemon=True).start()
            self._json(200, {"ok": True, "name": name, "files": files, "dir": str(out_dir)})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dir", type=Path, default=Path("captures"))
    ap.add_argument("--on-upload", default=None,
                    help="shell command run after each upload; {dir} and {name} are substituted")
    args = ap.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(args.dir.resolve(), args.on_upload))
    print(f"PhoneScan receiver on http://{args.host}:{args.port}, saving to {args.dir.resolve()}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
