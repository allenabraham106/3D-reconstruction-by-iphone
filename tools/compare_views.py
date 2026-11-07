#!/usr/bin/env python3
"""Side-by-side comparison of every system on the same scan, from the same camera.

For a few moments of each walk: the real iPhone frame, then what each reconstruction looks
like from exactly that camera (each system's own pose for that moment):

    real photo | CPU pipeline mesh | ViPE mesh | ViPE crisp points | Gaussian splat

plus a dollhouse overview row (CPU mesh | ViPE mesh | ViPE crisp points, ceiling cut).
CPU keyframes are matched to ViPE's frames by image similarity. Mesh / point views are
rendered in the repo viewer (window.__renderCam) through Playwright; splats with gsplat.

    python tools/compare_views.py --scans IMG_6075 IMG_6076 --out compare/
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TW, TH = 640, 360


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def serve(folder):
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                            cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    return proc, f"http://127.0.0.1:{port}/index.html"


def snapcam(node_dir, folder, jobs, query=""):
    proc, url = serve(folder)
    try:
        subprocess.run(["node", str(Path(node_dir) / "snapcam.js"), url + query, "1280", "720",
                        *[f"{i}:{p}" for i, p in jobs]], check=True, cwd=node_dir)
    finally:
        proc.kill()


def snap_overview(node_dir, folder, out, query):
    proc, url = serve(folder)
    try:
        subprocess.run(["node", str(Path(node_dir) / "snap.js"), url + query, str(out),
                        "0.5", "0.0", "0.55", "0.8", "1280", "720", "1"], check=True, cwd=node_dir)
    finally:
        proc.kill()


def label(im, text):
    im = cv2.resize(im, (TW, TH), interpolation=cv2.INTER_AREA)
    cv2.putText(im, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def thumb(img):
    g = cv2.cvtColor(cv2.resize(img, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def render_splat(ckpt_path, k, out, W=1280, H=720):
    import torch
    from gsplat import rasterization
    ck = torch.load(ckpt_path, map_location="cpu")
    dev = torch.device("cuda")
    P = {n: v.to(dev) for n, v in ck["params"].items()}
    K = ck["Ks"][k].clone()
    K[0] *= W / ck["W"]
    K[1] *= H / ck["H"]
    with torch.no_grad():
        o, _, _ = rasterization(P["means"], P["quats"], torch.exp(P["scales"]), torch.sigmoid(P["opacities"]),
                                torch.cat([P["sh0"], P["shN"]], 1), ck["viewmats"][k][None].to(dev),
                                K[None].to(dev), W, H, sh_degree=ck["sh_degree"], near_plane=0.05)
        rgb = (torch.einsum("ij,hwj->hwi", ck["expo_A"][k].to(dev), o[0, ..., :3]) + ck["expo_b"][k].to(dev)).clamp(0, 1)
    cv2.imwrite(str(out), (rgb * 255).byte().cpu().numpy()[..., ::-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans", nargs="+", default=["IMG_6075", "IMG_6077", "IMG_6076", "IMG_6070"])
    ap.add_argument("--cpu", type=Path, default=ROOT / "bench/work/cpu-5070box")
    ap.add_argument("--moments", type=float, nargs="+", default=[0.3, 0.7], help="fractions of the walk")
    ap.add_argument("--node-dir", required=True, help="folder with snapcam.js / snap.js and playwright")
    ap.add_argument("--cut", type=float, default=2.2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    for scan in a.scans:
        work = a.out / scan
        work.mkdir(exist_ok=True)
        cpu_dir = a.cpu / scan
        vdef = ROOT / "out_vipe/default" / scan / "viewer"
        vcrisp = ROOT / "out_vipe/crisp" / scan / "viewer_crisp"
        cpu_cams = json.loads((cpu_dir / "viewer/cameras.json").read_text())
        def_cams = json.loads((vdef / "cameras.json").read_text())
        crisp_cams = json.loads((vcrisp / "cameras.json").read_text())
        num = lambda n: int(Path(n).stem.split("_")[-1])
        # ViPE frames, for matching CPU keyframes by appearance
        cap = cv2.VideoCapture(str(ROOT / "out_vipe/default/vipe/rgb" / f"{scan}.mp4"))
        vth = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            vth.append(thumb(f))
        vth = np.stack(vth)
        # splat to use: ViPE poses for the office, CPU-pipeline poses for the apartments
        if (ROOT / "out_splat/cpuposes" / scan / "splat/ckpt.pt").exists():
            splat_ckpt, splat_label = ROOT / "out_splat/cpuposes" / scan / "splat/ckpt.pt", "Gaussian splat (CPU poses)"
            cache_names = sorted(p.name[:-4] for p in (cpu_dir / "dense/depth_cache").glob("*.npy"))
            splat_k = lambda cpu_name, vipe_i: cache_names.index(cpu_name)
        else:
            import torch
            splat_ckpt, splat_label = ROOT / "out_splat" / scan / "ckpt.pt", "Gaussian splat (ViPE poses)"
            ids = list(torch.load(splat_ckpt, map_location="cpu")["frame_ids"])
            splat_k = lambda cpu_name, vipe_i: ids.index(vipe_i)

        jobs_cpu, jobs_def, jobs_crisp, rows = [], [], [], []
        for mi, frac in enumerate(a.moments):
            ci = int(frac * (len(cpu_cams) - 1))
            cname = cpu_cams[ci]["name"]
            real = cv2.imread(str(cpu_dir / "frames" / cname))
            vi = int(np.argmax(vth.reshape(len(vth), -1) @ thumb(real).ravel()))
            di = int(np.argmin([abs(num(c["name"]) - vi) for c in def_cams]))
            ki = int(np.argmin([abs(num(c["name"]) - vi) for c in crisp_cams]))
            p = {s: work / f"m{mi}_{s}.png" for s in ("cpu", "vipe", "crisp", "splat")}
            jobs_cpu.append((ci, p["cpu"]))
            jobs_def.append((di, p["vipe"]))
            jobs_crisp.append((ki, p["crisp"]))
            render_splat(splat_ckpt, splat_k(cname, vi), p["splat"])
            rows.append((real, p, cname, vi))
            print(f"[compare] {scan} moment {frac:.0%}: CPU keyframe {cname} = ViPE frame {vi}", flush=True)
        snapcam(a.node_dir, cpu_dir / "viewer", jobs_cpu)
        snapcam(a.node_dir, vdef, jobs_def)
        snapcam(a.node_dir, vcrisp, jobs_crisp, "?show=points&ps=2")
        # overview: dollhouse, ceiling cut
        ov = {s: work / f"overview_{s}.png" for s in ("cpu", "vipe", "crisp")}
        snap_overview(a.node_dir, cpu_dir / "viewer", ov["cpu"], f"?cut={a.cut}")
        snap_overview(a.node_dir, vdef, ov["vipe"], f"?cut={a.cut}")
        snap_overview(a.node_dir, vcrisp, ov["crisp"], f"?show=points&ps=2&cut={a.cut}")

        grid = []
        for real, p, cname, vi in rows:
            grid.append(np.concatenate([label(real.copy(), "Real iPhone frame"),
                                        label(cv2.imread(str(p["cpu"])), "CPU pipeline mesh"),
                                        label(cv2.imread(str(p["vipe"])), "ViPE (GPU) mesh"),
                                        label(cv2.imread(str(p["crisp"])), "ViPE crisp points"),
                                        label(cv2.imread(str(p["splat"])), splat_label)], 1))
        blank = np.full((TH, TW, 3), 29, np.uint8)
        grid.append(np.concatenate([label(cv2.imread(str(ov["cpu"])), "Overview: CPU mesh"),
                                    label(cv2.imread(str(ov["vipe"])), "Overview: ViPE mesh"),
                                    label(cv2.imread(str(ov["crisp"])), "Overview: ViPE crisp points"),
                                    blank, blank], 1))
        sheet = np.concatenate(grid, 0)
        cv2.imwrite(str(a.out / f"{scan}_side_by_side.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"[compare] {scan} -> {a.out / (scan + '_side_by_side.jpg')}", flush=True)


if __name__ == "__main__":
    main()
