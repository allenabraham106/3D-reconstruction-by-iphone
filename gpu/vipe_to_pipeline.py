#!/usr/bin/env python3
"""Turn NVIDIA ViPE output into this repo's products: TSDF mesh, robot map, viewer.

ViPE (https://github.com/nv-tlabs/vipe) estimates camera poses, intrinsics and dense,
near-metric depth for every video frame on a GPU. This script reads its artifacts

    <vipe_out>/rgb/<name>.mp4          frames ViPE used
    <vipe_out>/pose/<name>.npz         data: (N, 4, 4) camera-to-world, inds: frame ids
    <vipe_out>/intrinsics/<name>.npz   data: (N, 4) fx, fy, cx, cy, inds: frame ids
    <vipe_out>/depth/<name>.zip        %05d.exr, half-float metric depth (channel Z)

and runs the CPU stages that follow stereo here: TSDF fusion -> mesh, gravity-aligned
robot map (scale kept: ViPE depth is metric, so no phone-height guess) and the viewer.

    python gpu/vipe_to_pipeline.py vipe_out/ IMG_6075 out_vipe/IMG_6075 [--every 2]
"""
import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recon"))


def read_exr_depth(blob):
    """Half-float single-channel EXR (as ViPE writes it) -> float32 array."""
    import Imath
    import OpenEXR
    with tempfile.NamedTemporaryFile(suffix=".exr") as f:
        f.write(blob)
        f.flush()
        exr = OpenEXR.InputFile(f.name)
        dw = exr.header()["dataWindow"]
        w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        z = exr.channel("Z", Imath.PixelType(Imath.PixelType.HALF))
        return np.frombuffer(z, np.float16).reshape(h, w).astype(np.float32)


def reliable_mask(D, window=5, ratio=0.1):
    """ViPE's own reliability rule: local depth range / local mean below `ratio`
    (drops depth edges / flying pixels); plus finite and positive."""
    ok = np.isfinite(D) & (D > 1e-3)
    Dz = np.where(ok, D, 0).astype(np.float32)
    k = np.ones((window, window), np.uint8)
    big = cv2.dilate(np.where(ok, Dz, 0), k)
    small = cv2.erode(np.where(ok, Dz, np.float32(1e9)), k)
    mean = cv2.blur(Dz, (window, window)) / np.maximum(cv2.blur(ok.astype(np.float32), (window, window)), 1e-6)
    return ok & ((big - small) / np.maximum(mean, 1e-6) < ratio)


def load_vipe(vipe_dir: Path, name: str, every=1, max_depth=8.0):
    pose = np.load(vipe_dir / "pose" / f"{name}.npz")
    intr = np.load(vipe_dir / "intrinsics" / f"{name}.npz")
    c2w = dict(zip(pose["inds"].tolist(), pose["data"]))
    K4 = dict(zip(intr["inds"].tolist(), intr["data"]))
    cap = cv2.VideoCapture(str(vipe_dir / "rgb" / f"{name}.mp4"))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    views = []
    with zipfile.ZipFile(vipe_dir / "depth" / f"{name}.zip") as z:
        for fn in sorted(z.namelist()):
            i = int(fn.split(".")[0])
            if i % every or i not in c2w or i >= len(frames):
                continue
            D = read_exr_depth(z.read(fn))
            T = c2w[i]
            R = T[:3, :3].T                 # cam_from_world (OpenCV camera axes)
            t = -R @ T[:3, 3]
            fx, fy, cx, cy = K4.get(i, next(iter(K4.values())))
            img = frames[i]
            if img.shape[:2] != D.shape:    # ViPE may resize; keep depth resolution
                img = cv2.resize(img, (D.shape[1], D.shape[0]), interpolation=cv2.INTER_AREA)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            D = np.where(D < max_depth, D, 0).astype(np.float32)
            views.append(SimpleNamespace(name=f"frame_{i:05d}.jpg", depth=D, K=K, R=R, t=t,
                                         C=T[:3, 3].copy(), img=img, h=D.shape[0], w=D.shape[1]))
    return views


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vipe_dir", type=Path)
    ap.add_argument("name", help="artifact name (the video's file stem)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--every", type=int, default=2, help="fuse every k-th frame (ViPE gives all)")
    ap.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel size in metres")
    a = ap.parse_args()

    from robot_map import build
    from tsdf import fuse_tsdf, write_mesh_ply
    from mvs import write_ply

    views = load_vipe(a.vipe_dir, a.name, a.every)
    masks = [reliable_mask(v.depth) for v in views]
    print(f"[vipe] {len(views)} frames, median reliable depth "
          f"{np.median([m.mean() for m in masks])*100:.0f}% of pixels", flush=True)
    dense = a.out / "dense"
    dense.mkdir(parents=True, exist_ok=True)
    V, F, C, vx = fuse_tsdf(views, masks, a.voxel, log=lambda m: print(m, flush=True))
    write_mesh_ply(dense / "mesh.ply", V, F, C)
    write_ply(dense / "dense.ply", V, C)
    cams = [{"name": v.name, "R": v.R.tolist(), "t": v.t.tolist(), "C": v.C.tolist(),
             "fx": float(v.K[0, 0]), "fy": float(v.K[1, 1]), "w": v.w, "h": v.h} for v in views]
    (dense / "cameras.json").write_text(json.dumps(cams))
    # for bench/benchmark.py: frames tried (every k-th video frame) and kept depth, so the
    # numbers line up with the CPU pipeline's frames/ folder and depth cache
    n_video = int(cv2.VideoCapture(str(a.vipe_dir / "rgb" / f"{a.name}.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    (dense / "bench_info.json").write_text(json.dumps({
        "frames_total": -(-n_video // a.every) if n_video > 0 else None,
        "depth_pixels_kept_median_pct": round(float(np.median([m.mean() for m in masks])) * 100, 1)}))
    meta = build(dense / "dense.ply", dense / "cameras.json", a.out / "robot", metric_scale=True)
    vdir = a.out / "viewer"
    if vdir.exists():
        shutil.rmtree(vdir)
    shutil.copytree(ROOT / "viewer", vdir)
    for src, dst in [("map_metric.ply", "dense.ply"), ("cameras_metric.json", "cameras.json"),
                     ("mesh_metric.ply", "mesh.ply"), ("inferred_floor.ply", "inferred_floor.ply")]:
        if (a.out / "robot" / src).exists():
            shutil.copy(a.out / "robot" / src, vdir / dst)
    (vdir / "view.json").write_text('{"up": [0, 0, 1]}')
    print(f"[vipe] done: mesh {len(F):,} triangles, camera height {meta['camera_height_median_m']:.2f} m "
          f"(metric from ViPE), viewer in {vdir}")


if __name__ == "__main__":
    main()
