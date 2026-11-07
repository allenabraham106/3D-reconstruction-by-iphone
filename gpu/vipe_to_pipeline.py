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


def backproject_points(views, masks, voxel, stride=2):
    """Trusted depth pixels -> world points with their own colour, one (mean) point per voxel."""
    P, C = [], []
    for v, m in zip(views, masks):
        ys, xs = np.nonzero(m[::stride, ::stride])
        ys, xs = ys * stride, xs * stride
        z = v.depth[ys, xs].astype(np.float64)
        Xc = np.stack([(xs - v.K[0, 2]) / v.K[0, 0] * z, (ys - v.K[1, 2]) / v.K[1, 1] * z, z], 1)
        P.append(((Xc - v.t) @ v.R).astype(np.float32))
        C.append(v.img[ys, xs, ::-1])
    P, C = np.concatenate(P), np.concatenate(C).astype(np.float32)
    key = np.floor((P - P.min(0)) / voxel).astype(np.int64)
    key = (key[:, 0] * 1_000_003 + key[:, 1]) * 1_000_033 + key[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    Pm = np.stack([np.bincount(inv, P[:, d], len(cnt)) for d in range(3)], 1)
    Cm = np.stack([np.bincount(inv, C[:, d], len(cnt)) for d in range(3)], 1)
    keep = cnt >= 2  # seen by at least two pixels: drops lone speckles
    return (Pm[keep] / cnt[keep, None]).astype(np.float32), (Cm[keep] / cnt[keep, None]).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vipe_dir", type=Path)
    ap.add_argument("name", help="artifact name (the video's file stem)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--every", type=int, default=2, help="fuse every k-th frame (ViPE gives all)")
    ap.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel size in metres (robot mesh)")
    ap.add_argument("--tsdf", choices=["auto", "gpu", "cpu"], default="auto",
                    help="TSDF fusion on the GPU (gpu/tsdf_gpu.py, same result, much faster) "
                         "or the CPU (recon/tsdf.py); auto = GPU if PyTorch sees CUDA")
    ap.add_argument("--max-depth", type=float, default=8.0, help="ignore depth beyond this (m)")
    ap.add_argument("--crisp", action="store_true",
                    help="also build sharp visuals for the viewer: strict edge filter + multi-view "
                         "agreement on ViPE depth (gpu/depth_filter.py), a fine TSDF mesh and a "
                         "per-pixel coloured point cloud. The robot map is unchanged.")
    ap.add_argument("--crisp-voxel", type=float, default=0.01, help="--crisp mesh voxel (m)")
    ap.add_argument("--crisp-max-depth", type=float, default=5.0, help="--crisp depth range (m)")
    ap.add_argument("--points-voxel", type=float, default=0.006, help="--crisp point spacing (m)")
    a = ap.parse_args()

    from robot_map import build, to_map
    from tsdf import fuse_tsdf, write_mesh_ply
    from mvs import write_ply
    log = lambda m: print(m, flush=True)

    views = load_vipe(a.vipe_dir, a.name, a.every, a.max_depth)
    masks = [reliable_mask(v.depth) for v in views]
    print(f"[vipe] {len(views)} frames, median reliable depth "
          f"{np.median([m.mean() for m in masks])*100:.0f}% of pixels", flush=True)
    dense = a.out / "dense"
    dense.mkdir(parents=True, exist_ok=True)
    use_gpu = a.tsdf == "gpu"
    if a.tsdf == "auto":
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
    sys.path.insert(0, str(ROOT / "gpu"))
    if use_gpu:
        from tsdf_gpu import fuse_tsdf_gpu as fuse
    else:
        fuse = fuse_tsdf
    V, F, C, vx = fuse(views, masks, a.voxel, log=log)
    write_mesh_ply(dense / "mesh.ply", V, F, C)
    write_ply(dense / "dense.ply", V, C)
    cams = [{"name": v.name, "R": v.R.tolist(), "t": v.t.tolist(), "C": v.C.tolist(),
             "fx": float(v.K[0, 0]), "fy": float(v.K[1, 1]), "w": v.w, "h": v.h} for v in views]
    (dense / "cameras.json").write_text(json.dumps(cams))
    # for bench/benchmark.py: frames tried (every k-th video frame) and kept depth, so the
    # numbers line up with the CPU pipeline's frames/ folder and depth cache
    n_video = int(cv2.VideoCapture(str(a.vipe_dir / "rgb" / f"{a.name}.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    (dense / "bench_info.json").write_text(json.dumps({
        "frames_total": -(-n_video // a.every) if n_video > 0 else None, "every": a.every,
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
          f"(metric from ViPE), viewer in {vdir}", flush=True)

    if a.crisp:
        from depth_filter import consistency_masks, edge_mask
        cm = [m & edge_mask(v.depth) & (v.depth < a.crisp_max_depth) for v, m in zip(views, masks)]
        cm = consistency_masks(views, cm, log=log)
        kept = float(np.median([m.mean() for m in cm])) * 100
        print(f"[crisp] depth kept after edge + multi-view filter: {kept:.0f}% of pixels (median)", flush=True)
        T = json.loads((a.out / "robot" / "map_transform.json").read_text())
        Vc, Fc, Cc, _ = fuse(views, cm, a.crisp_voxel, log=log)
        Vc = to_map(Vc, T)
        P, PC = backproject_points(views, cm, a.points_voxel)
        P = to_map(P.astype(np.float64), T)
        above = P[:, 2] > -0.03                      # nothing under the floor (reflections)
        P, PC = P[above], PC[above]
        cdir = a.out / "crisp"
        cdir.mkdir(exist_ok=True)
        write_mesh_ply(cdir / "mesh_crisp_metric.ply", Vc, Fc, Cc)
        write_ply(cdir / "points_crisp_metric.ply", P, PC)
        (cdir / "crisp_info.json").write_text(json.dumps({
            "depth_kept_median_pct": round(kept, 1), "mesh_voxel_m": a.crisp_voxel,
            "mesh_triangles": int(len(Fc)), "points": int(len(P)), "points_voxel_m": a.points_voxel,
            "max_depth_m": a.crisp_max_depth}))
        cv = a.out / "viewer_crisp"
        if cv.exists():
            shutil.rmtree(cv)
        shutil.copytree(ROOT / "viewer", cv)
        shutil.copy(cdir / "points_crisp_metric.ply", cv / "dense.ply")
        shutil.copy(cdir / "mesh_crisp_metric.ply", cv / "mesh.ply")
        shutil.copy(a.out / "robot" / "cameras_metric.json", cv / "cameras.json")
        (cv / "view.json").write_text('{"up": [0, 0, 1]}')
        print(f"[crisp] mesh {len(Fc):,} triangles at {a.crisp_voxel*100:.0f} cm, {len(P):,} points; "
              f"viewer in {cv} (open index.html?show=points&cut=2.2)", flush=True)


if __name__ == "__main__":
    main()
