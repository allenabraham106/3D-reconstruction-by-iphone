#!/usr/bin/env python3
"""The CPU pipeline's result (run.py: keyframes, metric cameras, dense metric cloud) in ViPE's layout, so
gpu/splat_train.py can train a splat without ViPE:

    python gpu/cpu_to_vipe.py out/IMG_1429 out_cpu_vipe IMG_1429
    python gpu/splat_train.py out_cpu_vipe IMG_1429 IMG_1429.MOV out_splat/IMG_1429 --frames-dir out/IMG_1429/frames

pose/<name>.npz (camera-to-world, OpenCV axes), intrinsics/<name>.npz (fx fy cx cy), depth/<name>.zip (one
half-float EXR per keyframe: the dense cloud z-buffered into that camera at half resolution). Frame index i is
frames/frame_{i:05d}.jpg. Robot-map frame: metres, z up.
"""
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recon"))
from robot_map import read_ply  # noqa: E402


def write_exr(path, D):
    import Imath
    import OpenEXR
    h, w = D.shape
    hdr = OpenEXR.Header(w, h)
    hdr["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
    f = OpenEXR.OutputFile(str(path), hdr)
    f.writePixels({"Z": D.astype(np.float16).tobytes()})
    f.close()


def zbuffer(P, R, t, fx, fy, cx, cy, w, h, splat=1):
    """Nearest depth per pixel of the points P seen by camera (R, t: cam_from_world)."""
    Xc = P @ R.T + t
    z = Xc[:, 2]
    ok = z > 0.05
    u = np.round(Xc[ok, 0] / z[ok] * fx + cx).astype(int)
    v = np.round(Xc[ok, 1] / z[ok] * fy + cy).astype(int)
    z = z[ok]
    D = np.full((h, w), np.inf, np.float32)
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            uu, vv = u + dx, v + dy
            m = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            np.minimum.at(D, (vv[m], uu[m]), z[m])
    D[~np.isfinite(D)] = 0
    return D


def sparse_points(run_out):
    """SfM's sparse points in the robot-map frame (the fast path: no dense stereo needed)."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(run_out / "sfm" / "sparse_best"))
    P = np.array([p.xyz for p in rec.points3D.values()])
    T = json.loads((run_out / "robot" / "map_transform.json").read_text())
    P = P @ np.array(T["R"]).T
    P[:, 2] -= T["z0"]
    P *= T["scale"]
    P[:, :2] -= T["xy0"]
    return P


def main(run_out, out, name, scale=0.5, sparse=False):
    run_out, out = Path(run_out), Path(out)
    cams = json.loads((run_out / "robot" / "cameras_metric.json").read_text())
    P = sparse_points(run_out) if sparse else read_ply(run_out / "robot" / "map_metric.ply")[0].astype(np.float64)
    print(f"{len(P):,} {'sparse SfM' if sparse else 'dense'} points")
    for d in ("pose", "intrinsics", "depth"):
        (out / d).mkdir(parents=True, exist_ok=True)
    inds, poses, intr = [], [], []
    with zipfile.ZipFile(out / "depth" / f"{name}.zip", "w") as z, tempfile.TemporaryDirectory() as tmp:
        for c in cams:
            i = int(Path(c["name"]).stem.split("_")[-1])
            R, t = np.array(c["R"]), np.array(c["t"])
            w, h = int(c["w"]), int(c["h"])
            fx, fy = float(c["fx"]), float(c["fy"])
            cx, cy = w / 2, h / 2  # recon/mvs.py writes centred principal points (fx fy w h only)
            T = np.eye(4); T[:3, :3], T[:3, 3] = R.T, -R.T @ t
            inds.append(i); poses.append(T)
            # intrinsics at depth resolution (splat_train scales them to the frame size by the depth map's size)
            intr.append([fx * scale, fy * scale, cx * scale, cy * scale])
            D = zbuffer(P, R, t, fx * scale, fy * scale, cx * scale, cy * scale, int(w * scale), int(h * scale),
                        splat=2 if sparse else 1)
            write_exr(Path(tmp) / "d.exr", D)
            z.write(Path(tmp) / "d.exr", f"{i:05d}.exr")
    o = np.argsort(inds)
    np.savez(out / "pose" / f"{name}.npz", inds=np.array(inds)[o], data=np.array(poses)[o])
    np.savez(out / "intrinsics" / f"{name}.npz", inds=np.array(inds)[o], data=np.array(intr)[o])
    print(f"{len(inds)} keyframes -> {out} (depth at {scale:g}x)")


if __name__ == "__main__":
    main(*sys.argv[1:4], sparse="--sparse" in sys.argv)
