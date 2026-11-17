#!/usr/bin/env python3
"""The fast route, video -> Gaussian splat in ~10 minutes: keyframes, SfM poses, then a splat initialised from SfM's
sparse points (no dense stereo). IMG_1429 on one RTX 3090: 9.4 min, held-out 24.79 dB / LPIPS 0.187. 1.5M Gaussians
beat 3M (24.32 dB, 11.3 min): fewer, larger Gaussians generalise better from 287 frames.

    python gpu/fast_splat.py IMG_1429.MOV out_fast/IMG_1429 [--frames 300] [--steps 7000]
"""
import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "recon"), str(ROOT / "gpu")]
from cpu_to_vipe import write_exr, zbuffer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--sfm-features", type=int, default=4096, help="SIFT features per frame for SfM")
    ap.add_argument("--overlap", type=int, default=10, help="SfM sequential matching: neighbours per frame")
    a, rest = ap.parse_known_args()
    name, out = a.video.stem, a.out
    out.mkdir(parents=True, exist_ok=True)
    t = {"start": time.time()}

    from frames import extract
    extract(a.video, out / "frames", a.frames, 1280)
    t["frames"] = time.time()

    import pycolmap
    assert pycolmap.has_cuda, "fast route needs pycolmap-cuda12 (GPU SIFT + matching), not the CPU pycolmap"
    from sfm import run_sfm
    run_sfm(out / "frames", out / "sfm", continuous_video=True, max_features=a.sfm_features, overlap=a.overlap)
    t["sfm"] = time.time()

    rec = pycolmap.Reconstruction(str(out / "sfm" / "sparse_best"))
    P = np.array([p.xyz for p in rec.points3D.values()])
    v = out / "vipe"
    for d in ("pose", "intrinsics", "depth"):
        (v / d).mkdir(parents=True, exist_ok=True)
    inds, poses, intr = [], [], []
    with zipfile.ZipFile(v / "depth" / f"{name}.zip", "w") as z, tempfile.TemporaryDirectory() as tmp:
        for im in rec.images.values():
            i = int(Path(im.name).stem.split("_")[-1])
            cw = im.cam_from_world()  # pycolmap >= 4 (requirements.txt)
            R, tr = cw.rotation.matrix(), np.asarray(cw.translation)
            cam = rec.cameras[im.camera_id]
            fx, fy, cx, cy = cam.focal_length_x, cam.focal_length_y, cam.principal_point_x, cam.principal_point_y
            T = np.eye(4); T[:3, :3], T[:3, 3] = R.T, -R.T @ tr
            inds.append(i); poses.append(T); intr.append([fx / 2, fy / 2, cx / 2, cy / 2])
            write_exr(Path(tmp) / "d.exr", zbuffer(P, R, tr, fx / 2, fy / 2, cx / 2, cy / 2, cam.width // 2, cam.height // 2, splat=2))
            z.write(Path(tmp) / "d.exr", f"{i:05d}.exr")
    o = np.argsort(inds)
    np.savez(v / "pose" / f"{name}.npz", inds=np.array(inds)[o], data=np.array(poses)[o])
    np.savez(v / "intrinsics" / f"{name}.npz", inds=np.array(inds)[o], data=np.array(intr)[o])
    t["convert"] = time.time()

    subprocess.run([sys.executable, "-u", str(ROOT / "gpu" / "splat_train.py"), str(v), name, str(a.video), str(out / "splat"),
                    "--frames-dir", str(out / "frames"), "--steps", str(a.steps), "--aa", "1", "--cap", "1500000", "--video-frames", "240", *rest],
                   check=True)
    t["splat"] = time.time()

    from splat_map import run as splat_map  # mesh + nav2 robot map from the splat's depth, on the GPU
    splat_map(out)
    t["map"] = time.time()
    keys = list(t)
    times = {k: round(t[k] - t[keys[i - 1]], 1) for i, k in enumerate(keys) if i}
    times["total"] = round(t["map"] - t["start"], 1)
    (out / "timings.json").write_text(json.dumps(times, indent=1))
    print("[fast] seconds:", json.dumps(times), flush=True)


if __name__ == "__main__":
    main()
