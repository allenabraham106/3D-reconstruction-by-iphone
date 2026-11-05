#!/usr/bin/env python3
"""Write a scan as a COLMAP text model + images folder, the input Gaussian-splat trainers
expect (gsplat's examples/simple_trainer.py, nerfstudio splatfacto, ...).

    data_dir/images/frame_00000.jpg ...
    data_dir/sparse/0/{cameras,images,points3D}.txt

Two sources:
    python gpu/to_colmap.py data_dir --vipe out_vipe/vipe IMG_6077   # ViPE poses (metric)
    python gpu/to_colmap.py data_dir --cpu out/IMG_6077              # this repo's CPU SfM

Splat initial points: ViPE -> back-projected reliable depth; CPU -> the SfM points.
"""
import argparse
import shutil
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recon"))
sys.path.insert(0, str(ROOT / "gpu"))


def rot_to_quat(R):
    """Rotation matrix -> (w, x, y, z), COLMAP's order."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


MODEL_IDS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3, "OPENCV": 4}


def write_model_bin(sp, cams, images, points):
    """Same model in COLMAP's binary format. Trainers read the .bin files first, and the
    text reader bundled with gsplat stops at the first blank line (an image with no 2D
    points), so the binary files are the ones that matter."""
    with open(sp / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid, (model, w, h, params) in cams.items():
            f.write(struct.pack("<IiQQ", cid, MODEL_IDS[model], w, h))
            f.write(struct.pack(f"<{len(params)}d", *params))
    with open(sp / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for i, (name, cid, R, t) in enumerate(images, 1):
            f.write(struct.pack("<I4d3dI", i, *rot_to_quat(R), *np.asarray(t, float), cid))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))  # no 2D points
    xyz, rgb = points
    with open(sp / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i, (p, c) in enumerate(zip(xyz, rgb), 1):
            f.write(struct.pack("<Q3d3BdQ", i, *map(float, p), *map(int, c), 0.0, 0))


def write_model(out, cams, images, points):
    """cams: {id: (model, w, h, params)}; images: [(name, cam_id, R, t)] (cam_from_world);
    points: (N,3) xyz, (N,3) uint8 rgb."""
    sp = out / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    with open(sp / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid, (model, w, h, params) in cams.items():
            f.write(f"{cid} {model} {w} {h} " + " ".join(f"{p:.6f}" for p in params) + "\n")
    with open(sp / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n# POINTS2D[] (none)\n")
        for i, (name, cid, R, t) in enumerate(images, 1):
            q = rot_to_quat(R)
            f.write(f"{i} " + " ".join(f"{v:.9f}" for v in (*q, *t)) + f" {cid} {name}\n\n")
    xyz, rgb = points
    with open(sp / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, (p, c) in enumerate(zip(xyz, rgb), 1):
            f.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} 0\n")
    write_model_bin(sp, cams, images, points)
    return sp


def from_vipe(out, vipe_dir, name, every, n_points):
    from vipe_to_pipeline import load_vipe, reliable_mask
    import mvs
    views = load_vipe(vipe_dir, name, every=every)
    cams, images, P, C = {}, [], [], []
    for v in views:
        key = (v.w, v.h, round(v.K[0, 0], 2), round(v.K[1, 1], 2), round(v.K[0, 2], 2), round(v.K[1, 2], 2))
        cid = next((k for k, c in cams.items() if c[-1] == key), None)
        if cid is None:
            cid = len(cams) + 1
            cams[cid] = ("PINHOLE", v.w, v.h, [v.K[0, 0], v.K[1, 1], v.K[0, 2], v.K[1, 2]], key)
        cv2.imwrite(str(out / "images" / v.name), v.img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        images.append((v.name, cid, v.R, v.t))
        m = reliable_mask(v.depth) & (v.depth > 0)
        X, col = mvs.backproject(v, v.depth, m)
        P.append(X[::97]), C.append(col[::97])
    P, C = np.concatenate(P), np.concatenate(C)
    if len(P) > n_points:
        k = np.random.default_rng(0).choice(len(P), n_points, replace=False)
        P, C = P[k], C[k]
    return {k: c[:4] for k, c in cams.items()}, images, (P, C.astype(np.uint8))  # RGB


def from_cpu(out, scan):
    import pycolmap
    rec = pycolmap.Reconstruction(str(scan / "sfm" / "sparse_best"))
    cams = {}
    for cid, cam in rec.cameras.items():
        p = list(cam.params)
        if cam.model.name not in ("SIMPLE_RADIAL", "PINHOLE", "SIMPLE_PINHOLE", "RADIAL", "OPENCV"):
            raise SystemExit(f"camera model {cam.model.name} not handled")
        cams[cid] = (cam.model.name, cam.width, cam.height, p)
    images = []
    for im in sorted(rec.images.values(), key=lambda im: im.name):
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        images.append((im.name, im.camera_id, T.rotation.matrix(), T.translation))
        shutil.copy(scan / "frames" / im.name, out / "images" / im.name)
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    rgb = np.array([p.color for p in rec.points3D.values()], np.uint8)
    return cams, images, (xyz, rgb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--vipe", nargs=2, metavar=("VIPE_DIR", "NAME"))
    ap.add_argument("--cpu", type=Path, metavar="RUN_DIR", help="out/<name> from run.py")
    ap.add_argument("--every", type=int, default=2, help="ViPE: use every k-th frame")
    ap.add_argument("--points", type=int, default=200_000, help="ViPE: initial splat points")
    a = ap.parse_args()
    if bool(a.vipe) == bool(a.cpu):
        ap.error("give exactly one of --vipe or --cpu")
    if a.out.exists():
        shutil.rmtree(a.out)
    (a.out / "images").mkdir(parents=True)
    if a.vipe:
        cams, images, pts = from_vipe(a.out, Path(a.vipe[0]), a.vipe[1], a.every, a.points)
    else:
        cams, images, pts = from_cpu(a.out, a.cpu)
    sp = write_model(a.out, cams, images, pts)
    print(f"[to_colmap] {len(images)} images, {len(cams)} camera(s), {len(pts[0]):,} points -> {sp}")


if __name__ == "__main__":
    main()
