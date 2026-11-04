"""Ingest a folder recorded by the iOS ScanCapture app (ios/).

Two paths:
  * LiDAR iPhones (Pro models): fuse ARKit LiDAR depth with ARKit metric poses directly.
    No SfM needed; output is metric and gravity-aligned. Seconds on a laptop.
  * Non-LiDAR iPhones: run SfM + dense stereo on the RGB frames, then Sim(3)-align the
    SfM trajectory to the ARKit trajectory, which makes the result metric and gravity-aligned.

Both write <out>/dense.ply + <out>/cameras.json in the same format as mvs.py
(COLMAP convention: cam_from_world, camera x right / y down / z forward).
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np

from mvs import remove_outliers, voxel_downsample, write_ply

# ARKit camera (x right, y up, z back)  ->  CV camera (x right, y down, z forward)
FLIP = np.diag([1.0, -1.0, -1.0])


def load_transforms_json(cap: Path):
    """nerfstudio-style transforms.json (c2w in OpenGL axes == ARKit camera axes, metres)."""
    j = json.loads((cap / "transforms.json").read_text())
    out = []
    for i, f in enumerate(j["frames"]):
        g = lambda k, d=None: f.get(k, j.get(k, d))
        w, h = int(g("w")), int(g("h"))
        K = np.array([[g("fl_x"), 0, g("cx", w / 2)], [0, g("fl_y", g("fl_x")), g("cy", h / 2)], [0, 0, 1.0]])
        T = np.array(f["transform_matrix"], dtype=np.float64)  # row-major 4x4
        Rwc = T[:3, :3] @ FLIP
        C = T[:3, 3]
        R = Rwc.T
        e = {"index": i, "rgb": f["file_path"], "width": w, "height": h,
             "K": K, "R": R, "t": -R @ C, "C": C}
        if f.get("depth_file_path"):
            e["depth"] = f["depth_file_path"]
        conf = f.get("confidence_file_path") or f.get("confidence_path")
        if conf:
            e["confidence"] = conf
        out.append(e)
    return out


def read_depth(cap: Path, f):
    """Depth in metres + confidence (or None) for either capture format."""
    path = cap / f["depth"]
    if path.suffix == ".bin":
        D = np.fromfile(path, dtype=np.float32).reshape(f["depth_height"], f["depth_width"])
    else:  # 16-bit PNG in millimetres
        D = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    conf = None
    if "confidence" in f:
        cp = cap / f["confidence"]
        conf = (np.fromfile(cp, dtype=np.uint8).reshape(D.shape) if cp.suffix == ".bin"
                else cv2.imread(str(cp), cv2.IMREAD_UNCHANGED))
        if conf is not None and conf.ndim == 3:
            conf = conf[..., 0]
        if conf is not None and conf.shape != D.shape:
            conf = cv2.resize(conf, (D.shape[1], D.shape[0]), interpolation=cv2.INTER_NEAREST)
    return D, conf


def load_capture(cap: Path):
    if (cap / "transforms.json").exists() and not (cap / "frames.jsonl").exists():
        return load_transforms_json(cap)
    frames = [json.loads(l) for l in (cap / "frames.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for f in frames:
        T = np.array(f["world_from_camera"], dtype=np.float64).reshape(4, 4).T  # column-major
        K = np.array(f["intrinsics"], dtype=np.float64).reshape(3, 3).T
        Rwc = T[:3, :3] @ FLIP  # world_from_cvcam rotation
        C = T[:3, 3]
        R = Rwc.T  # cam_from_world
        t = -R @ C
        out.append({**f, "R": R, "t": t, "C": C, "K": K})
    return out


def fuse_lidar(cap: Path, frames, out: Path, voxel=0.01, min_conf=2, max_depth=5.0):
    allP, allC = [], []
    for f in frames:
        if "depth" not in f:
            continue
        D, conf = read_depth(cap, f)
        dh, dw = D.shape
        ok = np.isfinite(D) & (D > 0.1) & (D < max_depth)
        if conf is not None:
            ok &= conf >= min_conf
        img = cv2.imread(str(cap / f["rgb"]))
        h, w = img.shape[:2]
        sx, sy = dw / w, dh / h
        K = f["K"].copy()
        K[0] *= sx
        K[1] *= sy
        small = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
        ys, xs = np.nonzero(ok)
        z = D[ys, xs]
        pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs, dtype=float)], 1)
        Xc = (pix @ np.linalg.inv(K).T) * z[:, None]
        Xw = (Xc - f["t"]) @ f["R"]
        allP.append(Xw)
        allC.append(small[ys, xs, ::-1])
    if not allP:
        return False
    P = np.concatenate(allP)
    C = np.concatenate(allC)
    P, C, cnt = voxel_downsample(P, C, voxel)
    keep = cnt >= 2
    P, C = remove_outliers(P[keep], C[keep])
    write_ply(out / "dense.ply", P, C)
    print(f"[arkit] LiDAR fusion: {len(P):,} points (voxel {voxel*100:.0f} cm)")
    return True


def umeyama(src, dst):
    """Similarity (s, R, t) minimising ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = np.trace(np.diag(S) @ E) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def sfm_aligned(cap: Path, frames, out: Path):
    from mvs import run_mvs
    from robot_map import read_ply
    from sfm import run_sfm
    import pycolmap
    img_dir = cap / "rgb"
    run_sfm(img_dir, out / "sfm")
    P, C, views = run_mvs(out / "sfm" / "sparse_best", img_dir, out / "mvs")
    by_name = {Path(f["rgb"]).name: f for f in frames}
    src = np.array([v.C for v in views if v.name in by_name])
    dst = np.array([by_name[v.name]["C"] for v in views if v.name in by_name])
    s, R, t = umeyama(src, dst)
    err = np.linalg.norm((s * src @ R.T + t) - dst, axis=1)
    print(f"[arkit] Sim3 SfM->ARKit: scale={s:.4f}, trajectory RMSE={np.sqrt((err**2).mean())*100:.1f} cm")
    P, C = read_ply(out / "mvs" / "dense.ply")
    write_ply(out / "dense.ply", s * P @ R.T + t, C)
    return True


def ingest(cap: Path, out: Path, force_sfm=False):
    out.mkdir(parents=True, exist_ok=True)
    frames = load_capture(cap)
    has_depth = any("depth" in f for f in frames)
    print(f"[arkit] {len(frames)} frames, LiDAR depth: {has_depth}")
    if has_depth and not force_sfm:
        fuse_lidar(cap, frames, out)
    else:
        sfm_aligned(cap, frames, out)
    cams = [{"name": Path(f["rgb"]).name, "R": f["R"].tolist(), "t": f["t"].tolist(),
             "C": f["C"].tolist(), "fx": float(f["K"][0, 0]), "fy": float(f["K"][1, 1]),
             "w": f["width"], "h": f["height"]} for f in frames]
    (out / "cameras.json").write_text(json.dumps(cams))
    return out / "dense.ply", out / "cameras.json"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--force-sfm", action="store_true")
    a = ap.parse_args()
    ingest(a.capture, a.out, a.force_sfm)
