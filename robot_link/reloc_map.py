#!/usr/bin/env python3
"""Build a relocalization map from a run.py output folder.

The reconstruction already knows every keyframe's pose in the robot map frame
(`robot/cameras_metric.json`: metres, Z up, floor at Z = 0, the same frame as the nav2
occupancy grid). This script detects SIFT features in those keyframes, matches nearby frames,
and triangulates the matches with the known poses. The result is a set of 3D landmarks, each with
the descriptors it was seen with, saved as `robot/reloc_map.npz`.

Localizing later is then a single PnP problem: match a new image's features to the landmarks,
and solve for the camera pose. No SfM has to run on the robot.

    python -m robot_link.reloc_map out/living-room
    python -m robot_link.reloc_map out/living-room --images out/living-room/frames
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Keyframe:
    name: str
    path: Path
    K: np.ndarray        # 3x3 for the image as stored on disk
    R: np.ndarray        # cam_from_world rotation (OpenCV camera axes), map frame
    t: np.ndarray        # cam_from_world translation

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t[:, None]])

    @property
    def C(self) -> np.ndarray:
        return -self.R.T @ self.t


def root_sift(desc: np.ndarray) -> np.ndarray:
    """RootSIFT (Arandjelovic & Zisserman 2012): L1-normalize, then sqrt. Matches better than raw SIFT."""
    desc = desc.astype(np.float32)
    desc /= np.abs(desc).sum(1, keepdims=True) + 1e-7
    return np.sqrt(desc)


def detect(gray: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02)
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None or not kps:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.array([k.pt for k in kps], np.float32), root_sift(desc)


def find_images_dir(out_dir: Path) -> list[Path]:
    """Places run.py leaves keyframe images: video frames, or the app capture's images/."""
    cands = [out_dir / "frames", out_dir / "images", out_dir / "dense" / "images"]
    src = out_dir / "source.txt"  # optional pointer written by the robot link server
    if src.exists():
        cands.insert(0, Path(src.read_text().strip()) / "images")
    return [c for c in cands if c.is_dir()]


def load_keyframes(out_dir: Path, images: Path | None = None) -> list[Keyframe]:
    cams = json.loads((out_dir / "robot" / "cameras_metric.json").read_text())
    dirs = [images] if images else find_images_dir(out_dir)
    frames = []
    for c in cams:
        path = next((d / c["name"] for d in dirs if (d / c["name"]).exists()), None)
        if path is None:
            path = next((p for d in dirs for p in d.glob(Path(c["name"]).name)), None)
        if path is None:
            continue
        w, h = int(c["w"]), int(c["h"])
        img_w, img_h = _image_size(path)
        sx, sy = img_w / w, img_h / h
        K = np.array([[c["fx"] * sx, 0, c.get("cx", w / 2) * sx],
                      [0, c["fy"] * sy, c.get("cy", h / 2) * sy],
                      [0, 0, 1]], np.float64)
        frames.append(Keyframe(c["name"], path, K, np.array(c["R"], np.float64), np.array(c["t"], np.float64)))
    if not frames:
        raise SystemExit(f"no keyframe images found for {out_dir} (looked in {[str(d) for d in dirs]}); pass --images")
    return frames


def _image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return img.shape[1], img.shape[0]


def triangulate_pair(a: Keyframe, b: Keyframe, pa: np.ndarray, pb: np.ndarray,
                     max_reproj: float, min_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """DLT-triangulate matched pixels with known poses; returns (points, keep-mask)."""
    X = cv2.triangulatePoints(a.P, b.P, pa.T.astype(np.float64), pb.T.astype(np.float64))
    X = (X[:3] / X[3]).T
    keep = np.isfinite(X).all(1)
    for f, p in ((a, pa), (b, pb)):
        Xc = X @ f.R.T + f.t
        keep &= Xc[:, 2] > 0.05
        proj = Xc @ f.K.T
        uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)
        keep &= np.linalg.norm(uv - p, axis=1) < max_reproj
    ra = X - a.C
    rb = X - b.C
    cosang = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
    keep &= cosang < np.cos(np.radians(min_angle_deg))
    return X, keep


def build(out_dir: Path, images: Path | None = None, n_features: int = 3000, window: int = 3,
          ratio: float = 0.8, max_reproj: float = 2.0, min_angle_deg: float = 1.5,
          max_points: int = 250_000, verbose: bool = True) -> dict:
    t0 = time.time()
    frames = load_keyframes(out_dir, images)
    feats = []
    for f in frames:
        gray = cv2.imread(str(f.path), cv2.IMREAD_GRAYSCALE)
        feats.append(detect(gray, n_features))
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    pts, descs, owners, seen_by = [], [], [], []
    for i in range(len(frames)):
        for j in range(i + 1, min(len(frames), i + 1 + window)):
            (ka, da), (kb, db) = feats[i], feats[j]
            if len(da) < 8 or len(db) < 8:
                continue
            knn = matcher.knnMatch(da, db, k=2)
            good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
            if len(good) < 8:
                continue
            ia = np.array([m.queryIdx for m in good])
            ib = np.array([m.trainIdx for m in good])
            X, keep = triangulate_pair(frames[i], frames[j], ka[ia], kb[ib], max_reproj, min_angle_deg)
            if not keep.any():
                continue
            base = sum(len(p) for p in pts)
            n = int(keep.sum())
            pts.append(X[keep])
            # both observations' descriptors are kept: better recall from different viewpoints
            descs.extend([da[ia[keep]], db[ib[keep]]])
            owners.extend([np.arange(base, base + n), np.arange(base, base + n)])
            seen_by.append(np.stack([np.full(n, i), np.full(n, j)], 1))

    if not pts:
        raise SystemExit("could not triangulate any landmarks: are the images textured and the poses right?")
    P = np.concatenate(pts).astype(np.float32)
    D = np.concatenate(descs).astype(np.float32)
    O = np.concatenate(owners).astype(np.int32)
    S = np.concatenate(seen_by).astype(np.int32)
    if len(P) > max_points:
        rng = np.random.default_rng(0)
        keep_pts = np.sort(rng.choice(len(P), max_points, replace=False))
        remap = np.full(len(P), -1, np.int64)
        remap[keep_pts] = np.arange(len(keep_pts))
        km = remap[O] >= 0
        P, S, D, O = P[keep_pts], S[keep_pts], D[km], remap[O[km]].astype(np.int32)

    robot = out_dir / "robot"
    np.savez_compressed(
        robot / "reloc_map.npz", points=P, descriptors=D.astype(np.float16), owner=O, seen_by=S,
        cam_centres=np.array([f.C for f in frames], np.float32),
        frame_names=np.array([f.name for f in frames]),
    )
    info = {"keyframes": len(frames), "landmarks": int(len(P)), "descriptors": int(len(D)),
            "extent_m": (np.percentile(P, 99, 0) - np.percentile(P, 1, 0)).round(2).tolist(),
            "seconds": round(time.time() - t0, 1)}
    (robot / "reloc_meta.json").write_text(json.dumps(info, indent=2))
    if verbose:
        print("[reloc_map]", json.dumps(info))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="a run.py output folder (contains robot/cameras_metric.json)")
    ap.add_argument("--images", type=Path, default=None, help="keyframe image folder, if not auto-detected")
    ap.add_argument("--features", type=int, default=3000)
    ap.add_argument("--window", type=int, default=3, help="match each keyframe with the next N")
    a = ap.parse_args()
    build(a.out_dir, a.images, a.features, a.window)


if __name__ == "__main__":
    main()
