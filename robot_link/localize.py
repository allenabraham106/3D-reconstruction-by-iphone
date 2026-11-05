#!/usr/bin/env python3
"""Localize a camera image inside a relocalization map (see reloc_map.py).

SIFT features in the query image are matched to the map's landmarks, then PnP + RANSAC solves
for the camera pose. The pose is in the robot map frame: metres, Z up, floor at Z = 0, the same
frame as robot/occupancy.yaml. So a result can be drawn straight onto the nav2 map, or published
as the robot's initial pose.

    python -m robot_link.localize out/living-room photo.jpg --K 1450,1450,960,720
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .reloc_map import detect


@dataclass
class Pose:
    ok: bool
    reason: str
    x: float = 0.0             # camera position in the map frame, metres
    y: float = 0.0
    z: float = 0.0             # height above the floor
    yaw_deg: float = 0.0       # heading of the camera's view direction, CCW from +X
    pitch_deg: float = 0.0     # + looking up
    matches: int = 0           # 2D-3D matches tried
    inliers: int = 0           # matches consistent with the pose
    reproj_px: float = 0.0     # mean reprojection error of the inliers
    R: list | None = None      # cam_from_world rotation (OpenCV axes)
    t: list | None = None
    grid_px: list | None = None  # [col, row] in occupancy.pgm, if the map has one

    def to_dict(self) -> dict:
        return asdict(self)


class Localizer:
    def __init__(self, out_dir: Path, min_inliers: int = 25, ratio: float = 0.85):
        self.out_dir = Path(out_dir)
        data = np.load(self.out_dir / "robot" / "reloc_map.npz")
        self.points = data["points"].astype(np.float64)
        self.desc = data["descriptors"].astype(np.float32)
        self.owner = data["owner"]
        self.min_inliers = min_inliers
        self.ratio = ratio
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        self.matcher.add([self.desc])
        self.matcher.train()
        self.grid = self._load_grid()

    def _load_grid(self) -> dict | None:
        y = self.out_dir / "robot" / "occupancy.yaml"
        pgm = self.out_dir / "robot" / "occupancy.pgm"
        if not (y.exists() and pgm.exists()):
            return None
        meta = {}
        for line in y.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        origin = [float(v) for v in meta["origin"].strip("[]").split(",")[:2]]
        img = cv2.imread(str(pgm), cv2.IMREAD_GRAYSCALE)
        return {"resolution": float(meta["resolution"]), "origin": origin,
                "width": int(img.shape[1]), "height": int(img.shape[0])}

    def localize(self, image: np.ndarray, K: np.ndarray, n_features: int = 4000) -> Pose:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, desc = detect(gray, n_features)
        if len(desc) < self.min_inliers:
            return Pose(False, f"only {len(desc)} features in the image: point the camera at a textured area")
        knn = self.matcher.knnMatch(desc, k=2)
        best: dict[int, tuple[float, int]] = {}
        for m in knn:
            if len(m) < 2 or m[0].distance >= self.ratio * m[1].distance:
                continue
            p = int(self.owner[m[0].trainIdx])
            if p not in best or m[0].distance < best[p][0]:
                best[p] = (m[0].distance, m[0].queryIdx)
        if len(best) < self.min_inliers:
            return Pose(False, f"only {len(best)} matches with the map: is this the room that was scanned?",
                        matches=len(best))
        pid = np.fromiter(best.keys(), int)
        qid = np.array([best[p][1] for p in pid])
        obj = self.points[pid]
        img = kps[qid].astype(np.float64)
        thr = max(4.0, 0.004 * max(gray.shape))  # ~8 px at 1920
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=4000,
                                                 reprojectionError=thr, confidence=0.9995,
                                                 flags=cv2.SOLVEPNP_SQPNP)
        if not ok or inl is None or len(inl) < self.min_inliers:
            n = 0 if inl is None else len(inl)
            return Pose(False, f"no consistent pose ({n} inliers of {len(pid)} matches)",
                        matches=len(pid), inliers=n)
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()
        proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, K, None)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - img[inl], axis=1).mean())
        C = -R.T @ t
        fwd = R.T @ np.array([0.0, 0.0, 1.0])  # camera +Z (view direction) in the map frame
        yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))
        pose = Pose(True, "ok", float(C[0]), float(C[1]), float(C[2]), yaw, pitch,
                    len(pid), len(inl), err, R.tolist(), t.tolist())
        if self.grid:
            g = self.grid
            col = (C[0] - g["origin"][0]) / g["resolution"]
            row = g["height"] - 1 - (C[1] - g["origin"][1]) / g["resolution"]
            pose.grid_px = [round(float(col), 1), round(float(row), 1)]
        return pose


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("image", type=Path)
    ap.add_argument("--K", required=True, help="fx,fy,cx,cy for this image")
    a = ap.parse_args()
    fx, fy, cx, cy = map(float, a.K.split(","))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    img = cv2.imread(str(a.image))
    print(json.dumps(Localizer(a.out_dir).localize(img, K).to_dict(), indent=2))


if __name__ == "__main__":
    main()
