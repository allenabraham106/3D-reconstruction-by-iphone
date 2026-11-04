#!/usr/bin/env python3
"""Fuse PhoneScan LiDAR depth maps + ARKit poses into one coloured, metric point cloud (PLY).

No SfM needed: every depth pixel is back-projected with its frame's intrinsics and pose, then the
cloud is voxel-downsampled. Good for a quick look, robot navigation maps, or as an initial point
cloud for Gaussian splatting.

    python3 tools/fuse_depth.py CAPTURE_DIR                 # -> CAPTURE_DIR/fused.ply
    python3 tools/fuse_depth.py CAPTURE_DIR --voxel 0.01 --min-confidence 2 --max-depth 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phonescan_io import load_capture, write_ply_points  # noqa: E402


def backproject(depth_mm: np.ndarray, K_depth: np.ndarray, c2w_cv: np.ndarray,
                mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_mm.shape
    v, u = np.nonzero(mask)
    z = depth_mm[v, u].astype(np.float64) / 1000.0
    # pixel centres
    x = (u + 0.5 - K_depth[0, 2]) / K_depth[0, 0] * z
    y = (v + 0.5 - K_depth[1, 2]) / K_depth[1, 1] * z
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    pts_world = (c2w_cv @ pts_cam.T).T[:, :3]
    return pts_world, np.stack([u, v], axis=1)


def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    if voxel <= 0 or len(xyz) == 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n = len(counts)
    xyz_out = np.zeros((n, 3))
    rgb_out = np.zeros((n, 3))
    np.add.at(xyz_out, inverse, xyz)
    np.add.at(rgb_out, inverse, rgb.astype(np.float64))
    return xyz_out / counts[:, None], (rgb_out / counts[:, None]).round().astype(np.uint8)


def fuse(capture: Path, voxel: float, min_conf: int, max_depth: float, stride: int, every: int):
    frames = [f for f in load_capture(capture) if f.depth is not None][::every]
    if not frames:
        raise SystemExit("capture has no depth maps (iPhone without LiDAR?)")
    all_xyz, all_rgb = [], []
    for f in frames:
        depth = np.array(Image.open(f.depth), dtype=np.uint16)
        dh, dw = depth.shape
        sx, sy = dw / f.width, dh / f.height
        K_d = f.K.copy()
        K_d[0, 0] *= sx; K_d[0, 2] *= sx
        K_d[1, 1] *= sy; K_d[1, 2] *= sy

        mask = (depth > 0) & (depth <= max_depth * 1000)
        if f.confidence is not None and f.confidence.exists():
            mask &= np.array(Image.open(f.confidence), dtype=np.uint8) >= min_conf
        if stride > 1:
            sub = np.zeros_like(mask)
            sub[::stride, ::stride] = True
            mask &= sub

        xyz, uv = backproject(depth, K_d, f.c2w_cv, mask)
        rgb_img = np.array(Image.open(f.image).convert("RGB").resize((dw, dh), Image.BILINEAR))
        rgb = rgb_img[uv[:, 1], uv[:, 0]]
        all_xyz.append(xyz)
        all_rgb.append(rgb)

    xyz = np.concatenate(all_xyz)
    rgb = np.concatenate(all_rgb)
    raw = len(xyz)
    xyz, rgb = voxel_downsample(xyz, rgb, voxel)
    return xyz, rgb, raw, len(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: CAPTURE/fused.ply")
    ap.add_argument("--voxel", type=float, default=0.01, help="voxel size in metres (0 = keep all)")
    ap.add_argument("--min-confidence", type=int, default=2, choices=[0, 1, 2],
                    help="ARKit confidence: 0 low, 1 medium, 2 high")
    ap.add_argument("--max-depth", type=float, default=5.0, help="metres; LiDAR is reliable to ~5 m")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth depth pixel")
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame")
    args = ap.parse_args()

    xyz, rgb, raw, nframes = fuse(args.capture, args.voxel, args.min_confidence, args.max_depth,
                                  args.stride, args.every)
    out = args.out or args.capture / "fused.ply"
    write_ply_points(out, xyz, rgb)
    print(f"fused {nframes} frames: {raw} raw points -> {len(xyz)} after {args.voxel} m voxels -> {out}")


if __name__ == "__main__":
    main()
