#!/usr/bin/env python3
"""Mesh + robot map from a trained splat, on the GPU: render the splat's depth at every keyframe, TSDF-fuse those
depths in torch, marching cubes, then recon/robot_map.py (floor, gravity, phone-height scale, nav2 occupancy).
Replaces the CPU pipeline's plane-sweep stereo + numpy TSDF (IMG_1429: 10.5 + 7.4 min).

    python gpu/splat_map.py out_fast/IMG_1429      # a gpu/fast_splat.py output: splat/ckpt.pt, frames/

Writes dense/{mesh.ply, dense.ply, cameras.json} and robot/ (same layout as run.py, so robot_link and the
viewer read it unchanged).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recon"))
from mvs import write_ply  # noqa: E402
from tsdf import write_mesh_ply  # noqa: E402


def render_depths(ck, scale, min_alpha, dev):
    """Expected depth per keyframe at scale x the training resolution; 0 where the splat is not opaque.
    (Leaving out the largest Gaussians removes the floor too: robot_map then cannot find it.)"""
    from gsplat import rasterization
    p = {k: v.to(dev) for k, v in ck["params"].items()}
    colors = torch.cat([p["sh0"], p["shN"]], 1)
    W, H = round(ck["W"] * scale), round(ck["H"] * scale)
    Ks = ck["Ks"].to(dev).clone()
    Ks[:, :2] *= scale
    vms = ck["viewmats"].to(dev)
    D = torch.empty(len(vms), H, W, device=dev)
    with torch.no_grad():
        for k in range(len(vms)):
            out, alpha, _ = rasterization(p["means"], p["quats"], torch.exp(p["scales"]),
                                          torch.sigmoid(p["opacities"]), colors, vms[k:k + 1], Ks[k:k + 1], W, H,
                                          sh_degree=ck["sh_degree"], render_mode="ED", near_plane=0.05)
            D[k] = torch.where(alpha[0, ..., 0] > min_alpha, out[0, ..., 0], 0)
    return D, Ks, vms


def fuse(D, Ks, vms, imgs, min_weight, voxel_frac=0.0022, trunc_vox=4.0, max_voxels=60_000_000):
    """TSDF (KinectFusion style) over all keyframes at once on the GPU. Returns (verts, faces, colors, voxel)."""
    from skimage.measure import marching_cubes
    dev = D.device
    N, h, w = D.shape
    v_pix, u_pix = torch.meshgrid(torch.arange(h, device=dev) + 0.5, torch.arange(w, device=dev) + 0.5, indexing="ij")
    R, t = vms[:, :3, :3], vms[:, :3, 3]

    sample = []  # scene bounds from a sample of back-projected depth
    for k in range(0, N, 4):
        m = D[k] > 0
        z = D[k][m]
        Xc = torch.stack([(u_pix[m] - Ks[k, 0, 2]) / Ks[k, 0, 0] * z, (v_pix[m] - Ks[k, 1, 2]) / Ks[k, 1, 1] * z, z], 1)
        sample.append(((Xc - t[k]) @ R[k])[::37])
    P = torch.cat(sample)
    lo, hi = torch.quantile(P, 0.005, 0), torch.quantile(P, 0.995, 0)
    p2, p98 = torch.quantile(P, 0.02, 0), torch.quantile(P, 0.98, 0)
    voxel = float(torch.linalg.norm(p98 - p2)) * voxel_frac
    lo, hi = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
    dims = torch.ceil((hi - lo) / voxel).long()
    while int(dims.prod()) > max_voxels:
        voxel *= 1.15
        dims = torch.ceil((hi - lo) / voxel).long()
    trunc = trunc_vox * voxel
    nx, ny, nz = dims.tolist()
    print(f"[splat_map] grid {nx}x{ny}x{nz} = {nx * ny * nz / 1e6:.1f}M voxels, voxel {voxel:.4f}", flush=True)

    axes = [lo[i] + (torch.arange(n, device=dev) + 0.5) * voxel for i, n in enumerate((nx, ny, nz))]
    X = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    tsdf = torch.ones(len(X), device=dev)
    wsum = torch.zeros(len(X), device=dev)
    col = torch.zeros(len(X), 3, device=dev)
    ccount = torch.zeros(len(X), device=dev)
    for k in range(N):
        Xc = X @ R[k].T + t[k]
        z = Xc[:, 2]
        u = torch.floor(Ks[k, 0, 0] * Xc[:, 0] / z + Ks[k, 0, 2]).long()
        v = torch.floor(Ks[k, 1, 1] * Xc[:, 1] / z + Ks[k, 1, 2]).long()
        idx = torch.nonzero((z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)).squeeze(1)
        u, v = u[idx], v[idx]
        sdf = D[k, v, u] - z[idx]
        keep = (D[k, v, u] > 0) & (sdf > -trunc)  # far behind the surface: unknown, not free
        idx, u, v, sdf = idx[keep], u[keep], v[keep], sdf[keep]
        wo = wsum[idx]
        tsdf[idx] = (tsdf[idx] * wo + torch.clamp(sdf / trunc, max=1.0)) / (wo + 1)
        wsum[idx] = wo + 1
        s = sdf.abs() < trunc
        col[idx[s]] += imgs[k, v[s], u[s]].float()
        ccount[idx[s]] += 1

    tsdf[wsum < min_weight] = 1.0
    vol = tsdf.reshape(nx, ny, nz).cpu().numpy()
    verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=(voxel,) * 3, allow_degenerate=False)
    gi = np.clip(np.round(verts / voxel).astype(int), 0, np.array([nx, ny, nz]) - 1)
    flat = torch.from_numpy((gi[:, 0] * ny + gi[:, 1]) * nz + gi[:, 2]).to(dev)
    colors = (col[flat] / ccount[flat].clamp(min=1)[:, None]).clamp(0, 255).byte().cpu().numpy()
    verts = verts + lo.cpu().numpy() + 0.5 * voxel
    return verts, faces, colors, voxel


def run(out, dst=None, scale=0.5, min_alpha=0.9, min_weight=6, phone_height=1.4):
    """min_weight 6 (vs 2): far fewer false obstacles from splat floaters on IMG_1429 (9110 -> 6802 cells;
    the CPU stereo map has 5731)."""
    t0 = time.time()
    dev = torch.device("cuda")
    ck = torch.load(out / "splat" / "ckpt.pt", map_location="cpu", weights_only=False)
    names = [f"frame_{i:05d}.jpg" for i in ck["frame_ids"]]
    D, Ks, vms = render_depths(ck, scale, min_alpha, dev)
    h, w = D.shape[1:]
    imgs = torch.from_numpy(np.stack([cv2.resize(cv2.imread(str(out / "frames" / n))[..., ::-1], (w, h),
                                                 interpolation=cv2.INTER_AREA) for n in names])).to(dev)
    print(f"[splat_map] {len(names)} depth maps at {w}x{h}, coverage {(D > 0).float().mean():.2f}, "
          f"{time.time() - t0:.0f}s", flush=True)
    verts, faces, colors, voxel = fuse(D, Ks, vms, imgs, min_weight)
    print(f"[splat_map] mesh {len(verts):,} vertices, {len(faces):,} faces, {time.time() - t0:.0f}s", flush=True)

    dst = dst or out
    dense = dst / "dense"
    dense.mkdir(parents=True, exist_ok=True)
    write_mesh_ply(dense / "mesh.ply", verts, faces, colors)
    write_ply(dense / "dense.ply", verts, colors)
    Kf, V = ck["Ks"].numpy(), ck["viewmats"].numpy()
    cams = [{"name": n, "R": V[k, :3, :3].tolist(), "t": V[k, :3, 3].tolist(),
             "C": (-V[k, :3, :3].T @ V[k, :3, 3]).tolist(), "fx": float(Kf[k, 0, 0]), "fy": float(Kf[k, 1, 1]),
             "w": int(ck["W"]), "h": int(ck["H"])} for k, n in enumerate(names)]
    (dense / "cameras.json").write_text(json.dumps(cams))

    from robot_map import build
    build(dense / "dense.ply", dense / "cameras.json", dst / "robot", phone_height)
    print(f"[splat_map] done in {time.time() - t0:.0f}s -> {dst / 'robot'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, help="gpu/fast_splat.py output folder")
    ap.add_argument("--dst", type=Path, default=None, help="write dense/ and robot/ here (default: out)")
    ap.add_argument("--scale", type=float, default=0.5, help="depth render size relative to the training frames")
    ap.add_argument("--min-alpha", type=float, default=0.9, help="opacity a pixel needs for its depth to count")
    ap.add_argument("--min-weight", type=int, default=6, help="views that must see a voxel before it counts")
    ap.add_argument("--phone-height", type=float, default=1.4, help="metric scale: median camera height (m)")
    a = ap.parse_args()
    run(a.out, a.dst, a.scale, a.min_alpha, a.min_weight, a.phone_height)


if __name__ == "__main__":
    main()
