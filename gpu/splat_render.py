#!/usr/bin/env python3
"""Demo videos from a trained splat (gpu/splat_train.py's ckpt.pt).

    python gpu/splat_render.py out_splat/IMG_6075/ckpt.pt out_vipe/default/IMG_6075/robot/map_transform.json \
        out_splat/IMG_6075/demo --walk --orbit

Cameras are planned in the robot map's frame (metres, z-up, floor at z=0; the transform comes
from robot/map_transform.json) and converted back to the splat's frame, so the splat itself is
never re-oriented.

  --walk   the recorded walk, smoothed over ~2 s, eye height kept, pulled back a little
           behind the phone so the view isn't pressed against walls
  --orbit  dollhouse: a slow circle above the room looking down at its centre, with every
           Gaussian above --cut metres hidden (the ceiling), like the viewer's ?cut=
"""
import argparse
import math
import subprocess
from pathlib import Path

import json
import numpy as np
import torch


def look_at(eye, target, up=(0, 0, 1)):
    """camera-to-world (OpenCV: x right, y down, z forward) looking from eye to target."""
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.array(up, float))
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0, 0])
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, d, f, eye
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("map_transform", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--walk", action="store_true")
    ap.add_argument("--orbit", action="store_true")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cut", type=float, default=2.3, help="orbit: hide Gaussians above this height (m)")
    ap.add_argument("--turns", type=float, default=0.5, help="orbit: fraction of a full circle")
    ap.add_argument("--pullback", type=float, default=0.35, help="walk: metres behind the phone")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--hfov", type=float, default=80, help="degrees")
    a = ap.parse_args()
    from gsplat import rasterization
    from scipy.ndimage import gaussian_filter1d

    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location="cpu")
    P = {k: v.to(dev) for k, v in ck["params"].items()}
    T = json.loads(a.map_transform.read_text())
    R, z0, s, xy0 = np.array(T["R"]), T["z0"], T["scale"], np.array(T["xy0"])

    def map_from_splat(X):
        Y = X @ R.T
        Y[..., 2] -= z0
        Y = Y * s
        Y[..., :2] -= xy0
        return Y

    def splat_from_map_c2w(Tm):
        """camera-to-world in map frame -> world-to-camera (viewmat) in the splat frame."""
        C = Tm[:3, 3].copy()
        C[:2] += xy0
        C = C / s
        C[2] += z0
        Rw = R.T @ Tm[:3, :3]
        Ts = np.eye(4)
        Ts[:3, :3], Ts[:3, 3] = Rw, R.T @ C
        return np.linalg.inv(Ts)

    means_map = map_from_splat(P["means"].cpu().numpy())
    # exposure: the average of the training frames' colour correction
    if "expo_A" in ck:
        tr = ck["train"]
        eA, eb = ck["expo_A"][tr].mean(0).to(dev), ck["expo_b"][tr].mean(0).to(dev)
    else:
        eA, eb = torch.eye(3, device=dev), torch.zeros(3, device=dev)
    W, H = a.width, a.height
    f = 0.5 * W / math.tan(math.radians(a.hfov) / 2)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32, device=dev)[None]
    n = int(a.seconds * a.fps)
    a.out.mkdir(parents=True, exist_ok=True)

    # recorded cameras in the map frame
    vm = ck["viewmats"].numpy()
    c2w = np.linalg.inv(vm)
    Cm = map_from_splat(c2w[:, :3, 3])
    Fm = (R @ c2w[:, :3, 2].T).T  # viewing directions (rotation only)

    def render(viewmats, keep, name):
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                              "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-crf", "15", "-preset", "slow",
                              "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out / f"{name}.mp4")],
                             stdin=subprocess.PIPE)
        kt = torch.from_numpy(keep).to(dev)
        colors = torch.cat([P["sh0"], P["shN"]], 1)[kt]
        args = (P["means"][kt], P["quats"][kt], torch.exp(P["scales"][kt]), torch.sigmoid(P["opacities"][kt]), colors)
        stills = {int(n * q) for q in (0.15, 0.5, 0.85)}
        with torch.no_grad():
            for i, V in enumerate(viewmats):
                out, _, _ = rasterization(*args, torch.tensor(V, dtype=torch.float32, device=dev)[None], K, W, H,
                                          sh_degree=ck["sh_degree"], near_plane=0.05)
                rgb = (torch.einsum("ij,hwj->hwi", eA, out[0, ..., :3]) + eb).clamp(0, 1)
                img = (rgb * 255).byte().cpu().numpy()
                p.stdin.write(img.tobytes())
                if i in stills:
                    import cv2
                    cv2.imwrite(str(a.out / f"{name}_{i:04d}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
        p.stdin.close()
        p.wait()
        print(f"[render] {a.out / (name + '.mp4')}: {len(viewmats)} frames", flush=True)

    if a.walk:
        sig = 20  # frames: ~2 s of a 10 fps walk
        C = gaussian_filter1d(Cm, sigma=sig, axis=0, mode="nearest")
        Fd = gaussian_filter1d(Fm, sigma=sig, axis=0, mode="nearest")
        Fd[:, 2] *= 0.3  # keep the gaze closer to level
        Fd /= np.linalg.norm(Fd, axis=1, keepdims=True)
        t = np.linspace(0, len(C) - 1, n)
        Ct = np.stack([np.interp(t, np.arange(len(C)), C[:, d]) for d in range(3)], 1)
        Ft = np.stack([np.interp(t, np.arange(len(C)), Fd[:, d]) for d in range(3)], 1)
        Ft /= np.linalg.norm(Ft, axis=1, keepdims=True)
        views = []
        for c, fdir in zip(Ct, Ft):
            eye = c - a.pullback * np.array([fdir[0], fdir[1], 0]) / max(np.linalg.norm(fdir[:2]), 1e-6)
            views.append(splat_from_map_c2w(look_at(eye, eye + fdir)))
        render(views, np.ones(len(means_map), bool), "walk")

    if a.orbit:
        # seen from above, the splat's loosely constrained parts show: hide what's above the
        # cut (ceiling), outside the room's footprint, very large, or nearly transparent
        sc = torch.exp(P["scales"]).max(1).values.cpu().numpy() * s
        op = torch.sigmoid(P["opacities"]).cpu().numpy()
        lo, hi = np.percentile(means_map[:, :2], 2, 0), np.percentile(means_map[:, :2], 98, 0)
        inside = np.all((means_map[:, :2] > lo) & (means_map[:, :2] < hi), 1)
        keep = (means_map[:, 2] < a.cut) & (means_map[:, 2] > -0.1) & inside & (sc < 0.15) & (op > 0.05)
        print(f"[render] orbit keeps {keep.mean()*100:.0f}% of Gaussians", flush=True)
        lo, hi = np.percentile(means_map[keep, :2], 5, 0), np.percentile(means_map[keep, :2], 95, 0)
        ctr = np.array([*(lo + hi) / 2, 0.5])
        rad = 0.42 * np.linalg.norm(hi - lo)
        views = []
        for i in range(n):
            th = 2 * math.pi * i / n * a.turns + 0.3
            eye = ctr + np.array([rad * math.cos(th), rad * math.sin(th), a.cut + 0.9 - ctr[2]])
            views.append(splat_from_map_c2w(look_at(eye, ctr)))
        render(views, keep, "orbit")


if __name__ == "__main__":
    main()
