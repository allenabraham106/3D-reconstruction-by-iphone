#!/usr/bin/env python3
"""Fill a gap in a walk from the 3D scene: render the missing frames from a splat that was
trained WITHOUT them (gpu/splat_train.py --exclude A:B), along a camera path interpolated
between the last pose before the gap and the first pose after it.

Unlike 2D in-betweening, this uses everything the rest of the walk saw of that part of the
room. Writes <out>/splat/frame_%05d.png at the size of the reference video (for gap_fill.py).

    python gpu/gap_splat.py out_gap/splat_no400/ckpt.pt --start 400 --length 24 \
        --size 1280x720 out_gap/IMG_6075_400
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fit-to", default=None,
                    help="folder of guide frames (e.g. LTX-Video's in-betweens): solve each frame's camera "
                         "pose so the splat render matches the guide, then render that pose sharp. The "
                         "video model supplies the motion, the 3D scene the detail.")
    ap.add_argument("--path", choices=["linear", "cubic"], default="linear",
                    help="camera path through the gap: straight line, or a cubic (Hermite) curve that "
                         "keeps the camera's velocity and turn rate from just before / after the gap")
    ap.add_argument("--true-poses", action="store_true",
                    help="render at the real (ViPE) poses of the gap frames instead of interpolating: "
                         "an upper bound that shows how much is lost to the unknown camera path")
    a = ap.parse_args()
    from gsplat import rasterization
    from scipy.spatial.transform import Rotation, Slerp

    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location="cpu")
    P = {k: v.to(dev) for k, v in ck["params"].items()}
    ids = list(ck["frame_ids"])
    pos = {i: k for k, i in enumerate(ids)}
    vm = ck["viewmats"].numpy()
    c2w = np.linalg.inv(vm)
    k0, k1 = pos[a.start - 1], pos[a.start + a.length]   # keyframes either side of the gap
    W, H = (int(x) for x in a.size.split("x"))
    sx, sy = W / ck["W"], H / ck["H"]
    K = ck["Ks"][k0].clone()
    K[0] *= sx
    K[1] *= sy
    K = K[None].to(dev)
    eA = 0.5 * (ck["expo_A"][k0] + ck["expo_A"][k1])
    eb = 0.5 * (ck["expo_b"][k0] + ck["expo_b"][k1])
    eA, eb = eA.to(dev), eb.to(dev)
    slerp = Slerp([0, 1], Rotation.from_matrix([c2w[k0][:3, :3], c2w[k1][:3, :3]]))
    name = "splat_truepose" if a.true_poses else ("splat_fit_" + Path(a.fit_to).name if a.fit_to else
                                                  ("splat_cubic" if a.path == "cubic" else "splat"))
    d = a.out / name
    d.mkdir(parents=True, exist_ok=True)
    colors = torch.cat([P["sh0"], P["shN"]], 1)

    # velocities from 3 frames either side (outside the gap), in gap units (t: 0 -> 1)
    span = a.length + 1
    p0, p1 = c2w[k0][:3, 3], c2w[k1][:3, 3]
    v0 = (c2w[k0][:3, 3] - c2w[k0 - 3][:3, 3]) / 3 * span
    v1 = (c2w[k1 + 3][:3, 3] - c2w[k1][:3, 3]) / 3 * span
    r0, r1 = Rotation.from_matrix(c2w[k0][:3, :3]), Rotation.from_matrix(c2w[k1][:3, :3])
    w0 = (r0 * Rotation.from_matrix(c2w[k0 - 3][:3, :3]).inv()).as_rotvec() / 3 * span   # world-frame turn rate
    w1 = (Rotation.from_matrix(c2w[k1 + 3][:3, :3]) * r1.inv()).as_rotvec() / 3 * span
    dr = (r1 * r0.inv()).as_rotvec()

    def interp(t):
        T = np.eye(4)
        if a.path == "cubic":
            h00, h10, h01, h11 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2
            T[:3, 3] = h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
            # rotation: Hermite on the rotation vector relative to r0 (small-angle blend of rates)
            rv = h10 * w0 + h01 * dr + h11 * w1
            T[:3, :3] = (Rotation.from_rotvec(rv) * r0).as_matrix()
        else:
            T[:3, :3] = slerp(t).as_matrix()
            T[:3, 3] = (1 - t) * p0 + t * p1
        return np.linalg.inv(T)

    def fit_pose(V0, guide, iters=200, w=320):
        """Photometric pose fit: small rotation + translation in camera space so the low-res,
        blurred splat render matches the low-res, blurred guide frame."""
        h = int(round(w * H / W))
        Kl = K.clone()
        Kl[:, 0] *= w / W
        Kl[:, 1] *= h / H
        g = cv2.GaussianBlur(cv2.resize(guide, (w, h), interpolation=cv2.INTER_AREA), (0, 0), 1.5)
        g = torch.from_numpy(g).float().to(dev) / 255
        d6 = torch.zeros(6, device=dev, requires_grad=True)
        opt = torch.optim.Adam([d6], lr=2e-3)
        V0t = torch.tensor(V0, dtype=torch.float32, device=dev)
        k = torch.tensor([[1, 4, 6, 4, 1]], dtype=torch.float32, device=dev)
        k = (k.T @ k / 256)[None, None].repeat(3, 1, 1, 1)
        for _ in range(iters):
            wv, tv = d6[:3], d6[3:]
            th = wv.norm().clamp(min=1e-9)
            kx = wv / th
            Kx = torch.zeros(3, 3, device=dev)
            Kx[0, 1], Kx[0, 2], Kx[1, 2] = -kx[2], kx[1], -kx[0]
            Kx = Kx - Kx.T
            R = torch.eye(3, device=dev) + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)
            D = torch.eye(4, device=dev)
            D = torch.cat([torch.cat([R, tv[:, None]], 1), D[3:]], 0)
            out, _, _ = rasterization(P["means"], P["quats"], torch.exp(P["scales"]), torch.sigmoid(P["opacities"]),
                                      colors, (D @ V0t)[None], Kl, w, h, sh_degree=ck["sh_degree"], near_plane=0.05)
            r = (torch.einsum("ij,hwj->hwi", eA, out[0, ..., :3]) + eb).clamp(0, 1)
            r = torch.nn.functional.conv2d(r.permute(2, 0, 1)[None], k, padding=2, groups=3)[0].permute(1, 2, 0)
            loss = (r - g).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return (D @ V0t).cpu().numpy(), float(loss)

    prev_delta = np.eye(4)
    for i in range(a.length):
        t = (i + 1) / (a.length + 1)
        if a.true_poses:
            V = vm[pos[a.start + i]]
        elif a.fit_to:
            guide = cv2.imread(str(Path(a.fit_to) / f"frame_{a.start + i:05d}.png"))[..., ::-1].copy()
            # start from the interpolated pose carried by the previous frame's correction
            V, err = fit_pose(prev_delta @ interp(t), guide)
            prev_delta = V @ np.linalg.inv(interp(t))
        else:
            V = interp(t)
        with torch.no_grad():
            out, _, _ = rasterization(P["means"], P["quats"], torch.exp(P["scales"]), torch.sigmoid(P["opacities"]),
                                      colors, torch.tensor(V, dtype=torch.float32, device=dev)[None], K, W, H,
                                      sh_degree=ck["sh_degree"], near_plane=0.05)
            rgb = (torch.einsum("ij,hwj->hwi", eA, out[0, ..., :3]) + eb).clamp(0, 1)
            cv2.imwrite(str(d / f"frame_{a.start + i:05d}.png"), (rgb * 255).byte().cpu().numpy()[..., ::-1])
    print(f"[gap_splat] {a.length} frames -> {d}")


if __name__ == "__main__":
    main()
