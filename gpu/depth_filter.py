"""Make ViPE's dense depth crisp before fusion (GPU, PyTorch).

ViPE gives every pixel a depth from learned monocular models. That depth is smooth across
object edges (flying pixels between foreground and background) and a few cm inconsistent
from frame to frame, so fusing all of it smears edges, and blank ceilings / glass come
through as a soft shell. The CPU pipeline looks sharp because stereo keeps only depth that
several views agree on. This applies the same idea to ViPE depth:

  1. edge filter: local depth range / mean in a 5x5 window below `edge_ratio` (ViPE's own
     rule uses 10 %; 3 % drops the edge bands),
  2. range: depth within `max_depth` (learned metric depth degrades with distance),
  3. multi-view agreement: back-project each pixel, project it into neighbouring frames and
     keep it only if at least `min_agree` of them see the same surface (depth within
     `rel_tol`), as MVS fusion does.

    keep = consistency_masks(views, base_masks)   # list of bool (h, w), one per view
"""
import numpy as np


def edge_mask(D, window=5, ratio=0.03):
    import cv2
    ok = np.isfinite(D) & (D > 1e-3)
    Dz = np.where(ok, D, 0).astype(np.float32)
    k = np.ones((window, window), np.uint8)
    big = cv2.dilate(Dz, k)
    small = cv2.erode(np.where(ok, Dz, np.float32(1e9)), k)
    return ok & ((big - small) / np.maximum(Dz, 1e-6) < ratio)


def consistency_masks(views, base_masks, offsets=(-6, -4, -2, 2, 4, 6), rel_tol=0.015,
                      min_agree=2, device="cuda", log=print):
    import torch
    dev = torch.device(device)
    n = len(views)
    h, w = views[0].depth.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    D = [torch.from_numpy(np.where(m, v.depth, 0).astype(np.float32)).to(dev) for v, m in zip(views, base_masks)]
    R = [torch.tensor(v.R, dtype=torch.float32, device=dev) for v in views]
    t = [torch.tensor(v.t, dtype=torch.float32, device=dev) for v in views]
    K = [v.K for v in views]
    out = []
    for i in range(n):
        Di = D[i]
        valid = Di > 0
        fx, fy, cx, cy = K[i][0, 0], K[i][1, 1], K[i][0, 2], K[i][1, 2]
        Xc = torch.stack([(xs - cx) / fx * Di, (ys - cy) / fy * Di, Di], -1)   # camera i
        Xw = (Xc - t[i]) @ R[i]                                                 # world
        agree = torch.zeros((h, w), device=dev, dtype=torch.int32)
        for o in offsets:
            j = i + o
            if j < 0 or j >= n:
                continue
            Xj = Xw @ R[j].T + t[j]
            z = Xj[..., 2]
            fxj, fyj, cxj, cyj = K[j][0, 0], K[j][1, 1], K[j][0, 2], K[j][1, 2]
            zs = torch.where(z > 1e-3, z, torch.ones_like(z))
            u = torch.round(fxj * Xj[..., 0] / zs + cxj).long()
            v = torch.round(fyj * Xj[..., 1] / zs + cyj).long()
            inb = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            dj = torch.zeros_like(z)
            dj[inb] = D[j][v[inb], u[inb]]
            ok = inb & (dj > 0) & ((dj - z).abs() < rel_tol * z)
            agree += ok.int()
        out.append((valid & (agree >= min_agree)).cpu().numpy())
        if i % 200 == 0:
            log(f"[filter] {i + 1}/{n} views, kept {out[-1].mean()*100:.0f}% of pixels")
    return out
