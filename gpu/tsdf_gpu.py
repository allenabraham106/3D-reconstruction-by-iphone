"""TSDF fusion on an NVIDIA GPU (PyTorch CUDA): same algorithm and parameters as
recon/tsdf.py's fuse_tsdf, so meshes match the CPU version, but every view is integrated
over the whole voxel grid in one pass on the GPU instead of slab by slab in numpy.

    from tsdf_gpu import fuse_tsdf_gpu
    V, F, C, voxel = fuse_tsdf_gpu(views, masks, voxel=0.01)

Same inputs as fuse_tsdf: views with .depth/.K/.R/.t/.img (cam_from_world, OpenCV axes),
masks of trusted depth. Marching cubes stays on the CPU (scikit-image).
"""
import numpy as np


def _bounds(views, masks, voxel, pad, max_voxels):
    pts = []
    for v, m in zip(views, masks):
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        sel = np.random.default_rng(0).choice(len(ys), min(len(ys), 4000), replace=False)
        ys, xs = ys[sel], xs[sel]
        z = v.depth[ys, xs]
        pix = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float64)
        Xc = (pix @ np.linalg.inv(v.K).T) * z[:, None]
        pts.append((Xc - v.t) @ v.R)
    P = np.concatenate(pts)
    lo, hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    dims = np.ceil((hi - lo) / voxel).astype(int)
    while np.prod(dims) > max_voxels:
        voxel *= 1.15
        dims = np.ceil((hi - lo) / voxel).astype(int)
    return lo, dims, voxel


def fuse_tsdf_gpu(views, masks, voxel, trunc_vox=4.0, max_voxels=200_000_000, min_weight=2.0,
                  pad=0.05, device="cuda", log=print):
    import torch
    from skimage.measure import marching_cubes

    lo, dims, voxel = _bounds(views, masks, voxel, pad, max_voxels)
    trunc = trunc_vox * voxel
    nx, ny, nz = (int(d) for d in dims)
    log(f"[tsdf-gpu] grid {dims.tolist()} = {np.prod(dims)/1e6:.1f}M voxels, voxel {voxel:.4f}, "
        f"truncation {trunc:.4f}")
    dev = torch.device(device)
    nvox = nx * ny * nz
    tsdf = torch.ones(nvox, dtype=torch.float32, device=dev)
    weight = torch.zeros(nvox, dtype=torch.float32, device=dev)
    col = torch.zeros((nvox, 3), dtype=torch.float32, device=dev)
    ccount = torch.zeros(nvox, dtype=torch.float32, device=dev)

    gx = torch.tensor(lo[0] + (np.arange(nx) + 0.5) * voxel, dtype=torch.float32, device=dev)
    gy = torch.tensor(lo[1] + (np.arange(ny) + 0.5) * voxel, dtype=torch.float32, device=dev)
    gz = torch.tensor(lo[2] + (np.arange(nz) + 0.5) * voxel, dtype=torch.float32, device=dev)
    # process the grid in x-slabs so temporaries stay small even at 1 cm on a big room
    slab = max(1, int(40_000_000 // (ny * nz)))
    YY, ZZ = torch.meshgrid(gy, gz, indexing="ij")
    YY, ZZ = YY.reshape(-1), ZZ.reshape(-1)

    for vi, (v, m) in enumerate(zip(views, masks)):
        D = np.where(m, v.depth, 0).astype(np.float32)
        if not (D > 0).any():
            continue
        h, w = D.shape
        Dt = torch.from_numpy(D).to(dev)
        img = torch.from_numpy(np.ascontiguousarray(v.img[..., ::-1])).to(dev).float()  # BGR -> RGB
        R = torch.tensor(v.R, dtype=torch.float32, device=dev)
        t = torch.tensor(v.t, dtype=torch.float32, device=dev)
        fx, fy, cx, cy = (float(v.K[0, 0]), float(v.K[1, 1]), float(v.K[0, 2]), float(v.K[1, 2]))
        for x0 in range(0, nx, slab):
            xs_ = gx[x0:x0 + slab]
            n_s = len(xs_)
            X = torch.stack([xs_[:, None].expand(n_s, ny * nz).reshape(-1),
                             YY.repeat(n_s), ZZ.repeat(n_s)], 1)
            Xc = X @ R.T + t
            z = Xc[:, 2]
            front = z > 1e-6
            zs = torch.where(front, z, torch.ones_like(z))
            u = torch.round(fx * Xc[:, 0] / zs + cx).long()
            vv = torch.round(fy * Xc[:, 1] / zs + cy).long()
            inb = front & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
            idx = torch.nonzero(inb, as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            uu, vy = u[idx], vv[idx]
            d = Dt[vy, uu]
            has = d > 0
            idx, d, uu, vy = idx[has], d[has], uu[has], vy[has]
            sdf = d - z[idx]
            near = sdf > -trunc
            idx, sdf, uu, vy = idx[near], sdf[near], uu[near], vy[near]
            if idx.numel() == 0:
                continue
            obs = torch.clamp(sdf / trunc, max=1.0)
            g = idx + x0 * ny * nz  # flat grid index (x-major, like the slab layout)
            w_old = weight[g]
            tsdf[g] = (tsdf[g] * w_old + obs) / (w_old + 1)
            weight[g] = w_old + 1
            surf = sdf.abs() < trunc
            if surf.any():
                gs = g[surf]
                col[gs] += img[vy[surf], uu[surf]]
                ccount[gs] += 1
        if vi % 100 == 0:
            log(f"[tsdf-gpu] integrated {vi + 1}/{len(views)} views")

    observed = weight >= min_weight
    tsdf[~observed] = 1.0
    T = tsdf.reshape(nx, ny, nz).cpu().numpy()
    colv = col.cpu().numpy().reshape(nx, ny, nz, 3)
    cc = ccount.cpu().numpy().reshape(nx, ny, nz)
    del tsdf, weight, col, ccount
    torch.cuda.empty_cache()
    verts, faces, _, _ = marching_cubes(T, level=0.0, spacing=(voxel, voxel, voxel),
                                        allow_degenerate=False)
    verts = verts + lo + 0.5 * voxel
    vi_ = np.clip(np.round((verts - lo - 0.5 * voxel) / voxel).astype(int), 0, dims - 1)
    n = cc[vi_[:, 0], vi_[:, 1], vi_[:, 2]][:, None]
    colors = np.clip(colv[vi_[:, 0], vi_[:, 1], vi_[:, 2]] / np.maximum(n, 1), 0, 255).astype(np.uint8)
    log(f"[tsdf-gpu] mesh: {len(verts):,} vertices, {len(faces):,} faces")
    return verts, faces, colors, voxel
