"""TSDF fusion: average every view's depth into one volume, extract a clean surface.

Plane-sweep depth from a walking phone is noisy along each viewing ray (several cm at
3 m with short baselines). Fusing per point keeps that noise, so walls look smeared.
A truncated signed distance field (KinectFusion / nvblox style) averages the signed
distance to the surface over all views that saw each voxel, so independent noise
cancels (~1/sqrt(N)), then marching cubes extracts the zero crossing as a mesh.
CPU / numpy only.
"""
import numpy as np


def fuse_tsdf(views, masks, voxel, trunc_vox=4.0, max_voxels=45_000_000, min_weight=2.0,
              pad=0.05, log=print):
    """views: mvs.View list with .depth; masks: per-view bool masks of trusted depth.

    Returns (verts, faces, colors uint8, voxel_used)."""
    from skimage.measure import marching_cubes

    # bounds from the trusted depth points (robust percentiles)
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
    lo = np.percentile(P, 0.5, 0)
    hi = np.percentile(P, 99.5, 0)
    span = hi - lo
    lo -= pad * span
    hi += pad * span
    dims = np.ceil((hi - lo) / voxel).astype(int)
    while np.prod(dims) > max_voxels:  # keep memory bounded on big scenes
        voxel *= 1.15
        dims = np.ceil((hi - lo) / voxel).astype(int)
    trunc = trunc_vox * voxel
    log(f"[tsdf] grid {dims.tolist()} = {np.prod(dims)/1e6:.1f}M voxels, voxel {voxel:.4f}, "
        f"truncation {trunc:.4f}")
    nx, ny, nz = dims
    gx = lo[0] + (np.arange(nx) + 0.5) * voxel
    gy = lo[1] + (np.arange(ny) + 0.5) * voxel
    gz = lo[2] + (np.arange(nz) + 0.5) * voxel
    tsdf = np.ones(dims, np.float32)
    weight = np.zeros(dims, np.float32)
    col = np.zeros(tuple(dims) + (3,), np.float32)
    ccount = np.zeros(dims, np.float32)  # colour samples per voxel

    # integrate slab by slab along x to bound temporary memory
    slab = max(1, int(6_000_000 // (ny * nz)))
    YY, ZZ = np.meshgrid(gy, gz, indexing="ij")
    for vi, (v, m) in enumerate(zip(views, masks)):
        D = np.where(m, v.depth, 0).astype(np.float32)
        if not (D > 0).any():
            continue
        R = v.R.astype(np.float32)
        t = v.t.astype(np.float32)
        K = v.K
        h, w = D.shape
        img = v.img
        for x0 in range(0, nx, slab):
            xs_ = gx[x0:x0 + slab]
            X = np.stack(np.broadcast_arrays(xs_[:, None, None], YY[None], ZZ[None]), -1)
            X = X.reshape(-1, 3).astype(np.float32)
            Xc = X @ R.T + t
            z = Xc[:, 2]
            front = z > 1e-6
            u = np.full(len(z), -1, np.int64)
            vv = np.full(len(z), -1, np.int64)
            u[front] = np.round(K[0, 0] * Xc[front, 0] / z[front] + K[0, 2]).astype(np.int64)
            vv[front] = np.round(K[1, 1] * Xc[front, 1] / z[front] + K[1, 2]).astype(np.int64)
            inb = front & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
            if not inb.any():
                continue
            idx = np.nonzero(inb)[0]
            d = D[vv[idx], u[idx]]
            has = d > 0
            idx, d = idx[has], d[has]
            sdf = d - z[idx]
            near = sdf > -trunc  # behind the surface by more than trunc: unknown
            idx, sdf = idx[near], sdf[near]
            if len(idx) == 0:
                continue
            obs = np.minimum(1.0, sdf / trunc)
            # flat index into the slab -> grid index
            sl_shape = (len(xs_), ny, nz)
            ii, jj, kk = np.unravel_index(idx, sl_shape)
            ii = ii + x0
            w_old = weight[ii, jj, kk]
            tsdf[ii, jj, kk] = (tsdf[ii, jj, kk] * w_old + obs) / (w_old + 1)
            weight[ii, jj, kk] = w_old + 1
            surf = np.abs(sdf) < trunc
            if surf.any():
                rgb = img[vv[idx][surf], u[idx][surf], ::-1].astype(np.float32)
                a, b, c = ii[surf], jj[surf], kk[surf]
                col[a, b, c] += rgb  # indices are unique within one view
                ccount[a, b, c] += 1
        if vi % 40 == 0:
            log(f"[tsdf] integrated {vi + 1}/{len(views)} views")

    tsdf[weight < min_weight] = 1.0  # unobserved / weakly observed -> empty, no surface
    verts, faces, _, _ = marching_cubes(tsdf, level=0.0, spacing=(voxel, voxel, voxel),
                                        allow_degenerate=False)
    verts = verts + lo + 0.5 * voxel
    # vertex colours from the nearest voxel's average colour
    vi_ = np.clip(np.round((verts - lo - 0.5 * voxel) / voxel).astype(int), 0, dims - 1)
    n = ccount[vi_[:, 0], vi_[:, 1], vi_[:, 2]][:, None]
    colors = np.clip(col[vi_[:, 0], vi_[:, 1], vi_[:, 2]] / np.maximum(n, 1), 0, 255).astype(np.uint8)
    log(f"[tsdf] mesh: {len(verts):,} vertices, {len(faces):,} faces")
    return verts, faces, colors, voxel


def fuse_tsdf_fast(views, masks, voxel, trunc_vox=4.0, max_voxels=45_000_000, min_weight=2.0,
                   pad=0.05, free_step_vox=4.0, pixel_stride=1, free_stride=2, log=print):
    """Approximation of fuse_tsdf, ~3x faster: only voxels along each depth pixel's ray
    are touched -- near-surface samples (|sdf| < trunc) plus sparse free-space samples
    between the camera and the surface (they clear floaters, as full integration does).
    Per view, each voxel gets the mean of the samples that fell in it (weight 1).
    Sparser free-space carving leaves more floaters than fuse_tsdf (real apartment:
    7.4 vs 20 min, floor cells seen 4134 vs 6199), so it is for quick previews."""
    from skimage.measure import marching_cubes

    pts = []
    for v, m in zip(views, masks):
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        sel = np.random.default_rng(0).choice(len(ys), min(len(ys), 4000), replace=False)
        z = v.depth[ys[sel], xs[sel]]
        pix = np.stack([xs[sel], ys[sel], np.ones(len(sel))], 1)
        pts.append(((pix @ np.linalg.inv(v.K).T) * z[:, None] - v.t) @ v.R)
    P = np.concatenate(pts)
    lo, hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    dims = np.ceil((hi - lo) / voxel).astype(int)
    while np.prod(dims) > max_voxels:
        voxel *= 1.15
        dims = np.ceil((hi - lo) / voxel).astype(int)
    trunc = trunc_vox * voxel
    nvox = int(np.prod(dims))
    log(f"[tsdf-fast] grid {dims.tolist()} = {nvox/1e6:.1f}M voxels, voxel {voxel:.4f}")
    tsum = np.zeros(nvox, np.float32)   # sum over views of per-view mean tsdf
    wsum = np.zeros(nvox, np.float32)   # number of views that observed the voxel
    csum = np.zeros((nvox, 3), np.float32)
    ccnt = np.zeros(nvox, np.float32)
    near_offsets = np.arange(-trunc_vox, trunc_vox + 1e-6, 1.0) * voxel  # along-ray, metres

    def to_flat(X):
        ijk = np.floor((X - lo) / voxel).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < dims), 1)
        flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
        return flat, ok

    for vi, (v, m) in enumerate(zip(views, masks)):
        ys, xs = np.nonzero(m)
        if pixel_stride > 1:
            keep = (ys % pixel_stride == 0) & (xs % pixel_stride == 0)
            ys, xs = ys[keep], xs[keep]
        if len(ys) == 0:
            continue
        d = v.depth[ys, xs].astype(np.float64)
        pix = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float64)
        ray = pix @ np.linalg.inv(v.K).T             # camera ray with z = 1
        rn = np.linalg.norm(ray, axis=1)
        rw = ray @ v.R                               # ray direction in world (unnormalised)
        C = -v.R.T @ v.t
        rgb = v.img[ys, xs, ::-1].astype(np.float32)
        # near-surface samples: depth z = d + o * (1/|ray|) so steps are 1 voxel along the ray
        Z = d[:, None] + near_offsets[None, :] / rn[:, None]
        X = C + Z[..., None] * rw[:, None, :]
        sdf = (d[:, None] - Z) * rn[:, None]         # metric distance along the ray
        obs = np.clip(sdf / trunc, -1, 1).ravel()
        flat, ok = to_flat(X.reshape(-1, 3))
        surf = (np.abs(sdf) < 1.5 * voxel).ravel()
        col_rgb = np.repeat(rgb, Z.shape[1], 0)
        # free-space samples from the camera to just before the truncation band
        # (on a sparser pixel grid: clearing floaters doesn't need every ray)
        fsel = (ys % free_stride == 0) & (xs % free_stride == 0)
        nfree = np.maximum(((d[fsel] * rn[fsel] - trunc) / (free_step_vox * voxel)).astype(int), 0)
        nmax = int(min(nfree.max(), 200)) if len(nfree) else 0
        if nmax > 0:
            k = np.arange(1, nmax + 1)
            Zf = (k[None, :] * free_step_vox * voxel) / rn[fsel][:, None]
            valid_f = k[None, :] <= nfree[:, None]
            Xf = C + Zf[..., None] * rw[fsel][:, None, :]
            ff, okf = to_flat(Xf[valid_f])
            flat = np.concatenate([flat, ff])
            ok = np.concatenate([ok, okf])
            obs = np.concatenate([obs, np.ones(len(ff), np.float32)])
            surf = np.concatenate([surf, np.zeros(len(ff), bool)])
            col_rgb = np.concatenate([col_rgb, np.zeros((len(ff), 3), np.float32)])
        flat, obs, surf, col_rgb = flat[ok], obs[ok], surf[ok], col_rgb[ok]
        # per-view mean observation per voxel, then add with weight 1 (bincount: linear)
        s_ = np.bincount(flat, weights=obs, minlength=nvox)
        n_ = np.bincount(flat, minlength=nvox)
        u = np.flatnonzero(n_)
        tsum[u] += (s_[u] / n_[u]).astype(np.float32)
        wsum[u] += 1.0
        if surf.any():
            fs = flat[surf]
            for c in range(3):
                csum[:, c] += np.bincount(fs, weights=col_rgb[surf, c], minlength=nvox).astype(np.float32)
            ccnt += np.bincount(fs, minlength=nvox).astype(np.float32)
        if vi % 40 == 0:
            log(f"[tsdf-fast] integrated {vi + 1}/{len(views)} views")

    tsdf = np.where(wsum >= min_weight, tsum / np.maximum(wsum, 1), 1.0).astype(np.float32)
    tsdf = tsdf.reshape(dims)
    verts, faces, _, _ = marching_cubes(tsdf, level=0.0, spacing=(voxel, voxel, voxel),
                                        allow_degenerate=False)
    verts = verts + lo + 0.5 * voxel
    vi_ = np.clip(np.round((verts - lo - 0.5 * voxel) / voxel).astype(int), 0, dims - 1)
    fl = (vi_[:, 0] * dims[1] + vi_[:, 1]) * dims[2] + vi_[:, 2]
    colors = np.clip(csum[fl] / np.maximum(ccnt[fl], 1)[:, None], 0, 255).astype(np.uint8)
    log(f"[tsdf-fast] mesh: {len(verts):,} vertices, {len(faces):,} faces")
    return verts, faces, colors, voxel


def write_mesh_ply(path, verts, faces, colors):
    verts = verts.astype(np.float32)
    colors = colors.astype(np.uint8)
    faces = faces.astype(np.int32)
    vrec = np.empty(len(verts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vrec["x"], vrec["y"], vrec["z"] = verts.T
    vrec["red"], vrec["green"], vrec["blue"] = colors.T
    frec = np.empty(len(faces), dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    frec["n"] = 3
    frec["a"], frec["b"], frec["c"] = faces.T
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
                  "end_header\n").encode())
        fh.write(vrec.tobytes())
        fh.write(frec.tobytes())


def read_mesh_ply(path):
    """Read a binary PLY written by write_mesh_ply: (verts float64, faces int, colors uint8)."""
    data = open(path, "rb").read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    head = data[:end].decode()
    nv = int(head.split("element vertex ")[1].split()[0])
    nf = int(head.split("element face ")[1].split()[0]) if "element face" in head else 0
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v = np.frombuffer(data, vdt, count=nv, offset=end)
    fdt = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    f = np.frombuffer(data, fdt, count=nf, offset=end + nv * vdt.itemsize)
    V = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    C = np.stack([v["r"], v["g"], v["b"]], 1)
    F = np.stack([f["a"], f["b"], f["c"]], 1)
    return V, F, C
