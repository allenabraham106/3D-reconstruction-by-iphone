"""Stage 3: dense depth via pairwise rectified stereo (OpenCV SGBM) + multi-view fusion.

CPU replacement for a GPU MVS / learned-depth step. For every registered frame:
  1. pick the best stereo partner (good baseline, small viewing-angle change, shared points)
  2. rectify the pair from the SfM poses, run SGBM with a left-right check
  3. turn disparity into a depth map in the reference camera
Then every depth map is checked against its neighbours' depth maps (geometric
consistency) and the surviving pixels are back-projected into one coloured point cloud.
"""
import argparse, json, time
from pathlib import Path

import cv2
import numpy as np
import pycolmap


# ---------------------------------------------------------------- camera helpers
class View:
    def __init__(self, image, camera, image_dir, scale=1.0, rec=None, tag=0):
        # tag = which reconstruction this view came from (main model = 0, attached
        # sub-models 1, 2, ...); ids are namespaced by it so models never collide.
        self.name = image.name
        self.id = (tag, image.image_id)
        self.rec = rec
        T = image.cam_from_world().matrix()  # 3x4
        self.R = T[:, :3].copy()
        self.t = T[:, 3].copy()
        self.C = -self.R.T @ self.t  # camera centre in world
        K = camera.calibration_matrix().copy()
        dist = np.zeros(5)
        if camera.model_name in ("SIMPLE_RADIAL", "RADIAL"):
            dist[0] = camera.params[3]
            if camera.model_name == "RADIAL":
                dist[1] = camera.params[4]
        elif camera.model_name == "OPENCV":
            dist[:4] = camera.params[4:8]
        img = cv2.imread(str(image_dir / image.name))
        img = cv2.undistort(img, K, dist)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            K = K.copy()
            K[:2] *= scale
        self.K = K
        self.img = img
        self.h, self.w = img.shape[:2]
        pts = [image.points2D[i].point3D_id for i in range(len(image.points2D))
               if image.points2D[i].has_point3D()]
        self.point_ids = {(tag, p) for p in pts}
        self.depth = None
        self.conf_mask = None

    @property
    def axis(self):
        return self.R[2]  # viewing direction in world


def sparse_depths(view, rec=None):
    if not view.point_ids:
        return np.array([1.0])
    rec = view.rec if view.rec is not None else rec
    X = np.array([rec.points3D[i].xyz for _, i in view.point_ids])
    z = (X @ view.R.T + view.t)[:, 2]
    return z[z > 0]


# ---------------------------------------------------------------- pair selection
def choose_partners(views, rec, k_consistency=4):
    med = {v.id: np.median(sparse_depths(v, rec)) for v in views}
    stereo, neigh = {}, {}
    for v in views:
        scored = []
        for u in views:
            if u.id == v.id:
                continue
            shared = len(v.point_ids & u.point_ids)
            if shared < 30:
                continue
            angle = np.degrees(np.arccos(np.clip(v.axis @ u.axis, -1, 1)))
            ratio = np.linalg.norm(v.C - u.C) / med[v.id]
            scored.append((u, shared, angle, ratio))
        if not scored:
            continue
        neigh[v.id] = [s[0] for s in sorted(scored, key=lambda s: -s[1])[:k_consistency]]
        good = [s for s in scored if s[2] < 15 and 0.02 < s[3] < 0.18]
        # prefer ~0.07 baseline/depth and many shared points; keep a ranked list of fallbacks
        good.sort(key=lambda s: -s[1] * np.exp(-((np.log(s[3] / 0.07)) ** 2)))
        if good:
            stereo[v.id] = [g[0] for g in good[:3]]
    return stereo, neigh, med


# ---------------------------------------------------------------- stereo for one pair
def stereo_depth(ref, src, rec):
    # relative pose: X_src = R X_ref + T
    R = src.R @ ref.R.T
    T = src.t - R @ ref.t
    size = (ref.w, ref.h)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(ref.K, np.zeros(5), src.K, np.zeros(5), size, R,
                                                 T.reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY,
                                                 alpha=1)
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])
    ax = 1 if vertical else 0
    f = P1[ax, ax]
    B = abs(P2[ax, 3]) / P2[ax, ax]
    z = sparse_depths(ref, rec)
    zmin, zmax = np.percentile(z, 2), np.percentile(z, 98)
    dmax = f * B / (zmin * 0.7)
    dmin = f * B / (zmax * 1.5)
    if dmax > 300 or dmax < 8:
        return None
    m1 = cv2.initUndistortRectifyMap(ref.K, np.zeros(5), R1, P1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(src.K, np.zeros(5), R2, P2, size, cv2.CV_32FC1)
    gl = cv2.cvtColor(cv2.remap(ref.img, *m1, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(cv2.remap(src.img, *m2, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
    valid_rect = cv2.remap(np.full((ref.h, ref.w), 255, np.uint8), *m1, cv2.INTER_NEAREST) > 0
    if vertical:  # make the epipolar lines horizontal
        gl, gr, valid_rect = gl.T.copy(), gr.T.copy(), valid_rect.T.copy()
    flip = P2[ax, 3] > 0  # partner sits on the "left": mirror so disparity is positive
    if flip:
        gl, gr, valid_rect = gl[:, ::-1].copy(), gr[:, ::-1].copy(), valid_rect[:, ::-1].copy()
    min_disp = int(np.floor(max(dmin, 0)))
    num = int(np.clip(np.ceil((dmax - min_disp) / 16.0) * 16, 16, 320))
    bs = 5
    mk = lambda: cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num, blockSize=bs,
                                       P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=-1,
                                       uniquenessRatio=12, speckleWindowSize=100, speckleRange=2,
                                       mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    dL = mk().compute(gl, gr).astype(np.float32) / 16.0
    # right-image disparity (mirror both and swap) for a left-right consistency check
    dR = mk().compute(gr[:, ::-1].copy(), gl[:, ::-1].copy()).astype(np.float32)[:, ::-1] / 16.0
    H, W = dL.shape
    xs = np.arange(W)[None, :].repeat(H, 0)
    ok = dL > max(min_disp, 0.5)
    xr = np.clip(np.round(xs - dL).astype(int), 0, W - 1)
    ok &= np.abs(dL - dR[np.arange(H)[:, None], xr]) <= 1.0
    ok &= valid_rect
    Z = np.where(ok, f * B / np.maximum(dL, 1e-6), 0).astype(np.float32)
    if flip:
        Z = Z[:, ::-1]
    if vertical:
        Z = Z.T
    f = P1[0, 0]  # P1 has equal fx, fy

    # rectified-ref pixels -> 3D in ref camera -> splat into the ref image grid
    ys, xs = np.nonzero(Z > 0)
    Zs = Z[ys, xs]
    Xr = np.stack([(xs - P1[0, 2]) * Zs / f, (ys - P1[1, 2]) * Zs / f, Zs], 1)
    Xc = Xr @ R1  # R1^T applied row-wise
    keep = Xc[:, 2] > 0
    Xc = Xc[keep]
    uv = (Xc @ ref.K.T)
    u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
    v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
    inb = (u >= 0) & (u < ref.w) & (v >= 0) & (v < ref.h)
    D = np.full((ref.h, ref.w), np.inf, np.float32)
    np.minimum.at(D, (v[inb], u[inb]), Xc[inb, 2].astype(np.float32))
    D[~np.isfinite(D)] = 0
    # sanity clamp against the sparse depth range
    D[(D < zmin * 0.5) | (D > zmax * 2.0)] = 0
    return D


# ---------------------------------------------------------------- plane-sweep stereo
def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def planesweep_depth(ref, srcs, rec=None, scale=0.5, n_planes=96, win=3, min_ncc=0.5,
                     min_std=2.0, margin=0.02, excl=4, stats=None):
    """Depth for `ref` by sweeping fronto-parallel planes through `srcs` (any motion).

    Unlike rectified stereo this works when the camera walks forward (epipole inside
    the image). Matching cost is ZNCC over a (2*win+1)^2 window, which ignores the
    brightness/exposure changes phones make between frames. Per plane, each source's
    cost is computed and the best two are averaged (robust to one source being occluded).
    """
    z = sparse_depths(ref, rec)
    zmin, zmax = np.percentile(z, 2) * 0.7, np.percentile(z, 98) * 1.4
    inv = np.linspace(1.0 / zmin, 1.0 / zmax, n_planes)  # uniform in inverse depth
    h, w = int(ref.h * scale), int(ref.w * scale)
    S = np.diag([scale, scale, 1.0])
    Kr = S @ ref.K
    Kr_inv = np.linalg.inv(Kr)
    I = cv2.resize(cv2.cvtColor(ref.img, cv2.COLOR_BGR2GRAY), (w, h),
                   interpolation=cv2.INTER_AREA).astype(np.float32)
    muI = _box(I, win)
    sdI = np.sqrt(np.maximum(_box(I * I, win) - muI * muI, 0))
    textured = sdI > min_std
    src_data = []
    for s_ in srcs:
        J = cv2.resize(cv2.cvtColor(s_.img, cv2.COLOR_BGR2GRAY), (w, h),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        R = s_.R @ ref.R.T
        t = s_.t - R @ ref.t
        src_data.append((J, S @ s_.K, R, t))
    ones = np.ones((h, w), np.float32)
    best = np.full((h, w), np.inf, np.float32)
    best_i = np.zeros((h, w), np.int32)
    costs = np.empty((n_planes, h, w), np.float32)
    n = np.array([0.0, 0.0, 1.0])
    for k, idep in enumerate(inv):
        per_src = []
        for J, Ks, R, t in src_data:
            H = Ks @ (R + np.outer(t, n) * idep) @ Kr_inv  # ref pixel -> src pixel
            Jw = cv2.warpPerspective(J, H, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            muJ = _box(Jw, win)
            sdJ = np.sqrt(np.maximum(_box(Jw * Jw, win) - muJ * muJ, 0))
            ncc = (_box(I * Jw, win) - muI * muJ) / (sdI * sdJ + 1e-3)
            c = 1.0 - ncc
            c[(valid < 0.5) | (sdJ < min_std)] = 2.0
            per_src.append(c)
        P = np.sort(np.stack(per_src), axis=0)
        cost = P[:2].mean(0) if len(per_src) >= 2 else P[0]
        costs[k] = cost
        upd = cost < best
        best[upd] = cost[upd]
        best_i[upd] = k
    # sub-plane refinement (parabola through neighbours, in inverse depth)
    k0 = np.clip(best_i, 1, n_planes - 2)
    yy, xx = np.mgrid[0:h, 0:w]
    cm, c0, cp = costs[k0 - 1, yy, xx], costs[k0, yy, xx], costs[k0 + 1, yy, xx]
    denom = cm - 2 * c0 + cp
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(denom) > 1e-6, 0.5 * (cm - cp) / denom, 0.0)
    off = np.clip(off, -0.5, 0.5)
    step = inv[1] - inv[0]
    idep = inv[k0] + off * step
    # confidence: good match, textured, not at the sweep's ends, clearly better than far planes
    far = costs.copy()
    for d in range(-excl, excl + 1):  # a broad, correct minimum is not ambiguity
        far[np.clip(best_i + d, 0, n_planes - 1), yy, xx] = np.inf
    second = far.min(0)
    c_match = best < 1.0 - min_ncc
    c_inside = (best_i > 0) & (best_i < n_planes - 1)
    c_unique = (second - best) > margin
    ok = c_match & textured & c_inside & c_unique
    if stats is not None:
        stats.update(match=c_match.mean(), textured=textured.mean(), inside=c_inside.mean(),
                     unique=c_unique.mean(), anysrc=(best < 2.0).mean(), ok=ok.mean())
    D = np.where(ok, 1.0 / np.maximum(idep, 1e-9), 0).astype(np.float32)
    return cv2.resize(D, (ref.w, ref.h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------- fusion
def backproject(view, D, mask):
    ys, xs = np.nonzero(mask)
    z = D[ys, xs]
    Kinv = np.linalg.inv(view.K)
    pix = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float64)
    Xc = (pix @ Kinv.T) * z[:, None]
    Xw = (Xc - view.t) @ view.R  # R^T (Xc - t)
    col = view.img[ys, xs, ::-1]  # BGR -> RGB
    return Xw, col


def consistency_mask(view, neighbours, rel_tol=0.02, min_agree=1):
    D = view.depth
    m = D > 0
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return m
    Xw, _ = backproject(view, D, m)
    agree = np.zeros(len(ys), np.int32)
    for n in neighbours:
        if n.depth is None:
            continue
        Xc = Xw @ n.R.T + n.t
        z = Xc[:, 2]
        uv = Xc @ n.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(uv[:, 0] / uv[:, 2])
            v = np.round(uv[:, 1] / uv[:, 2])
        inb = (z > 0) & (u >= 0) & (u < n.w) & (v >= 0) & (v < n.h)
        dn = np.zeros(len(z))
        dn[inb] = n.depth[v[inb].astype(int), u[inb].astype(int)]
        agree += (dn > 0) & (np.abs(dn - z) < rel_tol * z)
    out = np.zeros_like(m)
    out[ys, xs] = agree >= min_agree
    return out


def voxel_downsample(P, C, voxel):
    q = np.floor(P / voxel).astype(np.int64)
    _, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = cnt.size
    Ps = np.zeros((n, 3)); Cs = np.zeros((n, 3))
    np.add.at(Ps, inv, P); np.add.at(Cs, inv, C.astype(np.float64))
    return Ps / cnt[:, None], (Cs / cnt[:, None]).astype(np.uint8), cnt


def remove_outliers(P, C, k=12, std_ratio=2.0):
    """Statistical outlier removal: drop points whose mean k-NN distance is unusually large."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(P).query(P, k=k + 1, workers=-1)
    m = d[:, 1:].mean(1)
    keep = m < m.mean() + std_ratio * m.std()
    return P[keep], C[keep]


def write_ply(path, P, C):
    P = P.astype(np.float32); C = C.astype(np.uint8)
    rec = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = P.T
    rec["red"], rec["green"], rec["blue"] = C.T
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        fh.write(rec.tobytes())


def merge_depths(A, B, rel_tol=0.02):
    """Fill A's holes from B; where both have depth but disagree, trust neither."""
    out = A.copy()
    fill = (A <= 0) & (B > 0)
    out[fill] = B[fill]
    both = (A > 0) & (B > 0)
    conflict = both & (np.abs(A - B) > rel_tol * A)
    out[conflict] = 0
    agree = both & ~conflict
    out[agree] = 0.5 * (A[agree] + B[agree])
    return out


def run_mvs(sparse_dir: Path, image_dir: Path, out_dir: Path, scale=1.0, voxel_frac=0.0015,
            rel_tol=0.015, min_agree=2, max_views=None, n_partners=2, extra_sparse=(),
            method="sweep"):
    """extra_sparse: sub-models already aligned into the main model's frame (see
    sfm.attach_submodels); their frames that the main model lacks are added as views."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction(str(sparse_dir))
    imgs = sorted([im for im in rec.images.values() if im.has_pose], key=lambda im: im.name)
    if max_views:
        imgs = imgs[:: max(1, len(imgs) // max_views)]
    views = [View(im, rec.cameras[im.camera_id], image_dir, scale, rec, 0) for im in imgs]
    have = {v.name for v in views}
    for tag, extra_dir in enumerate(extra_sparse, start=1):
        er = pycolmap.Reconstruction(str(extra_dir))
        added = [View(im, er.cameras[im.camera_id], image_dir, scale, er, tag)
                 for im in sorted(er.images.values(), key=lambda im: im.name)
                 if im.has_pose and im.name not in have]
        # keep the sub-model's overlap frames too: they are its stereo/consistency partners
        overlap = [View(im, er.cameras[im.camera_id], image_dir, scale, er, tag)
                   for im in er.images.values() if im.has_pose and im.name in have]
        for v in overlap:
            v.depth_only_support = True
        views += added + overlap
        have |= {v.name for v in added}
        print(f"[mvs] attached sub-model {extra_dir.name}: +{len(added)} new frames", flush=True)
    print(f"[mvs] {len(views)} views loaded {time.time()-t0:.0f}s", flush=True)
    stereo, neigh, med = choose_partners(views, rec)
    for i, v in enumerate(views):
        if method == "sweep":  # plane sweep: any camera motion, incl. walking forward
            srcs = stereo.get(v.id, [])[:3] or neigh.get(v.id, [])[:3]
            if srcs:
                v.depth = planesweep_depth(v, srcs)
        else:  # rectified SGBM: fast, but only for sideways / orbiting motion
            got = 0
            for partner in stereo.get(v.id, []):
                D = stereo_depth(v, partner, rec)
                if D is None or (D > 0).mean() < 0.05:
                    continue
                v.depth = D if v.depth is None else merge_depths(v.depth, D)
                got += 1
                if got >= n_partners:
                    break
        if i % 25 == 0:
            cov = 0 if v.depth is None else (v.depth > 0).mean()
            print(f"[mvs] stereo {i+1}/{len(views)} coverage={cov:.2f} {time.time()-t0:.0f}s", flush=True)
    allP, allC, cov_raw, cov_fused = [], [], [], []
    for v in views:
        if v.depth is None or getattr(v, "depth_only_support", False):
            continue  # overlap copies only serve as partners for a sub-model's new frames
        m = consistency_mask(v, neigh.get(v.id, []), rel_tol, min_agree)
        cov_raw.append((v.depth > 0).mean()); cov_fused.append(m.mean())
        P, C = backproject(v, v.depth, m)
        allP.append(P); allC.append(C)
    P = np.concatenate(allP); C = np.concatenate(allC)
    print(f"[mvs] per-view coverage: stereo median {np.median(cov_raw):.2f}, "
          f"after consistency median {np.median(cov_fused):.2f}", flush=True)
    print(f"[mvs] fused raw points: {len(P):,} {time.time()-t0:.0f}s", flush=True)
    # scene-relative voxel size
    extent = np.linalg.norm(np.percentile(P, 98, 0) - np.percentile(P, 2, 0))
    voxel = extent * voxel_frac
    P, C, cnt = voxel_downsample(P, C, voxel)
    keep = cnt >= 2  # supported by >1 observation
    P, C = P[keep], C[keep]
    P, C = remove_outliers(P, C)
    print(f"[mvs] after voxel ({voxel:.4f}) + support filter: {len(P):,}", flush=True)
    write_ply(out_dir / "dense.ply", P, C)
    # cameras for the viewer
    cams = [{"name": v.name, "R": v.R.tolist(), "t": v.t.tolist(), "C": v.C.tolist(),
             "fx": float(v.K[0, 0]), "fy": float(v.K[1, 1]), "w": v.w, "h": v.h}
            for v in sorted(views, key=lambda v: v.name)
            if not getattr(v, "depth_only_support", False)]
    (out_dir / "cameras.json").write_text(json.dumps(cams))
    print(f"[mvs] wrote {out_dir/'dense.ply'} in {time.time()-t0:.0f}s")
    return P, C, views


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sparse", type=Path)
    ap.add_argument("images", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-views", type=int, default=None)
    a = ap.parse_args()
    run_mvs(a.sparse, a.images, a.out, a.scale, max_views=a.max_views)
