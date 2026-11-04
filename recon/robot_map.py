"""Stage 4: turn the reconstruction into something a (humanoid) robot can use.

  * gravity alignment: world +Z = up, floor at Z = 0 (RANSAC floor plane)
  * metric scale: video-only SfM has no absolute scale, so we fix it from the phone's
    height above the floor while filming (default 1.4 m, typical for a hand-held phone).
    ARKit captures (see ingest_arkit.py) are already metric -> pass --metric.
  * 2D occupancy grid for navigation (ROS nav2 map_server format: PGM + YAML)
  * 3D voxel occupancy (npz) for footstep / collision planning
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np

from mvs import write_ply


def read_ply(path):
    data = open(path, "rb").read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    head = data[:end].decode()
    n = int(head.split("element vertex ")[1].split()[0])
    rec = np.frombuffer(data[end:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                           ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n)
    P = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)
    C = np.stack([rec["r"], rec["g"], rec["b"]], 1)
    return P, C


def rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), a @ b
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def _ransac_plane(cand, up, thr, rng, iters):
    """Plane within 15 deg of horizontal with the most inliers (or None)."""
    best, best_n = None, 0
    for _ in range(iters):
        s = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if n @ up < 0:
            n = -n
        if n @ up < np.cos(np.radians(15)):
            continue
        d = -n @ s[0]
        inl = np.abs(cand @ n + d) < thr
        if inl.sum() > best_n:
            best_n, best = inl.sum(), (n, d, inl)
    if best is None:
        return None
    n, d, inl = best
    Q = cand[inl]  # least-squares refine on inliers
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    n2 = vt[2] if vt[2] @ up > 0 else -vt[2]
    return n2, -n2 @ c, np.abs(cand @ n2 - n2 @ c) < thr


def fit_floor(P, up, cam_centres, iters=600, seed=0, min_rel_support=0.10, min_drop_ratio=1.25):
    """The floor = the LOWEST well-supported horizontal plane below the cameras.

    Tables, beds and counters are big horizontal planes too, and in a close-up scan
    one of them can out-number the floor. So after the biggest plane, keep looking
    underneath it and take the lowest plane with >= 10% of its support.
    """
    rng = np.random.default_rng(seed)
    cam_h = np.median(cam_centres @ up)
    cand = P[P @ up < cam_h]  # floor is below the phone
    if len(cand) > 200000:
        cand = cand[rng.choice(len(cand), 200000, replace=False)]
    total = len(cand)
    thr = max(np.ptp(cand @ up) * 0.01, 1e-6)
    plane = _ransac_plane(cand, up, thr, rng, iters)
    if plane is None:
        raise ValueError("no horizontal plane found")
    n, d, inl = plane
    first_support = inl.sum()
    for _ in range(3):
        below = cand[(cand @ n + d) < -3 * thr]  # strictly under the current plane
        if len(below) < max(50, min_rel_support * first_support):
            break
        lower = _ransac_plane(below, up, thr, rng, iters)
        if lower is None or lower[2].sum() < min_rel_support * first_support:
            break
        # A table/counter -> floor step is a big drop (phone ~2x farther from the floor).
        # Reflections in glossy floors and noise make a thin layer just *under* the
        # true floor; ignore lower planes that are not clearly farther from the cameras.
        h_cur = np.median(cam_centres @ n + d)
        h_low = np.median(cam_centres @ lower[0] + lower[1])
        if h_low < min_drop_ratio * h_cur:
            break
        n, d, inl = lower
    return n, d, inl.sum() / total


def build(ply, cameras_json, out, phone_height=1.4, metric=False, cell=0.05,
          robot_height=1.8, step_height=0.08, walk_radius=0.3, infer_gap=0.5, carve_radius=0.2):
    out.mkdir(parents=True, exist_ok=True)
    P, C = read_ply(ply)
    cams = json.loads(Path(cameras_json).read_text())
    Rs = np.array([c["R"] for c in cams])
    Cs = np.array([c["C"] for c in cams])
    up0 = np.array([0.0, 1.0, 0.0])  # ARKit world: +Y is gravity-up, exactly
    if metric:
        n, d, frac = fit_floor(P, up0, Cs)
    else:
        # Image-up is usually world-up (phone held upright), but not if the video's
        # rotation flag was wrong or the phone was sideways. Try every camera axis as
        # gravity and keep the best-supported floor that lies below the cameras.
        best = None
        for axis, sign in ((1, -1), (1, 1), (0, -1), (0, 1)):
            cand = sign * Rs[:, axis, :].mean(0)
            if np.linalg.norm(cand) < 0.3:  # axis not consistent across the capture
                continue
            cand /= np.linalg.norm(cand)
            try:
                n_, d_, f_ = fit_floor(P, cand, Cs)
            except (TypeError, ValueError):
                continue
            above = np.median(Cs @ n_ + d_) > 0
            # People film roughly horizontally: viewing directions run along the floor.
            # If the cameras mostly look *into* the candidate plane, it is a wall.
            look = np.median(np.abs(Rs[:, 2, :] @ n_))
            print(f"[robot_map] up candidate axis {axis}{'+' if sign > 0 else '-'}: floor support "
                  f"{f_:.3f}, cameras above: {above}, view-dir alignment {look:.2f}")
            if not above or look > 0.75:
                continue
            if best is None or f_ > best[2] * 1.2:
                best = (n_, d_, f_)
        if best is None:
            raise ValueError("could not find a floor plane below the cameras")
        n, d, frac = best
    # rotate floor normal to +Z, translate floor to Z=0
    R = rot_between(n, np.array([0, 0, 1.0]))
    Pw = P @ R.T
    Cw = Cs @ R.T
    z0 = -d  # n.x + d = 0  ->  height of floor along n is -d
    Pw[:, 2] -= z0
    Cw[:, 2] -= z0
    cam_h = np.median(Cw[:, 2])
    scale = 1.0 if metric else phone_height / cam_h
    Pw *= scale
    Cw *= scale
    # centre XY on the scene
    xy0 = np.median(Pw[:, :2], 0)
    Pw[:, :2] -= xy0
    Cw[:, :2] -= xy0
    write_ply(out / "map_metric.ply", Pw, C)

    # transformed cameras (metric, z-up) for the viewer / robot
    Rw = np.einsum("nij,kj->nik", Rs, R)  # cam_from_world' = R_c R^T
    cams_m = []
    for c, Rc, Cc in zip(cams, Rw, Cw):
        cams_m.append({**c, "R": Rc.tolist(), "C": Cc.tolist(), "t": (-Rc @ Cc).tolist()})
    (out / "cameras_metric.json").write_text(json.dumps(cams_m))

    # ---------------- 2D occupancy grid (nav2 conventions) ----------------
    lo, hi = np.percentile(Pw[:, :2], 0.5, 0) - 0.5, np.percentile(Pw[:, :2], 99.5, 0) + 0.5
    W, H = np.ceil((hi - lo) / cell).astype(int)
    ij = np.floor((Pw[:, :2] - lo) / cell).astype(int)
    inb = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
    z = Pw[:, 2]
    floor_pts = inb & (np.abs(z) < 0.05)
    obst_pts = inb & (z > step_height) & (z < robot_height)
    free = np.zeros((H, W), np.int32)
    occ = np.zeros((H, W), np.int32)
    np.add.at(free, (ij[floor_pts, 1], ij[floor_pts, 0]), 1)
    np.add.at(occ, (ij[obst_pts, 1], ij[obst_pts, 0]), 1)
    occ_mask = occ >= 3
    seen_free = (free >= 1) & ~occ_mask
    # fill small holes in observed floor (sparse stereo on textureless floor)
    seen_free = cv2.morphologyEx(seen_free.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((5, 5), np.uint8)).astype(bool) & ~occ_mask
    # Real evidence stereo can miss: the person carrying the phone walked here, so the
    # floor exists and was free (a ~0.3 m footprint around the projected camera path).
    path_px = np.floor((Cw[:, :2] - lo) / cell).astype(np.int32)
    walked = np.zeros((H, W), np.uint8)
    foot = max(1, int(round(walk_radius / cell)))
    for a_, b_ in zip(path_px[:-1], path_px[1:]):
        if np.linalg.norm((b_ - a_) * cell) < 1.0:  # skip jumps across SfM gaps
            cv2.line(walked, tuple(int(x) for x in a_), tuple(int(x) for x in b_), 1, 2 * foot)
    # The walker's body swept this space: obstacles reported in the core of the path
    # (< carve_radius) are false (depth streaks) and get cleared.
    core = np.zeros((H, W), np.uint8)
    cr = max(1, int(round(carve_radius / cell)))
    for a_, b_ in zip(path_px[:-1], path_px[1:]):
        if np.linalg.norm((b_ - a_) * cell) < 1.0:
            cv2.line(core, tuple(int(x) for x in a_), tuple(int(x) for x in b_), 1, 2 * cr)
    carved = occ_mask & core.astype(bool)
    occ_mask = occ_mask & ~carved
    walked = walked.astype(bool) & ~occ_mask
    free_mask = seen_free | walked
    # Inferred ("imagined") floor: gaps up to ~infer_gap between known-free cells with no
    # obstacle seen. Kept in a separate layer: plausible, but never observed.
    g = max(3, int(round(infer_gap / cell)) | 1)
    closed = cv2.morphologyEx(free_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g))).astype(bool)
    inferred = closed & ~free_mask & ~occ_mask

    def write_grid(name, free_cells):
        grid = np.full((H, W), 205, np.uint8)  # unknown
        grid[free_cells] = 254
        grid[occ_mask] = 0
        cv2.imwrite(str(out / f"{name}.pgm"), grid[::-1])  # PGM row 0 = max y
        (out / f"{name}.yaml").write_text(
            f"image: {name}.pgm\nmode: trinary\nresolution: {cell}\n"
            f"origin: [{lo[0]:.3f}, {lo[1]:.3f}, 0.0]\nnegate: 0\n"
            f"occupied_thresh: 0.65\nfree_thresh: 0.25\n")
    write_grid("occupancy", free_mask)                         # observed + walked
    write_grid("occupancy_with_inferred", free_mask | inferred)  # + imagined gaps
    # preview: white seen, light green walked, light blue inferred, black occupied
    prev = np.full((H, W, 3), 205, np.uint8)
    prev[seen_free] = (255, 255, 255)
    prev[walked & ~seen_free] = (200, 240, 200)
    prev[inferred] = (245, 225, 190)
    prev[occ_mask] = (0, 0, 0)
    prev = prev[::-1].copy()
    pts = path_px.copy()
    pts[:, 1] = H - 1 - pts[:, 1]
    cv2.polylines(prev, [pts.reshape(-1, 1, 2)], False, (255, 160, 60), 1, cv2.LINE_AA)
    k = max(1, int(np.ceil(600 / max(W, H))))
    prev = cv2.resize(prev, (W * k, H * k), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out / "occupancy_preview.png"), prev)
    # inferred floor as points (for the viewer), coloured like the observed floor, z = 0
    fill = (walked & ~seen_free) | inferred
    if fill.any() and floor_pts.any():
        fy, fx = np.nonzero(fill)
        Pi = np.stack([lo[0] + (fx + 0.5) * cell, lo[1] + (fy + 0.5) * cell, np.zeros(len(fx))], 1)
        # sub-sample each cell a little so it renders as a surface, not a dot grid
        jit = np.random.default_rng(0).uniform(-0.5, 0.5, (len(Pi) * 4, 2)) * cell
        Pi = np.repeat(Pi, 4, 0); Pi[:, :2] += jit
        col = np.median(C[floor_pts], 0)
        write_ply(out / "inferred_floor.ply", Pi, np.tile(col, (len(Pi), 1)).astype(np.uint8))

    # ---------------- 3D voxel occupancy ----------------
    vox = 0.05
    vmin = Pw.min(0)
    vi = np.unique(np.floor((Pw - vmin) / vox).astype(np.int32), axis=0)
    np.savez_compressed(out / "voxels.npz", indices=vi, origin=vmin, voxel_size=vox)

    warnings = []
    if frac > 0.5:
        warnings.append("over half the scene lies on one plane: this looks like a close-up/tabletop "
                        "scan where the real floor was barely seen. The 'floor' may be a table, so "
                        "scale and the occupancy map are unreliable. Walk the room with the floor in view.")
    if metric and cam_h < 0.9:
        warnings.append(f"the phone was only {cam_h:.2f} m above the detected floor: that 'floor' is "
                        "probably a table or bed, not the ground.")
    for w in warnings:
        print("[robot_map] WARNING:", w)
    meta = {"scale_applied": float(scale), "metric_source": "arkit" if metric else
            f"phone_height={phone_height}m", "floor_inlier_fraction": float(frac),
            "grid_cell_m": cell, "grid_size": [int(W), int(H)],
            "free_cells": int(free_mask.sum()), "free_cells_seen": int(seen_free.sum()),
            "free_cells_walked_only": int((walked & ~seen_free).sum()),
            "inferred_cells": int(inferred.sum()), "occupied_cells": int(occ_mask.sum()),
            "false_obstacles_carved_by_walk": int(carved.sum()),
            "points": int(len(Pw)), "voxels_5cm": int(len(vi)),
            "scene_extent_m": (Pw.max(0) - Pw.min(0)).round(2).tolist(),
            "camera_height_median_m": float(np.median(Cw[:, 2])), "warnings": warnings}
    (out / "map_meta.json").write_text(json.dumps(meta, indent=2))
    print("[robot_map]", json.dumps(meta))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    ap.add_argument("cameras", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--phone-height", type=float, default=1.4)
    ap.add_argument("--metric", action="store_true", help="input is already metric (ARKit)")
    ap.add_argument("--cell", type=float, default=0.05)
    a = ap.parse_args()
    build(a.ply, a.cameras, a.out, a.phone_height, a.metric, a.cell)
