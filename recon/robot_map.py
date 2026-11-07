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


def find_glass_walls(Pw, Cw, z_band=(0.2, 2.2), thr=0.05, min_inliers=200, min_len=1.5,
                     behind_min=1.0, behind_ratio=1.0, max_walls=6, suppress=0.3, seed=0,
                     gap_bin=0.1, cliff_frac=0.1, low_frac=0.05, low_len=0.5, beyond_frac=0.05):
    """Vertical planes (lines in top view) that look like glass: things are seen *behind*
    them, on the side the walker never was.

    An opaque wall hides what's behind it, so almost no points lie beyond it. A glass wall
    or window has a frame/mullions (some points on the line) and the world behind it
    (many points beyond). Returns [(p0, p1, ratio_behind)] segment endpoints in XY.

    The wall is placed at the density cliff where the room's contents end (stereo puts
    almost nothing on glass), and only when that cliff is followed by >= 0.5 m of
    near-empty space and then real geometry beyond; otherwise the line is skipped, so the
    detector only draws walls it is confident about."""
    rng = np.random.default_rng(seed)
    z = Pw[:, 2]
    Q = Pw[(z > z_band[0]) & (z < z_band[1]), :2]
    if len(Q) > 300000:
        Q = Q[rng.choice(len(Q), 300000, replace=False)]
    cams = Cw[:, :2]
    remaining = np.ones(len(Q), bool)
    walls = []
    for _ in range(max_walls * 3):
        idx = np.flatnonzero(remaining)
        if len(idx) < min_inliers or len(walls) >= max_walls:
            break
        best = None
        for _ in range(300):
            a, b = Q[rng.choice(idx, 2, replace=False)]
            d = b - a
            L = np.linalg.norm(d)
            if L < 0.3:
                continue
            n = np.array([-d[1], d[0]]) / L
            dist = (Q[idx] - a) @ n
            inl = np.abs(dist) < thr
            if best is None or inl.sum() > best[2].sum():
                best = (a, n, inl)
        if best is None or best[2].sum() < min_inliers:
            break
        a, n, inl = best
        pts = Q[idx[inl]]
        dist_r = (Q - a) @ n
        remaining &= np.abs(dist_r) > suppress  # drop the whole band: no duplicate lines
        t_dir = np.array([n[1], -n[0]])
        t = (pts - a) @ t_dir
        t0, t1 = np.percentile(t, 2), np.percentile(t, 98)
        if t1 - t0 < min_len:
            continue
        # only cameras facing this stretch of wall count; none may be behind it (a room
        # the walker went into is not "the other side of glass")
        c_along = (cams - a) @ t_dir
        c_dist = (cams - a) @ n
        span = (c_along > t0 - 0.5) & (c_along < t1 + 0.5)
        if span.sum() < 5:
            continue
        walk_side = np.sign(np.median(c_dist[span]))
        if np.any(c_dist[span] * walk_side < 0) or np.any(np.abs(c_dist) < 0.2):
            continue
        along = (Q - a) @ t_dir
        within = (along > t0) & (along < t1)
        behind = within & (dist_r * walk_side < -behind_min)
        ratio = behind.sum() / max(inl.sum(), 1)
        if ratio <= behind_ratio:
            continue
        # Where is the glass? Walking from the cameras outward: room contents (dense, the
        # peak is furniture against the glass), then a sharp density cliff (stereo puts
        # almost nothing on glass), a long low stretch (reflections, see-through haze),
        # and the world beyond. Wall goes at the cliff -- only if all of that is clearly
        # there; otherwise this line is skipped (no guess).
        sd = dist_r[within] * walk_side                 # + = walker's side
        cam_s = np.median(c_dist[span] * walk_side)
        edges = np.arange(-4.0, cam_s + gap_bin, gap_bin)
        hist, edges = np.histogram(sd, bins=edges)
        room_side = edges[:-1] >= 0
        if not room_side.any() or hist[room_side].max() < min_inliers:
            continue
        peak_k = np.flatnonzero(room_side)[np.argmax(hist[room_side])]
        peak = hist[peak_k]
        found = None
        for k in range(peak_k - 1, -1, -1):              # outward from the peak
            if hist[k] < cliff_frac * peak:
                n_low = int(round(low_len / gap_bin))
                lo_k = k - n_low                         # the cliff bin itself may be on the slope
                if lo_k < 0:
                    break
                if np.all(hist[lo_k:k] < low_frac * peak):
                    beyond = hist[:lo_k].sum()
                    room = hist[k + 1:].sum()
                    if beyond >= beyond_frac * room:
                        found = (edges[k + 1], room, beyond)
                break                                    # first cliff decides
        if found is None:
            continue                                     # not certain: draw nothing
        cliff, room, beyond = found
        off = cliff * walk_side
        p0 = a + t0 * t_dir + off * n
        p1 = a + t1 * t_dir + off * n
        dup = False
        for q0, q1, _ in walls:  # same glass found twice (parallel, close): keep the first
            dq = (q1 - q0) / max(np.linalg.norm(q1 - q0), 1e-9)
            nq = np.array([-dq[1], dq[0]])
            mid = 0.5 * (p0 + p1)
            if abs(dq @ t_dir) > np.cos(np.radians(10)) and abs((mid - q0) @ nq) < 0.4:
                dup = True
                break
        if not dup:
            walls.append((p0, p1, float(beyond / max(room, 1))))
    return walls


def to_map(X, transform):
    """Points in the reconstruction's frame -> the robot map's metric, z-up frame
    (transform: robot/map_transform.json as a dict)."""
    R = np.array(transform["R"])
    Y = X @ R.T
    Y[:, 2] -= transform["z0"]
    Y *= transform["scale"]
    Y[:, :2] -= np.array(transform["xy0"])
    return Y


def build(ply, cameras_json, out, phone_height=1.4, metric=False, cell=0.05,
          robot_height=1.8, step_height=0.08, walk_radius=0.3, infer_gap=0.5, carve_radius=0.2,
          below_floor=0.05, obst_ratio=0.5, reach_radius=3.5, detect_glass=True,
          metric_scale=False, floor_ref=None):
    out.mkdir(parents=True, exist_ok=True)
    P, C = read_ply(ply)
    # Floor / gravity can come from a larger cloud in the same coordinates (e.g. the full
    # scan when mapping a sub-section of it): a subset may barely see the floor.
    P_floor = read_ply(floor_ref)[0] if floor_ref else P
    cams = json.loads(Path(cameras_json).read_text())
    Rs = np.array([c["R"] for c in cams])
    Cs = np.array([c["C"] for c in cams])
    up0 = np.array([0.0, 1.0, 0.0])  # ARKit world: +Y is gravity-up, exactly
    if metric:
        n, d, frac = fit_floor(P_floor, up0, Cs)
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
                n_, d_, f_ = fit_floor(P_floor, cand, Cs)
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
    # metric: ARKit (scale + gravity known). metric_scale: scale known (e.g. ViPE's metric
    # depth) but gravity still estimated from the floor. Otherwise: phone-height guess.
    scale = 1.0 if (metric or metric_scale) else phone_height / cam_h
    Pw *= scale
    Cw *= scale
    # centre XY on the scene
    xy0 = np.median(Pw[:, :2], 0)
    Pw[:, :2] -= xy0
    Cw[:, :2] -= xy0
    # the transform itself, so other clouds in the input frame can be brought into the map:
    # X_map = scale * (R @ X - [0, 0, z0]) - [xy0, 0]   (to_map() below)
    (out / "map_transform.json").write_text(json.dumps(
        {"R": R.tolist(), "z0": float(z0), "scale": float(scale), "xy0": xy0.tolist()}))
    # same metric, z-up transform for the TSDF mesh, if the dense step produced one
    for src_name, dst_name in (("mesh.ply", "mesh_metric.ply"),
                               ("mesh_inferred.ply", "mesh_inferred_metric.ply")):
        mesh_in = Path(ply).parent / src_name
        if not mesh_in.exists():
            continue
        from tsdf import read_mesh_ply, write_mesh_ply
        V, F, Cm = read_mesh_ply(mesh_in)
        V = V @ R.T
        V[:, 2] -= z0
        V *= scale
        V[:, :2] -= xy0
        write_mesh_ply(out / dst_name, V, F, Cm)
    # Nothing can be under the floor of a single-level space: points well below it are
    # reflections in glossy floors (stereo matches the mirror image) or noise.
    under = Pw[:, 2] < -below_floor
    Pw, C = Pw[~under], C[~under]
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
    # Obstacle if enough points stand in the robot's height band *relative to* what the
    # cell holds overall: with dense depth, a thin haze of noise above a well-seen floor
    # would otherwise exceed any fixed count.
    occ_mask = (occ >= 3) & (occ >= obst_ratio * free)
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
    # Obstacles far from anywhere the walker went are mostly seen *through* glass walls
    # / windows (an atrium, the street) or are reflection ghosts. The robot can't reach
    # them through the glass anyway: mark them unknown rather than blocked.
    far = np.zeros((H, W), np.uint8)
    rr = max(1, int(round(reach_radius / cell)))
    for a_, b_ in zip(path_px[:-1], path_px[1:]):
        if np.linalg.norm((b_ - a_) * cell) < 1.0:
            cv2.line(far, tuple(int(x) for x in a_), tuple(int(x) for x in b_), 1, 2 * rr)
    beyond = occ_mask & ~far.astype(bool)
    occ_mask = occ_mask & ~beyond
    walked = walked.astype(bool) & ~occ_mask
    free_mask = seen_free | walked
    # Inferred ("imagined") floor: gaps up to ~infer_gap between known-free cells with no
    # obstacle seen. Kept in a separate layer: plausible, but never observed.
    g = max(3, int(round(infer_gap / cell)) | 1)
    closed = cv2.morphologyEx(free_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g))).astype(bool)
    inferred = closed & ~free_mask & ~occ_mask

    # glass walls / windows: draw them as walls (stereo sees through glass, so the map
    # would otherwise have a gap there, or "obstacles" out in the room beyond)
    glass = find_glass_walls(Pw, Cw) if detect_glass else []
    glass_cells = np.zeros((H, W), np.uint8)
    for p0, p1, _ in glass:
        a_ = tuple(int(x) for x in np.floor((p0 - lo) / cell))
        b_ = tuple(int(x) for x in np.floor((p1 - lo) / cell))
        cv2.line(glass_cells, a_, b_, 1, 2)
    glass_mask = glass_cells.astype(bool)
    occ_mask = occ_mask | glass_mask
    free_mask = free_mask & ~glass_mask
    inferred = inferred & ~glass_mask
    seen_free = seen_free & ~glass_mask
    walked = walked & ~glass_mask

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
    prev[glass_mask] = (230, 120, 0)  # glass walls drawn in blue (BGR)
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
    # A hand-held phone stays at about the same height. If the camera heights spread much
    # more, the reconstruction is bent or tilted (monocular drift), whatever the floor fit
    # says. Clean walk-throughs: 0.05-0.4 m; bent ones measured 0.5-2.1 m.
    h_spread = float(np.percentile(Cw[:, 2], 95) - np.percentile(Cw[:, 2], 5))
    if h_spread > 0.5:
        warnings.append(f"camera heights spread over {h_spread:.2f} m (p5-p95): the map is probably "
                        "bent or tilted (camera-path drift), so walls and floor may not line up. "
                        "Re-run, or capture with the app (ARKit) for drift-free poses.")
    for w in warnings:
        print("[robot_map] WARNING:", w)
    meta = {"scale_applied": float(scale), "metric_source": "arkit" if metric else "input (metric depth)" if metric_scale else
            f"phone_height={phone_height}m", "floor_inlier_fraction": float(frac),
            "grid_cell_m": cell, "grid_size": [int(W), int(H)],
            "free_cells": int(free_mask.sum()), "free_cells_seen": int(seen_free.sum()),
            "free_cells_walked_only": int((walked & ~seen_free).sum()),
            "inferred_cells": int(inferred.sum()), "occupied_cells": int(occ_mask.sum()),
            "false_obstacles_carved_by_walk": int(carved.sum()),
            "points_below_floor_removed": int(under.sum()),
            "obstacles_beyond_reach_unknown": int(beyond.sum()),
            "glass_walls": [{"from": [round(float(v), 2) for v in p0], "to": [round(float(v), 2) for v in p1],
                             "length_m": round(float(np.linalg.norm(p1 - p0)), 2),
                             "behind_ratio": round(r, 2)} for p0, p1, r in glass],
            "points": int(len(Pw)), "voxels_5cm": int(len(vi)),
            "scene_extent_m": (Pw.max(0) - Pw.min(0)).round(2).tolist(),
            "camera_height_median_m": float(np.median(Cw[:, 2])),
            "camera_height_spread_m": round(h_spread, 3), "warnings": warnings}
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
    ap.add_argument("--floor-ref", type=Path, default=None,
                    help="larger cloud in the same coordinates to take the floor/gravity from")
    a = ap.parse_args()
    build(a.ply, a.cameras, a.out, a.phone_height, a.metric, a.cell, floor_ref=a.floor_ref)
