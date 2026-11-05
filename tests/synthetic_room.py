"""Tiny numpy ray caster for a furnished living room, used by the robot_link tests and docs renders.

Cameras use ARKit's convention: camera-to-world 4x4, OpenGL axes (x right, y up, -z forward),
world y-up, metres.
"""
import numpy as np

ROOM = (np.array([-2.25, 0.0, -2.0]), np.array([2.25, 2.6, 2.0]))

# (min, max, rgb, name)
BOXES = [
    ((-1.6, 0.0, -2.0), (0.4, 0.45, -1.1), (0.36, 0.42, 0.52), "sofa"),
    ((-1.6, 0.45, -2.0), (0.4, 0.88, -1.72), (0.33, 0.39, 0.49), "sofa_back"),
    ((-1.75, 0.0, -2.0), (-1.6, 0.62, -1.1), (0.31, 0.37, 0.47), "arm_l"),
    ((0.4, 0.0, -2.0), (0.55, 0.62, -1.1), (0.31, 0.37, 0.47), "arm_r"),
    ((-1.05, 0.36, -0.75), (0.0, 0.42, -0.2), (0.55, 0.38, 0.24), "table_top"),
    ((-1.0, 0.0, -0.7), (-0.94, 0.36, -0.64), (0.4, 0.27, 0.17), "leg1"),
    ((-0.11, 0.0, -0.7), (-0.05, 0.36, -0.64), (0.4, 0.27, 0.17), "leg2"),
    ((-1.0, 0.0, -0.31), (-0.94, 0.36, -0.25), (0.4, 0.27, 0.17), "leg3"),
    ((-0.11, 0.0, -0.31), (-0.05, 0.36, -0.25), (0.4, 0.27, 0.17), "leg4"),
    ((1.85, 0.0, -1.3), (2.25, 1.85, -0.2), (0.78, 0.74, 0.68), "shelf"),
    ((1.82, 0.62, -1.28), (1.86, 0.66, -0.22), (0.6, 0.56, 0.5), "shelf_edge1"),
    ((1.82, 1.22, -1.28), (1.86, 1.26, -0.22), (0.6, 0.56, 0.5), "shelf_edge2"),
    ((1.45, 0.0, 1.35), (1.8, 0.35, 1.7), (0.62, 0.5, 0.42), "pot"),
    ((1.48, 0.35, 1.38), (1.77, 1.15, 1.67), (0.25, 0.45, 0.24), "plant"),
    ((-2.25, 0.0, 0.9), (-1.85, 0.75, 1.6), (0.5, 0.36, 0.26), "side_table"),
    ((-0.3, 0.0, 1.0), (0.25, 0.46, 1.55), (0.72, 0.45, 0.3), "chair_seat"),
    ((-0.3, 0.46, 1.48), (0.25, 0.95, 1.55), (0.68, 0.42, 0.28), "chair_back"),
]
LIGHT = np.array([0.35, 0.85, 0.4]); LIGHT /= np.linalg.norm(LIGHT)


def _hash(ix, iz):
    return (np.sin(ix * 12.9898 + iz * 78.233) * 43758.5453) % 1.0


def render(c2w, W, H, fx, fy, cx, cy):
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u)], -1)
    R, o = c2w[:3, :3], c2w[:3, 3]
    dw = d @ R.T
    inv = 1.0 / np.where(np.abs(dw) < 1e-9, 1e-9, dw)

    lo, hi = ROOM
    t1 = (lo - o) * inv; t2 = (hi - o) * inv
    texit = np.maximum(t1, t2)
    t = texit.min(-1)
    axis = texit.argmin(-1)
    sign = np.take_along_axis(np.sign(dw), axis[..., None], -1)[..., 0]
    normal = np.zeros(dw.shape); np.put_along_axis(normal, axis[..., None], -sign[..., None], -1)
    obj = np.full(t.shape, -1)

    for k, (bmin, bmax, _, _) in enumerate(BOXES):
        a = (np.array(bmin) - o) * inv; b = (np.array(bmax) - o) * inv
        tn = np.minimum(a, b); tf = np.maximum(a, b)
        tin = tn.max(-1); tout = tf.min(-1)
        hit = (tin <= tout) & (tin > 1e-4) & (tin < t)
        if not hit.any():
            continue
        ax = tn.argmax(-1)
        s = np.take_along_axis(np.sign(dw), ax[..., None], -1)[..., 0]
        n = np.zeros(dw.shape); np.put_along_axis(n, ax[..., None], -s[..., None], -1)
        t = np.where(hit, tin, t); obj = np.where(hit, k, obj)
        normal = np.where(hit[..., None], n, normal)

    p = o + dw * t[..., None]
    col = np.zeros(p.shape)
    # walls / floor / ceiling
    floor = (obj == -1) & (normal[..., 1] > 0.5)
    ceil = (obj == -1) & (normal[..., 1] < -0.5)
    wall = (obj == -1) & ~floor & ~ceil
    plank = np.floor(p[..., 0] / 0.18); seg = np.floor((p[..., 2] + plank * 0.37) / 1.2)
    tint = 0.85 + 0.15 * _hash(plank, seg)
    wood = np.stack([0.62 * tint, 0.46 * tint, 0.31 * tint], -1)
    grain = 1 - 0.06 * (np.sin(p[..., 2] * 40 + plank * 3) ** 2)
    col[floor] = (wood * grain[..., None])[floor]
    rug = floor & (p[..., 0] > -1.45) & (p[..., 0] < 0.45) & (p[..., 2] > -1.05) & (p[..., 2] < 0.35)
    stripe = (np.floor((p[..., 0] + 1.45) / 0.12) % 2)
    col[rug] = np.stack([0.78 - 0.08 * stripe, 0.74 - 0.08 * stripe, 0.66 - 0.06 * stripe], -1)[rug]
    col[wall] = (0.86, 0.85, 0.82)
    # mottled paint + a few posters: real rooms are never perfectly flat-coloured
    ax_u = np.where(np.abs(normal[..., 0]) > 0.5, p[..., 2], p[..., 0])
    cu, cv = np.floor(ax_u / 0.07), np.floor(p[..., 1] / 0.07)
    mott = 0.88 + 0.14 * _hash(cu, cv) + 0.06 * _hash(np.floor(ax_u / 0.23), np.floor(p[..., 1] / 0.23))
    col[wall] = (col * mott[..., None])[wall]
    for (u0, u1, v0, v1, k) in [(-1.9, -1.3, 0.9, 1.5, 1), (0.6, 1.4, 1.1, 1.7, 2), (-0.4, 0.2, 0.6, 1.0, 3)]:
        post = wall & (ax_u > u0) & (ax_u < u1) & (p[..., 1] > v0) & (p[..., 1] < v1)
        pu, pv = np.floor((ax_u - u0) / 0.05), np.floor((p[..., 1] - v0) / 0.05)
        pc = np.stack([_hash(pu, pv + k), _hash(pu + k, pv), _hash(pu * 3 + k, pv * 7)], -1)
        col[post] = pc[post]
    col[ceil] = (0.93, 0.93, 0.92)
    # window on x = -2.25 wall, door on z = +2 wall, frame on z = -2 wall
    win = wall & (normal[..., 0] > 0.5) & (p[..., 1] > 1.0) & (p[..., 1] < 2.15) & (p[..., 2] > -1.0) & (p[..., 2] < 0.6)
    mull = win & ((np.abs(p[..., 2] + 0.2) < 0.02) | (np.abs(p[..., 1] - 1.57) < 0.02))
    col[win] = (0.78, 0.88, 0.97); col[mull] = (0.95, 0.95, 0.95)
    door = wall & (normal[..., 2] < -0.5) & (p[..., 0] > 0.8) & (p[..., 0] < 1.7) & (p[..., 1] < 2.05)
    col[door] = (0.74, 0.66, 0.56)
    art = wall & (normal[..., 2] > 0.5) & (p[..., 0] > -1.1) & (p[..., 0] < -0.1) & (p[..., 1] > 1.3) & (p[..., 1] < 1.95)
    col[art] = np.stack([0.2 + 0.5 * ((p[..., 0] + 1.1)), 0.35 + 0.2 * np.sin(p[..., 1] * 9), 0.55 + 0.0 * p[..., 0]], -1)[art]
    for k, (_, _, rgb, _) in enumerate(BOXES):
        m = obj == k
        col[m] = rgb
    # books on shelf
    books = (obj == BOXES.index(next(b for b in BOXES if b[3] == "shelf"))) & (normal[..., 0] < -0.5)
    bk = np.floor(p[..., 2] / 0.06)
    bc = np.stack([0.3 + 0.5 * _hash(bk, 1), 0.25 + 0.4 * _hash(bk, 2), 0.25 + 0.5 * _hash(bk, 3)], -1)
    shelf_rows = ((p[..., 1] > 0.68) & (p[..., 1] < 0.98)) | ((p[..., 1] > 1.28) & (p[..., 1] < 1.6)) | ((p[..., 1] > 0.08) & (p[..., 1] < 0.5))
    col[books & shelf_rows & (_hash(bk, 9) > 0.15)] = bc[books & shelf_rows & (_hash(bk, 9) > 0.15)]

    lam = np.clip((normal * LIGHT).sum(-1), 0, 1)
    sky = 0.55 + 0.25 * normal[..., 1]
    win_light = np.exp(-np.linalg.norm(p - np.array([-2.25, 1.6, -0.2]), axis=-1) / 2.2)
    shade = 0.32 + 0.38 * lam + 0.25 * sky + 0.35 * win_light
    shade = np.where(win, 1.0, shade)
    # contact darkening near floor
    shade *= np.where(wall, 1 - 0.18 * np.exp(-p[..., 1] / 0.12), 1)
    rgb = np.clip(col * shade[..., None], 0, 1)
    depth = t  # along -z because d.z == -1
    return rgb, depth, normal, p, obj


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(np.asarray, (eye, target, up))
    back = eye - target; back = back / np.linalg.norm(back)
    right = np.cross(up, back); right /= np.linalg.norm(right)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, np.cross(back, right), back, eye
    return T
