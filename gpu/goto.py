#!/usr/bin/env python3
""""Go to the red toolbox": a text prompt -> a place in the scan -> a planned path -> a video.

    python gpu/goto.py out_splat/IMG_6075/ckpt.pt out_vipe/default/vipe IMG_6075 \
        out_vipe/crisp/IMG_6075/robot out_goto/IMG_6075 "the red toolbox" "the whiteboard"

1. Find it: every 3rd ViPE frame is cut into crops (whole frame + 3x3 overlapping tiles) and
   embedded with CLIP (open_clip, cached in <out>/clip_index.npz). The prompt is matched against
   all crops; the best crops are lifted to 3D with ViPE's metric depth at their centres, and the
   3D cluster with the most match score wins. So the target is a point in the room, not a frame.
2. Plan: A* on the robot's 5 cm occupancy grid (robot/occupancy.pgm, obstacles inflated by
   --radius) from the start to the reachable free cell nearest --standoff metres from the target.
3. Show: the splat rendered from a camera at --eye height walking the path at --speed, turning
   to face the target on the way in, then holding on it; plus a floor plan with the path.

Outputs per prompt in <out>/<slug>/: goto.mp4, plan.png, match.jpg (best matching crop),
result.json (target in map metres, path length, score). Needs the `splat` env (gsplat, torch)
plus `pip install open_clip_torch`.
"""
import argparse
import heapq
import json
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu"))
from splat_render import look_at  # noqa: E402

TEMPLATES = ["a photo of {}.", "a photo of {} in a room.", "a close-up photo of {}.", "a cropped photo of {}.", "{}"]
# what most crops of a room look like: a crop has to beat these to count as the thing asked for
DISTRACTORS = ["a wall", "the floor", "the ceiling", "a room", "a person", "a blurry photo", "clutter",
               "a table", "a chair", "a door", "a computer monitor", "cables"]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "target"


# ----------------------------------------------------------------------------- scan
class Scan:
    """The splat checkpoint (its cameras are ViPE's, one per ViPE frame) + the robot map frame."""

    def __init__(self, ckpt, robot):
        self.ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        T = json.loads((robot / "map_transform.json").read_text())
        self.R, self.z0, self.s, self.xy0 = np.array(T["R"]), T["z0"], T["scale"], np.array(T["xy0"])
        self.c2w = np.linalg.inv(self.ck["viewmats"].numpy())  # splat (ViPE) frame
        self.K = self.ck["Ks"].numpy()
        self.W, self.H = self.ck["W"], self.ck["H"]
        y = (robot / "occupancy.yaml").read_text()
        self.res = float(re.search(r"resolution:\s*([\d.]+)", y).group(1))
        self.origin = np.array([float(v) for v in re.search(r"origin:\s*\[([^\]]+)\]", y).group(1).split(",")[:2]])
        self.occ = cv2.imread(str(robot / "occupancy.pgm"), cv2.IMREAD_UNCHANGED)  # 0 occ, 254 free, 205 unknown

    def map_from_splat(self, X):
        Y = np.asarray(X, float) @ self.R.T
        Y[..., 2] -= self.z0
        Y = Y * self.s
        Y[..., :2] -= self.xy0
        return Y

    def viewmat(self, Tm):
        """camera-to-world in the map frame -> world-to-camera in the splat frame."""
        C = Tm[:3, 3].copy()
        C[:2] += self.xy0
        C = C / self.s
        C[2] += self.z0
        Ts = np.eye(4)
        Ts[:3, :3], Ts[:3, 3] = self.R.T @ Tm[:3, :3], self.R.T @ C
        return np.linalg.inv(Ts)

    # grid <-> metres (pgm row 0 is the top = largest y, as in ROS map_server)
    def cell(self, xy):
        c = np.floor((np.asarray(xy)[..., :2] - self.origin) / self.res).astype(int)
        return np.stack([self.occ.shape[0] - 1 - c[..., 1], c[..., 0]], -1)  # (row, col)

    def xy(self, rc):
        rc = np.asarray(rc)
        return np.stack([(rc[..., 1] + 0.5) * self.res + self.origin[0],
                         (self.occ.shape[0] - 1 - rc[..., 0] + 0.5) * self.res + self.origin[1]], -1)


# ----------------------------------------------------------------------------- 1. find
def crops_of(img, scales=(1.0, 0.55, 0.35)):
    """Square tiles (CLIP centre-crops to a square, so wide crops would lose their sides) at
    three sizes relative to the frame height, overlapping by half: big things, mid-size and
    small objects each get a tile that's mostly them."""
    h, w = img.shape[:2]
    out = []
    for sc in scales:
        t = int(h * sc)
        nx, ny = max(1, math.ceil((w - t) / (t / 2)) + 1), max(1, math.ceil((h - t) / (t / 2)) + 1)
        for y in np.linspace(0, h - t, ny).astype(int):
            for x in np.linspace(0, w - t, nx).astype(int):
                out.append(((int(x), int(y), t, t), img[y:y + t, x:x + t]))
    return out


def build_index(vipe, name, every, model, pre, dev, cache):
    if cache.exists():
        d = np.load(cache)
        return d["feat"], d["frame"], d["box"]
    cap = cv2.VideoCapture(str(vipe / "rgb" / f"{name}.mp4"))
    from PIL import Image
    feats, frames, boxes, k = [], [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % every == 0:
            rgb = f[..., ::-1]
            cs = crops_of(rgb)
            batch = torch.stack([pre(Image.fromarray(np.ascontiguousarray(c))) for _, c in cs]).to(dev)
            with torch.no_grad(), torch.autocast("cuda"):
                e = torch.cat([model.encode_image(batch[i:i + 64]).float() for i in range(0, len(batch), 64)])
            feats.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
            frames += [k] * len(cs)
            boxes += [b for b, _ in cs]
        k += 1
    feat, frame, box = np.concatenate(feats), np.array(frames), np.array(boxes)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, feat=feat.astype(np.float16), frame=frame, box=box)
    print(f"[goto] indexed {len(set(frames))} frames, {len(frame)} crops", flush=True)
    return feat, frame, box


def lift(scan, depth_zip, zmap, frame, box, vid_wh):
    """Crop centre -> 3D point in the map frame, from ViPE depth (median of the centre patch)."""
    from vipe_to_pipeline import read_exr_depth
    if frame not in zmap:
        return None
    D = read_exr_depth(depth_zip.read(zmap[frame])).astype(np.float32)
    x, y, w, h = box
    sx, sy = D.shape[1] / vid_wh[0], D.shape[0] / vid_wh[1]
    u0, v0 = int((x + w * 0.4) * sx), int((y + h * 0.4) * sy)
    u1, v1 = int((x + w * 0.6) * sx) + 1, int((y + h * 0.6) * sy) + 1
    patch = D[v0:v1, u0:u1]
    patch = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 8)]
    if len(patch) < 5:
        return None
    z = float(np.median(patch))
    u, v = (x + w / 2) / vid_wh[0] * scan.W, (y + h / 2) / vid_wh[1] * scan.H
    K = scan.K[frame]
    Xc = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    Xw = scan.c2w[frame, :3, :3] @ Xc + scan.c2w[frame, :3, 3]
    return scan.map_from_splat(Xw)


def embed_text(model, tok, dev, phrase):
    with torch.no_grad():
        t = model.encode_text(tok([tp.format(phrase) for tp in TEMPLATES]).to(dev)).float()
    t = torch.nn.functional.normalize(t, dim=-1).mean(0)
    return torch.nn.functional.normalize(t, dim=0).cpu().numpy()


def find(scan, prompt, models, frame, box, dev, vipe, name, top=40, radius=0.6):
    """-> target (map metres), crop score, best frame, its crop box, how many top crops agree,
    and the share of the prompt over the distractors on the best crop (low = not in the room)."""
    sims, probs = [], []
    for feat, model, tok in models:  # several CLIP models: average them (each one's misses get outvoted)
        feat = feat.astype(np.float32)
        t = embed_text(model, tok, dev, prompt)
        negs = np.stack([embed_text(model, tok, dev, d) for d in DISTRACTORS if d not in prompt.lower()])
        sim = feat @ t
        logits = 100 * np.concatenate([sim[:, None], feat @ negs.T], 1)
        logits -= logits.max(1, keepdims=True)
        probs.append(np.exp(logits[:, 0]) / np.exp(logits).sum(1))  # share of the prompt vs distractors
        sims.append(sim)
    sim, prob = np.mean(sims, 0), np.mean(probs, 0)
    rank = sim + 0.02 * prob                               # raw match first, distractor check breaks ties
    order = np.argsort(-rank)[:top]
    zf = zipfile.ZipFile(vipe / "depth" / f"{name}.zip")
    zmap = {int(Path(n).stem): n for n in zf.namelist()}
    vw, vh = box[:, 0].max() + box[box[:, 0].argmax(), 2], box[:, 1].max() + box[box[:, 1].argmax(), 3]
    pts, w, idx = [], [], []
    for i in order:
        P = lift(scan, zf, zmap, int(frame[i]), box[i], (vw, vh))
        if P is not None:
            pts.append(P)
            w.append(math.exp((rank[i] - rank[order[0]]) * 50))
            idx.append(i)
    if not pts:
        raise SystemExit(f"[goto] '{prompt}': no match with usable depth")
    pts, w = np.array(pts), np.array(w)
    # the 3D spot most of the best matches agree on (not just the single best crop)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    support = (np.exp(-(d / radius) ** 2) * w[None]).sum(1)
    b = int(np.argmax(support))
    near = d[b] < radius
    target = np.average(pts[near], axis=0, weights=w[near])
    best = idx[int(np.argmax(np.where(near, w, -1)))]
    # where the phone stood when it saw it: the splat is sharpest from near those cameras
    seen_from = scan.map_from_splat(np.array([scan.c2w[int(frame[idx[k]]), :3, 3] for k in np.nonzero(near)[0]]))
    return target, float(sim[best]), int(frame[best]), box[best], int(near.sum()), float(prob[best]), seen_from


# ----------------------------------------------------------------------------- 2. plan
def plan(scan, start_xy, target, standoff, radius, walked_xy=None, seen_from=None):
    """Dijkstra on a nav2-style costmap: obstacles are walls, cells within `radius` of one are
    allowed but cost more (so paths keep clear when they can), and anywhere the phone was
    carried is drivable (it was free when filmed, even if the map left it unknown)."""
    occ = scan.occ
    obst = occ < 50
    free = occ >= 250
    if walked_xy is not None:
        wc = scan.cell(walked_xy)
        wc = wc[(wc[:, 0] >= 0) & (wc[:, 0] < occ.shape[0]) & (wc[:, 1] >= 0) & (wc[:, 1] < occ.shape[1])]
        walked = np.zeros(occ.shape, np.uint8)
        walked[wc[:, 0], wc[:, 1]] = 1
        k = max(1, int(round(0.2 / scan.res)))
        walked = cv2.dilate(walked, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
        free |= walked
    # distance (m) to the nearest obstacle -> extra cost inside the robot radius and a soft margin
    dist_obs = cv2.distanceTransform((~obst).astype(np.uint8), cv2.DIST_L2, 5) * scan.res
    ok = free & (dist_obs > 0.08)
    extra = np.where(dist_obs < radius, 8.0, 0.0) + np.clip((radius + 0.25 - dist_obs) / 0.25, 0, 1) * 2.0
    s = tuple(scan.cell(start_xy))
    if not (0 <= s[0] < occ.shape[0] and 0 <= s[1] < occ.shape[1]) or not ok[s]:
        rr, cc = np.nonzero(ok)
        j = np.argmin((rr - s[0]) ** 2 + (cc - s[1]) ** 2)
        s = (int(rr[j]), int(cc[j]))
    H, W = ok.shape
    dist = np.full(ok.shape, np.inf)
    prev = -np.ones(ok.shape + (2,), int)
    dist[s] = 0
    pq = [(0.0, s)]
    steps = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    while pq:
        dd, (r, c) = heapq.heappop(pq)
        if dd > dist[r, c]:
            continue
        for dr, dc, cost in steps:
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < H and 0 <= c2 < W and ok[r2, c2]:
                nd = dd + cost * (1 + extra[r2, c2])
                if nd < dist[r2, c2]:
                    dist[r2, c2] = nd
                    prev[r2, c2] = (r, c)
                    heapq.heappush(pq, (nd, (r2, c2)))
    reach = np.isfinite(dist)
    rr, cc = np.nonzero(reach)
    gxy = scan.xy(np.stack([rr, cc], 1))
    dt = np.linalg.norm(gxy - target[:2], axis=1)
    # stop about `standoff` from it (0.8-2.5 m is fine), in open space, close to where the phone
    # was when it saw the target (the view there is sharp), cheap to reach
    score = np.clip(np.abs(dt - standoff) - 0.3, 0, None) * 4 + (dt < 0.6) * 10 + (dt > 2.5) * 10 \
        + dist[rr, cc] * scan.res * 0.02 + extra[rr, cc] * 0.3
    if seen_from is not None and len(seen_from):
        dv = np.min(np.linalg.norm(gxy[:, None] - seen_from[None, :, :2], axis=-1), axis=1)
        score = score + dv * 3
    j = int(np.argmin(score))
    g = (int(rr[j]), int(cc[j]))
    path = [g]
    while path[-1] != s:
        p = tuple(prev[path[-1]])
        if p[0] < 0:
            break
        path.append(p)
    path = path[::-1]
    return scan.xy(np.array(path)), float(dt[j]), ok & (dist_obs >= radius)


def smooth_path(P, step=0.02, sigma=0.25):
    from scipy.ndimage import gaussian_filter1d
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return P[:1]
    t = np.arange(0, s[-1], step)
    Q = np.stack([np.interp(t, s, P[:, d]) for d in range(2)], 1)
    sg = sigma / step
    Q = gaussian_filter1d(Q, sg, axis=0, mode="nearest")
    Q[0], Q[-1] = P[0], P[-1]
    return Q


# ----------------------------------------------------------------------------- 3. show
def plan_png(scan, ok, path, start, target, out, label):
    occ = scan.occ
    img = np.full(occ.shape + (3,), 245, np.uint8)
    img[occ == 205] = (225, 225, 225)
    img[(occ >= 250) & ~ok] = (215, 232, 228)
    img[occ < 50] = (40, 48, 46)
    rc = scan.cell(path)
    up = 4
    img = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
    pts = (rc[:, ::-1] * up + up // 2).astype(np.int32)
    cv2.polylines(img, [pts], False, (114, 124, 10), 3, cv2.LINE_AA)
    s = (scan.cell(start)[::-1] * up + up // 2)
    t = (scan.cell(target)[::-1] * up + up // 2)
    cv2.circle(img, tuple(int(v) for v in s), 9, (114, 124, 10), -1, cv2.LINE_AA)
    cv2.drawMarker(img, tuple(int(v) for v in t), (40, 40, 220), cv2.MARKER_STAR, 30, 3, cv2.LINE_AA)
    ys, xs = np.nonzero((cv2.resize(occ, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST) != 205))
    pad = 40
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    img = np.ascontiguousarray(img[y0:y1, x0:x1])
    cv2.putText(img, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), img)


def render_trip(scan, path, target, out, eye, speed, fps, W, H, hfov, hold):
    from gsplat import rasterization
    dev = torch.device("cuda")
    P = {k: v.to(dev) for k, v in scan.ck["params"].items()}
    tr = scan.ck["train"]
    eA, eb = scan.ck["expo_A"][tr].mean(0).to(dev), scan.ck["expo_b"][tr].mean(0).to(dev)
    f = 0.5 * W / math.tan(math.radians(hfov) / 2)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32, device=dev)[None]
    # camera positions at constant speed, eye height
    step = speed / fps
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1] + 1e-9, step) if s[-1] > 0 else np.zeros(1)
    C = np.stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1]), np.full(len(t), eye)], 1)
    # gaze: along the path, turning toward the target over the last 40 % (and at least 1.5 m)
    n = len(C)
    look_ahead = np.array([C[min(i + int(0.6 / step), n - 1)] - C[i] for i in range(n)])
    tgt = np.array([target[0], target[1], np.clip(target[2], 0.3, eye + 0.3)])
    views = []
    blend_from = min(s[-1] * 0.6, max(0.0, s[-1] - 1.5))
    from scipy.ndimage import gaussian_filter1d
    dirs = []
    for i in range(n):
        d_path = look_ahead[i].copy()
        d_path[2] = 0
        d_tgt = tgt - C[i]
        if np.linalg.norm(d_path) < 1e-3:
            d_path = d_tgt
        a = np.clip((t[i] - blend_from) / max(s[-1] - blend_from, 1e-6), 0, 1)
        a = a * a * (3 - 2 * a)
        d = (1 - a) * d_path / np.linalg.norm(d_path) + a * d_tgt / np.linalg.norm(d_tgt)
        dirs.append(d / np.linalg.norm(d))
    dirs = gaussian_filter1d(np.array(dirs), sigma=fps * 0.25, axis=0, mode="nearest")
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for c, d in zip(C, dirs):
        views.append(scan.viewmat(look_at(c, c + d)))
    views += [views[-1]] * int(hold * fps)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    colors = torch.cat([P["sh0"], P["shN"]], 1)
    args = (P["means"], P["quats"], torch.exp(P["scales"]), torch.sigmoid(P["opacities"]), colors)
    last = None
    with torch.no_grad():
        for V in views:
            o, _, _ = rasterization(*args, torch.tensor(V, dtype=torch.float32, device=dev)[None], K, W, H,
                                    sh_degree=scan.ck["sh_degree"], near_plane=0.05)
            rgb = (torch.einsum("ij,hwj->hwi", eA, o[0, ..., :3]) + eb).clamp(0, 1)
            last = (rgb * 255).byte().cpu().numpy()
            p.stdin.write(last.tobytes())
    p.stdin.close()
    p.wait()
    cv2.imwrite(str(out.with_name("arrived.jpg")), last[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    return len(views) / fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path, help="gpu/splat_train.py checkpoint")
    ap.add_argument("vipe", type=Path, help="ViPE artifact folder (rgb/, depth/)")
    ap.add_argument("name", help="ViPE artifact name (the video's file stem)")
    ap.add_argument("robot", type=Path, help="robot/ folder of the same scan (occupancy + map_transform)")
    ap.add_argument("out", type=Path)
    ap.add_argument("prompts", nargs="+", help='what to go to, e.g. "the red toolbox"')
    ap.add_argument("--start", type=float, nargs=2, help="start x y in map metres (default: where the walk began)")
    ap.add_argument("--every", type=int, default=3, help="index every k-th ViPE frame")
    ap.add_argument("--standoff", type=float, default=1.3, help="stop about this far from the target (m)")
    ap.add_argument("--radius", type=float, default=0.25, help="robot radius for obstacle inflation (m)")
    ap.add_argument("--eye", type=float, default=1.35, help="camera height (m): a humanoid's head")
    ap.add_argument("--speed", type=float, default=0.7, help="m/s")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds looking at the target at the end")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=75)
    ap.add_argument("--models", nargs="+", default=["ViT-L-14:openai", "ViT-B-16:openai"],
                    help="open_clip model:pretrained, averaged (one is fine too)")
    ap.add_argument("--min-prob", type=float, default=0.5,
                    help="below this the prompt is reported as not found (share vs distractor prompts)")
    ap.add_argument("--no-render", action="store_true", help="only find + plan")
    a = ap.parse_args()

    import open_clip
    dev = "cuda"
    scan = Scan(a.ckpt, a.robot)
    models = []
    for spec in a.models:
        mname, pretrained = spec.split(":")
        model, _, pre = open_clip.create_model_and_transforms(mname, pretrained=pretrained, device=dev)
        model.eval()
        feat, frame, box = build_index(a.vipe, a.name, a.every, model, pre, dev,
                                       a.out / f"clip_index_{mname}_{a.every}.npz")
        models.append((feat, model, open_clip.get_tokenizer(mname)))
    walked = scan.map_from_splat(scan.c2w[:, :3, 3])[:, :2]
    start = np.array(a.start) if a.start else walked[0]
    vid = cv2.VideoCapture(str(a.vipe / "rgb" / f"{a.name}.mp4"))
    summary = []
    for prompt in a.prompts:
        od = a.out / slug(prompt)
        od.mkdir(parents=True, exist_ok=True)
        target, score, fr, bx, votes, prob, seen_from = find(scan, prompt, models, frame, box, dev, a.vipe, a.name)
        path, stand, ok = plan(scan, start, target, a.standoff, a.radius, walked, seen_from)
        path = smooth_path(path)
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        vid.set(cv2.CAP_PROP_POS_FRAMES, fr)
        _, img = vid.read()
        x, y, w, h = (int(v) for v in bx)
        m = img.copy()
        cv2.rectangle(m, (x, y), (x + w, y + h), (40, 40, 220), 4)
        cv2.imwrite(str(od / "match.jpg"), m, [cv2.IMWRITE_JPEG_QUALITY, 90])
        plan_png(scan, ok, path, start, target, od / "plan.png", f'"{prompt}"  {length:.1f} m')
        res = {"prompt": prompt, "found": bool(prob >= a.min_prob), "confidence": round(prob, 3), "target_m": [round(float(v), 3) for v in target], "clip_score": round(score, 4),
               "matches_agreeing": votes, "matched_frame": fr, "start_m": [round(float(v), 3) for v in start],
               "stop_distance_m": round(stand, 2), "path_length_m": round(length, 2)}
        if not res["found"]:
            print(f"[goto] '{prompt}': probably not in this scan (confidence {prob:.2f} < {a.min_prob}); "
                  f"best guess kept in result.json", flush=True)
        if not a.no_render and res["found"]:
            res["video_s"] = round(render_trip(scan, path, target, od / "goto.mp4", a.eye, a.speed, a.fps,
                                               a.width, a.height, a.hfov, a.hold), 1)
        (od / "result.json").write_text(json.dumps(res, indent=1))
        summary.append(res)
        print(f"[goto] '{prompt}': target {res['target_m']} (score {score:.3f}, {votes} matches agree), "
              f"confidence {prob:.2f}, path {length:.1f} m, stop {stand:.2f} m away -> {od}", flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
