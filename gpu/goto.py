#!/usr/bin/env python3
""""Go to the red toolbox": a text prompt -> a place in the scan -> a planned path -> a video.

    python gpu/goto.py out_splat/IMG_6075/ckpt.pt out_vipe/default/vipe IMG_6075 \
        out_vipe/crisp/IMG_6075/robot out_goto/IMG_6075 "the red toolbox" "take me to the whiteboard"

1. Find it: every 3rd ViPE frame is cut into square crops at three sizes and embedded with CLIP
   (ViT-L/14 + ViT-B/16, averaged; cached in <out>/clip_index_*.npz). Each crop is scored against
   the prompt and against ~120 everyday indoor objects: a crop only counts if it looks more like
   the prompt than like anything else on that list (so "humanoid robot" stops landing on a robot
   arm). The best crops are lifted to 3D with ViPE's metric depth, and the 3D spot most of them
   agree on wins. A prompt nothing matches well is reported as not found.
2. Plan: Dijkstra on a nav2-style costmap of robot/occupancy.pgm (obstacles block, cells within
   --radius of one cost more, anywhere the phone was carried is drivable) to the floor under a
   camera the phone actually filmed the target from, 0.8-3 m away.
3. Show: the splat at a humanoid's eye height along the path; the last second eases into that real
   phone view, where the splat is sharpest (no blurry arrivals at glass or plain walls).

Outputs per prompt in <out>/<slug>/: goto.mp4, arrived.jpg, plan.png, match.jpg, result.json (target,
path and nav2 goal pose in map metres; ros2/pioneer_map's goto node sends it to a robot).
gpu/goto_server.py keeps a GotoEngine loaded and serves this live from a web page.
Needs the `splat` env (gsplat, torch) plus `pip install open_clip_torch`.
"""
import argparse
import heapq
import json
import math
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu"))
from splat_render import look_at  # noqa: E402

TEMPLATES = ["a photo of {}.", "a photo of {} in a room.", "a close-up photo of {}.", "a cropped photo of {}.", "{}"]
# what crops of a room usually are. A crop has to look more like the prompt than any of these
# (minus near-synonyms of the prompt) to count.
VOCAB = """wall, floor, ceiling, room, person, people, blurry photo, clutter, cables, window, glass wall,
door, doorway, hallway, table, desk, chair, office chair, stool, couch, sofa, armchair, bed, pillow,
blanket, cabinet, wooden cabinets, shelf, bookshelf, drawer, cupboard, wardrobe, locker, box, cardboard box,
bin, trash can, bag, backpack, suitcase, computer monitor, laptop, keyboard, computer mouse, computer,
television, speaker, headphones, phone, tablet, printer, 3D printer, camera, tripod, lamp, ring light,
ceiling light, light switch, power outlet, extension cord, fan, air conditioner, radiator, heater,
whiteboard, blackboard, poster, painting, picture frame, clock, sign, paper, notebook, book, pen, tape,
toolbox, tool chest, drill, screwdriver, wrench, hammer, tools, workbench, robot, robot arm, humanoid robot,
robot dog, drone, car, bicycle, wheel, motor, circuit board, battery, plant, potted plant, flower, vase,
bottle, water bottle, cup, mug, plate, bowl, kitchen, stove, oven, microwave, fridge, sink, faucet,
counter, kitchen counter, dishwasher, toilet, bathtub, shower, mirror, towel, curtain, blinds, rug, carpet,
stairs, railing, column, pipe, vent, fire extinguisher, coat, jacket, shoes, helmet, ball, toy""".replace("\n", " ")
VOCAB = [v.strip() for v in VOCAB.split(",") if v.strip()]
COMMAND = re.compile(r"^\s*(please\s+)?(can you\s+|could you\s+)?((go|walk|drive|move|head|navigate|take me|bring me|"
                     r"get me|show me|find|look at|go over|go back)\s+(to\s+|over\s+to\s+|towards?\s+|at\s+)?)?"
                     r"|^\s*(where('s| is)\s+)", re.I)


def clean_prompt(p):
    """'Please take me to the red toolbox!' -> 'the red toolbox'."""
    p = COMMAND.sub("", p.strip()).strip(" .!?")
    return p or "it"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "target"


# ----------------------------------------------------------------------------- scan
class Scan:
    """The splat checkpoint (its cameras are ViPE's, one per ViPE frame) + the robot map frame."""

    def __init__(self, ckpt, robot):
        self.ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        T = json.loads((robot / "map_transform.json").read_text())
        self.R, self.z0, self.s, self.xy0 = np.array(T["R"]), T["z0"], T["scale"], np.array(T["xy0"])
        self.ids = list(self.ck.get("frame_ids", range(len(self.ck["viewmats"]))))
        self.row = {f: k for k, f in enumerate(self.ids)}  # ViPE frame id -> checkpoint row
        self.c2w = np.linalg.inv(self.ck["viewmats"].numpy())  # splat (ViPE) frame
        self.K = self.ck["Ks"].numpy()
        self.W, self.H = self.ck["W"], self.ck["H"]
        y = (robot / "occupancy.yaml").read_text()
        self.res = float(re.search(r"resolution:\s*([\d.]+)", y).group(1))
        self.origin = np.array([float(v) for v in re.search(r"origin:\s*\[([^\]]+)\]", y).group(1).split(",")[:2]])
        self.occ = cv2.imread(str(robot / "occupancy.pgm"), cv2.IMREAD_UNCHANGED)  # 0 occ, 254 free, 205 unknown
        self.cams_map = self.map_c2w(self.c2w)  # every phone camera, in the map frame

    def map_from_splat(self, X):
        Y = np.asarray(X, float) @ self.R.T
        Y[..., 2] -= self.z0
        Y = Y * self.s
        Y[..., :2] -= self.xy0
        return Y

    def map_c2w(self, c2w):
        T = np.array(c2w, float).copy()
        T[..., :3, :3] = self.R @ T[..., :3, :3]
        T[..., :3, 3] = self.map_from_splat(T[..., :3, 3])
        return T

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
    three sizes relative to the frame height, overlapping by half."""
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
            cs = crops_of(f[..., ::-1])
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


class GotoEngine:
    """Everything loaded once (splat, map, CLIP models + crop index, depth), then any number of
    prompts: find() ~0.3 s, plan() ~0.5 s, render() ~10-30 s on an RTX 5070 Ti."""

    def __init__(self, ckpt, vipe, name, robot, cache_dir, models=("ViT-L-14:openai", "ViT-B-16:openai"),
                 every=3, dev="cuda", log=print):
        import open_clip
        t0 = time.time()
        self.vipe, self.name, self.dev, self.log = Path(vipe), name, dev, log
        self.scan = Scan(Path(ckpt), Path(robot))
        self.models = []
        for spec in models:
            mname, pretrained = spec.split(":")
            model, _, pre = open_clip.create_model_and_transforms(mname, pretrained=pretrained, device=dev)
            model.eval()
            feat, frame, box = build_index(self.vipe, name, every, model, pre, dev,
                                           Path(cache_dir) / f"clip_index_{mname}_{every}.npz")
            tok = open_clip.get_tokenizer(mname)
            vocab = np.stack([self._embed(model, tok, v) for v in VOCAB])
            self.models.append(dict(feat=torch.from_numpy(feat.astype(np.float32)).to(dev), model=model, tok=tok,
                                    vocab=torch.from_numpy(vocab).to(dev)))
        self.frame, self.box = frame, box
        self.vid_wh = (int((box[:, 0] + box[:, 2]).max()), int((box[:, 1] + box[:, 3]).max()))
        self.zf = zipfile.ZipFile(self.vipe / "depth" / f"{name}.zip")
        self.zmap = {int(Path(n).stem): n for n in self.zf.namelist()}
        self.walked = self.scan.cams_map[:, :2, 3]
        self.start = self.walked[0]
        self._gpu = None
        log(f"[goto] ready: {len(self.box):,} crops from {len(set(frame.tolist()))} frames, "
            f"{len(self.models)} CLIP models, {time.time() - t0:.0f}s")

    def _embed(self, model, tok, phrase):
        with torch.no_grad():
            t = model.encode_text(tok([tp.format(phrase) for tp in TEMPLATES]).to(self.dev)).float()
        t = torch.nn.functional.normalize(t, dim=-1).mean(0)
        return torch.nn.functional.normalize(t, dim=0).cpu().numpy()

    # -- 1. find
    def _lift(self, frame, box):
        """Crop centre -> 3D point in the map frame, from ViPE depth (median of the centre patch)."""
        from vipe_to_pipeline import read_exr_depth
        if frame not in self.zmap or frame not in self.scan.row:
            return None
        D = read_exr_depth(self.zf.read(self.zmap[frame])).astype(np.float32)
        x, y, w, h = box
        vw, vh = self.vid_wh
        sx, sy = D.shape[1] / vw, D.shape[0] / vh
        patch = D[int((y + h * 0.4) * sy):int((y + h * 0.6) * sy) + 1, int((x + w * 0.4) * sx):int((x + w * 0.6) * sx) + 1]
        patch = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 8)]
        if len(patch) < 5:
            return None
        z = float(np.median(patch))
        k = self.scan.row[frame]
        K = self.scan.K[k]
        u, v = (x + w / 2) / vw * self.scan.W, (y + h / 2) / vh * self.scan.H
        Xc = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        return self.scan.map_from_splat(self.scan.c2w[k, :3, :3] @ Xc + self.scan.c2w[k, :3, 3])

    def _refine(self, i, sim, slack=0.02):
        """Coarse to fine: a large tile's centre often isn't on the object (a robot in front of a wall:
        the centre depth is the wall). Use the best-matching smaller tile inside it instead, if it
        matches almost as well."""
        if not hasattr(self, "_by_frame"):
            self._by_frame = {}
            for j, f in enumerate(self.frame.tolist()):
                self._by_frame.setdefault(f, []).append(j)
        x, y, s = self.box[i][:3]
        inside = [j for j in self._by_frame[int(self.frame[i])] if self.box[j][2] < s
                  and self.box[j][0] >= x and self.box[j][1] >= y and self.box[j][0] + self.box[j][2] <= x + s
                  and self.box[j][1] + self.box[j][3] <= y + s]
        if not inside:
            return i
        j = max(inside, key=lambda j: sim[j])
        return self._refine(j, sim, slack) if sim[j] >= sim[i] - slack else i

    def find(self, prompt, top=40, radius=0.6, synonym_cos=0.9):
        """-> dict: target (map m), confidence (share of the prompt vs the vocabulary on the best crop;
        low = not in this scan), score, frame, box, votes, views (frames of the agreeing crops)."""
        q = clean_prompt(prompt)
        sims, probs = [], []
        for m in self.models:
            t = torch.from_numpy(self._embed(m["model"], m["tok"], q)).to(self.dev)
            keep = (m["vocab"] @ t) < synonym_cos  # drop near-synonyms of the prompt from the vocabulary
            with torch.no_grad():
                sim = m["feat"] @ t
                logits = 100 * torch.cat([sim[:, None], m["feat"] @ m["vocab"][keep].T], 1)
                probs.append(torch.softmax(logits, 1)[:, 0].cpu().numpy())
            sims.append(sim.cpu().numpy())
        sim, prob = np.mean(sims, 0), np.mean(probs, 0)
        # raw match, plus a bonus for looking more like the prompt than like any other object on the
        # list (0.2: moves "the humanoid robot" off a small arm onto the big robot; the 9 other office
        # prompts land in the same place as without it)
        rank = sim + 0.2 * prob
        order = np.argsort(-rank)[:top]
        pts, w, idx = [], [], []
        for i in order:
            i = self._refine(int(i), sim)  # big tile -> the best small tile inside it (where the thing is)
            P = self._lift(int(self.frame[i]), self.box[i])
            if P is not None:
                pts.append(P)
                w.append(math.exp((rank[i] - rank[order[0]]) * 50))
                idx.append(int(i))
        if not pts:
            return dict(prompt=prompt, query=q, found=False, confidence=0.0, reason="no match with usable depth")
        pts, w = np.array(pts), np.array(w)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        b = int(np.argmax((np.exp(-(d / radius) ** 2) * w[None]).sum(1)))
        near = d[b] < radius
        target = np.average(pts[near], axis=0, weights=w[near])
        agree = [idx[k] for k in np.nonzero(near)[0]]
        best = max(agree, key=lambda i: rank[i])
        conf = float(np.average(prob[agree], weights=w[near]))
        return dict(prompt=prompt, query=q, target=target, confidence=conf, score=float(sim[best]),
                    frame=int(self.frame[best]), box=[int(v) for v in self.box[best]], votes=int(near.sum()),
                    views=[int(self.frame[i]) for i in sorted(agree, key=lambda i: -rank[i])])

    # -- 2. plan
    def _costmap(self, radius):
        occ = self.scan.occ
        obst = occ < 50
        free = occ >= 250
        wc = self.scan.cell(self.walked)
        wc = wc[(wc[:, 0] >= 0) & (wc[:, 0] < occ.shape[0]) & (wc[:, 1] >= 0) & (wc[:, 1] < occ.shape[1])]
        walked = np.zeros(occ.shape, np.uint8)
        walked[wc[:, 0], wc[:, 1]] = 1
        k = max(1, int(round(0.2 / self.scan.res)))
        free |= cv2.dilate(walked, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
        dist_obs = cv2.distanceTransform((~obst).astype(np.uint8), cv2.DIST_L2, 5) * self.scan.res
        ok = free & (dist_obs > 0.08)
        extra = np.where(dist_obs < radius, 8.0, 0.0) + np.clip((radius + 0.25 - dist_obs) / 0.25, 0, 1) * 2.0
        return ok, extra, dist_obs

    def _dijkstra(self, ok, extra, s):
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
        return dist, prev

    def plan(self, found, start=None, radius=0.25, near=0.8, far=3.0):
        """Path to the floor under the best phone camera that saw the target from near..far metres.
        -> dict: path (N,2 map m), arrive_frame, goal {x, y, yaw}, stop_distance, length."""
        sc = self.scan
        target = np.asarray(found["target"])
        ok, extra, dist_obs = self._costmap(radius)
        s0 = np.asarray(start if start is not None else self.start, float)
        s = tuple(sc.cell(s0))
        if not (0 <= s[0] < ok.shape[0] and 0 <= s[1] < ok.shape[1]) or not ok[s]:
            rr, cc = np.nonzero(ok)
            j = np.argmin((rr - s[0]) ** 2 + (cc - s[1]) ** 2)
            s = (int(rr[j]), int(cc[j]))
        dist, prev = self._dijkstra(ok, extra, s)
        # candidate arrival views: the cameras of the agreeing matches first, then every camera that
        # looked at the target (it is in front of the camera and inside its image)
        cand = list(dict.fromkeys(found.get("views", [])))
        for k, f in enumerate(sc.ids):
            if f not in cand:
                T = sc.cams_map[k]
                Xc = T[:3, :3].T @ (target - T[:3, 3])
                if Xc[2] > 0.3 and abs(Xc[0] / Xc[2]) < 0.5 and abs(Xc[1] / Xc[2]) < 0.35:
                    cand.append(f)
        best, best_cost = None, np.inf
        for rank_i, f in enumerate(cand):
            T = sc.cams_map[sc.row[f]]
            d = float(np.linalg.norm(T[:2, 3] - target[:2]))
            c = tuple(sc.cell(T[:2, 3]))
            if not (0 <= c[0] < ok.shape[0] and 0 <= c[1] < ok.shape[1]) or not np.isfinite(dist[c]):
                continue
            Xc = T[:3, :3].T @ (target - T[:3, 3])  # where the target sits in that view: prefer it centred
            off = abs(Xc[0] / Xc[2]) + abs(Xc[1] / Xc[2]) if Xc[2] > 0.1 else 3.0
            cost = (0 if near <= d <= far else 5 + abs(d - np.clip(d, near, far)) * 5) + rank_i * 0.02 \
                + dist[c] * sc.res * 0.01 + off * 3
            if cost < best_cost:
                best, best_cost = f, cost
        if best is None:  # no filmed viewpoint reachable: nearest reachable cell at ~1.3 m
            rr, cc = np.nonzero(np.isfinite(dist))
            g = sc.xy(np.stack([rr, cc], 1))
            dt = np.linalg.norm(g - target[:2], axis=1)
            j = int(np.argmin(np.abs(dt - 1.3) + dist[rr, cc] * sc.res * 0.02))
            goal_cell, arrive = (int(rr[j]), int(cc[j])), None
            end_xy = sc.xy(np.array(goal_cell))
        else:
            arrive = sc.cams_map[sc.row[best]]
            goal_cell, end_xy = tuple(sc.cell(arrive[:2, 3])), arrive[:2, 3]
        path = [goal_cell]
        while path[-1] != s:
            p = tuple(prev[path[-1]])
            if p[0] < 0:
                break
            path.append(p)
        path = sc.xy(np.array(path[::-1]))
        path = np.vstack([s0[None], path[1:-1], end_xy[None]]) if len(path) > 2 else np.vstack([s0, end_xy])
        path = smooth_path(path)
        yaw = math.atan2(target[1] - path[-1, 1], target[0] - path[-1, 0])
        return dict(path=path, arrive=arrive, arrive_frame=best, length=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                    stop_distance=float(np.linalg.norm(path[-1] - target[:2])),
                    goal=dict(x=float(path[-1, 0]), y=float(path[-1, 1]), yaw=yaw), ok=ok & (dist_obs >= radius), start=s0)

    # -- 3. show
    def _gaussians(self):
        if self._gpu is None:
            ck = self.scan.ck
            P = {k: v.to(self.dev) for k, v in ck["params"].items()}
            tr = ck["train"]
            tr = tr if hasattr(tr, "__len__") else list(range(int(tr)))
            self._gpu = dict(args=(P["means"], P["quats"], torch.exp(P["scales"]), torch.sigmoid(P["opacities"]),
                                   torch.cat([P["sh0"], P["shN"]], 1)),
                             eA=ck["expo_A"][tr].mean(0).to(self.dev), eb=ck["expo_b"][tr].mean(0).to(self.dev))
        return self._gpu

    def views(self, plan, target, eye=1.35, speed=0.7, fps=30, hold=1.5, settle=1.2):
        """Camera-to-world (map frame) per video frame: walk the path at eye height looking ahead,
        turn toward the target, and over the last `settle` s ease into the real phone view."""
        from scipy.ndimage import gaussian_filter1d
        from scipy.spatial.transform import Rotation, Slerp
        path = plan["path"]
        step = speed / fps
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        t = np.arange(0, s[-1] + 1e-9, step) if s[-1] > 0 else np.zeros(1)
        C = np.stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1]), np.full(len(t), eye)], 1)
        n = len(C)
        tgt = np.array([target[0], target[1], np.clip(target[2], 0.3, eye + 0.3)])
        blend_from = min(s[-1] * 0.6, max(0.0, s[-1] - 1.5))
        dirs = []
        for i in range(n):
            d_path = C[min(i + int(0.6 / step), n - 1)] - C[i]
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
        T = [look_at(c, c + d) for c, d in zip(C, dirs)]
        A = plan.get("arrive")
        if A is not None:  # settle into the phone's own view: same place, height and orientation
            m = max(2, int(settle * fps))
            T0 = T[-1]
            rots = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], A[:3, :3]]))
            for i in range(1, m + 1):
                a = i / m
                a = a * a * (3 - 2 * a)
                Ti = np.eye(4)
                Ti[:3, :3] = rots([a]).as_matrix()[0]
                Ti[:3, 3] = (1 - a) * T0[:3, 3] + a * A[:3, 3]
                T.append(Ti)
        T += [T[-1]] * int(hold * fps)
        return T

    def render(self, plan, target, out, width=1280, height=720, hfov=75, fps=30, preset="medium", progress=None, **kw):
        from gsplat import rasterization
        g = self._gaussians()
        f = 0.5 * width / math.tan(math.radians(hfov) / 2)
        K = torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=torch.float32, device=self.dev)[None]
        Ts = self.views(plan, target, fps=fps, **kw)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                              "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "20", "-preset", preset,
                              "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        last = None
        with torch.no_grad():
            for i, Tm in enumerate(Ts):
                V = torch.tensor(self.scan.viewmat(Tm), dtype=torch.float32, device=self.dev)[None]
                o, _, _ = rasterization(*g["args"], V, K, width, height, sh_degree=self.scan.ck["sh_degree"], near_plane=0.05)
                rgb = (torch.einsum("ij,hwj->hwi", g["eA"], o[0, ..., :3]) + g["eb"]).clamp(0, 1)
                last = (rgb * 255).byte().cpu().numpy()
                p.stdin.write(last.tobytes())
                if progress and i % 10 == 0:
                    progress(i / len(Ts))
        p.stdin.close()
        p.wait()
        cv2.imwrite(str(out.with_name("arrived.jpg")), last[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        return len(Ts) / fps

    def match_jpg(self, found, out):
        cap = cv2.VideoCapture(str(self.vipe / "rgb" / f"{self.name}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, found["frame"])
        _, img = cap.read()
        x, y, w, h = found["box"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 220), 4)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 88])

    def run(self, prompt, out_dir, render=True, min_conf=0.14, start=None, **kw):
        """find + plan (+ render) one prompt; writes the per-prompt folder, returns result.json's dict."""
        t0 = time.time()
        found = self.find(prompt)
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        res = dict(prompt=prompt, query=found["query"], found=bool(found.get("target") is not None
                                                                    and found["confidence"] >= min_conf),
                   confidence=round(found["confidence"], 3))
        if found.get("target") is None:
            res["reason"] = found.get("reason")
            (od / "result.json").write_text(json.dumps(res, indent=1))
            return res
        pl = self.plan(found, start=start)
        self.match_jpg(found, od / "match.jpg")
        plan_png(self.scan, pl["ok"], pl["path"], pl["start"], found["target"], od / "plan.png",
                 f'"{found["query"]}"  {pl["length"]:.1f} m')
        res.update(target_m=[round(float(v), 3) for v in found["target"]], clip_score=round(found["score"], 4),
                   matches_agreeing=found["votes"], matched_frame=found["frame"], arrive_frame=pl["arrive_frame"],
                   start_m=[round(float(v), 3) for v in pl["start"]], path_length_m=round(pl["length"], 2),
                   stop_distance_m=round(pl["stop_distance"], 2),
                   goal=dict(frame_id="map", x=round(pl["goal"]["x"], 3), y=round(pl["goal"]["y"], 3),
                             yaw=round(pl["goal"]["yaw"], 4)),
                   path_m=[[round(float(x), 3), round(float(y), 3)] for x, y in pl["path"][::5]] + [
                       [round(float(pl["path"][-1, 0]), 3), round(float(pl["path"][-1, 1]), 3)]],
                   find_plan_s=round(time.time() - t0, 2))
        if render and res["found"]:
            t1 = time.time()
            res["video_s"] = round(self.render(pl, found["target"], od / "goto.mp4", **kw), 1)
            res["render_s"] = round(time.time() - t1, 1)
        (od / "result.json").write_text(json.dumps(res, indent=1))
        return res


def smooth_path(P, step=0.02, sigma=0.25):
    from scipy.ndimage import gaussian_filter1d
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return P[:1]
    t = np.arange(0, s[-1], step)
    Q = np.stack([np.interp(t, s, P[:, d]) for d in range(2)], 1)
    Q = gaussian_filter1d(Q, sigma / step, axis=0, mode="nearest")
    Q = np.vstack([P[:1], Q[1:], P[-1:]])
    return Q


def plan_png(scan, ok, path, start, target, out, label):
    occ = scan.occ
    img = np.full(occ.shape + (3,), 245, np.uint8)
    img[occ == 205] = (225, 225, 225)
    img[(occ >= 250) & ~ok] = (215, 232, 228)
    img[occ < 50] = (40, 48, 46)
    rc = scan.cell(path)
    up = 4
    img = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
    cv2.polylines(img, [(rc[:, ::-1] * up + up // 2).astype(np.int32)], False, (114, 124, 10), 3, cv2.LINE_AA)
    s = scan.cell(start)[::-1] * up + up // 2
    t = scan.cell(target)[::-1] * up + up // 2
    cv2.circle(img, tuple(int(v) for v in s), 9, (114, 124, 10), -1, cv2.LINE_AA)
    cv2.drawMarker(img, tuple(int(v) for v in t), (40, 40, 220), cv2.MARKER_STAR, 30, 3, cv2.LINE_AA)
    ys, xs = np.nonzero(cv2.resize(occ, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST) != 205)
    pad = 40
    img = np.ascontiguousarray(img[max(0, ys.min() - pad):min(img.shape[0], ys.max() + pad),
                                   max(0, xs.min() - pad):min(img.shape[1], xs.max() + pad)])
    cv2.putText(img, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), img)


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
    ap.add_argument("--min-conf", type=float, default=0.14,
                    help="below this the prompt is reported as not found (share of the prompt vs the vocabulary)")
    ap.add_argument("--eye", type=float, default=1.35, help="camera height (m): a humanoid's head")
    ap.add_argument("--speed", type=float, default=0.7, help="m/s")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--models", nargs="+", default=["ViT-L-14:openai", "ViT-B-16:openai"],
                    help="open_clip model:pretrained, averaged (one is fine too)")
    ap.add_argument("--no-render", action="store_true", help="only find + plan")
    a = ap.parse_args()
    eng = GotoEngine(a.ckpt, a.vipe, a.name, a.robot, a.out, a.models, a.every)
    summary = []
    for prompt in a.prompts:
        res = eng.run(prompt, a.out / slug(clean_prompt(prompt)), render=not a.no_render, min_conf=a.min_conf,
                      start=a.start, eye=a.eye, speed=a.speed, width=a.width, height=a.height, preset="slow")
        summary.append(res)
        if not res["found"]:
            print(f"[goto] '{prompt}': not in this scan (confidence {res['confidence']:.2f})", flush=True)
        else:
            print(f"[goto] '{prompt}': target {res['target_m']} (confidence {res['confidence']:.2f}, "
                  f"{res['matches_agreeing']} matches agree), path {res['path_length_m']:.1f} m, "
                  f"stop {res['stop_distance_m']:.2f} m away, arrive at phone frame {res['arrive_frame']}", flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
