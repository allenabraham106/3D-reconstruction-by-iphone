#!/usr/bin/env python3
"""Merge several scans of the same place into one map (e.g. different walks or lighting).

Each scan was processed on its own by run.py (frames/, sfm/, dense/depth_cache/). This
  1. solves all scans' cameras together: features for every frame, sequential matching
     within each walk, plus *cross-scan* pairs between frames that see the same area
     (given as frame ranges, e.g. the kitchen both walks passed through),
  2. reuses each scan's cached trusted depth -- no stereo re-run: per scan, a similarity
     transform between its own SfM cameras and the joint ones gives the depth scale,
  3. fuses all depth into one TSDF mesh -> robot map -> viewer.

    python recon/merge_scans.py out/merged \\
        --scan out/IMG_6070 --cross 10-66 \\
        --scan out/IMG_6076 --cross 0-20,240-290 --keep 0-20,240-290

Alignment is robust: frames where a scan's own solve and the joint solve disagree (one
of them drifted) are dropped by RANSAC, then the overlap is tightened with rigid ICP.
Limit: it can only fill gaps that another scan actually saw. It cannot undo drift inside
one scan; leave such stretches out with --keep (ARKit / ViPE poses don't drift like that).
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_ranges(spec):
    out = []
    for part in spec.split(","):
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out


def frame_idx(name):
    return int(name.split("frame_")[1][:5])


def sim3_umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = np.trace(np.diag(S) @ E) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def sim3_ransac(src, dst, thresh, iters=3000, seed=0):
    """Similarity fit that ignores frames where the two solves disagree (one of them
    drifted there). Returns (s, R, t, inlier mask)."""
    rng = np.random.default_rng(seed)
    best = np.ones(len(src), bool)
    best_n = 0
    for _ in range(iters if len(src) > 4 else 0):
        m = rng.choice(len(src), 4, replace=False)
        s, R, t = sim3_umeyama(src[m], dst[m])
        inl = np.linalg.norm(s * src @ R.T + t - dst, axis=1) < thresh
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    return (*sim3_umeyama(src[best], dst[best]), best)


def icp_rigid(P, tree, ref, iters=10, keep=60):
    """Trimmed point-to-point ICP of cloud P onto a KD-tree of ref, rotation + shift only
    (a free scale shrinks under trimming). Returns (R, t) to apply after the Sim3."""
    R, t = np.eye(3), np.zeros(3)
    for _ in range(iters):
        X = P @ R.T + t
        d, j = tree.query(X)
        m = d < np.percentile(d, keep)
        mu_x, mu_y = X[m].mean(0), ref[j[m]].mean(0)
        U, _, Vt = np.linalg.svd((ref[j[m]] - mu_y).T @ (X[m] - mu_x))
        E = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        dR = U @ E @ Vt
        R, t = dR @ R, dR @ (t - mu_x) + mu_y
    return R, t, float(np.median(tree.query(P @ R.T + t)[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--scan", type=Path, action="append", required=True,
                    help="a run.py output folder (frames/, sfm/, dense*/depth_cache/)")
    ap.add_argument("--cross", action="append", required=True,
                    help="frame ranges of this scan that overlap the others, e.g. 10-66 or 0-120,210-270")
    ap.add_argument("--cross-step", type=int, default=3, help="use every k-th frame of the ranges")
    ap.add_argument("--cache", action="append", default=None,
                    help="depth cache dir per scan (default <scan>/dense/depth_cache)")
    ap.add_argument("--keep", action="append", default=None,
                    help="per scan: frame ranges to fuse (default all), e.g. to leave out a "
                         "stretch where that scan's own solve drifted; 'all' keeps everything")
    ap.add_argument("--phone-height", type=float, default=1.4)
    ap.add_argument("--reuse-joint", action="store_true", help="skip the joint SfM if already done")
    a = ap.parse_args()
    assert len(a.scan) == len(a.cross), "one --cross per --scan"
    t0 = time.time()
    out = a.out
    img_dir = out / "frames"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    tags = [chr(ord("A") + i) for i in range(len(a.scan))]
    caches = a.cache or [str(s / "dense" / "depth_cache") for s in a.scan]
    cross = {}
    for tag, scan, spec in zip(tags, a.scan, a.cross):
        for f in sorted((scan / "frames").glob("*.jpg")):
            (img_dir / f"{tag}_{f.name}").symlink_to(f.resolve())
        rng = parse_ranges(spec)
        names = sorted(f"{tag}_{f.name}" for f in (scan / "frames").glob("*.jpg")
                       if any(lo <= frame_idx(f.name) < hi for lo, hi in rng))
        cross[tag] = names[::a.cross_step]

    # 1. joint SfM ------------------------------------------------------------------
    work = out / "sfm"
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if a.reuse_joint and (work / "sparse_best").exists():
        rec = pycolmap.Reconstruction(str(work / "sparse_best"))
        print(f"[merge] reusing joint model ({rec.num_reg_images()} frames)", flush=True)
    else:
        rec = joint_sfm(db, img_dir, work, tags, cross, t0)
    per = {t: sum(im.name.startswith(t + "_") for im in rec.images.values()) for t in tags}
    print(f"[merge] joint model: {rec.num_reg_images()} frames {per}", flush=True)
    if sum(v > 20 for v in per.values()) < 2:
        raise SystemExit("[merge] scans did not link up (cross-scan matches too weak); "
                         "widen the --cross ranges to frames that really see the same place")
    views = align_and_load(a, tags, caches, rec, img_dir)
    fuse_and_map(a, out, views, t0)


def joint_sfm(db, img_dir, work, tags, cross, t0):
    if db.exists():
        db.unlink()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = 1600
    ext.sift.max_num_features = 8192
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext, device=pycolmap.Device.cpu)
    print(f"[merge] features {time.time()-t0:.0f}s", flush=True)
    seq = pycolmap.SequentialPairingOptions()
    seq.overlap = 15
    seq.quadratic_overlap = True
    pycolmap.match_sequential(db, pairing_options=seq, device=pycolmap.Device.cpu)
    pairs = [(x, y) for i, ti in enumerate(tags) for tj in tags[i + 1:]
             for x in cross[ti] for y in cross[tj]]
    lst = work / "cross_pairs.txt"
    lst.write_text("".join(f"{x} {y}\n" for x, y in pairs))
    po = pycolmap.ImportedPairingOptions()
    po.match_list_path = lst
    pycolmap.match_image_pairs(db, pairing_options=po, device=pycolmap.Device.cpu)
    print(f"[merge] matching {time.time()-t0:.0f}s ({len(pairs):,} cross-scan pairs)", flush=True)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_global_frames_ratio = opts.ba_global_points_ratio = 1.4
    recs = pycolmap.incremental_mapping(db, img_dir, work / "sparse", options=opts)
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    (work / "sparse_best").mkdir(exist_ok=True)
    rec.write(work / "sparse_best")
    return rec


def align_and_load(a, tags, caches, rec, img_dir):
    """Put every frame of every scan into the joint frame.

    The joint solve usually only links the overlapping parts. Each scan's own solve
    covers its whole walk, so align that to the joint model through the frames both
    contain (similarity transform) and carry *all* its cameras over, with the cached
    depth scaled by the same factor."""
    import mvs
    from types import SimpleNamespace
    views = []
    for tag, scan, cache, spec in zip(tags, a.scan, caches, a.cross):
        joint = {im.name[2:]: im for im in rec.images.values() if im.name.startswith(tag + "_")}
        own_models = [scan / "sfm" / "sparse_best"] + sorted((scan / "sfm").glob("attached_*"))
        own_ims = {}
        for d in own_models:  # attached sub-models are already in the scan's main frame
            r = pycolmap.Reconstruction(str(d))
            for im in r.images.values():
                own_ims.setdefault(im.name, (im, r))
        common = sorted(set(own_ims) & set(joint))
        if len(common) < 5:
            print(f"[merge] scan {tag}: only {len(common)} frames in the joint model, skipped")
            continue
        src = np.array([own_ims[n][0].projection_center() for n in common])
        dst = np.array([joint[n].projection_center() for n in common])
        ext = np.linalg.norm(np.ptp(dst, 0))
        s, R, t, inl = sim3_ransac(src, dst, 0.01 * ext)
        err = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        print(f"[merge] scan {tag}: aligned via {inl.sum()}/{len(common)} shared frames "
              f"(rest disagree between the solves), scale x{s:.3f}, "
              f"median camera error {np.median(err[inl])/ext*100:.2f}% of overlap size", flush=True)
        scan_views = []
        n_used = 0
        keep = (a.keep or ["all"] * len(a.scan))[tags.index(tag)]
        for name, (im, r) in sorted(own_ims.items()):
            f = Path(cache) / (name + ".npy")
            if keep != "all" and not any(lo <= frame_idx(name) < hi for lo, hi in parse_ranges(keep)):
                continue
            if not f.exists():
                continue
            v = mvs.View(im, r.cameras[im.camera_id], scan / "frames", 1.0, r, 0)
            Rc = v.R @ R.T                         # cam_from_joint rotation
            C = s * R @ v.C + t                    # camera centre in the joint frame
            scan_views.append(SimpleNamespace(name=f"{tag}_{name}", K=v.K, R=Rc, t=-Rc @ C, C=C,
                                              img=v.img, h=v.h, w=v.w,
                                              depth=np.load(f).astype(np.float32) * s))
            n_used += 1
        if views:  # tighten onto the scans already placed, using only the overlap frames
            from scipy.spatial import cKDTree
            ref = np.concatenate([mvs.backproject(v, v.depth, v.depth > 0)[0][::40] for v in views])
            ov = [v for v in scan_views
                  if any(lo <= frame_idx(v.name) < hi for lo, hi in parse_ranges(spec))]
            P = np.concatenate([mvs.backproject(v, v.depth, v.depth > 0)[0][::40] for v in ov])
            tree = cKDTree(ref)
            before = float(np.median(tree.query(P)[0]))
            dR, dt, after = icp_rigid(P, tree, ref)
            for v in scan_views:
                v.C = dR @ v.C + dt
                v.R = v.R @ dR.T
                v.t = -v.R @ v.C
            print(f"[merge] scan {tag}: overlap ICP, median gap {before:.3f} -> {after:.3f}", flush=True)
        views += scan_views
        print(f"[merge] scan {tag}: {n_used} frames brought into the joint frame", flush=True)
    return views


def fuse_and_map(a, out, views, t0):
    import mvs
    # 3. fuse -> mesh -> robot map -> viewer --------------------------------------------
    from robot_map import build
    from tsdf import fuse_tsdf, write_mesh_ply
    dense = out / "dense"
    dense.mkdir(exist_ok=True)
    masks = [v.depth > 0 for v in views]
    P = np.concatenate([mvs.backproject(v, v.depth, m)[0][::50] for v, m in zip(views, masks)])
    extent = np.linalg.norm(np.percentile(P, 98, 0) - np.percentile(P, 2, 0))
    V, F, C, _ = fuse_tsdf(views, masks, extent * 0.0022, log=lambda m: print(m, flush=True))
    write_mesh_ply(dense / "mesh.ply", V, F, C)
    mvs.write_ply(dense / "dense.ply", V, C)
    cams = [{"name": v.name, "R": v.R.tolist(), "t": v.t.tolist(), "C": v.C.tolist(),
             "fx": float(v.K[0, 0]), "fy": float(v.K[1, 1]), "w": v.w, "h": v.h}
            for v in sorted(views, key=lambda v: v.name)]
    (dense / "cameras.json").write_text(json.dumps(cams))
    build(dense / "dense.ply", dense / "cameras.json", out / "robot", a.phone_height)
    vdir = out / "viewer"
    if vdir.exists():
        shutil.rmtree(vdir)
    shutil.copytree(Path(__file__).resolve().parents[1] / "viewer", vdir)
    for src, dst in [("map_metric.ply", "dense.ply"), ("cameras_metric.json", "cameras.json"),
                     ("mesh_metric.ply", "mesh.ply"), ("inferred_floor.ply", "inferred_floor.ply")]:
        if (out / "robot" / src).exists():
            shutil.copy(out / "robot" / src, vdir / dst)
    (vdir / "view.json").write_text('{"up": [0, 0, 1]}')
    print(f"[merge] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
