#!/usr/bin/env python3
"""Hybrid map: the CPU pipeline's camera poses + ViPE's dense depth.

The benchmark showed each half is good at one thing:
  * CPU SfM poses are accurate (0.09 % on the Mip-NeRF room) but stereo keeps only 15-36 %
    of each frame's pixels as depth;
  * ViPE keeps ~99 % of the pixels with metric depth, but its poses drift on long walks
    (4.4 m camera-height spread on IMG_6077) and fail on photo sets.
So: take every CPU keyframe, find the same moment in ViPE's output, and fuse ViPE's depth
from the CPU camera. Per frame, ViPE's depth is scaled to the SfM points that frame sees
(robust median), which also corrects ViPE's frame-to-frame scale wobble; ViPE's overall
metric scale then sets metres for the whole map, so no phone-height guess is needed.

    python gpu/hybrid.py out/IMG_6075 out_vipe/default/vipe IMG_6075 out_hybrid/IMG_6075

CPU_RUN is run.py's output (sfm/, frames/, dense/cameras.json, frame_times.json if recorded).
VIPE_DIR is ViPE's artifact folder (pose/, intrinsics/, depth/<name>.zip, rgb/<name>.mp4).
The output has the same layout as run.py's, so bench/benchmark.py scores it the same way.
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recon"))
sys.path.insert(0, str(ROOT / "gpu"))


def thumb(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def read_vipe_frames(vipe_dir, name):
    cap = cv2.VideoCapture(str(vipe_dir / "rgb" / f"{name}.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames


def pair_frames(cpu_names, cpu_imgs, times, vipe_thumbs, vipe_fps, window=2):
    """CPU keyframe -> ViPE frame index. With keyframe times: the nearest ViPE frames by
    time, refined by image similarity. Without: best match by similarity, kept in order."""
    pairs, last = {}, -1
    for n, img in zip(cpu_names, cpu_imgs):
        t = thumb(img)
        if times and n in times and vipe_fps:
            c = int(round(times[n] * vipe_fps))
            cand = range(max(0, c - window), min(len(vipe_thumbs), c + window + 1))
        else:  # search ahead of the previous match (the walk only moves forward in time)
            cand = range(max(0, last - 3), len(vipe_thumbs))
        if not len(cand):
            continue
        scores = [float(np.mean(np.abs(t - vipe_thumbs[k]))) for k in cand]
        k = cand[int(np.argmin(scores))]
        pairs[n] = (k, min(scores))
        last = k
    # drop pairs whose best match is much worse than typical (wrong moment)
    s = np.array([v[1] for v in pairs.values()])
    lim = np.median(s) * 2.5 if len(s) else 0
    return {n: k for n, (k, sc) in pairs.items() if sc <= lim}


def frame_scale(im, rec, D, min_pts=30):
    """Scale taking ViPE metres to SfM units for one frame: robust median of
    z_sfm / z_vipe over the SfM points this frame sees. Also returns their log spread."""
    T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    R, t = T.rotation.matrix(), T.translation
    cam = rec.cameras[im.camera_id]
    sx, sy = D.shape[1] / cam.width, D.shape[0] / cam.height
    zs, zv = [], []
    for p in im.points2D:
        if not p.has_point3D():
            continue
        X = rec.points3D[p.point3D_id].xyz
        z = (R @ X + t)[2]
        u, v = int(p.xy[0] * sx), int(p.xy[1] * sy)
        if z <= 0 or not (0 <= u < D.shape[1] and 0 <= v < D.shape[0]):
            continue
        d = D[v, u]
        if np.isfinite(d) and d > 0.05:
            zs.append(z)
            zv.append(d)
    if len(zs) < min_pts:
        return None, None, len(zs)
    r = np.log(np.array(zs) / np.array(zv))
    med = np.median(r)
    return float(np.exp(med)), float(np.median(np.abs(r - med))), len(zs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cpu_run", type=Path)
    ap.add_argument("vipe_dir", type=Path)
    ap.add_argument("name", help="ViPE artifact name (the video's file stem)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--vipe-fps", type=float, help="frame rate ViPE saw (default: vipe_fps.txt "
                    "next to the ViPE output, else match by image only)")
    ap.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel (m)")
    ap.add_argument("--max-depth", type=float, default=6.0, help="ignore ViPE depth beyond this (m)")
    ap.add_argument("--max-spread", type=float, default=0.15,
                    help="skip frames whose SfM-vs-ViPE depth ratios spread more than this (log MAD)")
    ap.add_argument("--filter", choices=["none", "edge", "consistent"], default="edge",
                    help="edge: drop depth edges (flying pixels); consistent: also multi-view "
                         "agreement (gpu/depth_filter.py, needs torch)")
    ap.add_argument("--tsdf", choices=["auto", "gpu", "cpu"], default="auto")
    a = ap.parse_args()

    import pycolmap
    from depth_filter import edge_mask
    from mvs import write_ply
    from robot_map import build
    from tsdf import fuse_tsdf, write_mesh_ply
    from vipe_to_pipeline import read_exr_depth, reliable_mask

    log = lambda m: print(m, flush=True)  # noqa: E731
    run = a.cpu_run
    rec = pycolmap.Reconstruction(str(run / "sfm" / "sparse_best"))
    ims = {im.name: im for im in rec.images.values()}
    names = sorted(ims)
    cpu_imgs = [cv2.imread(str(run / "frames" / n)) for n in names]
    tf = run / "frame_times.json"
    times = json.loads(tf.read_text())["times"] if tf.exists() else None
    fps = a.vipe_fps
    if fps is None:
        for cand in (a.vipe_dir.parent / a.name / "vipe_fps.txt", a.vipe_dir / "vipe_fps.txt"):
            if cand.exists():
                fps = float(cand.read_text().strip())
    log(f"[hybrid] {len(names)} CPU keyframes; times {'yes' if times else 'no'}; ViPE fps {fps}")

    vframes = read_vipe_frames(a.vipe_dir, a.name)
    vthumbs = [thumb(f) for f in vframes]
    pairs = pair_frames(names, cpu_imgs, times, vthumbs, fps)
    log(f"[hybrid] paired {len(pairs)}/{len(names)} keyframes with ViPE frames")

    zf = zipfile.ZipFile(a.vipe_dir / "depth" / f"{a.name}.zip")
    zmap = {int(Path(n).stem): n for n in zf.namelist()}
    views, scales, spreads, skipped = [], [], [], 0
    for n in names:
        if n not in pairs or pairs[n] not in zmap:
            skipped += 1
            continue
        im, img = ims[n], cpu_imgs[names.index(n)]
        D = read_exr_depth(zf.read(zmap[pairs[n]])).astype(np.float32)
        h, w = img.shape[:2]
        if D.shape != (h, w):  # same video frame, maybe another size: same pixels after resize
            D = cv2.resize(D, (w, h), interpolation=cv2.INTER_NEAREST)
        s, spread, npts = frame_scale(im, rec, D)
        if s is None or spread > a.max_spread:
            skipped += 1
            continue
        cam = rec.cameras[im.camera_id]
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        K = np.array([[f * w / cam.width, 0, cx * w / cam.width],
                      [0, f * h / cam.height, cy * h / cam.height], [0, 0, 1.0]])
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R, t = T.rotation.matrix(), np.asarray(T.translation)
        views.append(SimpleNamespace(name=n, K=K, R=R, t=t, C=-R.T @ t, img=img, h=h, w=w,
                                     depth_vipe=D, s=s))
        scales.append(s)
        spreads.append(spread)
    if len(views) < 10:
        raise SystemExit(f"[hybrid] only {len(views)} usable frames: are these the same video?")
    scales = np.array(scales)
    m_per_unit = 1.0 / np.median(scales)  # ViPE's metric scale for the SfM model
    log(f"[hybrid] {len(views)} frames fused, {skipped} skipped; ViPE->SfM scale median "
        f"{np.median(scales):.4f} (frame-to-frame spread {np.std(np.log(scales))*100:.1f} %), "
        f"depth agreement with SfM points: median {np.median(spreads)*100:.1f} % (log MAD); "
        f"1 SfM unit = {m_per_unit:.3f} m")

    # everything in metres: SfM poses scaled by ViPE's metric scale, ViPE depth re-scaled per
    # frame to agree with the SfM geometry
    masks = []
    for v in views:
        v.C = v.C * m_per_unit
        v.t = -v.R @ v.C
        v.depth = v.depth_vipe * v.s * m_per_unit
        m = reliable_mask(v.depth) & (v.depth > 0.1) & (v.depth < a.max_depth)
        if a.filter in ("edge", "consistent"):
            m &= edge_mask(v.depth)
        masks.append(m)
        del v.depth_vipe
    if a.filter == "consistent":
        from depth_filter import consistency_masks
        import torch
        masks = consistency_masks(views, masks, device="cuda" if torch.cuda.is_available() else "cpu", log=log)
    kept = float(np.median([m.mean() for m in masks])) * 100
    log(f"[hybrid] depth kept: {kept:.0f}% of pixels (median per frame)")

    use_gpu = a.tsdf == "gpu"
    if a.tsdf == "auto":
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
    if use_gpu:
        from tsdf_gpu import fuse_tsdf_gpu as fuse
    else:
        fuse = fuse_tsdf
    dense = a.out / "dense"
    dense.mkdir(parents=True, exist_ok=True)
    V, F, C, _ = fuse(views, masks, a.voxel, log=log)
    write_mesh_ply(dense / "mesh.ply", V, F, C)
    write_ply(dense / "dense.ply", V, C)
    cams = [{"name": v.name, "R": v.R.tolist(), "t": v.t.tolist(), "C": v.C.tolist(),
             "fx": float(v.K[0, 0]), "fy": float(v.K[1, 1]), "w": v.w, "h": v.h} for v in views]
    (dense / "cameras.json").write_text(json.dumps(cams))
    n_frames = len([p for p in (run / "frames").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    (dense / "bench_info.json").write_text(json.dumps({
        "frames_total": n_frames, "depth_pixels_kept_median_pct": round(kept, 1),
        "hybrid": {"frames_fused": len(views), "metres_per_sfm_unit": m_per_unit,
                   "vipe_scale_wobble_pct": round(float(np.std(np.log(scales))) * 100, 2),
                   "depth_vs_sfm_points_logmad_pct": round(float(np.median(spreads)) * 100, 2)}}))
    meta = build(dense / "dense.ply", dense / "cameras.json", a.out / "robot", metric_scale=True)
    vdir = a.out / "viewer"
    if vdir.exists():
        shutil.rmtree(vdir)
    shutil.copytree(ROOT / "viewer", vdir)
    for src, dst in [("map_metric.ply", "dense.ply"), ("cameras_metric.json", "cameras.json"),
                     ("mesh_metric.ply", "mesh.ply"), ("inferred_floor.ply", "inferred_floor.ply")]:
        if (a.out / "robot" / src).exists():
            shutil.copy(a.out / "robot" / src, vdir / dst)
    (vdir / "view.json").write_text('{"up": [0, 0, 1]}')
    log(f"[hybrid] done: {len(F):,} triangles, camera height {meta['camera_height_median_m']:.2f} m "
        f"(metric from ViPE, poses from SfM) -> {a.out}")


if __name__ == "__main__":
    main()
