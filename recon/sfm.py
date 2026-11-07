"""Stage 2: camera poses + intrinsics via COLMAP incremental SfM (CPU).

Stands in for ViPE's bundle-adjusted SLAM: sequential matching suits video,
and one shared camera matches a single phone lens.
"""
import argparse, time
from pathlib import Path
import numpy as np
import pycolmap


def _map(db, image_dir, out, use_priors=False, ba_ratio=1.4, seed=None):
    out.mkdir(parents=True, exist_ok=True)
    opts = pycolmap.IncrementalPipelineOptions()
    if seed is not None:  # the mapper is randomised: another seed gives another solve
        opts.random_seed = seed
    # Global bundle adjustment runs whenever the model grows by this factor.
    opts.ba_global_frames_ratio = ba_ratio
    opts.ba_global_points_ratio = ba_ratio
    if use_priors:  # ARKit positions as soft constraints in bundle adjustment
        opts.use_prior_position = True
        opts.use_robust_loss_on_prior_position = True
    recs = pycolmap.incremental_mapping(db, image_dir, out, options=opts)
    if not recs:
        return None
    return recs[max(recs, key=lambda k: recs[k].num_reg_images())]


def write_position_priors(db_path, priors, sigma=0.05):
    """Attach known camera positions (e.g. ARKit, metres) to the database images."""
    db = pycolmap.Database.open(str(db_path))
    n = 0
    cov = np.eye(3) * sigma ** 2
    for im in db.read_all_images():
        if im.name in priors:
            pp = pycolmap.PosePrior(position=np.asarray(priors[im.name], float),
                                    position_covariance=cov,
                                    coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
                                    corr_data_id=im.data_id)
            db.write_pose_prior(pp)
            n += 1
    db.close()
    return n


def bridge_pairs(work: Path, image_names, every=3):
    """Pairs that can link SfM pieces which share no frames (a clean split).

    Frames outside the largest piece are matched against every `every`-th frame of
    the largest piece (and of each other). ~8x fewer pairs than exhaustive matching on a
    277-frame video, and enough when the walk comes back past earlier spots."""
    pieces = []
    for d in sorted(work.glob("sparse/[0-9]*")):
        r = pycolmap.Reconstruction(str(d))
        pieces.append(sorted(im.name for im in r.images.values()))
    pieces.sort(key=len, reverse=True)
    main = set(pieces[0]) if pieces else set()
    others = sorted(set(image_names) - main)
    anchors = sorted(main)[::every]
    pairs = {(a, b) for a in others for b in anchors}
    pairs |= {(a, b) for i, a in enumerate(others) for b in others[i + 1:][::every]}
    return sorted(pairs)


def attach_submodels(main, work: Path, min_shared=5, max_center_err_frac=0.02):
    """Level-1 gap bridging: align leftover sub-models into the main model's frame.

    Incremental SfM often splits a hard video into pieces (a dark corner, a turn on the
    spot). Pieces that share frames with the main model can be aligned exactly through
    those frames' camera centres. Returns the dirs of aligned sub-models that add frames.
    """
    main_names = {im.name for im in main.images.values()}
    main_C = {im.name: im.projection_center() for im in main.images.values()}
    extent = np.linalg.norm(np.ptp(np.array(list(main_C.values())), 0))
    attached, covered = [], set(main_names)
    cands = []
    for d in sorted(work.glob("sparse*/[0-9]*")):
        r = pycolmap.Reconstruction(str(d))
        names = {im.name for im in r.images.values()}
        new, shared = names - covered, names & main_names
        if len(new) and len(shared) >= min_shared:
            cands.append((len(new), d, r, shared))
    for _, d, r, shared in sorted(cands, key=lambda c: -c[0]):
        new = {im.name for im in r.images.values()} - covered
        if not new:
            continue
        src = np.array([next(im for im in r.images.values() if im.name == n).projection_center()
                        for n in sorted(shared)])
        dst = np.array([main_C[n] for n in sorted(shared)])
        ransac = pycolmap.RANSACOptions()
        ransac.max_error = max_center_err_frac * extent
        sim = pycolmap.estimate_sim3d_robust(src, dst, ransac)
        if isinstance(sim, dict):  # some pycolmap versions return {"tgt_from_src": ..., ...}
            sim = sim.get("tgt_from_src")
        if sim is None:
            continue
        r.transform(sim)
        err = np.linalg.norm(np.array([next(im for im in r.images.values() if im.name == n)
                                       .projection_center() for n in sorted(shared)]) - dst, axis=1)
        rel = np.median(err) / extent
        if rel > max_center_err_frac:
            print(f"[sfm] sub-model {d.parent.name}/{d.name}: alignment too loose ({rel:.1%}), skipped")
            continue
        out = work / f"attached_{len(attached)}"
        out.mkdir(exist_ok=True)
        r.write(out)
        attached.append(out)
        covered |= new
        print(f"[sfm] attached {d.parent.name}/{d.name}: +{len(new)} frames via {len(shared)} shared "
              f"(median centre error {rel:.2%} of scene)")
    return attached


def _frame_no(name):
    digits = "".join(ch if ch.isdigit() else " " for ch in Path(name).stem).split()
    return int(digits[-1]) if digits else None


def find_pose_jumps(rec, factor=12.0):
    """Video frames move a little each step. A camera that 'teleports' between
    neighbouring frames means the mapper glued a piece on through bad matches (typically
    frames facing a window or a blank wall). Returns [(frame_before, frame_after)]."""
    ims = sorted((_frame_no(im.name), im.projection_center(), im.num_points3D)
                 for im in rec.images.values() if _frame_no(im.name) is not None)
    if len(ims) < 10:
        return []
    idx = np.array([i for i, _, _ in ims])
    C = np.array([c for _, c, _ in ims])
    step = np.linalg.norm(np.diff(C, axis=0), axis=1) / np.maximum(np.diff(idx), 1)
    med = np.median(step)
    return [(int(idx[k]), int(idx[k + 1])) for k in np.where(step > factor * med)[0]]


def max_step_ratio(rec, between=None):
    """Largest frame-to-frame camera step / median step (or the step of one jump)."""
    ims = sorted((_frame_no(im.name), im.projection_center())
                 for im in rec.images.values() if _frame_no(im.name) is not None)
    idx = np.array([i for i, _ in ims])
    C = np.array([c for _, c in ims])
    step = np.linalg.norm(np.diff(C, axis=0), axis=1) / np.maximum(np.diff(idx), 1)
    r = step / np.median(step)
    if between is not None:
        k = np.where(idx[:-1] == between[0])[0]
        return float(r[k[0]]) if len(k) else 0.0
    return float(r.max())


def weak_frames_near(rec, jumps, radius=8, rel=0.35):
    """Frames around each jump that saw few 3D points (the weak link)."""
    pts = {_frame_no(im.name): im.num_points3D for im in rec.images.values()}
    med = np.median(list(pts.values()))
    cut = set()
    for a, b in jumps:
        near = range(a - radius, b + radius + 1)
        weak = {i for i in near if pts.get(i, 0) < rel * med}  # unregistered counts as weak
        cut |= weak or set(range(a - 2, b + 3))
    return cut


def cut_frame_links(db_path, frames):
    """Drop every match touching the given frames, so the mapper can't use them as glue."""
    import sqlite3
    M = 2147483647
    con = sqlite3.connect(str(db_path))
    ids = [i for i, n in con.execute("select image_id, name from images") if _frame_no(n) in frames]
    if ids:
        q = ",".join(map(str, ids))
        for tbl in ("two_view_geometries", "matches"):
            con.execute(f"delete from {tbl} where (pair_id / {M}) in ({q}) or (pair_id % {M}) in ({q})")
        con.commit()
    con.close()
    return len(ids)


def fix_pose_jumps(rec, db, image_dir, work, use_priors=False, ba_ratio=1.4, n_images=None):
    """Re-map without the weak frames around camera jumps (see find_pose_jumps)."""
    n_images = n_images or rec.num_reg_images()
    # Cut the worst jump first and look again: removing one weak link often fixes a
    # second jump too (the mapper had been forced into a bad placement by the first).
    # If a jump survives its cut, the bad link reaches further: cut a wider window.
    tried = []
    for attempt in range(4):
        jumps = find_pose_jumps(rec)
        if not jumps:
            break
        worst = max(jumps, key=lambda j: max_step_ratio(rec, j))
        if any(abs(worst[0] - t[0]) <= 10 for t in tried):
            cut = set(range(worst[0] - 6, worst[1] + 7))
        else:
            cut = weak_frames_near(rec, [worst])
        tried.append(worst)
        print(f"[sfm] camera jump between frames {worst} (of {len(jumps)}): cutting the weak "
              f"links ({len(cut)} frames: {min(cut)}-{max(cut)}) and re-mapping")
        cut_frame_links(db, cut)
        import shutil
        shutil.rmtree(work / f"sparse_cut{attempt}", ignore_errors=True)  # from an earlier pass
        rec4 = _map(db, image_dir, work / f"sparse_cut{attempt}", use_priors, ba_ratio)
        if rec4 is None or rec4.num_reg_images() < 0.7 * rec.num_reg_images():
            print("[sfm] WARNING: re-mapping lost too many frames; keeping the previous model")
            break
        if max_step_ratio(rec4) > 0.5 * max_step_ratio(rec):
            print("[sfm] jump still there after the cut")
            continue
        rec = rec4
        print(f"[sfm] re-mapped: {rec.num_reg_images()}/{n_images}, "
              f"{len(find_pose_jumps(rec))} jump(s) left")
        # pieces from the earlier solves were glued by the bad links: don't attach them
        for d in work.glob("sparse*"):
            if d.is_dir() and d.name not in (f"sparse_cut{attempt}", "sparse_best"):
                shutil.rmtree(work / f"rejected_{d.name}", ignore_errors=True)
                d.rename(work / f"rejected_{d.name}")
    else:
        if find_pose_jumps(rec):
            print("[sfm] WARNING: camera jumps remain; the map may be torn there")
    return rec


def run_sfm(image_dir: Path, work: Path, overlap: int = 15, max_size: int = 1600,
            exhaustive_limit: int = 400, position_priors=None, max_features: int = 8192,
            ba_ratio: float = 1.4, continuous_video: bool = False):
    """continuous_video: frames come from one video (run.py sets it), so camera jumps
    between neighbouring frames are tears to repair (see fix_pose_jumps).
    position_priors: optional {image name: [x, y, z]} in metres (ARKit). When given,
    frames that were physically close are matched too (catches revisits cheaply) and the
    mapper keeps the reconstruction consistent with those positions."""
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    t = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = max_size
    ext.sift.max_num_features = max_features
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext,
                              device=pycolmap.Device.cpu)
    print(f"[sfm] features {time.time()-t:.0f}s")
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pairing.quadratic_overlap = True
    pycolmap.match_sequential(db, pairing_options=pairing, device=pycolmap.Device.cpu)
    use_priors = False
    if position_priors:
        n = write_position_priors(db, position_priors)
        sp = pycolmap.SpatialPairingOptions()
        sp.ignore_z = False
        sp.max_num_neighbors = 30
        sp.max_distance = 2.0  # metres
        pycolmap.match_spatial(db, pairing_options=sp, device=pycolmap.Device.cpu)
        use_priors = n > 0
        print(f"[sfm] {n} ARKit position priors; spatial matching added")
    print(f"[sfm] matching {time.time()-t:.0f}s")
    n_images = len([p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    rec = _map(db, image_dir, work / "sparse", use_priors, ba_ratio)
    print(f"[sfm] mapping {time.time()-t:.0f}s: {0 if rec is None else rec.num_reg_images()}/{n_images}")
    if (rec is None or rec.num_reg_images() < 0.9 * n_images) and n_images <= exhaustive_limit:
        # Split into several models (pauses, revisits, fast turns). Match every pair so
        # revisited places link the pieces together, then map again.
        names = sorted(p.name for p in image_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        pairs = bridge_pairs(work, names)
        n_exh = len(names) * (len(names) - 1) // 2
        if len(pairs) < 0.5 * n_exh:
            print(f"[sfm] incomplete -> bridge matching {len(pairs):,} pairs "
                  f"(exhaustive would be {n_exh:,})")
            lst = work / "bridge_pairs.txt"
            lst.write_text("".join(f"{a} {b}\n" for a, b in pairs))
            po = pycolmap.ImportedPairingOptions()
            po.match_list_path = lst
            pycolmap.match_image_pairs(db, pairing_options=po, device=pycolmap.Device.cpu)
        else:
            print("[sfm] incomplete -> adding exhaustive matching for loop closure")
            pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
        rec2 = _map(db, image_dir, work / "sparse_exhaustive", use_priors, ba_ratio)
        if rec2 is not None and (rec is None or rec2.num_reg_images() > rec.num_reg_images()):
            rec = rec2
        if len(pairs) < 0.5 * n_exh and rec.num_reg_images() < 0.9 * n_images:
            # bridge pairs weren't enough: fall back to matching everything
            print(f"[sfm] still {rec.num_reg_images()}/{n_images} -> exhaustive matching")
            pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
            rec3 = _map(db, image_dir, work / "sparse_exhaustive_full", use_priors, ba_ratio)
            if rec3 is not None and rec3.num_reg_images() > rec.num_reg_images():
                rec = rec3
        print(f"[sfm] re-mapping {time.time()-t:.0f}s: {rec.num_reg_images()}/{n_images}")
    if rec is None:
        raise RuntimeError("SfM failed: no reconstruction")
    if continuous_video and find_pose_jumps(rec):
        # Only for video: frames a fraction of a second apart can't really jump 12+ median
        # steps. A photo set can (the Mip-NeRF room has a genuine 46x step between passes).
        import shutil
        uncut = work / "database_uncut.db"
        shutil.copy(db, uncut)
        rec = fix_pose_jumps(rec, db, image_dir, work, use_priors, ba_ratio, n_images)
        # The mapper is random: the same video can tear on one run and not the next. If
        # cutting didn't help, solve again from the uncut matches with other seeds.
        best = rec
        for seed in (1, 2):
            if not find_pose_jumps(best):
                break
            print(f"[sfm] jumps remain: re-solving with seed {seed}")
            shutil.copy(uncut, db)
            rs = _map(db, image_dir, work / f"sparse_seed{seed}", use_priors, ba_ratio, seed=seed)
            if rs is None:
                continue
            rs = fix_pose_jumps(rs, db, image_dir, work, use_priors, ba_ratio, n_images)
            ok = rs.num_reg_images() >= 0.7 * best.num_reg_images()
            if ok and (not find_pose_jumps(rs) or max_step_ratio(rs) < max_step_ratio(best)):
                best = rs
        rec = best
        if find_pose_jumps(rec):
            print("[sfm] WARNING: camera jumps remain after retries; the map may be torn")
    out = work / "sparse_best"
    out.mkdir(exist_ok=True)
    rec.write(out)
    print(f"[sfm] registered {rec.num_reg_images()} images, {rec.num_points3D()} points -> {out}")
    print(rec.summary())
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("--overlap", type=int, default=15)
    a = ap.parse_args()
    run_sfm(a.images, a.work, a.overlap)
