"""Stage 2: camera poses + intrinsics via COLMAP incremental SfM (CPU).

Stands in for ViPE's bundle-adjusted SLAM: sequential matching suits video,
and one shared camera matches a single phone lens.
"""
import argparse, time
from pathlib import Path
import numpy as np
import pycolmap


def _map(db, image_dir, out, use_priors=False, ba_ratio=1.4):
    out.mkdir(parents=True, exist_ok=True)
    opts = pycolmap.IncrementalPipelineOptions()
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


def run_sfm(image_dir: Path, work: Path, overlap: int = 15, max_size: int = 1600,
            exhaustive_limit: int = 400, position_priors=None, max_features: int = 8192,
            ba_ratio: float = 1.4):
    """position_priors: optional {image name: [x, y, z]} in metres (ARKit). When given,
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
        print("[sfm] incomplete -> adding exhaustive matching for loop closure")
        pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
        rec2 = _map(db, image_dir, work / "sparse_exhaustive", use_priors, ba_ratio)
        if rec2 is not None and (rec is None or rec2.num_reg_images() > rec.num_reg_images()):
            rec = rec2
        print(f"[sfm] re-mapping {time.time()-t:.0f}s: {rec.num_reg_images()}/{n_images}")
    if rec is None:
        raise RuntimeError("SfM failed: no reconstruction")
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
