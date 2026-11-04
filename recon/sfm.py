"""Stage 2: camera poses + intrinsics via COLMAP incremental SfM (CPU).

Stands in for ViPE's bundle-adjusted SLAM: sequential matching suits video,
and one shared camera matches a single phone lens.
"""
import argparse, time
from pathlib import Path
import pycolmap


def run_sfm(image_dir: Path, work: Path, overlap: int = 15, max_size: int = 1600):
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    t = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = max_size
    ext.sift.max_num_features = 8192
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext,
                              device=pycolmap.Device.cpu)
    print(f"[sfm] features {time.time()-t:.0f}s")
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pairing.quadratic_overlap = True
    pycolmap.match_sequential(db, pairing_options=pairing, device=pycolmap.Device.cpu)
    print(f"[sfm] matching {time.time()-t:.0f}s")
    opts = pycolmap.IncrementalPipelineOptions()
    recs = pycolmap.incremental_mapping(db, image_dir, work / "sparse", options=opts)
    print(f"[sfm] mapping {time.time()-t:.0f}s")
    if not recs:
        raise RuntimeError("SfM failed: no reconstruction")
    best_id = max(recs, key=lambda k: recs[k].num_reg_images())
    rec = recs[best_id]
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
