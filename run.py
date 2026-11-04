#!/usr/bin/env python3
"""One command: iPhone video (or image folder, or ARKit capture) -> 3D map + viewer.

    python run.py IMG_1234.MOV                 # video from the Camera app
    python run.py my_frames/                    # folder of photos
    python run.py capture_2026-11-04/ --arkit   # folder exported by the iOS capture app

Outputs in out/<name>/ :
    viewer/index.html + dense.ply + cameras.json   open with `python -m http.server`
    robot/map_metric.ply                           gravity-aligned, metres, floor at z=0
    robot/occupancy.pgm/.yaml                      ROS nav2 2D map
    robot/voxels.npz                               5 cm 3D occupancy
"""
import argparse, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "recon"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--frames", type=int, default=300, help="keyframes to extract from video")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--phone-height", type=float, default=1.4,
                    help="approx. phone height above the floor while filming (sets metric scale)")
    ap.add_argument("--arkit", action="store_true", help="input is an iOS capture-app folder")
    ap.add_argument("--stereo-scale", type=float, default=1.0)
    ap.add_argument("--dense-every", type=int, default=1,
                    help="compute depth for every k-th frame (others still used as sources); "
                         "2 roughly halves the dense stage on video")
    ap.add_argument("--stereo", choices=["sweep", "sgbm"], default="sweep",
                    help="sweep: plane sweep, works for any motion (default); "
                         "sgbm: faster rectified stereo, only for sideways/orbit captures")
    a = ap.parse_args()

    name = a.input.stem if a.input.is_file() else a.input.name
    out = a.out / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if a.arkit:
        from ingest_arkit import ingest
        ply, cams = ingest(a.input, out / "dense")
        metric = True
    else:
        if a.input.is_file():
            from frames import extract
            img_dir = out / "frames"
            if img_dir.exists():
                shutil.rmtree(img_dir)
            extract(a.input, img_dir, a.frames, a.max_side)
        else:
            img_dir = a.input
        from sfm import attach_submodels, run_sfm
        rec = run_sfm(img_dir, out / "sfm")
        extra = attach_submodels(rec, out / "sfm")  # bridge pieces SfM couldn't join
        from mvs import run_mvs
        run_mvs(out / "sfm" / "sparse_best", img_dir, out / "dense", scale=a.stereo_scale,
                extra_sparse=extra, method=a.stereo, cache_depth=True,
                dense_every=a.dense_every)
        ply, cams = out / "dense" / "dense.ply", out / "dense" / "cameras.json"
        metric = False

    from robot_map import build
    build(ply, cams, out / "robot", a.phone_height, metric)

    # viewer bundle: show the metric, gravity-aligned map
    vdir = out / "viewer"
    if vdir.exists():
        shutil.rmtree(vdir)
    shutil.copytree(ROOT / "viewer", vdir)
    shutil.copy(out / "robot" / "map_metric.ply", vdir / "dense.ply")
    shutil.copy(out / "robot" / "cameras_metric.json", vdir / "cameras.json")
    (vdir / "view.json").write_text('{"up": [0, 0, 1]}')  # robot map is z-up
    if (out / "robot" / "mesh_metric.ply").exists():  # TSDF surface, shown by default
        shutil.copy(out / "robot" / "mesh_metric.ply", vdir / "mesh.ply")
    if (out / "robot" / "inferred_floor.ply").exists():  # imagined floor, toggled in viewer
        shutil.copy(out / "robot" / "inferred_floor.ply", vdir / "inferred_floor.ply")
    print(f"\nDone in {time.time()-t0:.0f}s.\n  View:  cd {vdir} && python3 -m http.server 8000"
          f"  ->  http://localhost:8000\n  Robot: {out/'robot'}")


if __name__ == "__main__":
    main()
