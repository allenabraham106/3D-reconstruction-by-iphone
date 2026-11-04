#!/usr/bin/env python3
"""Convert a PhoneScan capture into a COLMAP text model seeded with ARKit poses.

ARKit poses are metric and gravity-aligned, so you can skip pose estimation entirely and only
triangulate (much faster and more robust than full SfM on textureless rooms):

    python3 tools/arkit_to_colmap.py CAPTURE_DIR --out CAPTURE_DIR/colmap
    colmap feature_extractor --database_path CAPTURE_DIR/colmap/database.db \
        --image_path CAPTURE_DIR/images --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1
    python3 tools/arkit_to_colmap.py CAPTURE_DIR --out CAPTURE_DIR/colmap --sync-db   # match DB ids
    colmap sequential_matcher --database_path CAPTURE_DIR/colmap/database.db
    colmap point_triangulator --database_path CAPTURE_DIR/colmap/database.db \
        --image_path CAPTURE_DIR/images --input_path CAPTURE_DIR/colmap/sparse_arkit \
        --output_path CAPTURE_DIR/colmap/sparse
    # optional: colmap bundle_adjuster to refine ARKit drift

Output (COLMAP text format, OpenCV camera axes, world = ARKit world, metres):
    <out>/sparse_arkit/cameras.txt  one PINHOLE camera (or one per frame with --per-frame-intrinsics)
    <out>/sparse_arkit/images.txt   one entry per frame, qvec/tvec are world-to-camera
    <out>/sparse_arkit/points3D.txt empty
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phonescan_io import load_capture, rotmat_to_qvec  # noqa: E402


def write_model(capture: Path, out: Path, per_frame_intrinsics: bool, id_map: dict[str, int] | None = None,
                cam_map: dict[str, int] | None = None) -> int:
    frames = load_capture(capture)
    if not frames:
        raise SystemExit(f"no frames in {capture}")
    model = out / "sparse_arkit"
    model.mkdir(parents=True, exist_ok=True)

    cams_lines = ["# Camera list with one line of data per camera:",
                  "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    img_lines = ["# Image list with two lines of data per image:",
                 "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                 "#   POINTS2D[] as (X, Y, POINT3D_ID)"]

    if not per_frame_intrinsics:
        K = np.median(np.stack([f.K for f in frames]), axis=0)
        cam_id = 1 if cam_map is None else next(iter(cam_map.values()))
        cams_lines.append(f"{cam_id} PINHOLE {frames[0].width} {frames[0].height} "
                          f"{K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f}")

    for i, f in enumerate(frames, start=1):
        name = f.image.name
        image_id = id_map[name] if id_map else i
        if per_frame_intrinsics:
            cam_id = cam_map[name] if cam_map else i
            cams_lines.append(f"{cam_id} PINHOLE {f.width} {f.height} "
                              f"{f.K[0,0]:.6f} {f.K[1,1]:.6f} {f.K[0,2]:.6f} {f.K[1,2]:.6f}")
        w2c = f.w2c_cv
        q = rotmat_to_qvec(w2c[:3, :3])
        t = w2c[:3, 3]
        img_lines.append(f"{image_id} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                         f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {cam_id} {name}")
        img_lines.append("")

    (model / "cameras.txt").write_text("\n".join(cams_lines) + "\n")
    (model / "images.txt").write_text("\n".join(img_lines) + "\n")
    (model / "points3D.txt").write_text("# 3D point list (empty: run colmap point_triangulator)\n")
    return len(frames)


def read_db_ids(db: Path) -> tuple[dict[str, int], dict[str, int]]:
    con = sqlite3.connect(db)
    rows = con.execute("SELECT name, image_id, camera_id FROM images").fetchall()
    con.close()
    names = {Path(n).name: (iid, cid) for n, iid, cid in rows}
    return {n: v[0] for n, v in names.items()}, {n: v[1] for n, v in names.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path, help="PhoneScan capture folder (contains transforms.json)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: CAPTURE/colmap)")
    ap.add_argument("--per-frame-intrinsics", action="store_true",
                    help="one camera per image (ARKit autofocus changes focal length slightly)")
    ap.add_argument("--sync-db", action="store_true",
                    help="reuse image/camera ids from <out>/database.db (run after feature_extractor)")
    args = ap.parse_args()
    out = args.out or args.capture / "colmap"

    id_map = cam_map = None
    if args.sync_db:
        id_map, cam_map = read_db_ids(out / "database.db")
    n = write_model(args.capture, out, args.per_frame_intrinsics, id_map, cam_map)
    print(f"wrote {n} posed images to {out / 'sparse_arkit'}")


if __name__ == "__main__":
    main()
