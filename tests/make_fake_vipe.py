"""Write a ViPE-format artifact folder from one of our reconstructions (no GPU needed).

Tests gpu/vipe_to_pipeline.py's conventions: poses (camera-to-world), intrinsics, EXR
depth and the frame video all in ViPE's exact layout, scaled to metres.

    python tests/make_fake_vipe.py out/<scan> <metric scale> fake_vipe/ NAME
"""
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recon"))


def write_exr_half(path, depth):
    import Imath
    import OpenEXR
    h, w = depth.shape
    header = OpenEXR.Header(w, h)
    header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
    exr = OpenEXR.OutputFile(str(path), header)
    exr.writePixels({"Z": depth.astype(np.float16).tobytes()})
    exr.close()


def main(scan: Path, scale: float, out: Path, name: str):
    rec = pycolmap.Reconstruction(str(scan / "sfm" / "sparse_best"))
    cache = scan / "dense" / "depth_cache"
    ims = [im for im in sorted(rec.images.values(), key=lambda im: im.name)
           if (cache / (im.name + ".npy")).exists()]
    cam = rec.cameras[ims[0].camera_id]
    fx = fy = float(cam.params[0]); cx, cy = float(cam.params[1]), float(cam.params[2])
    for sub in ("rgb", "pose", "intrinsics", "depth"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(scan / "frames" / ims[0].name))
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(str(out / "rgb" / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    poses, intr = [], []
    with zipfile.ZipFile(out / "depth" / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as z, \
            tempfile.TemporaryDirectory() as td:
        for i, im in enumerate(ims):
            vw.write(cv2.imread(str(scan / "frames" / im.name)))
            T = im.cam_from_world().matrix()
            R, t = T[:, :3], T[:, 3]
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = (-R.T @ t) * scale
            poses.append(c2w)
            intr.append([fx, fy, cx, cy])
            D = np.load(cache / (im.name + ".npy")).astype(np.float32) * scale
            p = Path(td) / f"{i:05d}.exr"
            write_exr_half(p, D)
            z.write(p, f"{i:05d}.exr")
    vw.release()
    inds = np.arange(len(ims))
    np.savez(out / "pose" / f"{name}.npz", data=np.stack(poses), inds=inds)
    np.savez(out / "intrinsics" / f"{name}.npz", data=np.array(intr), inds=inds)
    print(f"fake ViPE output: {len(ims)} frames -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
