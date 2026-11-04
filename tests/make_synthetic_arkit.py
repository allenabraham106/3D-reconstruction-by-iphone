"""Build a fake ScanCapture folder (exact iOS app format) from an existing metric map.

Used to test ingest_arkit.py conventions without an iPhone: we know the ground-truth
geometry, write it out the way ARKit would (y-up world, camera y-up/z-back, column-major
matrices, 256x192 float32 LiDAR depth), ingest it, and compare.
"""
import argparse, json, shutil, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recon"))
from robot_map import read_ply  # noqa: E402

ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)  # (x,y,z)->(x,z,-y)
FLIP = np.diag([1.0, -1.0, -1.0])


def main(map_ply, cams_json, images, out, every=2, dw=256, dh=192, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    P, _ = read_ply(map_ply)
    Pw = P @ ZUP_TO_YUP.T
    cams = json.loads(Path(cams_json).read_text())[::every]
    if out.exists():
        shutil.rmtree(out)
    for s in ("rgb", "depth", "conf"):
        (out / s).mkdir(parents=True)
    lines = []
    for i, c in enumerate(cams):
        R = np.array(c["R"]) @ ZUP_TO_YUP.T  # cam_from_world in the y-up world
        C = ZUP_TO_YUP @ np.array(c["C"])
        img = cv2.imread(str(images / c["name"]))
        h, w = img.shape[:2]
        K = np.array([[c["fx"], 0, w / 2], [0, c["fy"], h / 2], [0, 0, 1.0]])
        name = f"{i:06d}"
        cv2.imwrite(str(out / "rgb" / f"{name}.jpg"), img)
        # render depth (z-buffer splat of the map) at LiDAR resolution
        Kd = K.copy(); Kd[0] *= dw / w; Kd[1] *= dh / h
        Xc = (Pw - C) @ R.T
        z = Xc[:, 2]
        m = z > 0.1
        uv = Xc[m] @ Kd.T
        u = (uv[:, 0] / uv[:, 2]).astype(int); v = (uv[:, 1] / uv[:, 2]).astype(int)
        inb = (u >= 0) & (u < dw) & (v >= 0) & (v < dh)
        D = np.full((dh, dw), np.inf, np.float32)
        np.minimum.at(D, (v[inb], u[inb]), z[m][inb].astype(np.float32))
        D[~np.isfinite(D)] = 0
        D = cv2.morphologyEx(D, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        D[D > 0] *= (1 + noise * rng.standard_normal((D > 0).sum())).astype(np.float32)
        D.astype(np.float32).tofile(out / "depth" / f"{name}.bin")
        ((D > 0) * 2).astype(np.uint8).tofile(out / "conf" / f"{name}.bin")
        # ARKit pose: world_from_camera with camera axes y-up / z-back
        T = np.eye(4)
        T[:3, :3] = R.T @ FLIP
        T[:3, 3] = C
        lines.append(json.dumps({
            "index": i, "timestamp": i / 6, "rgb": f"rgb/{name}.jpg", "width": w, "height": h,
            "intrinsics": K.T.ravel().tolist(), "world_from_camera": T.T.ravel().tolist(),
            "depth": f"depth/{name}.bin", "depth_width": dw, "depth_height": dh,
            "confidence": f"conf/{name}.bin"}))
    (out / "frames.jsonl").write_text("\n".join(lines) + "\n")
    (out / "meta.json").write_text(json.dumps({"device": "synthetic", "lidar": True,
                                                "frames": len(cams), "format": "scan-capture-v1"}))
    print(f"wrote {len(cams)} synthetic ARKit frames to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("map_ply", type=Path)
    ap.add_argument("cameras", type=Path)
    ap.add_argument("images", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    main(a.map_ply, a.cameras, a.images, a.out)
