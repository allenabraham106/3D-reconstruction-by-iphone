#!/usr/bin/env python3
"""End-to-end tests for the PhoneScan desktop tools using a synthetic capture.

The synthetic capture is a 4 x 2.6 x 3 m box room rendered *in ARKit's own convention*
(OpenGL camera axes, camera-to-world poses, intrinsics for a 1920x1440 image, 256x192 depth
in millimetres), written exactly the way the iOS app writes it. The tools then have to convert
to OpenCV/COLMAP conventions on their own, so a sign or axis mistake fails the tests.

    python3 tools/test_tools.py
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arkit_to_colmap  # noqa: E402
import fuse_depth  # noqa: E402
import receive_server  # noqa: E402
from phonescan_io import qvec_to_rotmat, read_ply_points  # noqa: E402

ROOM_MIN = np.array([-2.0, 0.0, -1.5])
ROOM_MAX = np.array([2.0, 2.6, 1.5])
W, H = 1920, 1440
DW, DH = 256, 192
FX = FY = 1450.0
CX, CY = 961.3, 718.9


def look_at_gl(eye, target, up=(0, 1, 0)):
    """camera-to-world with OpenGL axes: -z looks at target, +y roughly up."""
    eye, target, up = map(np.asarray, (eye, target, up))
    back = eye - target
    back = back / np.linalg.norm(back)
    right = np.cross(up, back); right /= np.linalg.norm(right)
    upv = np.cross(back, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, upv, back, eye
    return T


def render_depth_gl(c2w_gl):
    """Ray-cast the box room from inside, in the GL camera frame. Returns depth (z along -z) in metres."""
    s = DW / W
    fx, fy, cx, cy = FX * s, FY * s, CX * s, CY * s
    u, v = np.meshgrid(np.arange(DW) + 0.5, np.arange(DH) + 0.5)
    # GL camera: +x right, +y up (image v grows downward), looking down -z.
    d_cam = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u)], axis=-1)
    R, o = c2w_gl[:3, :3], c2w_gl[:3, 3]
    d_world = d_cam @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (ROOM_MIN - o) / d_world
        t2 = (ROOM_MAX - o) / d_world
    tmax = np.where(d_world > 0, t2, t1)       # exit distance per axis (camera is inside the box)
    tmax = np.where(np.isfinite(tmax), tmax, np.inf)
    t = tmax.min(axis=-1)                       # parameter along d_cam where -z component = 1
    return t                                    # since d_cam.z == -1, depth == t


def colour_for(points):
    return (np.clip((points - ROOM_MIN) / (ROOM_MAX - ROOM_MIN), 0, 1) * 255).astype(np.uint8)


def make_capture(root: Path, n=12) -> Path:
    cap = root / "2026-11-04_12-00-00"
    for sub in ("images", "depth", "confidence"):
        (cap / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        eye = np.array([0.6 * np.cos(ang), 1.4 + 0.1 * np.sin(3 * ang), 0.5 * np.sin(ang)])
        tilt = -0.35 if i % 2 == 0 else 0.8  # alternate looking at floor and ceiling
        target = eye + np.array([np.cos(ang + 0.4), tilt + 0.1 * rng.standard_normal(), np.sin(ang + 0.4)])
        c2w = look_at_gl(eye, target)
        depth_m = render_depth_gl(c2w)
        mm = np.clip(np.round(depth_m * 1000), 0, 65535).astype(np.uint16)
        mm[rng.random(mm.shape) < 0.01] = 0  # some invalid pixels, like real LiDAR
        name = f"frame_{i:05d}"
        Image.fromarray(mm).save(cap / "depth" / f"{name}.png")
        conf = np.full((DH, DW), 2, np.uint8); conf[:4] = 0
        Image.fromarray(conf).save(cap / "confidence" / f"{name}.png")
        Image.new("RGB", (W, H), (int(30 + 15 * i), 120, 200)).save(cap / "images" / f"{name}.jpg", quality=80)
        frames.append({
            "file_path": f"images/{name}.jpg",
            "depth_file_path": f"depth/{name}.png",
            "confidence_file_path": f"confidence/{name}.png",
            "transform_matrix": c2w.tolist(),
            "fl_x": FX, "fl_y": FY, "cx": CX, "cy": CY, "w": W, "h": H,
            "timestamp": 1000.0 + i * 0.3, "exposure_duration": 0.01,
        })
    meta = {"camera_model": "OPENCV", "fl_x": FX, "fl_y": FY, "cx": CX, "cy": CY, "w": W, "h": H,
            "k1": 0, "k2": 0, "p1": 0, "p2": 0, "depth_unit_scale_factor": 0.001, "frames": frames}
    (cap / "transforms.json").write_text(json.dumps(meta))
    (cap / "frames.jsonl").write_text("\n".join(json.dumps(f) for f in frames) + "\n")
    return cap


def dist_to_room_surface(p):
    d = np.minimum(np.abs(p - ROOM_MIN), np.abs(p - ROOM_MAX))
    return d.min(axis=1)


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cap = make_capture(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fuse_depth_lands_on_room_walls(self):
        xyz, rgb, raw, nframes = fuse_depth.fuse(self.cap, voxel=0.0, min_conf=2, max_depth=10, stride=1, every=1)
        self.assertEqual(nframes, 12)
        self.assertGreater(raw, 12 * DW * DH * 0.9)
        err = dist_to_room_surface(xyz)
        # mm quantisation + pixel discretisation only
        self.assertLess(np.percentile(err, 99), 0.004, f"99th pct error {np.percentile(err, 99):.4f} m")
        inside = np.all((xyz > ROOM_MIN - 0.01) & (xyz < ROOM_MAX + 0.01), axis=1)
        self.assertTrue(inside.all())
        # floor and ceiling both seen, so the up axis survived the conversion
        self.assertTrue((np.abs(xyz[:, 1] - 0.0) < 0.01).any())
        self.assertTrue((np.abs(xyz[:, 1] - 2.6) < 0.01).any())

    def test_fuse_cli_writes_downsampled_ply(self):
        out = Path(self.tmp.name) / "fused.ply"
        xyz, rgb, raw, _ = fuse_depth.fuse(self.cap, voxel=0.05, min_conf=2, max_depth=10, stride=2, every=2)
        from phonescan_io import write_ply_points
        write_ply_points(out, xyz, rgb)
        back = read_ply_points(out)
        self.assertEqual(len(back), len(xyz))
        self.assertLess(len(xyz), raw)
        self.assertLess(np.percentile(dist_to_room_surface(back), 95), 0.05)

    def test_colmap_model_matches_gl_projection(self):
        out = Path(self.tmp.name) / "colmap"
        n = arkit_to_colmap.write_model(self.cap, out, per_frame_intrinsics=False)
        self.assertEqual(n, 12)
        cams = [l for l in (out / "sparse_arkit/cameras.txt").read_text().splitlines() if l and not l.startswith("#")]
        self.assertEqual(len(cams), 1)
        _, model, w, h, fx, fy, cx, cy = cams[0].split()
        self.assertEqual((model, int(w), int(h)), ("PINHOLE", W, H))
        K = np.array([[float(fx), 0, float(cx)], [0, float(fy), float(cy)], [0, 0, 1]])

        lines = [l for l in (out / "sparse_arkit/images.txt").read_text().splitlines() if l and not l.startswith("#")]
        frames = json.loads((self.cap / "transforms.json").read_text())["frames"]
        self.assertEqual(len(lines), 12)
        world_pt = np.array([1.9, 1.0, 0.3])
        for line, fr in zip(lines, frames):
            parts = line.split()
            q = np.array(list(map(float, parts[1:5])))
            t = np.array(list(map(float, parts[5:8])))
            self.assertEqual(parts[9], Path(fr["file_path"]).name)
            R = qvec_to_rotmat(q)
            c2w = np.array(fr["transform_matrix"])
            # camera centre agrees
            np.testing.assert_allclose(-R.T @ t, c2w[:3, 3], atol=1e-6)
            # projection of a world point agrees with independent GL-convention projection
            p_cv = R @ world_pt + t
            p_gl = np.linalg.inv(c2w) @ np.append(world_pt, 1)
            if p_gl[2] >= -0.1:
                self.assertLessEqual(p_cv[2], 0.1)
                continue
            uv_cv = (K @ p_cv)[:2] / p_cv[2]
            uv_gl = np.array([FX * p_gl[0] / -p_gl[2] + CX, -FY * p_gl[1] / -p_gl[2] + CY])
            np.testing.assert_allclose(uv_cv, uv_gl, atol=1e-4)

    def test_receive_server_roundtrip(self):
        root = Path(self.tmp.name) / "server"
        root.mkdir()
        httpd = receive_server.ThreadingHTTPServer(("127.0.0.1", 0), receive_server.make_handler(root.resolve(), None))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:  # same layout as iOS: one top-level folder
                for p in self.cap.rglob("*"):
                    if p.is_file():
                        zf.write(p, f"{self.cap.name}/{p.relative_to(self.cap)}")
            body = buf.getvalue()
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/upload", data=body, method="POST",
                                         headers={"Content-Type": "application/zip", "X-Capture-Name": self.cap.name})
            resp = json.loads(urllib.request.urlopen(req).read())
            self.assertTrue(resp["ok"])
            dest = root / self.cap.name
            self.assertTrue((dest / "transforms.json").exists())
            self.assertEqual(len(list((dest / "images").glob("*.jpg"))), 12)

            # path traversal is rejected
            evil = io.BytesIO()
            with zipfile.ZipFile(evil, "w") as zf:
                zf.writestr("../../escape.txt", "x")
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/upload", data=evil.getvalue(),
                                         method="POST", headers={"X-Capture-Name": "evil"})
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 400)
            self.assertFalse((root.parent / "escape.txt").exists())
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
