#!/usr/bin/env python3
"""Tests for robot_link: build a relocalization map from a synthetic room, localize new views.

The room (tests/synthetic_room.py) is rendered with known camera poses, written in exactly the
layout run.py leaves behind (frames/ + robot/cameras_metric.json + robot/occupancy.*). New
viewpoints that were never keyframes are then localized and compared with the truth.

    python tests/test_robot_link.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from synthetic_room import look_at, render  # noqa: E402
from robot_link import server  # noqa: E402
from robot_link.localize import Localizer  # noqa: E402
from robot_link.reloc_map import build  # noqa: E402

W, H, F = 640, 480, 485.0
SCENE_TO_MAP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # scene y-up -> map z-up


def gl_to_map(c2w_gl):
    """Scene camera (OpenGL c2w, y-up) -> (R, t, C) cam_from_world in the z-up map frame, OpenCV axes."""
    c2w = c2w_gl @ np.diag([1, -1, -1, 1])
    Rw = SCENE_TO_MAP @ c2w[:3, :3]
    C = SCENE_TO_MAP @ c2w[:3, 3]
    R = Rw.T
    return R, -R @ C, C


def bgr(rgb):
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def make_map_dir(root: Path, n=28) -> Path:
    out = root / "living-room"
    (out / "frames").mkdir(parents=True)
    (out / "robot").mkdir()
    cams = []
    for i in range(n):
        a = 2 * np.pi * i / n
        eye = np.array([0.15 + 0.9 * np.cos(a), 1.35, 0.1 + 0.8 * np.sin(a)])
        look = a + np.pi * 0.55
        c2w = look_at(eye, eye + np.array([np.cos(look), [-0.35, 0.05, 0.4][i % 3], np.sin(look)]))
        rgb, *_ = render(c2w, W, H, F, F, W / 2, H / 2)
        name = f"frame_{i:05d}.jpg"
        cv2.imwrite(str(out / "frames" / name), bgr(rgb), [cv2.IMWRITE_JPEG_QUALITY, 92])
        R, t, C = gl_to_map(c2w)
        cams.append(dict(name=name, R=R.tolist(), t=t.tolist(), C=C.tolist(), fx=F, fy=F, w=W, h=H))
    (out / "robot" / "cameras_metric.json").write_text(json.dumps(cams))
    # a small nav2 grid covering the room, 5 cm cells, origin at the room's corner
    grid = np.full((84, 94), 254, np.uint8)
    grid[[0, -1], :] = 0
    grid[:, [0, -1]] = 0
    cv2.imwrite(str(out / "robot" / "occupancy.pgm"), grid)
    (out / "robot" / "occupancy.yaml").write_text(
        "image: occupancy.pgm\nmode: trinary\nresolution: 0.05\norigin: [-2.35, -2.10, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    return out


# Viewpoints that are NOT keyframes: different spots, heights, headings, and a different camera.
QUERIES = [([0.4, 1.1, 0.6], [-1.5, 0.7, -1.6]),
           ([-0.5, 1.6, -0.2], [1.8, 0.9, -0.8]),
           ([0.9, 1.0, 0.9], [-2.0, 1.2, 1.5])]
QW, QH, QF = 800, 600, 640.0
QK = np.array([[QF, 0, QW / 2], [0, QF, QH / 2], [0, 0, 1.0]])


def query(i):
    c2w = look_at(*QUERIES[i])
    rgb, *_ = render(c2w, QW, QH, QF, QF, QW / 2, QH / 2)
    R, t, C = gl_to_map(c2w)
    fwd = R.T @ np.array([0, 0, 1.0])
    return bgr(rgb), C, math.degrees(math.atan2(fwd[1], fwd[0]))


class RobotLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.out = make_map_dir(cls.root)
        cls.info = build(cls.out, verbose=False)
        cls.loc = Localizer(cls.out)
        cls.queries = [query(i) for i in range(len(QUERIES))]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_map_has_landmarks_in_the_room(self):
        self.assertGreater(self.info["landmarks"], 1500)
        P = np.load(self.out / "robot" / "reloc_map.npz")["points"]
        inside = (np.abs(P[:, 0]) < 2.4) & (np.abs(P[:, 1]) < 2.2) & (P[:, 2] > -0.1) & (P[:, 2] < 2.7)
        self.assertGreater(inside.mean(), 0.97)

    def test_localizes_new_viewpoints_to_centimetres(self):
        for i, (img, C, yaw) in enumerate(self.queries):
            p = self.loc.localize(img, QK)
            self.assertTrue(p.ok, f"query {i}: {p.reason}")
            err = np.linalg.norm(np.array([p.x, p.y, p.z]) - C)
            self.assertLess(err, 0.05, f"query {i}: {err * 100:.1f} cm off")
            dyaw = (p.yaw_deg - yaw + 180) % 360 - 180
            self.assertLess(abs(dyaw), 2.0, f"query {i}: heading {dyaw:.1f} deg off")
            self.assertGreaterEqual(p.inliers, 25)
            # grid pixel lines up with the nav2 origin/resolution
            col = (p.x + 2.35) / 0.05
            self.assertAlmostEqual(p.grid_px[0], col, delta=0.2)

    def test_rejects_images_from_somewhere_else(self):
        rng = np.random.default_rng(0)
        noise = (rng.random((QH, QW, 3)) * 255).astype(np.uint8)
        blank = np.full((QH, QW, 3), 128, np.uint8)
        for img in (noise, blank):
            p = self.loc.localize(img, QK)
            self.assertFalse(p.ok)
            self.assertTrue(p.reason)

    def test_topdown_map_matches_grid_extent(self):
        # a flat floor + one box, written the way robot_map.py writes map_metric.ply
        rng = np.random.default_rng(1)
        floor = np.c_[rng.uniform(-2.3, 2.3, 40000), rng.uniform(-2.0, 2.0, 40000), np.zeros(40000)]
        box = np.c_[rng.uniform(0.0, 0.5, 4000), rng.uniform(0.0, 0.5, 4000), np.full(4000, 0.45)]
        P = np.vstack([floor, box]).astype("<f4")
        C = np.vstack([np.full((40000, 3), 200), np.full((4000, 3), 60)]).astype("u1")
        arr = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        arr["x"], arr["y"], arr["z"] = P.T
        arr["red"], arr["green"], arr["blue"] = C.T
        ply = self.out / "robot" / "map_metric.ply"
        hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\n"
               "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        ply.write_bytes(hdr.encode() + arr.tobytes())
        try:
            link = server.RobotLink(self.root, "Pioneer", server.Camera(), None, self.root / "captures")
            cloud = link.cloud_ply()
            self.assertTrue(cloud.startswith(b"ply\n"))
            Pc, _ = server._read_ply_xyzrgb(self.out / "robot" / "cloud_app.ply")
            self.assertEqual(len(Pc), len(P))           # under the cap, nothing dropped
            self.assertLess(Pc[:, 2].max(), 1.4)
            path = link.scan_path()["points"]
            self.assertEqual(len(path), 28)
            img = cv2.imdecode(np.frombuffer(link.topdown_png(), np.uint8), cv2.IMREAD_GRAYSCALE)
            g = link.localizer.grid
            self.assertEqual(img.shape, (round(g["height"] * 5), round(g["width"] * 5)))  # 1 cm pixels
            # the box (x 0..0.5, y 0..0.5) is darker than the floor, at the right place
            col = lambda x: int((x - g["origin"][0]) / 0.01)
            row = lambda y: img.shape[0] - 1 - int((y - g["origin"][1]) / 0.01)
            self.assertLess(img[row(0.25), col(0.25)] + 40, img[row(-1.0), col(-1.0)])
        finally:
            ply.unlink()
            (self.out / "robot" / "topdown.png").unlink(missing_ok=True)
            (self.out / "robot" / "cloud_app.ply").unlink(missing_ok=True)

    def test_bring_your_own_humanoid(self):
        """Another team's URDF (REP-103 names, faces +x, box meshes) converts, is served, and walks."""
        import trimesh
        sys.path.insert(0, str(HERE.parent / "tools"))
        from urdf_to_app_model import convert
        d = self.root / "acme_bot"
        (d / "meshes").mkdir(parents=True)
        trimesh.creation.box((0.3, 0.2, 0.5)).export(d / "meshes" / "torso.stl")
        trimesh.creation.box((0.08, 0.08, 0.4)).export(d / "meshes" / "leg.stl")
        def link(n, mesh, z=0.0):
            return (f'<link name="{n}"><visual><origin xyz="0 0 {z}"/><geometry><mesh filename="meshes/{mesh}.stl"/>'
                    f'</geometry><material name="m"><color rgba="0.8 0.2 0.2 1"/></material></visual></link>')
        def joint(n, parent, child, xyz, lo, hi):
            return (f'<joint name="{n}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
                    f'<origin xyz="{xyz}"/><axis xyz="0 1 0"/><limit lower="{lo}" upper="{hi}" effort="1" velocity="1"/></joint>')
        parts = [link("base_link", "torso", 1.05)]
        for side, y in (("left", 0.1), ("right", -0.1)):
            parts += [link(f"{side}_thigh", "leg", -0.2), link(f"{side}_shin", "leg", -0.2), link(f"{side}_foot", "leg", -0.02),
                      joint(f"{side}_hip_pitch_joint", "base_link", f"{side}_thigh", f"0 {y} 0.8", -1.5, 0.5),
                      joint(f"{side}_knee_joint", f"{side}_thigh", f"{side}_shin", "0 0 -0.4", 0.0, 2.3),
                      joint(f"{side}_ankle_pitch_joint", f"{side}_shin", f"{side}_foot", "0 0 -0.4", -0.6, 0.6)]
        urdf = d / "acme_bot.urdf"
        urdf.write_text('<robot name="acme_bot">' + "".join(parts) + "</robot>")

        info = convert(urdf, d / "acme.robot", verbose=False)
        self.assertEqual(info["name"], "acme_bot")
        self.assertEqual(info["gait"]["hip_pitch"], {"left": "left_hip_pitch_joint", "right": "right_hip_pitch_joint"})
        self.assertEqual(info["gait"]["knee"]["right"], "right_knee_joint")
        self.assertEqual(info["gait"]["ankle_pitch"]["left"], "left_ankle_pitch_joint")

        cached = server.prepare_robot_model(urdf, self.root / "_robots", name="Acme Walker")
        self.assertEqual(server.prepare_robot_model(urdf, self.root / "_robots", name="Acme Walker"), cached)  # cached
        link_ = server.RobotLink(self.root, "Acme Walker", server.Camera(), None, self.root / "captures", cached)
        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(link_))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{httpd.server_port}"
            st = json.loads(urllib.request.urlopen(base + "/status").read())
            self.assertEqual(st["robot"]["name"], "Acme Walker")
            self.assertEqual(st["robot"]["joints"], 6)
            model = urllib.request.urlopen(base + "/robot/model").read()
            self.assertEqual(model, cached.read_bytes())
            self.assertEqual(model[:4], b"PRBT")
        finally:
            httpd.shutdown()
            httpd.server_close()
        # without a model the server says so, and the app falls back to its Pioneer
        bare = server.RobotLink(self.root, "Pioneer", server.Camera(), None, self.root / "captures")
        self.assertIsNone(bare.status()["robot"])

    def test_base_pose_from_camera_mount(self):
        # A camera 1.45 m up, looking straight along the robot's +x, puts the base at z=0 with the same heading.
        T = server.base_from_camera_matrix("0.08 0 1.45 0 0 0")
        R_cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)  # cam_from_world for "looking along +X"
        world_from_cam = np.eye(4)
        world_from_cam[:3, :3] = R_cam.T
        world_from_cam[:3, 3] = [1.0, 2.0, 1.45]
        world_from_base = world_from_cam @ np.linalg.inv(T)
        np.testing.assert_allclose(world_from_base[:3, 3], [0.92, 2.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(world_from_base[:3, 0], [1, 0, 0], atol=1e-9)  # base x points along +X
        np.testing.assert_allclose(world_from_base[:3, 2], [0, 0, 1], atol=1e-9)  # base z is up

    def test_http_server(self):
        img, C, _ = self.queries[0]
        robot_img = self.root / "robot_cam.jpg"
        cv2.imwrite(str(robot_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cam = server.FileCamera(str(robot_img), QK)
        link = server.RobotLink(self.root, "Pioneer", cam, None, self.root / "captures")
        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(link))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_port}"

        def call(path, data=None, headers=None):
            req = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET",
                                         headers=headers or {})
            with urllib.request.urlopen(req) as r:
                return r.read(), r.headers.get("Content-Type")
        try:
            st = json.loads(call("/status")[0])
            self.assertEqual(st["map"], "living-room")
            self.assertEqual(st["grid"]["width"], 94)
            png, ctype = call("/map/occupancy.png")
            self.assertEqual(ctype, "image/png")
            self.assertEqual(png[:4], b"\x89PNG")
            mono, _ = call("/map/occupancy.png?style=mono")
            self.assertEqual(mono[:4], b"\x89PNG")
            self.assertNotEqual(mono, png)
            top, ctype = call("/map/topdown.png")   # no point cloud in this fixture: falls back to the mono grid
            self.assertEqual((ctype, top), ("image/png", mono))

            r = json.loads(call("/localize?source=robot", b"")[0])
            self.assertTrue(r["ok"], r["reason"])
            self.assertLess(np.linalg.norm(np.array([r["x"], r["y"], r["z"]]) - C), 0.05)

            img2, C2, _ = self.queries[1]
            jpg = cv2.imencode(".jpg", img2, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            hdr = {"Content-Type": "image/jpeg", "X-Fx": str(QF), "X-Fy": str(QF), "X-Cx": str(QW / 2), "X-Cy": str(QH / 2)}
            r = json.loads(call("/localize?source=phone", jpg, hdr)[0])
            self.assertTrue(r["ok"], r["reason"])
            self.assertLess(np.linalg.norm(np.array([r["x"], r["y"], r["z"]]) - C2), 0.05)

            st = json.loads(call("/status")[0])
            self.assertEqual(set(st["last"]), {"robot", "phone"})

            with self.assertRaises(urllib.error.HTTPError) as ctx:  # image without intrinsics
                call("/localize?source=phone", jpg, {"Content-Type": "image/jpeg"})
            self.assertEqual(ctx.exception.code, 400)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
