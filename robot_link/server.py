#!/usr/bin/env python3
"""Robot link server: the bridge between the Pioneer Home app and the robot.

Run it on the robot's computer (or a laptop next to it):

    python -m robot_link.server --maps out --camera realsense            # robot with a D455
    python -m robot_link.server --maps out --camera ros:/camera/color/image_raw --ros
    python -m robot_link.server --maps out                                # phone-only testing

The app connects to http://<robot>:8770 and can:

    GET  /status                 robot name, active map, camera, last poses, build progress
    GET  /maps                   maps that are ready (out/<name>/robot/reloc_map.npz)
    POST /maps/activate?name=X   switch the active map
    GET  /map/info               occupancy grid size / resolution / origin (nav2 conventions)
    GET  /map/occupancy.png      the nav2 floor map (?style=mono for the app's paper look)
    GET  /map/topdown.png        photo-like top-down view of the scan, aligned with the grid
    GET  /map/cloud.ply          the coloured 3D scan for the app's 3D map (walls cut at 1.4 m, <= 400k points)
    GET  /map/path.json          the scan route: keyframe camera positions, in order
    GET  /robot/model            this robot's 3D model for the app (from --robot-urdf / --robot-model);
                                 404 if none, and the app shows its built-in Pioneer
    POST /localize?source=robot  grab a frame from the robot camera and localize it
    POST /localize?source=phone  body = JPEG, headers X-Fx X-Fy X-Cx X-Cy: localize the phone
    POST /upload                 body = zipped Pioneer Home capture, header X-Capture-Name:
                                 runs run.py --arkit, then builds the relocalization map

Any humanoid works: pass --robot-urdf (plus --robot-forward if it doesn't face +x, as REP-103
robots do) and the app draws that robot, with its name, instead of the Pioneer:

    python -m robot_link.server --maps out --robot-urdf ~/my_robot/urdf/robot.urdf --name "Atlas"

Every pose is in the robot map frame: metres, Z up, floor at Z = 0, the same frame as
robot/occupancy.yaml. With --ros, robot poses are published as /initialpose
(geometry_msgs/PoseWithCovarianceStamped, frame "map"), which nav2's AMCL and most SLAM
stacks accept to start (or re-start) in a known map, plus /pioneer_home/camera_pose.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .localize import Localizer, Pose

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from receive_server import safe_extract  # noqa: E402

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


# ----------------------------------------------------------------------------- cameras

class Camera:
    """Something that returns (BGR image, 3x3 K)."""
    description = "none"

    def grab(self) -> tuple[np.ndarray, np.ndarray]:
        raise RuntimeError("this robot link has no camera configured (start it with --camera)")


class FileCamera(Camera):
    """For testing: always returns the same image."""
    def __init__(self, path: str, K: np.ndarray):
        self.path, self.K = path, K
        self.description = f"file {Path(path).name}"

    def grab(self):
        img = cv2.imread(self.path)
        if img is None:
            raise RuntimeError(f"cannot read {self.path}")
        return img, self.K


class RealSenseCamera(Camera):
    """Intel RealSense (the Pioneer humanoid's D455) colour stream via pyrealsense2."""
    def __init__(self, width=1280, height=720, fps=30):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        prof = self.pipe.start(cfg)
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
        self.description = f"RealSense {width}x{height}"
        for _ in range(15):  # let auto-exposure settle
            self.pipe.wait_for_frames()

    def grab(self):
        frames = self.pipe.wait_for_frames(timeout_ms=3000)
        return np.asanyarray(frames.get_color_frame().get_data()).copy(), self.K


class RosCamera(Camera):
    """Latest frame from a ROS 2 image topic (+ its camera_info). Needs rclpy."""
    def __init__(self, node, topic: str):
        from sensor_msgs.msg import CameraInfo, Image
        self.topic = topic
        self.description = f"ROS {topic}"
        self.lock = threading.Lock()
        self.img = None
        self.K = None
        info_topic = topic.rsplit("/", 1)[0] + "/camera_info"
        node.create_subscription(Image, topic, self._on_image, 2)
        node.create_subscription(CameraInfo, info_topic, self._on_info, 2)

    def _on_info(self, msg):
        self.K = np.array(msg.k, np.float64).reshape(3, 3)

    def _on_image(self, msg):
        buf = np.frombuffer(bytes(msg.data), np.uint8)
        if msg.encoding in ("bgr8", "rgb8"):
            img = buf.reshape(msg.height, msg.width, 3)
            if msg.encoding == "rgb8":
                img = img[..., ::-1]
        elif msg.encoding in ("mono8",):
            img = buf.reshape(msg.height, msg.width)
        else:
            return
        with self.lock:
            self.img = img.copy()

    def grab(self):
        t0 = time.time()
        while time.time() - t0 < 3:
            with self.lock:
                if self.img is not None and self.K is not None:
                    return self.img.copy(), self.K
            time.sleep(0.05)
        raise RuntimeError(f"no image on {self.topic} (and camera_info) within 3 s")


# ----------------------------------------------------------------------------- ROS output

class RosPublisher:
    """Publishes robot poses so the robot's own stack can start in the phone-built map."""
    def __init__(self, node, base_from_camera: np.ndarray):
        from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
        self.node = node
        self.PoseStamped, self.PoseCov = PoseStamped, PoseWithCovarianceStamped
        self.initial = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
        self.camera = node.create_publisher(PoseStamped, "/pioneer_home/camera_pose", 1)
        self.base_from_camera = base_from_camera

    def publish(self, pose: Pose) -> None:
        R = np.array(pose.R)
        t = np.array(pose.t)
        world_from_cam = np.eye(4)
        world_from_cam[:3, :3] = R.T
        world_from_cam[:3, 3] = -R.T @ t
        world_from_base = world_from_cam @ np.linalg.inv(self.base_from_camera)
        stamp = self.node.get_clock().now().to_msg()

        cam = self.PoseStamped()
        cam.header.frame_id, cam.header.stamp = "map", stamp
        _fill_pose(cam.pose, world_from_cam)
        self.camera.publish(cam)

        msg = self.PoseCov()
        msg.header.frame_id, msg.header.stamp = "map", stamp
        _fill_pose(msg.pose.pose, world_from_base)
        cov = np.zeros(36)
        cov[0] = cov[7] = 0.05 ** 2       # x, y: 5 cm
        cov[14] = 0.05 ** 2
        cov[21] = cov[28] = math.radians(2) ** 2
        cov[35] = math.radians(3) ** 2    # yaw: 3 degrees
        msg.pose.covariance = cov.tolist()
        self.initial.publish(msg)


def _model_info(path: Path) -> dict:
    """Name, height, joint count and a content id for a .robot file (the app caches by id)."""
    import hashlib
    import struct
    data = path.read_bytes()
    if data[:4] != b"PRBT":
        raise SystemExit(f"{path} is not a robot model (convert a URDF with --robot-urdf)")
    n = struct.unpack("<I", data[4:8])[0]
    m = json.loads(data[8:8 + n])
    joints = sum(1 for L in m["links"] if L.get("joint") and L["joint"]["type"] in ("revolute", "continuous"))
    return {"name": m.get("name"), "height": round(m.get("height", 0), 2), "joints": joints,
            "id": hashlib.sha1(data).hexdigest()[:12], "bytes": len(data), "source": m.get("source")}


def prepare_robot_model(urdf: Path, cache_dir: Path, forward: str = "+x", gait: Path | None = None,
                        name: str | None = None, triangles: int = 90_000) -> Path:
    """Convert a URDF once (cached by the URDF's and its meshes' content) to the app's .robot format."""
    import hashlib
    sys.path.insert(0, str(ROOT / "tools"))
    from urdf_to_app_model import convert
    h = hashlib.sha1()
    h.update(urdf.read_bytes())
    for f in sorted(urdf.parent.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".stl", ".obj", ".dae", ".glb", ".gltf", ".ply"):
            h.update(f.name.encode())
            h.update(str(f.stat().st_size).encode())
    h.update(f"{forward}|{gait}|{name}|{triangles}".encode())
    out = cache_dir / f"{urdf.stem}-{h.hexdigest()[:10]}.robot"
    if not out.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        convert(urdf, out, triangles, None, forward, gait, name, source=urdf.name)
    return out


def _read_ply_xyzrgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Binary little-endian PLY with float x y z and uchar r g b (what robot_map.py writes)."""
    data = path.read_bytes()
    end = data.index(b"end_header") + len(b"end_header")
    end = data.index(b"\n", end) + 1
    header = data[:end].decode(errors="replace").splitlines()
    n = int(next(l for l in header if l.startswith("element vertex")).split()[-1])
    props = [l.split() for l in header if l.startswith("property")]
    tmap = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
            "int": "<i4", "short": "<i2", "ushort": "<u2"}
    dtype = [(p[2], tmap[p[1]]) for p in props if p[1] != "list"]
    arr = np.frombuffer(data, dtype=dtype, count=n, offset=end)
    P = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
    names = arr.dtype.names
    if {"red", "green", "blue"} <= set(names):
        C = np.stack([arr["red"], arr["green"], arr["blue"]], 1).astype(np.float32)
    else:
        C = np.full((n, 3), 160, np.float32)
    return P, C


def _fill_pose(p, T: np.ndarray) -> None:
    from scipy.spatial.transform import Rotation
    p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)


def base_from_camera_matrix(spec: str) -> np.ndarray:
    """'x y z roll pitch yaw' of the camera mount on the robot base (metres, degrees).

    The camera's optical frame (z forward, x right, y down) is first turned into a
    body frame (x forward, y left, z up), then rotated/offset by the mount.
    """
    from scipy.spatial.transform import Rotation
    x, y, z, roll, pitch, yaw = map(float, spec.split())
    body_from_optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix() @ body_from_optical
    T[:3, 3] = [x, y, z]
    return T


# ----------------------------------------------------------------------------- state

class RobotLink:
    def __init__(self, maps: Path, name: str, camera: Camera, publisher: RosPublisher | None,
                 captures: Path, robot_model: Path | None = None):
        self.maps_root = maps
        self.name = name
        self.robot_model = robot_model
        self.robot_info = _model_info(robot_model) if robot_model else None
        self.camera = camera
        self.publisher = publisher
        self.captures = captures
        self.lock = threading.Lock()
        self.active: str | None = None
        self.localizer: Localizer | None = None
        self.last: dict[str, dict] = {}
        self.job: dict = {"state": "idle"}
        ready = self.available_maps()
        if ready:
            self.activate(ready[0])

    def available_maps(self) -> list[str]:
        if not self.maps_root.exists():
            return []
        found = [p.parent.parent for p in self.maps_root.glob("*/robot/reloc_map.npz")]
        return [p.name for p in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)]

    def activate(self, name: str) -> None:
        loc = Localizer(self.maps_root / name)
        with self.lock:
            self.active, self.localizer = name, loc
            self.last = {}

    def status(self) -> dict:
        info = self.localizer.grid if self.localizer else None
        return {"name": self.name, "map": self.active, "maps": self.available_maps(),
                "robot": self.robot_info,
                "camera": self.camera.description, "ros": self.publisher is not None,
                "grid": info, "last": self.last, "job": self.job}

    def localize(self, source: str, image: np.ndarray | None = None, K: np.ndarray | None = None) -> dict:
        if self.localizer is None:
            return {"ok": False, "reason": "no map loaded yet: send a scan from the app first"}
        if image is None:
            image, K = self.camera.grab()
        t0 = time.time()
        pose = self.localizer.localize(image, K)
        out = pose.to_dict()
        out.update(source=source, map=self.active, seconds=round(time.time() - t0, 2), time=time.time())
        if pose.ok:
            self.last[source] = out
            if source == "robot" and self.publisher:
                self.publisher.publish(pose)
        return out

    def occupancy_png(self, style: str = "color") -> bytes:
        if not self.active:
            raise FileNotFoundError("no map")
        grid = cv2.imread(str(self.maps_root / self.active / "robot" / "occupancy.pgm"), cv2.IMREAD_GRAYSCALE)
        if grid is None:
            raise FileNotFoundError("map has no occupancy.pgm")
        lut = np.zeros((256, 3), np.uint8)
        if style == "mono":                # paper: free = off-white, occupied = ink, unknown = grey
            lut[:] = (178, 184, 186)
            lut[254] = (232, 238, 240)
            lut[0] = (36, 38, 40)
        else:                              # WATonomous night palette
            lut[:] = (48, 40, 30)          # unknown (BGR)
            lut[254] = (232, 228, 222)     # free
            lut[0] = (150, 94, 30)         # occupied: logo blue
        img = lut[grid]
        k = max(1, int(np.ceil(800 / max(grid.shape))))
        img = cv2.resize(img, (grid.shape[1] * k, grid.shape[0] * k), interpolation=cv2.INTER_NEAREST)
        return cv2.imencode(".png", img)[1].tobytes()

    def topdown_png(self, px_m: float = 0.01, ceiling_cut: float = 2.0) -> bytes:
        """Greyscale "aerial photo" of the scan: the highest point below the ceiling in each 1 cm
        cell, shaded by its colour and height. Same extent as the occupancy grid, so the app can
        draw poses with the grid's pixel coordinates."""
        if not self.active or not self.localizer or not self.localizer.grid:
            raise FileNotFoundError("no map")
        robot = self.maps_root / self.active / "robot"
        cache = robot / "topdown.png"
        if cache.exists():
            return cache.read_bytes()
        ply = robot / "map_metric.ply"
        if not ply.exists():
            raise FileNotFoundError("map has no map_metric.ply")
        P, C = _read_ply_xyzrgb(ply)
        g = self.localizer.grid
        W = int(round(g["width"] * g["resolution"] / px_m))
        H = int(round(g["height"] * g["resolution"] / px_m))
        keep = P[:, 2] < ceiling_cut
        P, C = P[keep], C[keep]
        col = ((P[:, 0] - g["origin"][0]) / px_m).astype(int)
        row = H - 1 - ((P[:, 1] - g["origin"][1]) / px_m).astype(int)
        ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        col, row, z, C = col[ok], row[ok], P[ok, 2], C[ok]
        order = np.argsort(z)                      # later (higher) points win
        lum = (0.299 * C[:, 0] + 0.587 * C[:, 1] + 0.114 * C[:, 2]).astype(np.float32)
        img = np.full((H, W), np.nan, np.float32)
        hgt = np.zeros((H, W), np.float32)
        img[row[order], col[order]] = lum[order]
        hgt[row[order], col[order]] = z[order]
        seen = ~np.isnan(img)
        img = np.where(seen, img, 0)
        # soften gaps between points, then shade: tall things a little darker and with a drop shadow
        k = np.ones((5, 5), np.float32)
        cnt = cv2.filter2D(seen.astype(np.float32), -1, k)
        img = np.where(seen, img, cv2.filter2D(img, -1, k) / np.maximum(cnt, 1e-6))
        hgt = np.where(seen, hgt, cv2.filter2D(hgt, -1, k) / np.maximum(cnt, 1e-6))
        covered = cnt > 0
        shade = 1 - 0.18 * np.clip(hgt / 1.2, 0, 1)
        shadow = cv2.GaussianBlur(np.roll(np.roll(hgt, 3, 0), 3, 1), (0, 0), 3)
        shade *= 1 - 0.25 * np.clip((shadow - hgt) / 0.6, 0, 1)
        out = img * shade
        lo, hi = np.percentile(out[covered], [2, 98]) if covered.any() else (0, 255)
        out = np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1) ** 1.15 * 225 + 18   # punchy, like an aerial photo
        edges = cv2.morphologyEx(covered.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        out[edges] *= 0.55                                                        # crisp outline of the scanned area
        paper = np.full((H, W), 214, np.uint8)
        grey = np.where(covered, out, paper).astype(np.uint8)
        grain = np.random.default_rng(0).normal(0, 3.0, grey.shape)
        grey = np.clip(grey + grain, 0, 255).astype(np.uint8)
        png = cv2.imencode(".png", cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR))[1].tobytes()
        cache.write_bytes(png)
        return png

    def cloud_ply(self, max_points: int = 400_000, wall_cut: float = 1.4) -> bytes:
        """Binary PLY (float x y z, uchar r g b) of the scan for the app's 3D map. Everything above
        `wall_cut` metres is dropped, so the room reads like a dollhouse: low walls, the floor and
        furniture in full view, and the 1.6 m robot standing taller than its walls."""
        if not self.active:
            raise FileNotFoundError("no map")
        robot = self.maps_root / self.active / "robot"
        cache = robot / "cloud_app.ply"
        if cache.exists():
            return cache.read_bytes()
        ply = robot / "map_metric.ply"
        if not ply.exists():
            raise FileNotFoundError("map has no map_metric.ply")
        P, C = _read_ply_xyzrgb(ply)
        keep = P[:, 2] < wall_cut
        P, C = P[keep], C[keep]
        if len(P) > max_points:
            idx = np.random.default_rng(0).choice(len(P), max_points, replace=False)
            P, C = P[np.sort(idx)], C[np.sort(idx)]
        arr = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        arr["x"], arr["y"], arr["z"] = P[:, 0], P[:, 1], P[:, 2]
        arr["red"], arr["green"], arr["blue"] = np.clip(C, 0, 255).astype(np.uint8).T
        header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        data = header.encode() + arr.tobytes()
        cache.write_bytes(data)
        return data

    def scan_path(self) -> dict:
        """Keyframe camera centres in capture order (map frame), for drawing the scan route."""
        if not self.active:
            raise FileNotFoundError("no map")
        cams = json.loads((self.maps_root / self.active / "robot" / "cameras_metric.json").read_text())
        return {"map": self.active, "points": [[round(float(v), 3) for v in c["C"]] for c in cams]}

    def ingest_upload(self, name: str, zip_path: Path) -> None:
        """Unzip a capture, reconstruct it and build its relocalization map, in the background."""
        def run():
            try:
                dest = self.captures / name
                self.job = {"state": "unzipping", "name": name}
                safe_extract(zip_path, dest)
                self.job = {"state": "reconstructing", "name": name, "started": time.time()}
                subprocess.run([sys.executable, str(ROOT / "run.py"), str(dest), "--arkit",
                                "--out", str(self.maps_root)], check=True, cwd=ROOT)
                out_dir = self.maps_root / dest.name
                (out_dir / "source.txt").write_text(str(dest))
                self.job = {"state": "building relocalization map", "name": name}
                from .reloc_map import build
                build(out_dir)
                self.activate(out_dir.name)
                self.job = {"state": "done", "name": name, "map": out_dir.name}
            except Exception as e:  # report to the app instead of dying silently
                self.job = {"state": "failed", "name": name, "error": str(e)}
        threading.Thread(target=run, daemon=True).start()


# ----------------------------------------------------------------------------- HTTP

def make_handler(link: RobotLink):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PioneerRobotLink/1.0"

        def log_message(self, fmt, *args):
            sys.stderr.write("[robot_link] " + fmt % args + "\n")

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n > 0 else b""

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/status":
                    return self._json(200, link.status())
                if u.path == "/maps":
                    return self._json(200, {"maps": link.available_maps(), "active": link.active})
                if u.path == "/map/info":
                    return self._json(200, {"map": link.active, "grid": link.localizer.grid if link.localizer else None})
                if u.path == "/map/occupancy.png":
                    return self._send(200, link.occupancy_png(parse_qs(u.query).get("style", ["color"])[0]), "image/png")
                if u.path == "/robot/model":
                    if not link.robot_model:
                        return self._json(404, {"ok": False, "reason": "no robot model configured: the app uses its Pioneer"})
                    return self._send(200, link.robot_model.read_bytes(), "application/octet-stream")
                if u.path == "/map/cloud.ply":
                    return self._send(200, link.cloud_ply(), "application/octet-stream")
                if u.path == "/map/path.json":
                    return self._json(200, link.scan_path())
                if u.path == "/map/topdown.png":
                    try:
                        return self._send(200, link.topdown_png(), "image/png")
                    except FileNotFoundError:  # no point cloud: fall back to the plain grid
                        return self._send(200, link.occupancy_png("mono"), "image/png")
            except FileNotFoundError as e:
                return self._json(404, {"ok": False, "reason": str(e)})
            self._json(404, {"ok": False, "reason": "unknown path"})

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/maps/activate":
                    name = q.get("name", [""])[0]
                    if name not in link.available_maps():
                        return self._json(404, {"ok": False, "reason": f"no map named {name!r}"})
                    link.activate(name)
                    return self._json(200, {"ok": True, "map": name})
                if u.path == "/localize":
                    source = q.get("source", ["robot"])[0]
                    body = self._body()
                    image = K = None
                    if body:
                        image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
                        if image is None:
                            return self._json(400, {"ok": False, "reason": "body is not a JPEG/PNG image"})
                        h = self.headers
                        try:
                            fx, fy = float(h["X-Fx"]), float(h["X-Fy"])
                            cx, cy = float(h.get("X-Cx", image.shape[1] / 2)), float(h.get("X-Cy", image.shape[0] / 2))
                        except (TypeError, KeyError, ValueError):
                            return self._json(400, {"ok": False, "reason": "send intrinsics as X-Fx, X-Fy, X-Cx, X-Cy headers"})
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
                    return self._json(200, link.localize(source, image, K))
                if u.path == "/upload":
                    name = SAFE_NAME.sub("_", self.headers.get("X-Capture-Name") or "capture").strip("._") or "capture"
                    link.captures.mkdir(parents=True, exist_ok=True)
                    zp = link.captures / f"{name}.zip"
                    zp.write_bytes(self._body())
                    link.ingest_upload(name, zp)
                    return self._json(200, {"ok": True, "name": name, "job": "started"})
            except RuntimeError as e:  # camera problems
                return self._json(503, {"ok": False, "reason": str(e)})
            self._json(404, {"ok": False, "reason": "unknown path"})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, default=ROOT / "out", help="run.py output root")
    ap.add_argument("--captures", type=Path, default=ROOT / "captures", help="where uploads are unzipped")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--name", default=None, help="robot name in the app (default: the model's, else Pioneer)")
    ap.add_argument("--robot-urdf", type=Path, default=None, help="your humanoid's URDF: the app draws it")
    ap.add_argument("--robot-forward", default="+x", choices=["+x", "-x", "+y", "-y"],
                    help="URDF axis the robot faces (REP-103: +x; the Pioneer: -y)")
    ap.add_argument("--robot-gait", type=Path, default=None,
                    help="JSON naming the hip/knee/ankle/shoulder joints if they aren't auto-detected")
    ap.add_argument("--robot-model", type=Path, default=None, help="an already converted .robot file")
    ap.add_argument("--camera", default="none",
                    help="none | realsense | ros:/camera/color/image_raw | file:img.jpg")
    ap.add_argument("--camera-K", default=None, help="fx,fy,cx,cy (for --camera file:...)")
    ap.add_argument("--ros", action="store_true", help="publish /initialpose and /pioneer_home/camera_pose")
    ap.add_argument("--base-from-camera", default="0.08 0 1.45 0 0 0",
                    help="camera mount on the robot base: 'x y z roll pitch yaw' (m, deg)")
    a = ap.parse_args()

    node = None
    if a.ros or a.camera.startswith("ros:"):
        import rclpy
        rclpy.init()
        node = rclpy.create_node("pioneer_home_robot_link")
        threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    if a.camera == "none":
        cam: Camera = Camera()
    elif a.camera == "realsense":
        cam = RealSenseCamera()
    elif a.camera.startswith("ros:"):
        cam = RosCamera(node, a.camera[4:])
    elif a.camera.startswith("file:"):
        fx, fy, cx, cy = map(float, (a.camera_K or "").split(","))
        cam = FileCamera(a.camera[5:], np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
    else:
        raise SystemExit(f"unknown --camera {a.camera}")

    pub = RosPublisher(node, base_from_camera_matrix(a.base_from_camera)) if a.ros else None
    model = a.robot_model
    if a.robot_urdf:
        model = prepare_robot_model(a.robot_urdf, a.maps / "_robots", a.robot_forward, a.robot_gait, a.name)
    name = a.name or (_model_info(model)["name"] if model else None) or "Pioneer"
    link = RobotLink(a.maps, name, cam, pub, a.captures, model)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(link))
    print(f"[robot_link] {name} on http://{a.host}:{a.port}  map={link.active}  camera={cam.description}"
          f"  model={'built-in Pioneer' if not model else link.robot_info['name'] + ' (' + str(link.robot_info['joints']) + ' joints)'}"
          f"  ros={'on' if pub else 'off'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
