"""Shared helpers for reading PhoneScan captures (see ios/README.md for the format)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ARKit / nerfstudio cameras use OpenGL axes (x right, y up, looking down -z).
# OpenCV / COLMAP cameras use x right, y down, looking down +z.
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class Frame:
    image: Path
    depth: Path | None
    confidence: Path | None
    c2w_gl: np.ndarray  # 4x4 camera-to-world, OpenGL camera axes
    K: np.ndarray       # 3x3 intrinsics for the full-resolution image
    width: int
    height: int
    timestamp: float

    @property
    def c2w_cv(self) -> np.ndarray:
        return self.c2w_gl @ GL_TO_CV

    @property
    def w2c_cv(self) -> np.ndarray:
        return np.linalg.inv(self.c2w_cv)


def load_capture(capture_dir: str | Path) -> list[Frame]:
    capture_dir = Path(capture_dir)
    meta_path = capture_dir / "transforms.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        records = meta["frames"]
    else:  # capture was interrupted before transforms.json was written
        meta = {}
        records = [json.loads(l) for l in (capture_dir / "frames.jsonl").read_text().splitlines() if l.strip()]

    frames = []
    for r in records:
        fx = r.get("fl_x", meta.get("fl_x"))
        fy = r.get("fl_y", meta.get("fl_y"))
        cx = r.get("cx", meta.get("cx"))
        cy = r.get("cy", meta.get("cy"))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        frames.append(Frame(
            image=capture_dir / r["file_path"],
            depth=capture_dir / r["depth_file_path"] if r.get("depth_file_path") else None,
            confidence=capture_dir / r["confidence_file_path"] if r.get("confidence_file_path") else None,
            c2w_gl=np.array(r["transform_matrix"], dtype=np.float64),
            K=K,
            width=int(r.get("w", meta.get("w"))),
            height=int(r.get("h", meta.get("h"))),
            timestamp=float(r.get("timestamp", 0.0)),
        ))
    frames.sort(key=lambda f: f.image.name)
    return frames


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> unit quaternion (w, x, y, z), COLMAP convention, w >= 0."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def write_ply_points(path: str | Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    """Binary little-endian PLY point cloud."""
    xyz = np.asarray(xyz, dtype=np.float32)
    n = len(xyz)
    props = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if rgb is not None:
        props += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    arr = np.empty(n, dtype=props)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.uint8)
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(arr.tobytes())


def read_ply_points(path: str | Path) -> np.ndarray:
    """Reads x, y, z from a binary PLY written by write_ply_points (for tests and quick checks)."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode().splitlines()
    n = int(next(l for l in header if l.startswith("element vertex")).split()[-1])
    has_rgb = any("red" in l for l in header)
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + ([("r", "u1"), ("g", "u1"), ("b", "u1")] if has_rgb else [])
    arr = np.frombuffer(data[end:], dtype=dtype, count=n)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1)
