"""ROS-free helpers (unit-tested without ROS): load the pipeline's robot/ outputs."""
import numpy as np


def load_voxels(path):
    """voxels.npz -> (N, 3) float32 voxel centres in metres (map frame, Z up)."""
    d = np.load(path)
    idx = d["indices"].astype(np.float64)
    vs = float(d["voxel_size"])
    return (d["origin"] + (idx + 0.5) * vs).astype(np.float32), vs


def pointcloud2_fields_and_bytes(points):
    """Packed x,y,z float32 layout for sensor_msgs/PointCloud2 (12-byte points)."""
    pts = np.ascontiguousarray(points, dtype=np.float32)
    return [("x", 0), ("y", 4), ("z", 8)], 12, pts.tobytes()


def read_mesh_ply(path):
    """Binary PLY with xyz + rgb vertices and triangle faces -> (V, F, C)."""
    data = open(path, "rb").read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    head = data[:end].decode()
    nv = int(head.split("element vertex ")[1].split()[0])
    nf = int(head.split("element face ")[1].split()[0]) if "element face" in head else 0
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v = np.frombuffer(data, vdt, count=nv, offset=end)
    fdt = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    f = np.frombuffer(data, fdt, count=nf, offset=end + nv * vdt.itemsize)
    V = np.stack([v["x"], v["y"], v["z"]], 1)
    C = np.stack([v["r"], v["g"], v["b"]], 1)
    F = np.stack([f["a"], f["b"], f["c"]], 1)
    return V, F, C


def decimate_faces(F, max_faces):
    """Keep at most max_faces triangles (RViz chokes on millions)."""
    if len(F) <= max_faces:
        return F
    step = int(np.ceil(len(F) / max_faces))
    return F[::step]


def parse_map_yaml(text):
    """Minimal parser for the nav2 map YAML this pipeline writes (no PyYAML needed)."""
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.startswith("["):
            out[k.strip()] = [float(x) for x in v.strip("[]").split(",")]
        else:
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out
