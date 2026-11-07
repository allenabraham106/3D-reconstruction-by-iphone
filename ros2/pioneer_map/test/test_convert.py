"""ROS-free tests of the map loading (run: python -m pytest test/ with ROBOT_DIR set)."""
import os

import numpy as np
import pytest

from pioneer_map.convert import (decimate_faces, load_voxels, parse_map_yaml,
                                 pointcloud2_fields_and_bytes, read_mesh_ply)

ROBOT_DIR = os.environ.get("ROBOT_DIR", "")
needs_scan = pytest.mark.skipif(not os.path.isdir(ROBOT_DIR), reason="set ROBOT_DIR to a scan's robot/")


@needs_scan
def test_voxels_are_metric_and_above_floor():
    pts, vs = load_voxels(os.path.join(ROBOT_DIR, "voxels.npz"))
    assert pts.dtype == np.float32 and pts.shape[1] == 3 and len(pts) > 1000
    assert abs(vs - 0.05) < 1e-9
    assert np.percentile(pts[:, 2], 1) > -0.2           # floor at ~0, reflections removed
    assert np.ptp(pts[:, :2], 0).max() < 100            # metres, not SfM units


def test_pointcloud2_layout():
    pts = np.arange(12, dtype=np.float32).reshape(4, 3)
    fields, step, data = pointcloud2_fields_and_bytes(pts)
    assert [f for f, _ in fields] == ["x", "y", "z"] and step == 12 and len(data) == 48
    assert np.allclose(np.frombuffer(data, np.float32).reshape(4, 3), pts)


@needs_scan
def test_mesh_reads_and_decimates():
    V, F, C = read_mesh_ply(os.path.join(ROBOT_DIR, "mesh_metric.ply"))
    assert F.max() < len(V) and F.min() >= 0 and C.dtype == np.uint8
    assert len(decimate_faces(F, 1000)) <= 1000


@needs_scan
def test_map_yaml_has_nav2_fields():
    y = parse_map_yaml(open(os.path.join(ROBOT_DIR, "occupancy.yaml")).read())
    for k in ("image", "mode", "resolution", "origin", "negate", "occupied_thresh", "free_thresh"):
        assert k in y, k
    assert y["mode"] == "trinary" and len(y["origin"]) == 3 and y["resolution"] == 0.05
    assert os.path.exists(os.path.join(ROBOT_DIR, y["image"]))
