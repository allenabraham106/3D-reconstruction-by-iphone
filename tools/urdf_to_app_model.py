#!/usr/bin/env python3
"""Convert the Pioneer URDF (WATonomous/pioneer_humanoid) into the iOS app's articulated robot model.

Keeps every link and joint (so the app can move the legs, arms and fingers), decimates the CAD
meshes to a size a phone renders smoothly, and merges each link's visuals by colour.

    python3 tools/urdf_to_app_model.py ../pioneer_humanoid/assets/pioneer_whole_body_humanoid/pioneer_whole_body_humanoid.urdf \\
        ios/PhoneScan/Pioneer.robot --triangles 90000 --glb docs/pioneer_preview.glb

Output format (little-endian): b"PRBT", uint32 JSON length, JSON, then one binary buffer.
JSON: {"links": [{"name", "parent", "joint": {...} | null, "parts": [{"color": [r,g,b,a],
"v": [offset, count], "i": [offset, count]}]}], "foot_z": float, "height": float}
Vertices are float32 xyz + float32 normal xyz in the link frame (metres); indices uint32.
Joint: {"name", "type", "xyz", "rpy", "axis", "lower", "upper"} (URDF conventions).
"""
from __future__ import annotations

import argparse
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

try:
    import fast_simplification
except ImportError:  # pragma: no cover
    fast_simplification = None


def rpy_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def origin_of(el) -> tuple[list[float], list[float]]:
    o = el.find("origin") if el is not None else None
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return xyz, rpy


def T(xyz, rpy) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = rpy_matrix(rpy)
    M[:3, 3] = xyz
    return M


def cluster(m: trimesh.Trimesh, cell: float) -> trimesh.Trimesh:
    """Vertex clustering: snap vertices to a `cell`-metre grid, merge, drop collapsed faces.
    Robust on CAD triangle soup where edge-collapse simplifiers stall."""
    if cell <= 0 or len(m.faces) == 0:
        return m
    key = np.floor(m.vertices / cell).astype(np.int64)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inv, m.vertices)
    verts = sums / counts[:, None]
    faces = inv[m.faces]
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = np.unique(np.sort(faces[ok], axis=1), axis=0, return_index=True)[1]
    faces = inv[m.faces][ok][np.sort(faces)]
    return trimesh.Trimesh(verts, faces, process=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urdf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--triangles", type=int, default=90_000, help="total triangle budget")
    ap.add_argument("--glb", type=Path, default=None, help="also write a posed preview .glb")
    a = ap.parse_args()

    root = ET.parse(a.urdf).getroot()
    base = a.urdf.parent

    # ---- load every visual, in its link frame
    links = {}
    raw = []  # (link, color, mesh)
    for link in root.findall("link"):
        name = link.get("name")
        links[name] = {"name": name, "parent": None, "joint": None, "parts": []}
        for vis in link.findall("visual"):
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is None:
                continue
            m = trimesh.load(base / mesh_el.get("filename"), force="mesh", process=True)
            scale = [float(s) for s in mesh_el.get("scale", "1 1 1").split()]
            m.apply_scale(scale)
            m.apply_transform(T(*origin_of(vis)))
            c = vis.find("material/color")
            rgba = [float(v) for v in c.get("rgba").split()] if c is not None else [0.8, 0.8, 0.8, 1.0]
            raw.append((name, tuple(round(v, 3) for v in rgba), m))

    total = sum(len(m.faces) for _, _, m in raw)
    keep = min(1.0, a.triangles / max(total, 1))
    print(f"[urdf] {len(links)} links, {len(raw)} meshes, {total:,} triangles -> keep {keep:.1%}")

    # ---- decimate, then merge per (link, colour)
    merged: dict[tuple[str, tuple], list[trimesh.Trimesh]] = {}
    for name, rgba, m in raw:
        if keep < 1.0 and fast_simplification is not None and len(m.faces) > 60:
            target = max(0.0, 1.0 - max(keep, 40 / len(m.faces)))
            v, f = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int32),
                                                target_reduction=target)
            m = trimesh.Trimesh(v, f, process=True)
        merged.setdefault((name, rgba), []).append(m)

    # second pass: pick the coarsest-needed clustering grid that fits the triangle budget
    groups = {k: trimesh.util.concatenate(v) for k, v in merged.items()}
    n_now = sum(len(m.faces) for m in groups.values())
    if n_now > a.triangles:
        lo, hi = 0.0005, 0.03
        for _ in range(14):
            mid = (lo + hi) / 2
            n = sum(len(cluster(m, mid).faces) for m in groups.values())
            lo, hi = (lo, mid) if n <= a.triangles else (mid, hi)
        groups = {k: cluster(m, hi) for k, m in groups.items()}
        print(f"[urdf] clustered at {hi * 1000:.1f} mm cells: {n_now:,} -> "
              f"{sum(len(m.faces) for m in groups.values()):,} triangles")
    merged = {k: [m] for k, m in groups.items()}

    vbuf, ibuf = bytearray(), bytearray()
    vcount = icount = 0
    for (name, rgba), meshes in merged.items():
        m = trimesh.util.concatenate(meshes)
        m.merge_vertices()
        # split normals at CAD creases (> 35 deg) so flat panels stay flat and edges stay crisp
        m = trimesh.graph.smooth_shade(m, angle=np.radians(35))
        verts = np.hstack([m.vertices, m.vertex_normals]).astype("<f4")
        faces = m.faces.astype("<u4").ravel()
        links[name]["parts"].append({"color": list(rgba), "v": [vcount, len(verts)], "i": [icount, len(faces)]})
        vbuf += verts.tobytes()
        ibuf += faces.tobytes()
        vcount += len(verts)
        icount += len(faces)

    # ---- joints
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        parent = j.find("parent").get("link")
        xyz, rpy = origin_of(j)
        ax = j.find("axis")
        lim = j.find("limit")
        links[child]["parent"] = parent
        links[child]["joint"] = {
            "name": j.get("name"), "type": j.get("type"), "xyz": xyz, "rpy": rpy,
            "axis": [float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()],
            "lower": float(lim.get("lower", 0)) if lim is not None else 0.0,
            "upper": float(lim.get("upper", 0)) if lim is not None else 0.0,
        }

    # ---- forward kinematics at the zero pose: where are the feet, how tall is it
    world = {}
    def fk(name):
        if name in world:
            return world[name]
        L = links[name]
        M = np.eye(4) if L["parent"] is None else fk(L["parent"]) @ T(L["joint"]["xyz"], L["joint"]["rpy"])
        world[name] = M
        return M
    posed = []
    for (name, _), meshes in merged.items():
        for m in meshes:
            mm = m.copy()
            mm.apply_transform(fk(name))
            posed.append(mm)
    allv = np.vstack([m.vertices for m in posed])
    foot_z, height = float(allv[:, 2].min()), float(allv[:, 2].max() - allv[:, 2].min())
    print(f"[urdf] zero pose: feet at z={foot_z:.3f} m, height {height:.2f} m, "
          f"{icount // 3:,} triangles, {vcount:,} vertices")

    order = [n for n in links if links[n]["parent"] is None]
    seen = set(order)
    i = 0
    while i < len(order):  # parents before children
        for n, L in links.items():
            if L["parent"] == order[i] and n not in seen:
                order.append(n)
                seen.add(n)
        i += 1
    manifest = {"source": "WATonomous/pioneer_humanoid " + a.urdf.name + " (Apache-2.0)",
                "links": [links[n] for n in order], "foot_z": foot_z, "height": height,
                "vertex_stride": 24, "index_offset": len(vbuf)}
    js = json.dumps(manifest, separators=(",", ":")).encode()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(b"PRBT" + struct.pack("<I", len(js)) + js + bytes(vbuf) + bytes(ibuf))
    print(f"[urdf] wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")

    if a.glb:
        scene = trimesh.Scene()
        for (name, rgba), meshes in merged.items():
            m = trimesh.util.concatenate(meshes)
            m.apply_transform(fk(name))
            m.visual.face_colors = (np.array(rgba) * 255).astype(np.uint8)
            scene.add_geometry(m)
        a.glb.parent.mkdir(parents=True, exist_ok=True)
        scene.export(a.glb)
        print(f"[urdf] wrote preview {a.glb}")


if __name__ == "__main__":
    main()
