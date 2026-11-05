#!/usr/bin/env python3
"""Convert a humanoid's URDF into the Pioneer Home app's articulated robot model.

Works for any URDF with mesh visuals (STL / OBJ / DAE / GLB). The bundled model is the Pioneer
(WATonomous/pioneer_humanoid); other teams point robot_link at their own URDF and the app shows
their robot instead. Keeps every link and joint (so the app can move the legs, arms and fingers),
decimates the CAD meshes to a size a phone renders smoothly, and merges each link's visuals by colour.

The app animates walking with a few named roles (hip pitch, knee, ankle pitch, shoulder pitch,
finger curl). They are found from common joint names (`left_hip_pitch_joint`, `Knee_L`, `r_mcp`, ...);
pass --gait roles.json to set them by hand. --forward says which URDF axis the robot faces:
REP-103 robots face +x (the default); the Pioneer faces -y.

    python3 tools/urdf_to_app_model.py my_robot.urdf my_robot.robot --name "Atlas" --forward +x

    python3 tools/urdf_to_app_model.py ../pioneer_humanoid/assets/pioneer_whole_body_humanoid/pioneer_whole_body_humanoid.urdf \\
        ios/PhoneScan/Pioneer.robot --triangles 90000 --glb docs/pioneer_preview.glb

Output format (little-endian): b"PRBT", uint32 JSON length, JSON, then one binary buffer.
JSON: {"links": [{"name", "parent", "joint": {...} | null, "parts": [{"color": [r,g,b,a],
"v": [offset, count], "i": [offset, count]}]}], "foot_z": float, "height": float}
Vertices are float32 xyz + float32 normal xyz in the link frame (metres); indices uint32.
Joint: {"name", "type", "xyz", "rpy", "axis", "lower", "upper"} (URDF conventions).
Also: "name" (robot), "forward" [x, y] (facing direction in the URDF frame), "gait" (roles).
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from types import SimpleNamespace
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


FORWARD = {"+x": [1, 0], "-x": [-1, 0], "+y": [0, 1], "-y": [0, -1]}
ROLES = ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch", "elbow")


def side_of(name: str) -> str | None:
    n = name.lower()
    if re.search(r"(^|[_\-.])(l|left)([_\-.]|$)", n) or n.startswith("left") or n.endswith("_l") or n.startswith("l_"):
        return "left"
    if re.search(r"(^|[_\-.])(r|right)([_\-.]|$)", n) or n.startswith("right") or n.endswith("_r") or n.startswith("r_"):
        return "right"
    return None


def role_of(name: str) -> str | None:
    n = name.lower()
    if "hip" in n and ("pitch" in n or re.search(r"hip_f(_|$)", n) or "flex" in n):
        return "hip_pitch"
    if "knee" in n:
        return "knee"
    if "ankle" in n and ("pitch" in n or re.search(r"ankle_p(_|$)", n)):
        return "ankle_pitch"
    if "shoulder" in n and "pitch" in n:
        return "shoulder_pitch"
    if "elbow" in n:
        return "elbow"
    return None


def detect_gait(joint_names: list[str]) -> dict:
    """Map walking roles to joint names, plus the finger joints to curl when standing."""
    gait: dict = {r: {} for r in ROLES}
    for j in joint_names:
        r, s = role_of(j), side_of(j)
        if r and s and s not in gait[r]:
            gait[r][s] = j
    finger = re.compile(r"(mcp|pip|dip|finger|index|middle|ring|pinky|little|thumb_(flex|ip))")
    gait["curl"] = [j for j in joint_names if finger.search(j.lower()) and not re.search(r"(abd|yaw|roll|_c$)", j.lower())]
    return gait


def resolve_mesh(ref: str, urdf_dir: Path) -> Path:
    """Find a mesh the way ROS tools do, plus the usual breakage: `package://pkg/x.stl`, paths
    written for a URDF that has since moved (`../meshes/x.STL`), and different case."""
    ref = ref.replace("file://", "")
    cands = []
    if ref.startswith("package://"):
        rest = ref[len("package://"):].split("/", 1)[-1]
        for up in [urdf_dir, *urdf_dir.parents][:5]:
            cands.append(up / rest)
    else:
        cands.append(Path(ref) if Path(ref).is_absolute() else urdf_dir / ref)
    for c in cands:
        if c.exists():
            return c
    name = Path(ref).name.lower()
    for root in (urdf_dir, urdf_dir.parent):
        for f in root.rglob("*"):
            if f.name.lower() == name:
                return f
    raise FileNotFoundError(f"mesh {ref!r} not found near {urdf_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urdf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--triangles", type=int, default=90_000, help="total triangle budget")
    ap.add_argument("--glb", type=Path, default=None, help="also write a posed preview .glb")
    ap.add_argument("--name", default=None, help="robot name shown in the app (default: URDF robot name)")
    ap.add_argument("--forward", default="+x", choices=sorted(FORWARD), help="URDF axis the robot faces")
    ap.add_argument("--gait", type=Path, default=None, help="JSON overriding detected roles")
    ap.add_argument("--source", default=None, help="credit line stored in the model")
    a = ap.parse_args()
    convert(a.urdf, a.out, a.triangles, a.glb, a.forward, a.gait, a.name, a.source)


def convert(urdf: Path, out: Path, triangles: int = 90_000, glb: Path | None = None, forward: str = "+x",
            gait_override: Path | dict | None = None, name: str | None = None, source: str | None = None,
            verbose: bool = True) -> dict:
    a = SimpleNamespace(urdf=Path(urdf), out=Path(out), triangles=triangles, glb=glb)
    robot_name = name  # `name` is reused by the link loops below
    log = print if verbose else (lambda *_, **__: None)

    # ROS's URDF parser tolerates things strict XML doesn't (e.g. "--" inside comments), so drop comments first
    text = re.sub(r"<!--.*?-->", "", a.urdf.read_text(errors="replace"), flags=re.S)
    root = ET.fromstring(text)
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
            m = trimesh.load(resolve_mesh(mesh_el.get("filename"), base), force="mesh", process=True)
            scale = [float(s) for s in mesh_el.get("scale", "1 1 1").split()]
            m.apply_scale(scale)
            m.apply_transform(T(*origin_of(vis)))
            c = vis.find("material/color")
            rgba = [float(v) for v in c.get("rgba").split()] if c is not None else [0.8, 0.8, 0.8, 1.0]
            raw.append((name, tuple(round(v, 3) for v in rgba), m))

    total = sum(len(m.faces) for _, _, m in raw)
    keep = min(1.0, a.triangles / max(total, 1))
    log(f"[urdf] {len(links)} links, {len(raw)} meshes, {total:,} triangles -> keep {keep:.1%}")

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
        log(f"[urdf] clustered at {hi * 1000:.1f} mm cells: {n_now:,} -> "
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
    log(f"[urdf] zero pose: feet at z={foot_z:.3f} m, height {height:.2f} m, "
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
    joint_names = [L["joint"]["name"] for L in links.values() if L["joint"] and L["joint"]["type"] in ("revolute", "continuous")]
    gait = detect_gait(joint_names)
    if gait_override:
        over = gait_override if isinstance(gait_override, dict) else json.loads(Path(gait_override).read_text())
        gait.update(over)
    missing = [r for r in ROLES[:3] if len(gait[r]) < 2]
    log("[urdf] walking roles: " + ", ".join(f"{r}={gait[r].get('left', '?')}/{gait[r].get('right', '?')}" for r in ROLES)
        + f"; {len(gait['curl'])} finger joints" + (f"  (missing {missing}: legs won't animate; pass --gait)" if missing else ""))
    manifest = {"name": robot_name or root.get("name") or a.urdf.stem,
                "source": source or a.urdf.name,
                "forward": FORWARD[forward], "gait": gait,
                "links": [links[n] for n in order], "foot_z": foot_z, "height": height,
                "vertex_stride": 24, "index_offset": len(vbuf)}
    js = json.dumps(manifest, separators=(",", ":")).encode()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(b"PRBT" + struct.pack("<I", len(js)) + js + bytes(vbuf) + bytes(ibuf))
    log(f"[urdf] wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")

    if a.glb:
        scene = trimesh.Scene()
        for (name, rgba), meshes in merged.items():
            m = trimesh.util.concatenate(meshes)
            m.apply_transform(fk(name))
            m.visual.face_colors = (np.array(rgba) * 255).astype(np.uint8)
            scene.add_geometry(m)
        a.glb.parent.mkdir(parents=True, exist_ok=True)
        scene.export(a.glb)
        log(f"[urdf] wrote preview {a.glb}")
    return {"name": manifest["name"], "height": height, "links": len(links), "joints": len(joint_names),
            "triangles": icount // 3, "gait": gait}


if __name__ == "__main__":
    main()
