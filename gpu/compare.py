#!/usr/bin/env python3
"""Compare two runs of the same scan (e.g. CPU pipeline vs ViPE on a GPU).

    python gpu/compare.py out/IMG_6075 out_vipe/IMG_6075 out_vipe/IMG_6075_compare

Writes report.md (metrics side by side) and maps.png (both occupancy previews).
If run B is metric (ViPE / ARKit) and run A used the phone-height guess, the report also
states how far off that guess was.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def mesh_faces(robot: Path):
    p = robot / "mesh_metric.ply"
    if not p.exists():
        return None
    head = open(p, "rb").read(600).decode(errors="ignore")
    return int(head.split("element face ")[1].split()[0]) if "element face" in head else 0


def main(a: Path, b: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    ma = json.loads((a / "robot" / "map_meta.json").read_text())
    mb = json.loads((b / "robot" / "map_meta.json").read_text())
    rows = [
        ("scale source", ma["metric_source"], mb["metric_source"]),
        ("mesh triangles", mesh_faces(a / "robot"), mesh_faces(b / "robot")),
        ("floor cells seen (5 cm)", ma.get("free_cells_seen"), mb.get("free_cells_seen")),
        ("floor cells incl. walked", ma.get("free_cells"), mb.get("free_cells")),
        ("obstacle cells", ma.get("occupied_cells"), mb.get("occupied_cells")),
        ("scene extent (m)", ma.get("scene_extent_m"), mb.get("scene_extent_m")),
        ("camera height (m)", round(ma["camera_height_median_m"], 3), round(mb["camera_height_median_m"], 3)),
        ("glass walls", len(ma.get("glass_walls", [])), len(mb.get("glass_walls", []))),
        ("warnings", len(ma.get("warnings", [])), len(mb.get("warnings", []))),
    ]
    lines = [f"# {a.name}: `{a}` vs `{b}`", "", f"| | A | B |", "|---|---|---|"]
    lines += [f"| {k} | {va} | {vb} |" for k, va, vb in rows]
    if "phone_height" in ma["metric_source"] and "phone_height" not in mb["metric_source"]:
        assumed = ma["camera_height_median_m"]
        real = mb["camera_height_median_m"]
        lines += ["", f"**Phone-height guess:** A assumed {assumed:.2f} m; B (metric) measured "
                      f"{real:.2f} m, so A's scale is off by {(assumed / real - 1) * 100:+.1f} %."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    ia = cv2.imread(str(a / "robot" / "occupancy_preview.png"))
    ib = cv2.imread(str(b / "robot" / "occupancy_preview.png"))
    if ia is not None and ib is not None:
        h = max(ia.shape[0], ib.shape[0])
        pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 12, cv2.BORDER_CONSTANT,
                                            value=(40, 40, 40))
        cv2.imwrite(str(out / "maps.png"), np.hstack([pad(ia), pad(ib)]))
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
