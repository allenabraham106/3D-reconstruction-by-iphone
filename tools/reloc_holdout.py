#!/usr/bin/env python3
"""Can Pioneer find itself in a scan? Hold out every other keyframe, build robot_link's relocalization map from
the rest, localize the held-out ones and compare with their SfM poses. No ViPE needed (tools/reloc_test.py
compares against a ViPE run).

    python tools/reloc_holdout.py out/IMG_1429 [--features 3000 --query-features 4000 --ratio 0.85]

Prints and writes <scan>/reloc_holdout.json: localized fraction, within 25 cm / 10 deg, median position and
heading error, median inliers, map build seconds and milliseconds per query.
"""
import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_link.localize import Localizer  # noqa: E402
from robot_link.reloc_map import build  # noqa: E402


def yaw_of(R):
    f = np.asarray(R).T @ [0, 0, 1.0]
    return math.degrees(math.atan2(f[1], f[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", type=Path, help="run.py output: frames/ and robot/cameras_metric.json")
    ap.add_argument("--features", type=int, default=3000, help="SIFT features per keyframe for the map")
    ap.add_argument("--window", type=int, default=3, help="neighbouring keyframes matched when triangulating")
    ap.add_argument("--query-features", type=int, default=4000)
    ap.add_argument("--ratio", type=float, default=0.85, help="Lowe ratio when matching a query to the map")
    ap.add_argument("--min-inliers", type=int, default=25)
    a = ap.parse_args()

    cams = json.loads((a.scan / "robot" / "cameras_metric.json").read_text())
    cams.sort(key=lambda c: c["name"])
    mapped, held = cams[0::2], cams[1::2]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "robot").mkdir()
        (tmp / "robot" / "cameras_metric.json").write_text(json.dumps(mapped))
        for f in ("occupancy.yaml", "occupancy.pgm", "map_meta.json"):
            shutil.copy(a.scan / "robot" / f, tmp / "robot" / f)
        t0 = time.time()
        info = build(tmp, images=a.scan / "frames", n_features=a.features, window=a.window, verbose=False)
        build_s = time.time() - t0
        loc = Localizer(tmp, min_inliers=a.min_inliers, ratio=a.ratio)

        rows, ms = [], []
        for c in held:
            img = cv2.imread(str(a.scan / "frames" / c["name"]))
            sx, sy = img.shape[1] / c["w"], img.shape[0] / c["h"]
            K = np.array([[c["fx"] * sx, 0, img.shape[1] / 2], [0, c["fy"] * sy, img.shape[0] / 2], [0, 0, 1]])
            t1 = time.time()
            p = loc.localize(img, K, n_features=a.query_features)
            ms.append(1000 * (time.time() - t1))
            if not p.ok:
                rows.append(None)
                continue
            dpos = float(np.linalg.norm(np.array([p.x, p.y, p.z]) - np.array(c["C"])))
            dyaw = abs((p.yaw_deg - yaw_of(c["R"]) + 180) % 360 - 180)
            rows.append((dpos, dyaw, p.inliers))

    ok = [r for r in rows if r]
    res = {
        "queries": len(held), "localized": len(ok),
        "within_25cm_10deg": sum(1 for d, y, _ in ok if d < 0.25 and y < 10),
        "median_pos_err_m": round(float(np.median([r[0] for r in ok])), 3),
        "median_yaw_err_deg": round(float(np.median([r[1] for r in ok])), 2),
        "median_inliers": int(np.median([r[2] for r in ok])),
        "landmarks": info["landmarks"], "map_build_s": round(build_s, 1),
        "ms_per_query": round(float(np.median(ms)), 1),
        "params": {k: v for k, v in vars(a).items() if k != "scan"},
    }
    print(json.dumps(res))
    (a.scan / "reloc_holdout.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
