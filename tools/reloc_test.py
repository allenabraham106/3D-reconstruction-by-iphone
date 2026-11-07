#!/usr/bin/env python3
"""Can a robot localize in a scan? Relocalize held-out video frames and check the answer.

Builds robot_link's relocalization map (SIFT landmarks triangulated from the keyframes'
poses) for a CPU-pipeline scan and for the ViPE scan of the same video, then localizes video
frames that neither map was built from (odd ViPE frames; ViPE's map uses every 2nd), and
compares each answer with the true pose: the neighbouring keyframe poses, interpolated.

    python tools/reloc_test.py IMG_6075 --every 20
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def thumb(img):
    g = cv2.cvtColor(cv2.resize(img, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def interp_pose(cams_by_t, t):
    """cams_by_t: sorted list of (time index, C, R). Linear C, nearest R for heading."""
    ts = np.array([c[0] for c in cams_by_t])
    j = np.searchsorted(ts, t)
    if j <= 0 or j >= len(ts):
        return None
    t0, C0, R0 = cams_by_t[j - 1]
    t1, C1, R1 = cams_by_t[j]
    if t1 - t0 > 6:       # keyframes too far apart (CPU dropped frames here): no reliable truth
        return None
    f = (t - t0) / (t1 - t0)
    return (1 - f) * C0 + f * C1, (R0 if f < 0.5 else R1)


def heading(R):
    z = np.array(R)[2]                     # camera viewing direction in the map frame
    return np.degrees(np.arctan2(z[1], z[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan")
    ap.add_argument("--every", type=int, default=20, help="query every k-th odd ViPE frame")
    ap.add_argument("--cpu", type=Path, default=ROOT / "bench/work/cpu-5070box")
    a = ap.parse_args()
    from robot_link.localize import Localizer
    from robot_link import reloc_map

    vipe_out = ROOT / "out_vipe/default" / a.scan
    cpu_out = a.cpu / a.scan
    cap = cv2.VideoCapture(str(ROOT / "out_vipe/default/vipe/rgb" / f"{a.scan}.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    # ViPE keyframe images (every 2nd frame), for its relocalization map
    vcams = json.loads((vipe_out / "robot/cameras_metric.json").read_text())
    img_dir = vipe_out / "frames"
    img_dir.mkdir(exist_ok=True)
    for c in vcams:
        p = img_dir / c["name"]
        if not p.exists():
            cv2.imwrite(str(p), frames[int(Path(c["name"]).stem.split("_")[-1])], [cv2.IMWRITE_JPEG_QUALITY, 95])
    results = {}
    for label, out, cams_file in [("cpu", cpu_out, cpu_out / "robot/cameras_metric.json"),
                                  ("vipe", vipe_out, vipe_out / "robot/cameras_metric.json")]:
        if not (out / "robot/reloc_map.npz").exists():
            sys.argv = ["reloc_map", str(out)]
            reloc_map.main()
        cams = json.loads(cams_file.read_text())
        # time index of each keyframe on the ViPE frame clock
        if label == "vipe":
            tix = [int(Path(c["name"]).stem.split("_")[-1]) for c in cams]
        else:
            th = np.stack([thumb(f) for f in frames]).reshape(len(frames), -1)
            tix = [int(np.argmax(th @ thumb(cv2.imread(str(out / "frames" / c["name"]))).ravel())) for c in cams]
        by_t = sorted(((t, np.array(c["C"]), c["R"]) for t, c in zip(tix, cams)), key=lambda r: r[0])
        fx = float(cams[0]["fx"]) * frames[0].shape[1] / float(cams[0]["w"])
        K = (fx, fx, frames[0].shape[1] / 2, frames[0].shape[0] / 2)
        m = Localizer(out)
        rows = []
        for q in range(1, len(frames), 2)[:: a.every]:
            truth = interp_pose(by_t, q)
            if truth is None:
                continue
            pose = m.localize(frames[q], np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1.0]]))
            if not pose.ok:
                rows.append({"frame": q, "ok": False, "reason": pose.reason})
                continue
            err = float(np.linalg.norm(np.array([pose.x, pose.y, pose.z]) - truth[0]))
            dh = (pose.yaw_deg - heading(truth[1]) + 180) % 360 - 180
            rows.append({"frame": q, "ok": True, "pos_err_m": round(err, 3), "yaw_err_deg": round(abs(dh), 2),
                         "inliers": pose.inliers})
        ok = [r for r in rows if r["ok"]]
        good = [r for r in ok if r["pos_err_m"] < 0.25 and r["yaw_err_deg"] < 10]
        results[label] = {
            "queries": len(rows), "localized": len(ok), "within_25cm_10deg": len(good),
            "median_pos_err_m": round(float(np.median([r["pos_err_m"] for r in ok])), 3) if ok else None,
            "median_yaw_err_deg": round(float(np.median([r["yaw_err_deg"] for r in ok])), 2) if ok else None,
            "median_inliers": int(np.median([r["inliers"] for r in ok])) if ok else None, "rows": rows}
        print(label, {k: v for k, v in results[label].items() if k != "rows"}, flush=True)
    (ROOT / "bench/results" / f"reloc_{a.scan}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
