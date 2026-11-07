#!/usr/bin/env python3
"""Benchmark the reconstruction on a fixed set of scans, so runs can be compared over time
(e.g. today's CPU pipeline vs ViPE on a GPU next week).

    # run the suite with the CPU pipeline and record the results under a label
    python bench/benchmark.py run --label cpu-2026-11 --data ~/bench_data

    # same suite through ViPE on an NVIDIA machine (gpu/run_gpu.sh)
    python bench/benchmark.py run --label gpu-vipe --pipeline gpu --data ~/bench_data

    # score an output folder you already have (any pipeline, same layout as run.py's)
    python bench/benchmark.py metrics out/IMG_6075 --scan IMG_6075 --label cpu-2026-11

    # side-by-side table of two labels
    python bench/benchmark.py compare cpu-2026-11 gpu-vipe

Results: bench/results/<label>/<scan>.json and bench/results/<label>/README.md. Commit them:
the baseline has to travel to the GPU machine. The scans are listed in bench/suite.json;
--data is the folder that holds them (see bench/README.md).

Every metric is computed from the output folder only (dense/cameras.json, robot/*), so the
CPU pipeline and the ViPE converter are scored the same way. Only the room scene has a
reference (its dataset's camera poses); the iPhone scans are scored on consistency checks
that need no ground truth (see bench/README.md for what each number means).
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
sys.path.insert(0, str(ROOT / "recon"))


# ----------------------------------------------------------------------------- helpers
def frame_no(name):
    m = re.findall(r"\d+", Path(name).stem)
    return int(m[-1]) if m else None


def read_ply_points(path):
    from robot_map import read_ply
    return read_ply(path)[0]


def ply_faces(path):
    if not path.exists():
        return None
    head = open(path, "rb").read(800).decode(errors="ignore")
    return int(head.split("element face ")[1].split()[0]) if "element face" in head else 0


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = np.trace(np.diag(S) @ E) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def machine_info():
    info = {"python": platform.python_version(), "os": platform.platform(), "cpus": os.cpu_count()}
    try:
        cpu = [l for l in open("/proc/cpuinfo") if l.startswith("model name")]
        info["cpu"] = cpu[0].split(":", 1)[1].strip() if cpu else platform.processor()
        mem = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")]
        info["ram_gb"] = round(int(mem[0].split()[1]) / 1e6, 1)
    except OSError:
        info["cpu"] = platform.processor()
    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True)
        info["gpu"] = r.stdout.strip() or None
    return info


# ----------------------------------------------------------------------------- metrics
def pose_accuracy(cams, ref_dir):
    """Camera error against a reference COLMAP model, after a similarity alignment.
    Matched by image name; frame_%05d names (video frames, e.g. ViPE) are matched to the
    reference images in sorted order."""
    import pycolmap
    ref = pycolmap.Reconstruction(str(ref_dir))
    G = {}
    for im in ref.images.values():
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        G[Path(im.name).stem.lower()] = (T.rotation.matrix(), im.projection_center())
    E = {Path(c["name"]).stem.lower(): (np.array(c["R"]), np.array(c["C"])) for c in cams}
    if not set(E) & set(G):
        order = sorted(G)
        E = {order[frame_no(k)]: v for k, v in E.items()
             if frame_no(k) is not None and frame_no(k) < len(order)}
    names = sorted(set(E) & set(G))
    if len(names) < 3:
        return {"matched": len(names)}
    src = np.array([E[n][1] for n in names])
    dst = np.array([G[n][1] for n in names])
    s, R, t = umeyama(src, dst)
    pos = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
    size = np.linalg.norm(np.ptp(dst, 0))
    rot = []
    for n in names:
        Ra = E[n][0] @ R.T  # estimated cam_from_world in the reference frame
        c = np.clip((np.trace(G[n][0] @ Ra.T) - 1) / 2, -1, 1)
        rot.append(np.degrees(np.arccos(c)))
    return {"matched": len(names), "reference_images": len(G),
            "position_error_median_pct": round(float(np.median(pos) / size * 100), 4),
            "position_error_max_pct": round(float(pos.max() / size * 100), 4),
            "rotation_error_median_deg": round(float(np.median(rot)), 3),
            "rotation_error_max_deg": round(float(np.max(rot)), 3)}


def compute_metrics(out: Path, ref=None, frames_total=None, continuous=True):
    out = Path(out)
    robot = out / "robot"
    meta = json.loads((robot / "map_meta.json").read_text())
    cams = json.loads((out / "dense" / "cameras.json").read_text())
    cams_m = json.loads((robot / "cameras_metric.json").read_text())
    m = {"scale_source": meta.get("metric_source")}

    info = out / "dense" / "bench_info.json"  # written by gpu/vipe_to_pipeline.py
    info = json.loads(info.read_text()) if info.exists() else {}
    # coverage: how many video frames got a camera
    if frames_total is None:
        frames_total = info.get("frames_total")
    elif info.get("every", 1) > 1:  # GPU path fuses every k-th frame: count against those
        frames_total = -(-frames_total // info["every"])
    if frames_total is None and (out / "frames").is_dir():
        frames_total = len(list((out / "frames").glob("*.jpg")))
    m["frames_total"] = frames_total
    m["frames_with_camera"] = len(cams)
    m["frames_with_camera_pct"] = round(100 * len(cams) / frames_total, 1) if frames_total else None

    # tears: the biggest frame-to-frame camera jump, in median steps (a clean walk: ~2-5)
    seq = sorted((frame_no(c["name"]), np.array(c["C"])) for c in cams_m if frame_no(c["name"]) is not None)
    if len(seq) > 5:
        idx = np.array([i for i, _ in seq])
        C = np.array([c for _, c in seq])
        step = np.linalg.norm(np.diff(C, axis=0), axis=1) / np.maximum(np.diff(idx), 1)
        # only meaningful for video: a photo set can genuinely jump between photos
        m["max_jump_x_median_step"] = (round(float(step.max() / max(np.median(step), 1e-9)), 1)
                                       if continuous else None)
        # a hand-held phone stays at about the same height: spread = tilt / bend / drift
        z = C[:, 2]
        m["camera_height_spread_m"] = round(float(np.percentile(z, 95) - np.percentile(z, 5)), 3)
    m["camera_height_median_m"] = round(float(meta.get("camera_height_median_m", np.nan)), 3)

    # floor flatness: robust spread of the points within 10 cm of the fitted floor
    P = read_ply_points(robot / "map_metric.ply")
    fz = P[np.abs(P[:, 2]) < 0.10, 2]
    if len(fz) > 100:
        mad = np.median(np.abs(fz - np.median(fz)))
        m["floor_spread_mm"] = round(float(1.4826 * mad * 1000), 1)
    m["floor_support"] = round(float(meta.get("floor_inlier_fraction", np.nan)), 3)

    # dense depth (CPU: cached trusted depth per frame)
    cache = out / "dense" / "depth_cache"
    files = sorted(cache.glob("*.npy")) if cache.is_dir() else []
    if files:
        pick = files[:: max(1, len(files) // 60)]
        cov = [float((np.load(f) > 0).mean()) for f in pick]
        m["depth_pixels_kept_median_pct"] = round(100 * float(np.median(cov)), 1)
    elif "depth_pixels_kept_median_pct" in info:
        m["depth_pixels_kept_median_pct"] = info["depth_pixels_kept_median_pct"]

    # surfaces + robot map (5 cm cells)
    m["mesh_triangles"] = ply_faces(robot / "mesh_metric.ply")
    m["map_points"] = meta.get("points")
    m["floor_cells_seen"] = meta.get("free_cells_seen")
    m["floor_cells_walked_only"] = meta.get("free_cells_walked_only")
    m["obstacle_cells"] = meta.get("occupied_cells")
    m["false_obstacles_cleared_by_walk"] = meta.get("false_obstacles_carved_by_walk")
    seen, walked = meta.get("free_cells_seen") or 0, meta.get("free_cells_walked_only") or 0
    m["floor_seen_pct"] = round(100 * seen / (seen + walked), 1) if seen + walked else None
    m["scene_extent_m"] = meta.get("scene_extent_m")
    m["glass_walls"] = len(meta.get("glass_walls", []))
    m["warnings"] = meta.get("warnings", [])

    t = out / "timings.json"
    m["timings_s"] = json.loads(t.read_text()) if t.exists() else None
    if ref:
        m["accuracy"] = pose_accuracy(cams, ref)
    return m


# ----------------------------------------------------------------------------- suite
def load_suite():
    return json.loads((BENCH / "suite.json").read_text())["scans"]


def run_scan(scan, data: Path, pipeline, work: Path, log):
    src = data / scan["input"]
    if not src.exists():
        raise FileNotFoundError(f"{src} (see bench/README.md: where to put the scans)")
    name = src.stem if src.is_file() else src.name
    t0 = time.time()
    if pipeline == "cpu":
        out = work / name
        if out.exists():
            shutil.rmtree(out)
        cmd = [sys.executable, "-u", str(ROOT / "run.py"), str(src), "--out", str(work)]
    else:
        if src.is_dir():  # ViPE reads video: turn an image folder into one, in name order
            mp4 = work / f"{name}.mp4"
            work.mkdir(parents=True, exist_ok=True)
            lst = work / f"{name}_frames.txt"
            lst.write_text("".join(f"file '{p.resolve()}'\nduration 0.1\n"
                                   for p in sorted(src.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-vf", "scale=-2:720,format=yuv420p", "-r", "10", "-c:v", "libx264", str(mp4)], check=True)
            src = mp4
        vipe_out = Path(os.environ.get("OUT_VIPE", ROOT / "out_vipe"))
        out = vipe_out / name
        cmd = ["bash", str(ROOT / "gpu" / "run_gpu.sh"), str(src)]
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if r.returncode:
        raise RuntimeError(f"{scan['name']}: pipeline failed (exit {r.returncode}), see {log}")
    wall = round(time.time() - t0, 1)
    return out, wall


def save(label, scan_name, m):
    d = BENCH / "results" / label
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{scan_name}.json").write_text(json.dumps(m, indent=1, default=float))


def cmd_run(a):
    data = Path(a.data).expanduser()
    work = Path(a.work).expanduser() if a.work else ROOT / "bench" / "work" / a.label
    work.mkdir(parents=True, exist_ok=True)
    if a.pipeline == "gpu":  # ViPE pipeline + its own output folder, so variants don't collide
        os.environ["VIPE_PIPELINE"] = a.vipe_pipeline
        os.environ.setdefault("OUT_VIPE", str(ROOT / "out_vipe" / a.vipe_pipeline))
    machine = machine_info()
    for scan in load_suite():
        if a.scans and scan["name"] not in a.scans:
            continue
        print(f"[bench] {scan['name']}: running the {a.pipeline} pipeline ...", flush=True)
        out, wall = run_scan(scan, data, a.pipeline, work, work / f"{scan['name']}.log")
        ref = data / scan["reference"] if scan.get("reference") else None
        src = data / scan["input"]
        n_photos = (len([p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
                    if src.is_dir() else None)
        m = compute_metrics(out, ref, n_photos, continuous=src.is_file())
        m.update(scan=scan["name"], description=scan.get("description"),
                 pipeline=a.pipeline if a.pipeline == "cpu" else f"gpu (ViPE {a.vipe_pipeline})",
                 wall_clock_s=wall, machine=machine, date=time.strftime("%Y-%m-%d %H:%M"))
        save(a.label, scan["name"], m)
        print(f"[bench] {scan['name']}: done in {wall/60:.1f} min", flush=True)
    write_report(a.label)


def cmd_metrics(a):
    suite = {s["name"]: s for s in load_suite()}
    ref = a.ref
    if ref is None and a.scan in suite and suite[a.scan].get("reference") and a.data:
        ref = Path(a.data).expanduser() / suite[a.scan]["reference"]
    m = compute_metrics(Path(a.out), ref, a.frames_total, continuous=not a.photos)
    m.update(scan=a.scan, description=suite.get(a.scan, {}).get("description"),
             pipeline=a.pipeline, machine=machine_info(), date=time.strftime("%Y-%m-%d %H:%M"))
    if a.wall_clock:
        m["wall_clock_s"] = a.wall_clock
    save(a.label, a.scan, m)
    write_report(a.label)
    print(json.dumps(m, indent=1, default=float))


# ----------------------------------------------------------------------------- tables
ROWS = [  # (label, key path, format, better)
    ("Frames with a camera", "frames_with_camera_pct", "{}%", "higher"),
    ("Biggest camera jump (x median step)", "max_jump_x_median_step", "{}", "lower"),
    ("Camera height spread, p5-p95 (m)", "camera_height_spread_m", "{}", "lower"),
    ("Floor spread (mm)", "floor_spread_mm", "{}", "lower"),
    ("Floor support (share of low points on the floor plane)", "floor_support", "{}", "higher"),
    ("Depth pixels kept, median per frame", "depth_pixels_kept_median_pct", "{}%", "higher"),
    ("Mesh triangles", "mesh_triangles", "{:,}", "info"),
    ("Floor cells seen (5 cm)", "floor_cells_seen", "{:,}", "higher"),
    ("Floor seen vs only walked", "floor_seen_pct", "{}%", "higher"),
    ("Obstacle cells", "obstacle_cells", "{:,}", "info"),
    ("Scene extent (m)", "scene_extent_m", "{}", "info"),
    ("Scale from", "scale_source", "{}", "info"),
    ("Position error vs reference", "accuracy.position_error_median_pct", "{}%", "lower"),
    ("Rotation error vs reference, median", "accuracy.rotation_error_median_deg", "{} deg", "lower"),
    ("Map warnings (e.g. bent map)", "warnings_count", "{}", "lower"),
    ("Total time (min)", "wall_clock_min", "{}", "lower"),
]


def get(m, key):
    if key == "warnings_count":
        return len(m.get("warnings") or [])
    if key == "wall_clock_min":
        w = m.get("wall_clock_s") or (m.get("timings_s") or {}).get("total")
        return round(w / 60, 1) if w else None
    for k in key.split("."):
        m = m.get(k) if isinstance(m, dict) else None
    return m


def fmt(v, f):
    if v is None:
        return "-"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


def load_label(label):
    d = BENCH / "results" / label
    if not d.is_dir():
        raise SystemExit(f"no results for label '{label}' in {d}")
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}


def write_report(label):
    res = load_label(label)
    scans = list(res)
    lines = [f"# Benchmark: {label}", ""]
    any_m = next(iter(res.values()))
    mc = any_m.get("machine", {})
    lines += [f"Pipeline: {any_m.get('pipeline')} · machine: {mc.get('cpu')} ({mc.get('cpus')} threads, "
              f"{mc.get('ram_gb')} GB RAM){', GPU ' + mc['gpu'] if mc.get('gpu') else ''} · {any_m.get('date')}", ""]
    lines += ["| | " + " | ".join(scans) + " |", "|---|" + "---|" * len(scans)]
    lines += ["| *scan* | " + " | ".join(res[s].get("description") or "" for s in scans) + " |"]
    for name, key, f, _ in ROWS:
        vals = [fmt(get(res[s], key), f) for s in scans]
        if all(v == "-" for v in vals):
            continue
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    order = ["frames", "arkit_ingest", "vipe", "sfm", "dense", "robot_map", "viewer"]
    stages = sorted({k for s in scans for k in (res[s].get("timings_s") or {}) if k != "total"},
                    key=lambda k: order.index(k) if k in order else len(order))
    if stages:
        lines += ["", "Time per stage (min):", "", "| stage | " + " | ".join(scans) + " |",
                  "|---|" + "---|" * len(scans)]
        for st in stages:
            cells = []
            for s in scans:
                v = (res[s].get("timings_s") or {}).get(st)
                cells.append(fmt(round(v / 60, 1) if v is not None else None, "{}"))
            lines.append(f"| {st} | " + " | ".join(cells) + " |")
    (BENCH / "results" / label / "README.md").write_text("\n".join(lines) + "\n")


def cmd_compare(a):
    A, B = load_label(a.a), load_label(a.b)
    scans = [s for s in A if s in B]
    if not scans:
        raise SystemExit("the two labels share no scans")
    lines = [f"# {a.a} vs {a.b}", "",
             "✓ = B better, ✗ = B worse (by the metric's direction, changes under 2% ignored), blank = info only.", ""]
    for s in scans:
        lines += [f"## {s}" + (f" ({A[s].get('description')})" if A[s].get("description") else ""), "",
                  f"| | {a.a} | {a.b} | |", "|---|---|---|---|"]
        for name, key, f, better in ROWS:
            va, vb = get(A[s], key), get(B[s], key)
            if va is None and vb is None:
                continue
            mark = ""
            if better in ("higher", "lower") and isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
                    and abs(vb - va) > 0.02 * max(abs(va), 1e-9):  # ignore < 2% changes
                mark = "✓" if (vb > va) == (better == "higher") else "✗"
            lines.append(f"| {name} | {fmt(va, f)} | {fmt(vb, f)} | {mark} |")
        lines.append("")
    text = "\n".join(lines)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the suite and record results")
    r.add_argument("--label", required=True)
    r.add_argument("--data", required=True, help="folder with the suite's scans (bench/README.md)")
    r.add_argument("--pipeline", choices=["cpu", "gpu"], default="cpu")
    r.add_argument("--vipe-pipeline", default="default",
                   help="gpu only: ViPE pipeline (default, dav3, static_vda, no_vda, ...)")
    r.add_argument("--scans", nargs="*", help="only these scans (names from bench/suite.json)")
    r.add_argument("--work", help="where outputs go (default bench/work/<label>)")
    r.set_defaults(fn=cmd_run)
    m = sub.add_parser("metrics", help="score an existing output folder")
    m.add_argument("out")
    m.add_argument("--scan", required=True)
    m.add_argument("--label", required=True)
    m.add_argument("--pipeline", default="cpu")
    m.add_argument("--ref", type=Path, help="reference COLMAP model (default: from the suite + --data)")
    m.add_argument("--data", help="suite data folder, to find the reference")
    m.add_argument("--frames-total", type=int)
    m.add_argument("--photos", action="store_true", help="input was a photo set, not a video")
    m.add_argument("--wall-clock", type=float, help="seconds, if the run was timed elsewhere")
    m.set_defaults(fn=cmd_metrics)
    c = sub.add_parser("compare", help="side-by-side table of two labels")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", help="also write the table to this file")
    c.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
