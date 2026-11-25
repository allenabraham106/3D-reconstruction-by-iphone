"""Film data from a real robot map (run.py / robot_link output): a subsampled coloured cloud, the scan's camera path,
and for every point the first keyframe that saw it (so the film can reveal the room the way the walk captured it).
    python3 prep.py OUT_DIR/<map>   ->  room.json"""
import json, sys
from pathlib import Path
import numpy as np

m = Path(sys.argv[1]) / "robot"
data = m.joinpath("map_metric.ply").read_bytes()
end = data.index(b"end_header\n") + 11
n = int(data[:end].decode().split("element vertex ")[1].split()[0])
rec = np.frombuffer(data[end:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n)
rng = np.random.default_rng(0)
idx = rng.choice(n, 70000, replace=False)
P = np.c_[rec["x"], rec["y"], rec["z"]][idx].astype(float)
C = np.c_[rec["r"], rec["g"], rec["b"]][idx]
cams = json.loads(m.joinpath("cameras_metric.json").read_text())
first = np.full(len(P), len(cams), int)
for j, c in reversed(list(enumerate(cams))):
    Cj = np.array(c["C"]); fwd = np.array(c["R"]).T @ [0, 0, 1.0]
    d = P - Cj; dist = np.linalg.norm(d, axis=1)
    seen = (dist < 3.4) & ((d @ fwd) / np.maximum(dist, 1e-6) > np.cos(np.radians(38)))
    first[seen] = j
order = np.argsort(first, kind="stable")
out = {
    "points": np.round(P[order], 3).ravel().tolist(),
    "colors": C[order].ravel().tolist(),
    "seen": (first[order] / len(cams)).round(3).tolist(),         # 0..1 along the walk; 1 = never seen
    "path": [np.round(c["C"], 3).tolist() for c in cams],
    "look": [np.round(np.array(c["R"]).T @ [0, 0, 1.0], 3).tolist() for c in cams],
}
Path("room.json").write_text(json.dumps(out, separators=(",", ":")))
print(len(P), "points,", len(cams), "keyframes,", (first < len(cams)).mean().round(2), "seen")
