"""Compare an estimated reconstruction's camera poses to a reference one (matched by image name).

Reports absolute trajectory error after Sim(3) alignment (as a % of the scene's size,
since SfM has no metric scale) and rotation error in degrees.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recon"))
from ingest_arkit import umeyama  # noqa: E402


def poses(path):
    rec = pycolmap.Reconstruction(str(path))
    out = {}
    for im in rec.images.values():
        if im.has_pose:
            T = im.cam_from_world().matrix()
            R = T[:, :3]
            out[Path(im.name).stem.lower()] = (R, -R.T @ T[:, 3])
    return out


def main(est, ref):
    E, G = poses(est), poses(ref)
    names = sorted(set(E) & set(G))
    src = np.array([E[n][1] for n in names]); dst = np.array([G[n][1] for n in names])
    s, R, t = umeyama(src, dst)
    err = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
    extent = np.linalg.norm(dst.max(0) - dst.min(0))
    rot = []
    for n in names:
        Re = E[n][0] @ R.T  # estimated cam_from_world expressed in the reference frame
        d = Re @ G[n][0].T
        rot.append(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
    rot = np.array(rot)
    print(f"matched {len(names)} / ref {len(G)} / est {len(E)} images")
    print(f"ATE RMSE: {np.sqrt((err**2).mean())/extent*100:.2f}% of trajectory extent "
          f"(median {np.median(err)/extent*100:.2f}%)")
    print(f"rotation error: median {np.median(rot):.3f} deg, max {rot.max():.3f} deg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("estimate", type=Path)
    ap.add_argument("reference", type=Path)
    a = ap.parse_args()
    main(a.estimate, a.reference)
