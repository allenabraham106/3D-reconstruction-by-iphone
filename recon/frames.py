"""Stage 1: iPhone video -> sharp, evenly spaced keyframes.

Handles iPhone 15/16/17 footage: HEVC (H.265) .MOV, 10-bit HDR (HLG / Dolby Vision)
tone-mapped to SDR, rotation metadata (ffmpeg auto-rotates), 4K downscaled.
Picks the sharpest frame in each time window (variance of Laplacian) to avoid motion blur.
"""
import argparse, json, shutil, subprocess, tempfile
from pathlib import Path

import cv2
import numpy as np


def probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,color_transfer,codec_name,nb_frames:"
                          "format=duration", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True)
    j = json.loads(out.stdout)
    s = j["streams"][0]
    s["duration"] = float(j["format"].get("duration", 0))
    return s


def extract(video: Path, out_dir: Path, target_frames=300, max_side=1280, candidates_per_pick=4):
    info = probe(video)
    hdr = info.get("color_transfer") in ("arib-std-b67", "smpte2084")
    dur = max(info["duration"], 1e-3)
    sample_fps = min(30.0, target_frames * candidates_per_pick / dur)
    vf = [f"fps={sample_fps:.4f}"]
    if hdr:
        # HLG/PQ 10-bit -> SDR BT.709 so features/colours look normal
        vf.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                  "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
    vf.append(f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'")
    tmp = Path(tempfile.mkdtemp(prefix="frames_"))
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", ",".join(vf), "-q:v", "2",
           str(tmp / "c_%06d.jpg")]
    r = subprocess.run(cmd)
    if r.returncode != 0 and hdr:  # ffmpeg without zimg: fall back to plain conversion
        cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf",
               f"fps={sample_fps:.4f},format=yuv420p," + vf[-1], "-q:v", "2", str(tmp / "c_%06d.jpg")]
        subprocess.run(cmd, check=True)
    cands = sorted(tmp.glob("c_*.jpg"))
    sharp = []
    for c in cands:
        g = cv2.imread(str(c), cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2))
        sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
    sharp = np.array(sharp)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in range(0, len(cands), candidates_per_pick):
        w = sharp[s:s + candidates_per_pick]
        best = s + int(np.argmax(w))
        if sharp[best] < 0.25 * np.median(sharp):  # whole window blurred: skip
            continue
        shutil.copy(cands[best], out_dir / f"frame_{n:05d}.jpg")
        n += 1
    shutil.rmtree(tmp)
    print(f"[frames] {video.name}: {info.get('codec_name')} {info['width']}x{info['height']} "
          f"{dur:.1f}s hdr={hdr} -> {n} keyframes in {out_dir}")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--max-side", type=int, default=1280)
    a = ap.parse_args()
    extract(a.video, a.out, a.frames, a.max_side)
