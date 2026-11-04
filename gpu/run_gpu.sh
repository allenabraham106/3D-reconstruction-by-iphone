#!/usr/bin/env bash
# One command on an NVIDIA GPU machine: ViPE on a video -> mesh, robot map, viewer, video,
# plus a comparison against this repo's CPU result for the same video (if present).
#
#   bash gpu/run_gpu.sh path/to/IMG_6075.MOV [cpu_out_dir]
#
# cpu_out_dir defaults to out/<name> (what `python run.py <video>` wrote).
# Needs: NVIDIA GPU + CUDA driver, Python 3.10+, ffmpeg. Optional: node + playwright (video).
set -euo pipefail
VIDEO="$1"
NAME="$(basename "${VIDEO%.*}")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CPU_OUT="${2:-$ROOT/out/$NAME}"
WORK="$ROOT/out_vipe"
PIPELINE="${VIPE_PIPELINE:-default}"   # try VIPE_PIPELINE=wide_angle for 0.5x ultra-wide clips

nvidia-smi >/dev/null || { echo "No NVIDIA GPU / driver found (nvidia-smi failed)."; exit 1; }

# 1. ViPE (pip package; it downloads its model weights from Hugging Face on first run)
if ! command -v vipe >/dev/null; then
  python3 -m pip install --upgrade nvidia-vipe
fi
pip install -q -r "$ROOT/requirements.txt" OpenEXR imageio imageio-ffmpeg

# ViPE reads MP4: transcode the iPhone .MOV (HEVC, HDR) to SDR H.264 first, same tone-map
# as recon/frames.py so colours match the CPU pipeline.
mkdir -p "$WORK/input"
MP4="$WORK/input/$NAME.mp4"
if [ ! -f "$MP4" ]; then
  ffmpeg -v error -y -i "$VIDEO" -vf "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,scale=-2:720" \
    -c:v libx264 -crf 16 -an "$MP4" \
  || ffmpeg -v error -y -i "$VIDEO" -vf "format=yuv420p,scale=-2:720" -c:v libx264 -crf 16 -an "$MP4"
fi

time vipe infer "$MP4" -o "$WORK/vipe" -p "$PIPELINE"

# 2. ViPE artifacts -> TSDF mesh, gravity-aligned metric robot map, viewer
python3 "$ROOT/gpu/vipe_to_pipeline.py" "$WORK/vipe" "$NAME" "$WORK/$NAME" --every 2

# 3. Side-by-side with the CPU pipeline's result, if it exists
if [ -d "$CPU_OUT/robot" ]; then
  python3 "$ROOT/gpu/compare.py" "$CPU_OUT" "$WORK/$NAME" "$WORK/${NAME}_compare"
fi

# 4. Fly-around video (optional: needs node + playwright)
if command -v node >/dev/null && node -e "require('playwright')" 2>/dev/null; then
  (cd "$WORK/$NAME/viewer" && python3 -m http.server 8090 >/dev/null 2>&1 & echo $! > /tmp/vipe_http.pid)
  sleep 1
  node "$ROOT/tools/record_video.js" http://localhost:8090/index.html "$WORK/${NAME}_frames" 240 0.5
  kill "$(cat /tmp/vipe_http.pid)" || true
  ffmpeg -v error -y -framerate 24 -i "$WORK/${NAME}_frames/f_%04d.png" -c:v libx264 -pix_fmt yuv420p "$WORK/${NAME}_flyaround.mp4"
fi

echo
echo "Done. Viewer: cd $WORK/$NAME/viewer && python3 -m http.server 8000"
echo "Robot map:   $WORK/$NAME/robot   Comparison: $WORK/${NAME}_compare/report.md"
