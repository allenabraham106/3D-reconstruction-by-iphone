#!/usr/bin/env bash
# Gaussian splat of a scan on an NVIDIA GPU (gsplat v1.4.0): a photoreal fly-through.
# The robot map stays the TSDF mesh / occupancy grid; splats are for looking, not navigating.
#
#   bash gpu/splat.sh IMG_6077            # ViPE poses from gpu/run_gpu.sh (out_vipe/vipe)
#   SOURCE=cpu bash gpu/splat.sh IMG_6077 # poses from the CPU pipeline (out/IMG_6077)
#   STEPS=7000 bash gpu/splat.sh IMG_6077 # quick preview (default 30000)
#
# Output: out_splat/<name>/  (ckpts/*.pt, renders/, videos/traj_*.mp4, stats/)
set -euo pipefail
NAME="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="${SOURCE:-vipe}"
STEPS="${STEPS:-30000}"
WORK="$ROOT/out_splat"
SRC="$WORK/gsplat_src"
DATA="$WORK/${NAME}_data"

nvidia-smi >/dev/null || { echo "No NVIDIA GPU / driver found (nvidia-smi failed)."; exit 1; }
mkdir -p "$WORK"

# 1. gsplat (CUDA kernels build on first use) + its example trainer, pinned to v1.4.0
python3 -c "import torch; assert torch.cuda.is_available()" \
  || { echo "Needs PyTorch with CUDA: https://pytorch.org/get-started/locally/"; exit 1; }
pip install -q gsplat==1.4.0
[ -d "$SRC" ] || git clone -q --depth 1 --branch v1.4.0 https://github.com/nerfstudio-project/gsplat "$SRC"
pip install -q -r "$SRC/examples/requirements.txt"

# 2. scan -> COLMAP text model + images (what the trainer reads)
if [ "$SOURCE" = vipe ]; then
  python3 "$ROOT/gpu/to_colmap.py" "$DATA" --vipe "$ROOT/out_vipe/vipe" "$NAME" --every 2
else
  python3 "$ROOT/gpu/to_colmap.py" "$DATA" --cpu "$ROOT/out/$NAME"
fi

# 3. train (frames are already 1280 px or ViPE's size: no downscale)
cd "$SRC/examples"
python3 simple_trainer.py default \
  --data_dir "$DATA" --data_factor 1 --result_dir "$WORK/$NAME" \
  --max_steps "$STEPS" --eval_steps "$STEPS" --save_steps "$STEPS" \
  --disable_viewer

echo
echo "Done. Fly-through video: $WORK/$NAME/videos/   metrics: $WORK/$NAME/stats/"
echo "Interactive: python3 $SRC/examples/simple_viewer.py --ckpt $WORK/$NAME/ckpts/ckpt_$((STEPS-1))_rank0.pt"
