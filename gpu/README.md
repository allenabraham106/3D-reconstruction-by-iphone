# GPU session: NVIDIA ViPE on the iPhone scans

[ViPE](https://github.com/nv-tlabs/vipe) (NVIDIA's Video Pose Engine, the stack in the demo this repo set out to reproduce) estimates camera poses, intrinsics and **dense metric depth for every frame** with learned models on a GPU. Its depth fills the places where CPU stereo struggles (plain walls, dark corners, glass) and it has real scale, so no phone-height guess. This folder runs it and feeds its output into the same mesh / robot-map / viewer stages as the CPU pipeline, so the two are directly comparable.

## Run (one command per video)

```bash
git clone https://github.com/allenabraham106/3d-reconstruction-by-iphone && cd 3d-reconstruction-by-iphone
bash gpu/run_gpu.sh /path/to/IMG_6075.MOV            # office
bash gpu/run_gpu.sh /path/to/IMG_6070.MOV            # apartment
```

`run_gpu.sh` does the following:

1. Installs `nvidia-vipe` (pip) if `vipe` isn't on the PATH.
2. Transcodes the iPhone `.MOV` (HEVC, HDR) to a 720p SDR H.264 MP4, with the same tone mapping as the CPU pipeline.
3. Runs `vipe infer` (pipeline `default`; set `VIPE_PIPELINE=wide_angle` for 0.5x ultra-wide clips).
4. Runs `gpu/vipe_to_pipeline.py`: ViPE's depth goes through the same reliability filter ViPE itself uses (local depth range / mean < 10 %), then TSDF fusion, the mesh, a gravity-aligned robot map (scale kept from ViPE) and the viewer.
5. Runs `gpu/compare.py`, which compares against `out/<name>` from the CPU run: triangles, floor / obstacle cells, extent, camera height, and **how far off the CPU's 1.4 m phone-height guess was** -> `out_vipe/<name>_compare/report.md` + `maps.png`.
6. Renders a fly-around video, if Node + Playwright are installed.

Outputs land in `out_vipe/<name>/` with the same layout as `out/<name>/` (`viewer/`, `robot/`, `dense/`), so `ros2/pioneer_map` works on them unchanged.

## Tested on an RTX 5070 Ti (16 GB, 62 GB RAM)

Notes from running all of this for real (Nov 2026):

- **ViPE 1.2.0 install:** it pins a `gtsam-develop` build that is gone from PyPI; install the newest nightly and ViPE with `--no-deps --no-build-isolation`. Blackwell (sm_120) needs PyTorch cu128 and nvcc 12.8. The PyPI sdist is missing `vipe/priors/depth/dav3/model/dinov2/` (no `__init__.py`, so setuptools skips it); copy it from the v1.2.0 tag if you want the `dav3` pipeline.
- **RAM:** ViPE keeps the whole clip in memory, about 65 MB a frame at 720p (852 frames peaked at 54 GB). `run_gpu.sh` feeds it 10 fps, lowered for long clips to stay under `VIPE_MAX_FRAMES=760`, and `VIPE_MEM_MAX=52G` caps it so only ViPE is killed if it overruns.
- **VRAM:** `default` fits in 16 GB. `static_vda` (large video depth) and `dav3` (Depth Anything 3 GIANT, even with a 2-frame window) do not.
- **HDR:** the iPhone videos are HLG; the tone map needs an ffmpeg with `zscale` (the static build that ships with `imageio-ffmpeg` has it).
- **TSDF on the GPU:** `vipe_to_pipeline.py` fuses with `gpu/tsdf_gpu.py` when PyTorch sees CUDA: same mesh as `recon/tsdf.py` (median vertex difference 0.000 mm), 32x faster.

## Crisp visuals (`--crisp`)

```bash
python gpu/vipe_to_pipeline.py out_vipe/default/vipe IMG_6075 out_vipe/crisp/IMG_6075 --crisp --every 1
cd out_vipe/crisp/IMG_6075/viewer_crisp && python3 -m http.server 8000   # open index.html?show=points&cut=2.2
```

ViPE's learned depth is soft across object edges and a few cm different from frame to frame, so fusing all of it smears edges, and ceilings / glass come through as a shell. `--crisp` adds a second, visual-only layer (`crisp/`, `viewer_crisp/`): `gpu/depth_filter.py` drops depth at edges (5x5 range / depth < 3 %), beyond 5 m, and wherever fewer than two neighbouring frames agree within 1.5 % (as MVS fusion does); then a 1 cm TSDF mesh and a per-pixel coloured point cloud. The robot map is built as before. The viewer takes `?cut=<metres>` (dollhouse: hide everything higher), `?show=points` and `?ps=<point size>`.

## Gaussian splat, trained straight from ViPE (`splat_train.py`)

```bash
python gpu/splat_train.py out_vipe/default/vipe IMG_6075 ~/bench_data/IMG_6075.MOV out_splat/IMG_6075
python gpu/splat_render.py out_splat/IMG_6075/ckpt.pt out_vipe/crisp/IMG_6075/robot/map_transform.json out_splat/IMG_6075/demo --walk
```

A self-contained gsplat (PyPI 1.4.0) trainer, no COLMAP step: ViPE poses and intrinsics, the original video re-decoded at full 1080p, Gaussians initialised from ViPE's metric depth, 0.8 L1 + 0.2 D-SSIM + a fading inverse-depth loss, MCMC densification (3 M cap), a per-frame colour affine for auto-exposure, and people (from ViPE's instance masks) left out of the loss so they don't leave ghosts. Every 8th frame is held out. Office, 30k steps: 27.0 dB PSNR / SSIM 0.928 on held-out frames (static scene), 5.2 GB of VRAM. `splat_render.py --walk` renders a smoothed walk through the scan; `--orbit` (dollhouse from above) works poorly, since no training view was taken from up there.

## Recreating a gap in a walk (`gap_fill.py`, `gap_splat.py`)

```bash
python gpu/splat_train.py ... out_gap/splat_no400 --steps 7000 --exclude 400:424
python gpu/gap_splat.py out_gap/splat_no400/ckpt.pt out_gap/IMG_6075_406_12 --start 406 --length 12 --path cubic
python gpu/gap_fill.py out_vipe/default/vipe/rgb/IMG_6075.mp4 out_gap/IMG_6075_406_12 --start 406 --length 12 \
    --methods flow,ltx,splat_cubic
```

A stretch is cut from the walk and recreated from only the frames before and after it, then scored against the hidden real frames (PSNR, SSIM, LPIPS). Methods: optical-flow interpolation, LTX-Video 0.9.5 keyframe conditioning (diffusers `LTXConditionPipeline`; the T5 text encoder runs on the CPU), and the splat trained without the gap, rendered along a straight or cubic (velocity-matched) camera path. On the office walk the cubic-path splat had the lowest LPIPS at 0.6, 1.2 and 2.4 s; with the true camera path it is better still (0.32-0.35), so the camera path is what limits it.

## Gaussian splat with the reference trainer (photoreal fly-through)

```bash
bash gpu/splat.sh IMG_6077              # after run_gpu.sh: ViPE poses + depth-initialised splats
SOURCE=cpu bash gpu/splat.sh IMG_6077   # or the CPU pipeline's poses (out/IMG_6077)
STEPS=7000 bash gpu/splat.sh IMG_6077   # quick preview
```

`gpu/to_colmap.py` writes the scan as a COLMAP model (binary + text) plus an images folder. `splat.sh` then trains [gsplat](https://github.com/nerfstudio-project/gsplat) v1.4.0's `simple_trainer.py` on it. Results go to `out_splat/<name>/`: a fly-through `videos/traj_*.mp4`, PSNR/SSIM in `stats/`, and a checkpoint for gsplat's interactive `simple_viewer.py`. The splat is for looking at. The robot still navigates on the TSDF mesh and occupancy grid: splat geometry is soft blobs and has the same blind spots on plain walls and glass.

Tested here (no GPU): both exports (CPU SfM and the fake ViPE folder) read back with the same binary layout gsplat's COLMAP loader uses, and camera poses match the source to within 1e-13. The training itself has not been run here.

## What to bring / expect

- **GPU:** any recent NVIDIA card with >= 12 GB VRAM should do. ViPE downloads its model weights from Hugging Face on the first run (needs internet).
- **Videos:** `IMG_6070.MOV` (apartment) and `IMG_6075.MOV` (office) are in the repo's draft release.
- **CPU results to compare against:** run `python run.py IMG_6075.MOV` first on any machine, or copy `out/` over. The comparison step is skipped if they're missing.
- If ViPE runs out of memory on long clips, try `VIPE_PIPELINE=no_vda`, or ViPE's `pose_only_long` pipeline (poses only, no dense depth).

## Tested here (no GPU)

`tests/make_fake_vipe.py` writes a ViPE-format artifact folder (camera-to-world poses, intrinsics, half-float EXR depth zip, frame MP4) from one of our CPU reconstructions. `vipe_to_pipeline.py` reads it back through TSDF -> robot map -> viewer. That checks the file format and coordinate conventions end to end. ViPE itself (`vipe infer`) has not been run here.
