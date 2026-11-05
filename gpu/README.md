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

## Gaussian splat (photoreal fly-through)

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
