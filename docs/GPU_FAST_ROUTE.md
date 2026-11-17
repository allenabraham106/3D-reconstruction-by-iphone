# GPU fast route: results record

`gpu/fast_splat.py VIDEO OUT` turns one phone video into a Gaussian splat, a mesh and a nav2 robot map.
All numbers are for `IMG_1429.MOV` (114 s HEVC HDR, 1920x1080) on one WATcloud RTX 3090.

Steps: frames (ffmpeg, HDR tone-map) -> pycolmap SfM -> gsplat MCMC splat (`gpu/splat_train.py`)
-> `gpu/splat_map.py` (splat depth -> GPU TSDF -> mesh -> `recon/robot_map.py`).

## End-to-end time and quality

Quality is held-out test-view PSNR / LPIPS of the splat (every 8th frame held out).

| Run | Frames | SfM | Splat | Map | Total | PSNR / LPIPS |
|---|---|---|---|---|---|---|
| CPU pipeline (`run.py`, stereo + TSDF) | | | | 18 min | ~22 min | (mesh only) |
| ref: 3M Gaussians, 7000 steps | 59 s | 252 s | 357 s | | 11.3 min | 24.32 / 0.191 |
| lite: 1.5M Gaussians, anti-aliased | 63 s | 246 s | 245 s | | 9.4 min | **24.79 / 0.187** |
| c32i: + scale-before-tone-map frames | 27 s | 197 s | 237 s | | 7.8 min | 24.55 / 0.192 |
| A: quadratic matching off, 5000 steps | 31 s | 371 s | 175 s | 21 s | 10.1 min | 24.47 / 0.204 |
| B: as A, 220 frames | 21 s | 236 s | 164 s | 20 s | 7.4 min | 23.66 / 0.210 |
| C: as B, 4000 steps | 25 s | 229 s | 127 s | 21 s | 6.8 min | 23.20 / 0.225 |
| D: GPU SIFT + matching (`pycolmap-cuda12`), 5000 steps | | | | | pending | pending |

c32i SfM breakdown: SIFT 13 s, matching ~106 s, mapping 78 s. Matching is what run D moves to the GPU.

## What won and lost

Kept (now the defaults):
- 1.5M Gaussian cap + anti-aliasing: better than 3M (24.79 vs 24.32 dB) and 2 min faster.
- Downscale before the HDR tone-map: frames 46 s -> 24 s, same keyframes.
- Quadratic overlap in sequential matching: without it only 97/287 frames register,
  bridge matching kicks in and SfM gets slower (runs A-C).
- GPU map (`splat_map.py`): 25 s instead of 18 min of CPU stereo + TSDF.

Lost:
- GLOMAP global mapper: 145 s vs 78 s mapping, 24.16 dB.
- 32 CPU cores: matching stays ~104 s.
- Dropping the largest Gaussians before depth rendering: removes the floor, robot_map's floor fit fails, scale off x8.

Splat sweep on lite poses (7000 steps): cap 1M 24.58 dB / 3.0 min, SH degree 1 24.58 / 3.1 min,
depth weight 0 24.73, depth weight 0.5 24.77, baseline 24.79 / 3.7 min.

## Robot map: GPU vs CPU

| | Obstacle cells | Free cells seen | Extent | Floor inliers | Time |
|---|---|---|---|---|---|
| CPU stereo map | 5731 | 2433 | 7.17 x 5.28 m | 0.10 | 18 min |
| GPU splat map, min_weight 2 | 9110 | 1812 | 7.47 x 7.16 m | 0.05 | 25 s |
| GPU splat map, min_weight 6 (default) | 6802 | 1952 | 7.38 x 7.03 m | 0.05 | 25 s |

The GPU map still has more false obstacles (splat floaters) and a weaker floor fit than the CPU map.

## Localization (`robot_link/reloc_map.py`)

- Map build: 46 s -> 21 s (threaded SIFT + matmul ratio-test matching), same 232,956 landmarks.
- Holdout (`tools/reloc_holdout.py`): 136/139 frames localized, 3 mm median error, 69.6 ms per query.

## Next

- Read run D; if total is still over 5 min, try cap 1M / SH degree 1 (~40 s, -0.2 dB) and fewer fly-through frames.
- Use the phone's ARKit poses instead of SfM for app uploads; `robot_link/server.py` still runs the CPU `run.py`.
