# 3D reconstruction from iPhone footage

Walk through a room or floor with an iPhone. Get a dense colored 3D point cloud, the camera path, and a **metric, gravity-aligned map a robot can navigate with** (ROS nav2 occupancy grid + 3D voxels).

![overview](docs/img/full_overview.png)

*311 handheld frames → 1.15 M points. The camera path is blue and the live camera frustum is white. CPU only, about 5 minutes.*

This reproduces the kind of result in the NVIDIA-stack demo (ViPE-style pose and dense-depth reconstruction from phone video). It runs on a normal laptop CPU, with no GPU and no downloaded neural-network weights.

## Two ways to capture

| | **A. Any video** (Camera app) | **B. ScanCapture app** (in `ios/`) — recommended for robots |
|---|---|---|
| What you record | `.MOV` (HEVC, HDR OK, any orientation) | RGB + ARKit 6-DoF pose + intrinsics (+ LiDAR depth on Pro models) |
| Scale | estimated from phone height above the floor (`--phone-height`, default 1.4 m) | **true metric** from ARKit |
| Gravity | estimated (camera orientation + floor plane) | **exact** (ARKit gravity) |
| Processing | SfM + dense stereo, ~5–10 min on CPU | LiDAR: **~20 s**; no LiDAR: SfM + stereo, then aligned to ARKit |
| Command | `python run.py IMG_1234.MOV` | `python run.py scan_20261104_101500/ --arkit` |

iPhone 17 Pro / Pro Max have LiDAR. The base iPhone 17 and iPhone Air don't; the app detects this and still records ARKit poses.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pycolmap, opencv, numpy, scipy
# also needs ffmpeg on PATH for video input (brew install ffmpeg / apt install ffmpeg)

python run.py IMG_1234.MOV               # or a folder of photos, or an app capture with --arkit
cd out/IMG_1234/viewer && python3 -m http.server 8000   # open http://localhost:8000
```

The viewer works on desktop and in iPhone Safari (drag to orbit, pinch to zoom). *Follow camera* replays the capture path, and *All cameras* shows every frustum. You can also drop any `dense.ply` + `cameras.json` onto the page.

### Outputs (`out/<name>/`)

| File | What |
|---|---|
| `viewer/` | self-contained web viewer (three.js vendored, works offline) |
| `dense/dense.ply`, `dense/cameras.json` | raw reconstruction (SfM frame) |
| `robot/map_metric.ply` | point cloud in **metres, Z up, floor at Z=0** |
| `robot/occupancy.pgm` + `occupancy.yaml` | **ROS nav2 `map_server` map**, 5 cm cells (free / occupied / unknown) |
| `robot/occupancy_preview.png` | the grid with the capture path drawn on |
| `robot/voxels.npz` | 5 cm 3D occupancy (`indices`, `origin`, `voxel_size`) for collision / footstep planning |
| `robot/cameras_metric.json` | camera poses in the robot map frame |
| `robot/map_meta.json` | scale source, extent, cell counts |

Occupancy grid for the demo room (white = walkable floor, black = obstacles between 8 cm and 1.8 m high, grey = unseen, blue = capture path):

<img src="docs/img/occupancy_preview.png" width="260">

## How it works

```
iPhone video ─► frames.py ─► sfm.py ─────────► mvs.py ─────────────► robot_map.py ─► viewer + nav map
                sharp        COLMAP SIFT,       per-frame stereo        floor RANSAC,
                keyframes,   sequential         partner, rectify,       gravity align,
                HDR→SDR      matching,          SGBM + L/R check,       metric scale,
                             incremental BA     multi-view consistency  occupancy grid
                                                fusion

ScanCapture app ─► ingest_arkit.py ─► (LiDAR) fuse depth with ARKit poses ─► robot_map.py --metric
                                     (no LiDAR) sfm + mvs, Sim(3)-align to ARKit trajectory
```

1. **`recon/frames.py`** samples the video at about 4× the target keyframe count and keeps the sharpest frame in each window (variance of Laplacian), which drops motion-blurred frames. HLG / Dolby Vision HDR is tone-mapped to SDR (iPhones record HDR by default), the rotation flag is applied, and frames are downscaled to 1280 px.
2. **`recon/sfm.py`** runs COLMAP via pycolmap: SIFT features with one shared camera (one lens), sequential matching with quadratic overlap (suited to video), and incremental mapping with bundle adjustment. Outputs are camera poses, intrinsics and lens distortion, and sparse points.
3. **`recon/mvs.py`** produces dense depth on CPU. For each frame it picks the stereo partner with the best baseline-to-depth ratio (about 0.07), small rotation and many shared points. It undistorts, rectifies from the SfM poses (vertical layouts are transposed, mirrored layouts are flipped), and runs OpenCV semi-global matching with a left-right consistency check. The disparity range comes from the SfM depths. Each depth pixel is kept only if it **agrees within 1.5% with at least 2 neighboring views' depth maps**. The survivors are back-projected, voxel-averaged, and cleaned with statistical outlier removal.
4. **`recon/robot_map.py`** finds the floor with RANSAC (the largest plane roughly perpendicular to gravity, below the cameras), rotates it to Z=0, and sets metric scale. It then writes the nav2 occupancy grid and voxel map.
5. **`viewer/index.html`** is a three.js point-cloud viewer with a frustum animation.

## Measured results (Mip-NeRF 360 *room* scene, 311 handheld frames, 4-core CPU)

| | |
|---|---|
| Frames registered by SfM | **311 / 311** |
| Camera position error vs dataset reference poses (after similarity alignment) | **0.10 %** of trajectory size (≈ 5 mm in a 5 m room) |
| Camera rotation error | median **0.10°**, max 0.36° |
| Dense points | 1.15 M (from 12.9 M raw depth samples) |
| Time | features 53 s, matching 36 s, mapping 97 s, dense 90 s, map 15 s |
| Simulated iPhone clip (HEVC Main10, HLG HDR, `.MOV`) through `run.py` | see `docs/RESULTS.md` |
| ARKit path, synthetic capture in the app's exact format, 1 % depth noise | fused cloud within **6.5 mm median** (17.5 mm p90) of ground truth, camera height 1.38 m vs 1.40 m |

Reproduce: `python tests/eval_poses.py out/<name>/sfm/sparse_best <reference sparse dir>`; ARKit round-trip test: `tests/make_synthetic_arkit.py`.

## iOS app (`ios/`)

SwiftUI + ARKit + RealityKit, iOS 17+. While scanning it shows the live LiDAR mesh (or feature points on non-LiDAR phones), the tracking status, frame count and metres walked. A frame is saved only after ≥5 cm of movement or ≥5° of rotation, at most 6 per second. Captures go to the app's Documents folder, which is visible in the Files app. Share them with the share button (AirDrop to your Mac).

```bash
brew install xcodegen
cd ios && xcodegen && open ScanCapture.xcodeproj
# Xcode: select your team under Signing & Capabilities, plug in the iPhone, Run.
```

Capture format (`scan_<date>/`): `frames.jsonl` (one line per frame: `world_from_camera` column-major 4×4 in ARKit convention, `intrinsics` column-major 3×3, file names), `rgb/*.jpg`, `depth/*.bin` (float32 metres, 256×192), `conf/*.bin` (uint8 0–2), `meta.json`.

> The app was written without access to a Mac in this environment. The Python side was tested against synthetic captures in the exact same format, but the Swift code has **not been compiled yet**. Expect a small fix or two on first build.

## Capture tips

- Walk **slowly**, and move the phone **sideways** through the space rather than standing still and panning. Stereo and SfM need parallax.
- Cover the floor. The robot map's free space comes from seeing the floor.
- Avoid blank white walls filling the frame, mirrors, and fast turns.
- For a whole floor, keep loops short and come back past earlier spots so SfM can close loops.
- 1–2 minutes per room is plenty. Video mode extracts about 300 keyframes by default (`--frames`).

## Known limitations / next steps

- **Video-only scale** comes from an assumed phone height. Use the app (ARKit) for real metres.
- **Textureless surfaces** (plain walls, glossy floors) get sparse stereo coverage. LiDAR fixes this on Pro phones. A learned monocular-depth prior (as ViPE uses) would help without LiDAR; it was left out because this environment couldn't fetch model weights safely.
- Large multi-room floors: SfM time grows with frame count. Next steps are vocabulary-tree loop closure and `pose_only_long`-style windowed processing.
- GPU path: on an NVIDIA machine, NVIDIA's ViPE (`pip install nvidia-vipe`) can replace `sfm.py` + `mvs.py`, and nvblox can do real-time TSDF/ESDF mapping on the robot itself.
