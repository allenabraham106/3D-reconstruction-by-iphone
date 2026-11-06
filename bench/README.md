# Benchmark

A fixed set of scans, scored the same way every time, so pipeline versions can be compared: today's CPU pipeline against ViPE on a GPU, or before and after a change.

```bash
python bench/benchmark.py run --label cpu-baseline --data ~/bench_data                  # CPU, ~4 h on 4 cores
python bench/benchmark.py run --label gpu-vipe --pipeline gpu --data ~/bench_data       # NVIDIA machine
python bench/benchmark.py compare cpu-baseline gpu-vipe --out bench/results/cpu_vs_gpu.md
```

`run --scans room IMG_6075` runs a subset. `metrics <output folder> --scan <name> --label <label>` scores a folder you already have. Results go to `bench/results/<label>/` (one JSON per scan and a `README.md` table). Commit them, so the baseline is there on the GPU machine.

## The scans (`--data` folder)

| name | file in `--data` | what it tests |
|---|---|---|
| `room` | `room/images_4/`, `room/sparse/0/` | [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) *room*: 311 photos with the dataset's reference camera poses. The only scan with a known right answer. |
| `IMG_6075` | `IMG_6075.MOV` | Robotics lab / office, 85 s. Busy, well lit, glass wall. |
| `IMG_6077` | `IMG_6077.MOV` | Apartment, latest walk, 115 s. |
| `IMG_6076` | `IMG_6076.MOV` | Apartment, bright, faces a big window. This is where the camera path used to tear. |
| `IMG_6070` | `IMG_6070.MOV` | Apartment, dim evening light. |

The four videos are in the repo's draft releases. For the room scene, download `360_v2.zip` from the Mip-NeRF 360 page and copy `room/`.

## What the numbers mean

The iPhone scans have no ground truth, so most numbers check that the result is **self-consistent**, using things that must hold in any real room:

| metric | better | meaning |
|---|---|---|
| Frames with a camera | higher | share of video frames the pipeline placed |
| Biggest camera jump | lower | largest frame-to-frame camera move, in median steps. A clean walk is 2–5; a tear (a frame facing a window glues the rest of the walk on wrongly) shows as 20–200. |
| Camera height spread | lower | p5–p95 of the phone's height above the floor. A hand-held phone stays within ~0.3 m; more means the map is tilted or bent. |
| Floor spread | lower | robust spread (1.48 × MAD) of the points within 10 cm of the floor. A flat floor that's reconstructed well is a few mm to ~3 cm. |
| Floor support | higher | share of the low points that lie on the fitted floor plane |
| Depth pixels kept | higher | median share of each frame's pixels with trusted depth (CPU: multi-view agreement; ViPE: its reliability mask) |
| Floor cells seen / seen vs only walked | higher | robot map: 5 cm floor cells seen by the camera, against those known free only because the phone was carried through them |
| Position / rotation error vs reference | lower | `room` only: camera error after a similarity alignment, as a % of the trajectory size, and in degrees |
| Total time | lower | wall clock, plus a table per stage. CPU: frames, sfm, dense (stereo + fusion), robot_map. GPU: frames (transcode), vipe, dense (fusion + map). |

Scale is not compared directly: the CPU pipeline guesses it from a 1.4 m phone height, while ViPE's depth is metric. `gpu/compare.py` reports how far off the guess was.

Scans are processed one after another, and each run starts from scratch, so timings are comparable on the same machine. The machine (CPU, threads, RAM, GPU) is recorded with every result.

## What the CPU baseline showed

- **Accuracy where it can be checked (room):** cameras within 0.09% of the room's size of the reference, rotation 0.18° median.
- **Run-to-run variation:** the camera solver is randomised, and the latest apartment walk (IMG_6077) came out differently on four runs of the same video. One was clean (0.39 m camera height spread, 9.5 x 8.7 m). The others were torn or bent (spread 0.5-2.1 m). The pipeline now repairs tears, re-solves with other seeds if that fails, and flags a bent map (camera height spread over 0.5 m) as a warning in `robot/map_meta.json`. The baseline keeps the bent run with its warning, because that is what the CPU pipeline does on this video today. ViPE / ARKit poses should not drift like this: check that row first when comparing.
- **Where the time goes:** dense stereo + fusion is 16-22 min of every scan; the camera solve is 2-18 min, more when the walk splits and needs bridge matching.
