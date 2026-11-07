# Benchmark: cpu-baseline

Pipeline: cpu · machine: Intel(R) Xeon(R) Processor @ 2.10GHz (4 threads, 16.5 GB RAM) · 2026-11-06 20:03

| | IMG_6070 | IMG_6075 | IMG_6076 | IMG_6077 | room |
|---|---|---|---|---|---|
| *scan* | apartment, dim evening light, 74 s | robotics lab / office, 85 s | apartment, bright, faces a big window, 91 s | apartment, latest walk, 115 s | Mip-NeRF 360 room, 311 photos, reference poses |
| Frames with a camera | 88.3% | 100.0% | 87.2% | 93.4% | 100.0% |
| Biggest camera jump (x median step) | 7.0 | 4.2 | 3.6 | 8.4 | - |
| Camera height spread, p5-p95 (m) | 0.053 | 0.188 | 0.143 | 1.222 | 0.392 |
| Floor spread (mm) | 38.9 | 46.1 | 49.9 | 39.9 | 38.1 |
| Floor support (share of low points on the floor plane) | 0.188 | 0.052 | 0.197 | 0.097 | 0.078 |
| Depth pixels kept, median per frame | 15.6% | 35.6% | 19.2% | 19.5% | 48.6% |
| Mesh triangles | 1,310,586 | 2,018,386 | 1,203,008 | 1,702,178 | 1,748,588 |
| Floor cells seen (5 cm) | 6,906 | 2,518 | 5,660 | 760 | 2,599 |
| Floor seen vs only walked | 90.4% | 80.6% | 90.3% | 23.3% | 81.9% |
| Obstacle cells | 8,064 | 7,098 | 6,305 | 5,860 | 4,045 |
| Scene extent (m) | [7.66, 10.13, 3.71] | [6.86, 7.01, 2.28] | [8.67, 9.22, 2.9] | [5.74, 7.47, 3.07] | [3.51, 7.15, 1.88] |
| Scale from | phone_height=1.4m | phone_height=1.4m | phone_height=1.4m | phone_height=1.4m | phone_height=1.4m |
| Position error vs reference | - | - | - | - | 0.0884% |
| Rotation error vs reference, median | - | - | - | - | 0.184 deg |
| Map warnings (e.g. bent map) | 0 | 0 | 0 | 1 | 0 |
| Total time (min) | 19.9 | 35.4 | 38.0 | 31.1 | 23.5 |

Time per stage (min):

| stage | IMG_6070 | IMG_6075 | IMG_6076 | IMG_6077 | room |
|---|---|---|---|---|---|
| frames | 0.6 | 0.7 | 0.6 | 0.7 | - |
| sfm | 3.1 | 14.7 | 17.7 | 7.8 | 2.0 |
| dense | 16.0 | 19.9 | 19.5 | 22.4 | 21.4 |
| robot_map | 0.2 | 0.2 | 0.2 | 0.2 | 0.1 |
| viewer | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
