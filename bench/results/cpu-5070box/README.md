# Benchmark: cpu-5070box

Pipeline: cpu · machine: AMD Ryzen 7 5800X 8-Core Processor (16 threads, 65.7 GB RAM), GPU NVIDIA GeForce RTX 5070 Ti, 16303 MiB · 2026-11-07 04:34

| | IMG_6070 | IMG_6075 | IMG_6076 | IMG_6077 | room |
|---|---|---|---|---|---|
| *scan* | apartment, dim evening light, 74 s | robotics lab / office, 85 s | apartment, bright, faces a big window, 91 s | apartment, latest walk, 115 s | Mip-NeRF 360 room, 311 photos, reference poses |
| Frames with a camera | 88.3% | 100.0% | 90.3% | 97.8% | 100.0% |
| Biggest camera jump (x median step) | 7.0 | 3.8 | 4.5 | 4.7 | - |
| Camera height spread, p5-p95 (m) | 0.106 | 0.495 | 0.201 | 0.408 | 0.386 |
| Floor spread (mm) | 49.4 | 41.8 | 52.8 | 30.5 | 41.4 |
| Floor support (share of low points on the floor plane) | 0.177 | 0.063 | 0.225 | 0.178 | 0.081 |
| Depth pixels kept, median per frame | 15.8% | 35.6% | 14.9% | 22.8% | 47.9% |
| Mesh triangles | 1,314,216 | 2,034,830 | 1,141,472 | 990,092 | 1,756,100 |
| Floor cells seen (5 cm) | 5,164 | 712 | 4,736 | 6,690 | 2,683 |
| Floor seen vs only walked | 79.0% | 40.9% | 83.4% | 97.1% | 82.3% |
| Obstacle cells | 8,469 | 3,949 | 6,786 | 6,672 | 4,010 |
| Scene extent (m) | [7.48, 10.11, 2.76] | [4.54, 5.29, 2.08] | [9.99, 8.59, 2.71] | [9.65, 9.12, 2.14] | [4.27, 7.38, 1.92] |
| Scale from | phone_height=1.4m | phone_height=1.4m | phone_height=1.4m | phone_height=1.4m | phone_height=1.4m |
| Position error vs reference | - | - | - | - | 0.0885% |
| Rotation error vs reference, median | - | - | - | - | 0.094 deg |
| Map warnings (e.g. bent map) | 0 | 0 | 0 | 0 | 0 |
| Total time (min) | 9.2 | 17.6 | 12.4 | 10.9 | 9.9 |

Time per stage (min):

| stage | IMG_6070 | IMG_6075 | IMG_6076 | IMG_6077 | room |
|---|---|---|---|---|---|
| frames | 0.3 | 0.3 | 0.3 | 0.3 | - |
| sfm | 1.3 | 6.3 | 2.4 | 1.8 | 1.0 |
| dense | 7.4 | 10.9 | 9.5 | 8.7 | 8.8 |
| robot_map | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| viewer | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
