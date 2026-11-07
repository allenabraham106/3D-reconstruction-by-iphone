# Benchmark: gpu-vipe-default

Pipeline: gpu (ViPE default) · machine: AMD Ryzen 7 5800X 8-Core Processor (16 threads, 65.7 GB RAM), GPU NVIDIA GeForce RTX 5070 Ti, 16303 MiB · 2026-11-07 00:51

| | IMG_6070 | IMG_6075 | IMG_6076 | IMG_6077 | room |
|---|---|---|---|---|---|
| *scan* | apartment, dim evening light, 74 s | robotics lab / office, 85 s | apartment, bright, faces a big window, 91 s | apartment, latest walk, 115 s | Mip-NeRF 360 room, 311 photos, reference poses |
| Frames with a camera | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Biggest camera jump (x median step) | 5.1 | 2.6 | 2.1 | 2.7 | - |
| Camera height spread, p5-p95 (m) | 0.417 | 0.205 | 0.304 | 4.403 | 1.467 |
| Floor spread (mm) | 51.3 | 53.1 | 50.2 | 49.3 | 50.5 |
| Floor support (share of low points on the floor plane) | 0.063 | 0.047 | 0.054 | 0.066 | 0.053 |
| Depth pixels kept, median per frame | 99.2% | 97.5% | 99.1% | 99.2% | 97.2% |
| Mesh triangles | 3,883,328 | 2,677,556 | 4,719,239 | 4,813,380 | 984,154 |
| Floor cells seen (5 cm) | 5,731 | 2,867 | 6,487 | 4,660 | 1,398 |
| Floor seen vs only walked | 55.7% | 57.2% | 66.5% | 48.1% | 40.0% |
| Obstacle cells | 14,175 | 12,374 | 12,399 | 14,276 | 4,063 |
| Scene extent (m) | [13.12, 14.06, 3.36] | [9.6, 8.57, 3.32] | [12.18, 14.67, 2.92] | [12.5, 12.11, 8.03] | [6.15, 4.58, 4.84] |
| Scale from | input (metric depth) | input (metric depth) | input (metric depth) | input (metric depth) | input (metric depth) |
| Position error vs reference | - | - | - | - | 34.9618% |
| Rotation error vs reference, median | - | - | - | - | 148.233 deg |
| Map warnings (e.g. bent map) | 0 | 0 | 0 | 1 | 1 |
| Total time (min) | 6.9 | 6.5 | 7.8 | 8.0 | 4.2 |

Time per stage (min):

| stage | IMG_6070 | IMG_6075 | IMG_6076 | IMG_6077 | room |
|---|---|---|---|---|---|
| frames | 0.2 | 0.0 | 0.2 | 0.3 | 0.0 |
| vipe | 5.7 | 6.0 | 6.5 | 6.7 | 4.0 |
| dense | 1.0 | 0.6 | 1.1 | 1.0 | 0.2 |
