# Benchmark: hybrid

Pipeline: hybrid (SfM poses + ViPE depth) · machine: AMD Ryzen 7 5800X 8-Core Processor (16 threads, 65.7 GB RAM), GPU NVIDIA GeForce RTX 5070 Ti, 16303 MiB · 2026-11-24 23:59

| | IMG_6070 | IMG_6075 | IMG_6076 | IMG_6077 | IMG_6513 | room |
|---|---|---|---|---|---|---|
| *scan* | apartment, dim evening light, 74 s | robotics lab / office, 85 s | apartment, bright, faces a big window, 91 s | apartment, latest walk, 115 s | office, new walk (Nov 11), 76 s | Mip-NeRF 360 room, 311 photos, reference poses |
| Frames with a camera | 73.2% | 91.7% | 76.6% | 87.2% | 85.8% | 100.0% |
| Biggest camera jump (x median step) | 6.4 | 3.8 | 3.2 | 4.5 | 5.4 | 46.2 |
| Camera height spread, p5-p95 (m) | 0.47 | 0.998 | 0.314 | 0.772 | 0.435 | 0.558 |
| Floor spread (mm) | 52.6 | 50.8 | 54.4 | 50.4 | 52.9 | 53.0 |
| Floor support (share of low points on the floor plane) | 0.067 | 0.064 | 0.065 | 0.065 | 0.041 | 0.033 |
| Depth pixels kept, median per frame | 96.1% | 93.0% | 95.7% | 95.6% | 93.9% | 91.4% |
| Mesh triangles | 2,515,422 | 3,354,648 | 3,619,292 | 3,255,368 | 2,673,394 | 1,248,670 |
| Floor cells seen (5 cm) | 5,737 | 4,654 | 5,148 | 6,698 | 2,400 | 1,438 |
| Floor seen vs only walked | 61.7% | 71.9% | 63.5% | 69.2% | 54.8% | 48.4% |
| Obstacle cells | 10,753 | 13,359 | 14,315 | 11,972 | 11,989 | 5,349 |
| Scene extent (m) | [10.85, 12.02, 2.96] | [8.2, 11.5, 5.01] | [12.5, 11.24, 2.97] | [13.63, 11.52, 2.84] | [9.11, 9.07, 3.04] | [4.73, 6.89, 3.07] |
| Scale from | input (metric depth) | input (metric depth) | input (metric depth) | input (metric depth) | input (metric depth) | input (metric depth) |
| Position error vs reference | - | - | - | - | - | 0.0885% |
| Rotation error vs reference, median | - | - | - | - | - | 0.094 deg |
| Map warnings (e.g. bent map) | 0 | 1 | 0 | 1 | 0 | 1 |
