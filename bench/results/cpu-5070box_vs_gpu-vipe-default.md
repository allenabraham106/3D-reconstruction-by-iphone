# cpu-5070box vs gpu-vipe-default

✓ = B better, ✗ = B worse (by the metric's direction, changes under 2% ignored), blank = info only.

## IMG_6070 (apartment, dim evening light, 74 s)

| | cpu-5070box | gpu-vipe-default | |
|---|---|---|---|
| Frames with a camera | 88.3% | 100.0% | ✓ |
| Biggest camera jump (x median step) | 7.0 | 5.1 | ✓ |
| Camera height spread, p5-p95 (m) | 0.106 | 0.417 | ✗ |
| Floor spread (mm) | 49.4 | 51.3 | ✗ |
| Floor support (share of low points on the floor plane) | 0.177 | 0.063 | ✗ |
| Depth pixels kept, median per frame | 15.8% | 99.2% | ✓ |
| Mesh triangles | 1,314,216 | 3,883,328 |  |
| Floor cells seen (5 cm) | 5,164 | 5,731 | ✓ |
| Floor seen vs only walked | 79.0% | 55.7% | ✗ |
| Obstacle cells | 8,469 | 14,175 |  |
| Scene extent (m) | [7.48, 10.11, 2.76] | [13.12, 14.06, 3.36] |  |
| Scale from | phone_height=1.4m | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 0 |  |
| Total time (min) | 9.2 | 6.9 | ✓ |

## IMG_6075 (robotics lab / office, 85 s)

| | cpu-5070box | gpu-vipe-default | |
|---|---|---|---|
| Frames with a camera | 100.0% | 100.0% |  |
| Biggest camera jump (x median step) | 3.8 | 2.6 | ✓ |
| Camera height spread, p5-p95 (m) | 0.495 | 0.205 | ✓ |
| Floor spread (mm) | 41.8 | 53.1 | ✗ |
| Floor support (share of low points on the floor plane) | 0.063 | 0.047 | ✗ |
| Depth pixels kept, median per frame | 35.6% | 97.5% | ✓ |
| Mesh triangles | 2,034,830 | 2,677,556 |  |
| Floor cells seen (5 cm) | 712 | 2,867 | ✓ |
| Floor seen vs only walked | 40.9% | 57.2% | ✓ |
| Obstacle cells | 3,949 | 12,374 |  |
| Scene extent (m) | [4.54, 5.29, 2.08] | [9.6, 8.57, 3.32] |  |
| Scale from | phone_height=1.4m | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 0 |  |
| Total time (min) | 17.6 | 6.5 | ✓ |

## IMG_6076 (apartment, bright, faces a big window, 91 s)

| | cpu-5070box | gpu-vipe-default | |
|---|---|---|---|
| Frames with a camera | 90.3% | 100.0% | ✓ |
| Biggest camera jump (x median step) | 4.5 | 2.1 | ✓ |
| Camera height spread, p5-p95 (m) | 0.201 | 0.304 | ✗ |
| Floor spread (mm) | 52.8 | 50.2 | ✓ |
| Floor support (share of low points on the floor plane) | 0.225 | 0.054 | ✗ |
| Depth pixels kept, median per frame | 14.9% | 99.1% | ✓ |
| Mesh triangles | 1,141,472 | 4,719,239 |  |
| Floor cells seen (5 cm) | 4,736 | 6,487 | ✓ |
| Floor seen vs only walked | 83.4% | 66.5% | ✗ |
| Obstacle cells | 6,786 | 12,399 |  |
| Scene extent (m) | [9.99, 8.59, 2.71] | [12.18, 14.67, 2.92] |  |
| Scale from | phone_height=1.4m | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 0 |  |
| Total time (min) | 12.4 | 7.8 | ✓ |

## IMG_6077 (apartment, latest walk, 115 s)

| | cpu-5070box | gpu-vipe-default | |
|---|---|---|---|
| Frames with a camera | 97.8% | 100.0% | ✓ |
| Biggest camera jump (x median step) | 4.7 | 2.7 | ✓ |
| Camera height spread, p5-p95 (m) | 0.408 | 4.403 | ✗ |
| Floor spread (mm) | 30.5 | 49.3 | ✗ |
| Floor support (share of low points on the floor plane) | 0.178 | 0.066 | ✗ |
| Depth pixels kept, median per frame | 22.8% | 99.2% | ✓ |
| Mesh triangles | 990,092 | 4,813,380 |  |
| Floor cells seen (5 cm) | 6,690 | 4,660 | ✗ |
| Floor seen vs only walked | 97.1% | 48.1% | ✗ |
| Obstacle cells | 6,672 | 14,276 |  |
| Scene extent (m) | [9.65, 9.12, 2.14] | [12.5, 12.11, 8.03] |  |
| Scale from | phone_height=1.4m | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 1 | ✗ |
| Total time (min) | 10.9 | 8.0 | ✓ |

## room (Mip-NeRF 360 room, 311 photos, reference poses)

| | cpu-5070box | gpu-vipe-default | |
|---|---|---|---|
| Frames with a camera | 100.0% | 100.0% |  |
| Camera height spread, p5-p95 (m) | 0.386 | 1.467 | ✗ |
| Floor spread (mm) | 41.4 | 50.5 | ✗ |
| Floor support (share of low points on the floor plane) | 0.081 | 0.053 | ✗ |
| Depth pixels kept, median per frame | 47.9% | 97.2% | ✓ |
| Mesh triangles | 1,756,100 | 984,154 |  |
| Floor cells seen (5 cm) | 2,683 | 1,398 | ✗ |
| Floor seen vs only walked | 82.3% | 40.0% | ✗ |
| Obstacle cells | 4,010 | 4,063 |  |
| Scene extent (m) | [4.27, 7.38, 1.92] | [6.15, 4.58, 4.84] |  |
| Scale from | phone_height=1.4m | input (metric depth) |  |
| Position error vs reference | 0.0885% | 34.9618% | ✗ |
| Rotation error vs reference, median | 0.094 deg | 148.233 deg | ✗ |
| Map warnings (e.g. bent map) | 0 | 1 | ✗ |
| Total time (min) | 9.9 | 4.2 | ✓ |
