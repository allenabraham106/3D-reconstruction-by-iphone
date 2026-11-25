# cpu-5070box vs hybrid

✓ = B better, ✗ = B worse (by the metric's direction, changes under 2% ignored), blank = info only.

## IMG_6070 (apartment, dim evening light, 74 s)

| | cpu-5070box | hybrid | |
|---|---|---|---|
| Frames with a camera | 88.3% | 73.2% | ✗ |
| Biggest camera jump (x median step) | 7.0 | 6.4 | ✓ |
| Camera height spread, p5-p95 (m) | 0.106 | 0.47 | ✗ |
| Floor spread (mm) | 49.4 | 52.6 | ✗ |
| Floor support (share of low points on the floor plane) | 0.177 | 0.067 | ✗ |
| Depth pixels kept, median per frame | 15.8% | 96.1% | ✓ |
| Mesh triangles | 1,314,216 | 2,515,422 |  |
| Floor cells seen (5 cm) | 5,164 | 5,737 | ✓ |
| Floor seen vs only walked | 79.0% | 61.7% | ✗ |
| Obstacle cells | 8,469 | 10,753 |  |
| Scene extent (m) | [7.48, 10.11, 2.76] | [10.85, 12.02, 2.96] |  |
| Scale from | phone_height=1.4m | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 0 |  |
| Total time (min) | 9.2 | - |  |

## IMG_6075 (robotics lab / office, 85 s)

| | cpu-5070box | hybrid | |
|---|---|---|---|
| Frames with a camera | 100.0% | 91.7% | ✗ |
| Biggest camera jump (x median step) | 3.8 | 3.8 |  |
| Camera height spread, p5-p95 (m) | 0.495 | 0.998 | ✗ |
| Floor spread (mm) | 41.8 | 50.8 | ✗ |
| Floor support (share of low points on the floor plane) | 0.063 | 0.064 |  |
| Depth pixels kept, median per frame | 35.6% | 93.0% | ✓ |
| Mesh triangles | 2,034,830 | 3,354,648 |  |
| Floor cells seen (5 cm) | 712 | 4,654 | ✓ |
| Floor seen vs only walked | 40.9% | 71.9% | ✓ |
| Obstacle cells | 3,949 | 13,359 |  |
| Scene extent (m) | [4.54, 5.29, 2.08] | [8.2, 11.5, 5.01] |  |
| Scale from | phone_height=1.4m | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 1 | ✗ |
| Total time (min) | 17.6 | - |  |

## IMG_6076 (apartment, bright, faces a big window, 91 s)

| | cpu-5070box | hybrid | |
|---|---|---|---|
| Frames with a camera | 90.3% | 76.6% | ✗ |
| Biggest camera jump (x median step) | 4.5 | 3.2 | ✓ |
| Camera height spread, p5-p95 (m) | 0.201 | 0.314 | ✗ |
| Floor spread (mm) | 52.8 | 54.4 | ✗ |
| Floor support (share of low points on the floor plane) | 0.225 | 0.065 | ✗ |
| Depth pixels kept, median per frame | 14.9% | 95.7% | ✓ |
| Mesh triangles | 1,141,472 | 3,619,292 |  |
| Floor cells seen (5 cm) | 4,736 | 5,148 | ✓ |
| Floor seen vs only walked | 83.4% | 63.5% | ✗ |
| Obstacle cells | 6,786 | 14,315 |  |
| Scene extent (m) | [9.99, 8.59, 2.71] | [12.5, 11.24, 2.97] |  |
| Scale from | phone_height=1.4m | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 0 |  |
| Total time (min) | 12.4 | - |  |

## IMG_6077 (apartment, latest walk, 115 s)

| | cpu-5070box | hybrid | |
|---|---|---|---|
| Frames with a camera | 97.8% | 87.2% | ✗ |
| Biggest camera jump (x median step) | 4.7 | 4.5 | ✓ |
| Camera height spread, p5-p95 (m) | 0.408 | 0.772 | ✗ |
| Floor spread (mm) | 30.5 | 50.4 | ✗ |
| Floor support (share of low points on the floor plane) | 0.178 | 0.065 | ✗ |
| Depth pixels kept, median per frame | 22.8% | 95.6% | ✓ |
| Mesh triangles | 990,092 | 3,255,368 |  |
| Floor cells seen (5 cm) | 6,690 | 6,698 |  |
| Floor seen vs only walked | 97.1% | 69.2% | ✗ |
| Obstacle cells | 6,672 | 11,972 |  |
| Scene extent (m) | [9.65, 9.12, 2.14] | [13.63, 11.52, 2.84] |  |
| Scale from | phone_height=1.4m | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 1 | ✗ |
| Total time (min) | 10.9 | - |  |

## IMG_6513 (office, new walk (Nov 11), 76 s)

| | cpu-5070box | hybrid | |
|---|---|---|---|
| Frames with a camera | 96.9% | 85.8% | ✗ |
| Biggest camera jump (x median step) | 5.5 | 5.4 |  |
| Camera height spread, p5-p95 (m) | 0.378 | 0.435 | ✗ |
| Floor spread (mm) | 40.4 | 52.9 | ✗ |
| Floor support (share of low points on the floor plane) | 0.106 | 0.041 | ✗ |
| Depth pixels kept, median per frame | 31.8% | 93.9% | ✓ |
| Mesh triangles | 2,232,108 | 2,673,394 |  |
| Floor cells seen (5 cm) | 2,883 | 2,400 | ✗ |
| Floor seen vs only walked | 94.9% | 54.8% | ✗ |
| Obstacle cells | 7,134 | 11,989 |  |
| Scene extent (m) | [6.53, 6.91, 2.41] | [9.11, 9.07, 3.04] |  |
| Scale from | phone_height=1.4m | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 0 |  |
| Total time (min) | 17.1 | - |  |

## room (Mip-NeRF 360 room, 311 photos, reference poses)

| | cpu-5070box | hybrid | |
|---|---|---|---|
| Frames with a camera | 100.0% | 100.0% |  |
| Biggest camera jump (x median step) | - | 46.2 |  |
| Camera height spread, p5-p95 (m) | 0.386 | 0.558 | ✗ |
| Floor spread (mm) | 41.4 | 53.0 | ✗ |
| Floor support (share of low points on the floor plane) | 0.081 | 0.033 | ✗ |
| Depth pixels kept, median per frame | 47.9% | 91.4% | ✓ |
| Mesh triangles | 1,756,100 | 1,248,670 |  |
| Floor cells seen (5 cm) | 2,683 | 1,438 | ✗ |
| Floor seen vs only walked | 82.3% | 48.4% | ✗ |
| Obstacle cells | 4,010 | 5,349 |  |
| Scene extent (m) | [4.27, 7.38, 1.92] | [4.73, 6.89, 3.07] |  |
| Scale from | phone_height=1.4m | input (metric depth) |  |
| Position error vs reference | 0.0885% | 0.0885% |  |
| Rotation error vs reference, median | 0.094 deg | 0.094 deg |  |
| Map warnings (e.g. bent map) | 0 | 1 | ✗ |
| Total time (min) | 9.9 | - |  |
