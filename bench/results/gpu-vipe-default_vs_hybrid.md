# gpu-vipe-default vs hybrid

✓ = B better, ✗ = B worse (by the metric's direction, changes under 2% ignored), blank = info only.

## IMG_6070 (apartment, dim evening light, 74 s)

| | gpu-vipe-default | hybrid | |
|---|---|---|---|
| Frames with a camera | 100.0% | 73.2% | ✗ |
| Biggest camera jump (x median step) | 5.1 | 6.4 | ✗ |
| Camera height spread, p5-p95 (m) | 0.417 | 0.47 | ✗ |
| Floor spread (mm) | 51.3 | 52.6 | ✗ |
| Floor support (share of low points on the floor plane) | 0.063 | 0.067 | ✓ |
| Depth pixels kept, median per frame | 99.2% | 96.1% | ✗ |
| Mesh triangles | 3,883,328 | 2,515,422 |  |
| Floor cells seen (5 cm) | 5,731 | 5,737 |  |
| Floor seen vs only walked | 55.7% | 61.7% | ✓ |
| Obstacle cells | 14,175 | 10,753 |  |
| Scene extent (m) | [13.12, 14.06, 3.36] | [10.85, 12.02, 2.96] |  |
| Scale from | input (metric depth) | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 0 |  |
| Total time (min) | 6.9 | - |  |

## IMG_6075 (robotics lab / office, 85 s)

| | gpu-vipe-default | hybrid | |
|---|---|---|---|
| Frames with a camera | 100.0% | 91.7% | ✗ |
| Biggest camera jump (x median step) | 2.6 | 3.8 | ✗ |
| Camera height spread, p5-p95 (m) | 0.205 | 0.998 | ✗ |
| Floor spread (mm) | 53.1 | 50.8 | ✓ |
| Floor support (share of low points on the floor plane) | 0.047 | 0.064 | ✓ |
| Depth pixels kept, median per frame | 97.5% | 93.0% | ✗ |
| Mesh triangles | 2,677,556 | 3,354,648 |  |
| Floor cells seen (5 cm) | 2,867 | 4,654 | ✓ |
| Floor seen vs only walked | 57.2% | 71.9% | ✓ |
| Obstacle cells | 12,374 | 13,359 |  |
| Scene extent (m) | [9.6, 8.57, 3.32] | [8.2, 11.5, 5.01] |  |
| Scale from | input (metric depth) | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 1 | ✗ |
| Total time (min) | 6.5 | - |  |

## IMG_6076 (apartment, bright, faces a big window, 91 s)

| | gpu-vipe-default | hybrid | |
|---|---|---|---|
| Frames with a camera | 100.0% | 76.6% | ✗ |
| Biggest camera jump (x median step) | 2.1 | 3.2 | ✗ |
| Camera height spread, p5-p95 (m) | 0.304 | 0.314 | ✗ |
| Floor spread (mm) | 50.2 | 54.4 | ✗ |
| Floor support (share of low points on the floor plane) | 0.054 | 0.065 | ✓ |
| Depth pixels kept, median per frame | 99.1% | 95.7% | ✗ |
| Mesh triangles | 4,719,239 | 3,619,292 |  |
| Floor cells seen (5 cm) | 6,487 | 5,148 | ✗ |
| Floor seen vs only walked | 66.5% | 63.5% | ✗ |
| Obstacle cells | 12,399 | 14,315 |  |
| Scene extent (m) | [12.18, 14.67, 2.92] | [12.5, 11.24, 2.97] |  |
| Scale from | input (metric depth) | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 0 |  |
| Total time (min) | 7.8 | - |  |

## IMG_6077 (apartment, latest walk, 115 s)

| | gpu-vipe-default | hybrid | |
|---|---|---|---|
| Frames with a camera | 100.0% | 87.2% | ✗ |
| Biggest camera jump (x median step) | 2.7 | 4.5 | ✗ |
| Camera height spread, p5-p95 (m) | 4.403 | 0.772 | ✓ |
| Floor spread (mm) | 49.3 | 50.4 | ✗ |
| Floor support (share of low points on the floor plane) | 0.066 | 0.065 |  |
| Depth pixels kept, median per frame | 99.2% | 95.6% | ✗ |
| Mesh triangles | 4,813,380 | 3,255,368 |  |
| Floor cells seen (5 cm) | 4,660 | 6,698 | ✓ |
| Floor seen vs only walked | 48.1% | 69.2% | ✓ |
| Obstacle cells | 14,276 | 11,972 |  |
| Scene extent (m) | [12.5, 12.11, 8.03] | [13.63, 11.52, 2.84] |  |
| Scale from | input (metric depth) | input (metric depth) |  |
| Map warnings (e.g. bent map) | 1 | 1 |  |
| Total time (min) | 8.0 | - |  |

## IMG_6513 (office, new walk (Nov 11), 76 s)

| | gpu-vipe-default | hybrid | |
|---|---|---|---|
| Frames with a camera | 100.0% | 85.8% | ✗ |
| Biggest camera jump (x median step) | 3.6 | 5.4 | ✗ |
| Camera height spread, p5-p95 (m) | 0.353 | 0.435 | ✗ |
| Floor spread (mm) | 54.3 | 52.9 | ✓ |
| Floor support (share of low points on the floor plane) | 0.046 | 0.041 | ✗ |
| Depth pixels kept, median per frame | 97.9% | 93.9% | ✗ |
| Mesh triangles | 2,698,012 | 2,673,394 |  |
| Floor cells seen (5 cm) | 3,034 | 2,400 | ✗ |
| Floor seen vs only walked | 62.8% | 54.8% | ✗ |
| Obstacle cells | 13,642 | 11,989 |  |
| Scene extent (m) | [9.46, 9.19, 3.11] | [9.11, 9.07, 3.04] |  |
| Scale from | input (metric depth) | input (metric depth) |  |
| Map warnings (e.g. bent map) | 0 | 0 |  |
| Total time (min) | 6.9 | - |  |

## room (Mip-NeRF 360 room, 311 photos, reference poses)

| | gpu-vipe-default | hybrid | |
|---|---|---|---|
| Frames with a camera | 100.0% | 100.0% |  |
| Biggest camera jump (x median step) | - | 46.2 |  |
| Camera height spread, p5-p95 (m) | 1.467 | 0.558 | ✓ |
| Floor spread (mm) | 50.5 | 53.0 | ✗ |
| Floor support (share of low points on the floor plane) | 0.053 | 0.033 | ✗ |
| Depth pixels kept, median per frame | 97.2% | 91.4% | ✗ |
| Mesh triangles | 984,154 | 1,248,670 |  |
| Floor cells seen (5 cm) | 1,398 | 1,438 | ✓ |
| Floor seen vs only walked | 40.0% | 48.4% | ✓ |
| Obstacle cells | 4,063 | 5,349 |  |
| Scene extent (m) | [6.15, 4.58, 4.84] | [4.73, 6.89, 3.07] |  |
| Scale from | input (metric depth) | input (metric depth) |  |
| Position error vs reference | 34.9618% | 0.0885% | ✓ |
| Rotation error vs reference, median | 148.233 deg | 0.094 deg | ✓ |
| Map warnings (e.g. bent map) | 1 | 1 |  |
| Total time (min) | 4.2 | - |  |
