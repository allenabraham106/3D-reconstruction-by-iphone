# Pioneer Home documentation

<img src="images/pioneer-home/icon.png" width="96">

The full illustrated guide is published as an artifact. This file holds the images and a short index;
`ios/README.md` has the setup, format and reconstruction commands.

## Screens (mockups from the SwiftUI layout, not device captures)

| <img src="images/pioneer-home/home.webp" width="180"> | <img src="images/pioneer-home/mapping.webp" width="180"> | <img src="images/pioneer-home/scans.webp" width="180"> | <img src="images/pioneer-home/on-pioneer.webp" width="180"> |
|:-:|:-:|:-:|:-:|
| Home: 3D map, Pioneer localized | Mapping | My Scans | Maps on Pioneer |

<img src="images/pioneer-home/pioneer-model.webp" width="200"> <img src="images/pioneer-home/tutorial.webp" width="180">

## Real tool output (synthetic 48-frame capture in the app's exact format)

![Fused cloud, interactive viewer frame](images/cloud_viewer.png)
![Floor plan with occupancy and camera path](images/cloud_plan.webp)

One keyframe: ![rgb](images/kf_rgb.webp) ![depth](images/kf_depth.webp) ![confidence](images/kf_conf.webp)

`fuse_depth.py`: 48 frames, 2,342,201 raw points, 1,349,750 after 1 cm voxels. The cloud measures
4.60 × 2.65 × 4.09 m for a 4.5 × 2.6 × 4.0 m room.
