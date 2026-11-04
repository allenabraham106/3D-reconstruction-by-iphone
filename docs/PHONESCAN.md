# PhoneScan documentation

The full illustrated guide is published as an artifact. This file holds the images and a short index;
`ios/README.md` has the setup, format and reconstruction commands.

## Screens (mockups from the SwiftUI layout, not device captures)

| | | | |
|---|---|---|---|
| ![](images/05-no-lidar.webp) iPhone 17 starting | ![](images/05b-iphone17-recording.webp) iPhone 17 recording | ![](images/01-ready.webp) Pro ready | ![](images/02-recording.webp) Pro recording + mesh |
| ![](images/03-too-fast.webp) Tracking warning | ![](images/04-saved.webp) Saved | ![](images/06-library.webp) Captures | ![](images/07-settings.webp) Settings |

## Real tool output (synthetic 48-frame capture in the app's exact format)

![Fused cloud, interactive viewer frame](images/cloud_viewer.png)
![Floor plan with occupancy and camera path](images/cloud_plan.webp)

One keyframe: ![rgb](images/kf_rgb.webp) ![depth](images/kf_depth.webp) ![confidence](images/kf_conf.webp)

`fuse_depth.py`: 48 frames, 2,342,201 raw points, 1,349,750 after 1 cm voxels. The cloud measures
4.60 × 2.65 × 4.09 m for a 4.5 × 2.6 × 4.0 m room.
