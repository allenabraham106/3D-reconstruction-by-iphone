"""Robot link: localize a robot (or the phone) inside a map built from a Pioneer Home scan.

    reloc_map.py   build a relocalization map (3D landmarks + SIFT descriptors) from a run.py output
    localize.py    image + intrinsics -> camera pose in the robot map frame (PnP + RANSAC)
    server.py      HTTP server the Pioneer Home app talks to; optional ROS 2 / RealSense hooks
"""
