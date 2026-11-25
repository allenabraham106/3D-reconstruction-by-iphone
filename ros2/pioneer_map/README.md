# pioneer_map: serve an iPhone scan to ROS 2 / nav2

Takes the `out/<scan>/robot/` folder that `run.py` writes and serves it to a robot:

| Topic | Type | What |
|---|---|---|
| `/map` | `nav_msgs/OccupancyGrid` | 5 cm floor plan from nav2 `map_server` (free / occupied / unknown) |
| `/pioneer/voxels` | `sensor_msgs/PointCloud2` | 5 cm occupied voxels, latched: 3D costmaps, footstep / collision planning |
| `/pioneer/mesh` | `visualization_msgs/Marker` | the coloured TSDF mesh for RViz (decimated to 300 k triangles) |

All are in the `map` frame: metres, Z up, floor at Z = 0.

## Build and run

```bash
cp -r ros2/pioneer_map ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select pioneer_map && source install/setup.bash
ros2 launch pioneer_map map.launch.py robot_dir:=/path/to/out/IMG_6075/robot
```

Needs `nav2_map_server`, `nav2_lifecycle_manager` and `tf2_ros` (e.g. `sudo apt install ros-$ROS_DISTRO-navigation2`).

Launch arguments:

- `use_inferred:=true` serves `occupancy_with_inferred.yaml`, which also counts small unseen gaps (< 0.5 m) between known floor as free. It's more permissive; the default map only uses floor that was seen or walked on.
- `publish_odom_tf:=false` turns off the identity `map -> odom` transform once your robot's localization (e.g. AMCL against `/map`) publishes it.

In RViz, add *Map* (`/map`), *PointCloud2* (`/pioneer/voxels`) and *Marker* (`/pioneer/mesh`), with the fixed frame set to `map`.

## Tell the robot where to go (`goto` node)

Type a sentence, the robot drives there. `gpu/goto_server.py` (on the GPU machine) finds the place in the scan and plans the route; this node hands the goal to nav2.

```bash
python gpu/goto_server.py --host 0.0.0.0                                     # on the GPU machine
ros2 run pioneer_map goto --ros-args -p server:=http://<gpu-machine>:8765 -p scan:=IMG_6075
ros2 topic pub --once /pioneer/goto_text std_msgs/String "{data: 'take me to the red toolbox'}"
```

| Topic | Type | What |
|---|---|---|
| `/pioneer/goto_text` (in) | `std_msgs/String` | the sentence |
| `/goal_pose` | `geometry_msgs/PoseStamped` | nav2's goal topic (the one RViz's "Nav2 Goal" uses): where the phone stood when it filmed the target, facing it |
| `/pioneer/goto_path` | `nav_msgs/Path` | the planned route, for RViz |
| `/pioneer/goto_target` | `geometry_msgs/PointStamped` | the object, in 3D |
| `/pioneer/goto_status` | `std_msgs/String` | JSON: found or not, confidence, path length |

`-p use_action:=true` sends a nav2 `NavigateToPose` action instead (needs `nav2_msgs`) and logs when the robot arrives. `-p result:=<result.json>` replays a saved `gpu/goto.py` result with no server. Everything is in the `map` frame of the same scan that `map.launch.py` serves. Tested on ROS 2 Jazzy: "take me to the red toolbox" published the goal on `/goal_pose` and the route on `/pioneer/goto_path` (nav2 itself was not installed on that machine).

## Notes

- **Scale:** maps from plain video assume the phone was held about 1.4 m above the floor (`run.py --phone-height`), which is typically a few percent off. Captures from the Pioneer Home app are metric from ARKit.
- **Glass and windows** often show up as unknown cells rather than walls. Check the map before letting a robot drive near them.
- The map-loading helpers are unit-tested without ROS: `ROBOT_DIR=.../robot python -m pytest test/`. The node and launch file follow standard rclpy and nav2 patterns but haven't been run against a live ROS install yet.
