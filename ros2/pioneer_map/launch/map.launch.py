"""Serve an iPhone scan to nav2 + RViz.

    ros2 launch pioneer_map map.launch.py robot_dir:=/path/to/out/<scan>/robot

Starts:
  * nav2 map_server with robot/occupancy.yaml -> /map (nav_msgs/OccupancyGrid), managed by a
    lifecycle manager so it activates on its own
    (use_inferred:=true serves occupancy_with_inferred.yaml instead)
  * pioneer_map voxel_publisher -> /pioneer/voxels (PointCloud2) and /pioneer/mesh (Marker)
  * a static map -> odom transform (identity) so RViz / nav2 have a tree until your robot's
    localization takes over; disable with publish_odom_tf:=false
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    robot_dir = LaunchConfiguration("robot_dir").perform(context)
    inferred = LaunchConfiguration("use_inferred").perform(context).lower() == "true"
    odom_tf = LaunchConfiguration("publish_odom_tf").perform(context).lower() == "true"
    yaml_name = "occupancy_with_inferred.yaml" if inferred else "occupancy.yaml"
    yaml_path = os.path.join(robot_dir, yaml_name)
    nodes = [
        Node(package="nav2_map_server", executable="map_server", name="map_server",
             output="screen", parameters=[{"yaml_filename": yaml_path, "frame_id": "map"}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_map", output="screen",
             parameters=[{"autostart": True, "node_names": ["map_server"]}]),
        Node(package="pioneer_map", executable="voxel_publisher", name="pioneer_voxel_publisher",
             output="screen", parameters=[{"robot_dir": robot_dir}]),
    ]
    if odom_tf:
        nodes.append(Node(package="tf2_ros", executable="static_transform_publisher",
                          name="map_to_odom",
                          arguments=["--frame-id", "map", "--child-frame-id", "odom"]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_dir", description="out/<scan>/robot folder from run.py"),
        DeclareLaunchArgument("use_inferred", default_value="false",
                              description="serve the map that includes inferred floor"),
        DeclareLaunchArgument("publish_odom_tf", default_value="true"),
        OpaqueFunction(function=_nodes),
    ])
