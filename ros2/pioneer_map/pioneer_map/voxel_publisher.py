"""Publish the scan's 3D voxels (and optionally its mesh) for a robot / RViz.

  /pioneer/voxels   sensor_msgs/PointCloud2   5 cm occupied voxels, latched (transient local)
  /pioneer/mesh     visualization_msgs/Marker TRIANGLE_LIST of the TSDF mesh, latched

Both are in the `map` frame used by nav2's map_server (metres, Z up, floor at Z = 0).

    ros2 run pioneer_map voxel_publisher --ros-args -p robot_dir:=/path/to/out/<scan>/robot
"""
import os

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker

from pioneer_map.convert import (decimate_faces, load_voxels, pointcloud2_fields_and_bytes,
                                 read_mesh_ply)


class VoxelPublisher(Node):
    def __init__(self):
        super().__init__("pioneer_voxel_publisher")
        self.declare_parameter("robot_dir", "")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_mesh", True)
        self.declare_parameter("max_mesh_faces", 300000)
        robot_dir = self.get_parameter("robot_dir").value
        frame = self.get_parameter("frame_id").value
        if not robot_dir or not os.path.isdir(robot_dir):
            raise RuntimeError(f"robot_dir '{robot_dir}' not found (pass -p robot_dir:=.../robot)")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        header = Header(frame_id=frame, stamp=self.get_clock().now().to_msg())

        pts, vs = load_voxels(os.path.join(robot_dir, "voxels.npz"))
        fields, step, data = pointcloud2_fields_and_bytes(pts)
        cloud = PointCloud2(header=header, height=1, width=len(pts), is_dense=True,
                            is_bigendian=False, point_step=step, row_step=step * len(pts),
                            data=data,
                            fields=[PointField(name=n, offset=o, datatype=PointField.FLOAT32,
                                               count=1) for n, o in fields])
        self.voxel_pub = self.create_publisher(PointCloud2, "/pioneer/voxels", latched)
        self.voxel_pub.publish(cloud)
        self.get_logger().info(f"published {len(pts):,} voxels ({vs*100:.0f} cm) on /pioneer/voxels")

        mesh_path = os.path.join(robot_dir, "mesh_metric.ply")
        if self.get_parameter("publish_mesh").value and os.path.exists(mesh_path):
            V, F, C = read_mesh_ply(mesh_path)
            F = decimate_faces(F, int(self.get_parameter("max_mesh_faces").value))
            m = Marker(header=header, ns="pioneer", id=0, type=Marker.TRIANGLE_LIST,
                       action=Marker.ADD)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.a = 1.0
            idx = F.reshape(-1)
            m.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in V[idx]]
            m.colors = [ColorRGBA(r=r / 255.0, g=g / 255.0, b=b / 255.0, a=1.0)
                        for r, g, b in C[idx]]
            self.mesh_pub = self.create_publisher(Marker, "/pioneer/mesh", latched)
            self.mesh_pub.publish(m)
            self.get_logger().info(f"published mesh ({len(F):,} triangles) on /pioneer/mesh")


def main():
    rclpy.init()
    node = VoxelPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
