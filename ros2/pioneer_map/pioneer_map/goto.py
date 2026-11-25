"""Type a sentence, the robot drives there: gpu/goto_server.py finds the place in the scan and plans
the route; this node hands it to nav2.

    ros2 run pioneer_map goto --ros-args -p server:=http://gpu-box:8765 -p scan:=IMG_6075
    ros2 topic pub --once /pioneer/goto_text std_msgs/String "{data: 'take me to the red toolbox'}"

    # or replay a saved result (gpu/goto.py's result.json), no server needed:
    ros2 run pioneer_map goto --ros-args -p result:=/path/to/out_goto/IMG_6075/the_red_toolbox/result.json

On a sentence it publishes, in the map frame of the same scan (pioneer_map serves that map):
  /goal_pose            geometry_msgs/PoseStamped   nav2's goal topic (what RViz "Nav2 Goal" sends):
                                                    stop where the phone filmed the target, facing it
  /pioneer/goto_path    nav_msgs/Path               the planned route, for RViz
  /pioneer/goto_target  geometry_msgs/PointStamped  the object itself (3D)
  /pioneer/goto_status  std_msgs/String             JSON: found or not, confidence, path length
With nav2_msgs installed and use_action:=true it sends a NavigateToPose action instead of /goal_pose
and reports when the robot gets there.
"""
import json
import math
import threading
import urllib.parse
import urllib.request


# ----------------------------------------------------------------------------- ROS-free helpers
def yaw_quaternion(yaw):
    """(x, y, z, w) for a rotation of `yaw` radians about +Z."""
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def goal_from_result(res):
    """result.json -> (x, y, yaw) of the nav2 goal, or None when the prompt wasn't found."""
    if not res.get("found") or "goal" not in res:
        return None
    g = res["goal"]
    return float(g["x"]), float(g["y"]), float(g["yaw"])


def path_from_result(res):
    """result.json -> [(x, y, yaw)] along the route, yaw pointing to the next point."""
    P = [tuple(map(float, p)) for p in res.get("path_m", [])]
    out = []
    for i, (x, y) in enumerate(P):
        nx, ny = P[min(i + 1, len(P) - 1)]
        yaw = math.atan2(ny - y, nx - x) if (nx, ny) != (x, y) else (out[-1][2] if out else 0.0)
        out.append((x, y, yaw))
    if out and goal_from_result(res):
        out[-1] = goal_from_result(res)
    return out


def ask_server(server, scan, prompt, timeout=60):
    url = f"{server.rstrip('/')}/api/plan?" + urllib.parse.urlencode({"scan": scan, "prompt": prompt})
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


# ----------------------------------------------------------------------------- node
def main():
    import rclpy
    from geometry_msgs.msg import PointStamped, PoseStamped
    from nav_msgs.msg import Path
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import String

    class GotoNode(Node):
        def __init__(self):
            super().__init__("pioneer_goto")
            self.server = self.declare_parameter("server", "http://127.0.0.1:8765").value
            self.scan = self.declare_parameter("scan", "IMG_6075").value
            self.frame = self.declare_parameter("frame_id", "map").value
            result = self.declare_parameter("result", "").value
            use_action = self.declare_parameter("use_action", False).value
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
            self.path_pub = self.create_publisher(Path, "/pioneer/goto_path", latched)
            self.target_pub = self.create_publisher(PointStamped, "/pioneer/goto_target", latched)
            self.status_pub = self.create_publisher(String, "/pioneer/goto_status", latched)
            self.create_subscription(String, "/pioneer/goto_text", self.on_text, 10)
            self.action = None
            if use_action:
                try:
                    from nav2_msgs.action import NavigateToPose
                    from rclpy.action import ActionClient
                    self.NavigateToPose = NavigateToPose
                    self.action = ActionClient(self, NavigateToPose, "navigate_to_pose")
                except ImportError:
                    self.get_logger().warn("nav2_msgs not installed: publishing /goal_pose instead")
            if result:
                with open(result) as f:
                    self.create_timer(1.0, self._once(json.load(f)))
            self.get_logger().info(f"ready: publish a sentence on /pioneer/goto_text (scan {self.scan}, server {self.server})")

        def _once(self, res):
            done = []

            def fire():
                if not done:
                    done.append(1)
                    self.send(res)
            return fire

        def on_text(self, msg):
            text = msg.data.strip()
            self.get_logger().info(f"asked: {text!r}")

            def work():
                try:
                    self.send(ask_server(self.server, self.scan, text))
                except Exception as e:  # noqa: BLE001 - server down / bad reply: report, keep running
                    self.get_logger().error(f"goto server: {e}")
                    self.status_pub.publish(String(data=json.dumps({"prompt": text, "error": str(e)})))
            threading.Thread(target=work, daemon=True).start()

        def pose(self, x, y, yaw, stamp):
            p = PoseStamped()
            p.header.frame_id, p.header.stamp = self.frame, stamp
            p.pose.position.x, p.pose.position.y = x, y
            q = yaw_quaternion(yaw)
            p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = q
            return p

        def send(self, res):
            stamp = self.get_clock().now().to_msg()
            status = {k: res.get(k) for k in ("prompt", "query", "found", "confidence", "path_length_m", "stop_distance_m")}
            self.status_pub.publish(String(data=json.dumps(status)))
            goal = goal_from_result(res)
            if goal is None:
                self.get_logger().warn(f"{res.get('query', res.get('prompt'))!r} is not in this scan "
                                       f"(confidence {res.get('confidence', 0):.2f}): not moving")
                return
            path = Path()
            path.header.frame_id, path.header.stamp = self.frame, stamp
            path.poses = [self.pose(x, y, yaw, stamp) for x, y, yaw in path_from_result(res)]
            self.path_pub.publish(path)
            t = PointStamped()
            t.header.frame_id, t.header.stamp = self.frame, stamp
            t.point.x, t.point.y, t.point.z = (float(v) for v in res["target_m"])
            self.target_pub.publish(t)
            g = self.pose(*goal, stamp)
            if self.action is not None and self.action.wait_for_server(timeout_sec=2.0):
                req = self.NavigateToPose.Goal()
                req.pose = g
                fut = self.action.send_goal_async(req)
                fut.add_done_callback(lambda f: f.result().get_result_async().add_done_callback(
                    lambda r: self.get_logger().info(f"nav2 finished {res['query']!r}: status {r.result().status}")))
            else:
                self.goal_pub.publish(g)
            self.get_logger().info(f"going to {res['query']!r}: goal ({goal[0]:.2f}, {goal[1]:.2f}) m, "
                                   f"yaw {math.degrees(goal[2]):.0f} deg, path {res['path_length_m']:.1f} m")

    from rclpy.executors import ExternalShutdownException
    rclpy.init()
    node = GotoNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
