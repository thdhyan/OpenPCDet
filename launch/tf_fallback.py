"""Publish fallback static transforms on /tf (not /tf_static).

NVBLOX subscribes to /tf for frame lookups, but static_transform_publisher
publishes on /tf_static. This node uses TransformBroadcaster to publish
identity transforms on /tf so NVBLOX can always find required frames.

Published transforms:
    map → pelvis  (identity — CUVSLAM overrides when tracking)
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class TFFallback(Node):
    def __init__(self):
        super().__init__("tf_fallback")
        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.5, self._broadcast)  # 2 Hz
        self.get_logger().info("tf_fallback: broadcasting map→pelvis on /tf")

    def _broadcast(self):
        now = self.get_clock().now().to_msg()

        # map → pelvis (identity)
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "map"
        t.child_frame_id = "pelvis"
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)


def main():
    rclpy.init()
    node = TFFallback()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
