"""Convert depth image from 32FC1 (metres) to uint16 (millimetres).

Isaac Sim's D435 publishes depth as sensor_msgs/Image with encoding=32FC1
in metres. CUVSLAM in RGBD mode expects uint16 encoding in millimetres.

Subscribes:  /g1/camera/depth  (sensor_msgs/Image, 32FC1, metres)
Publishes:   /g1/camera/depth_uint16  (sensor_msgs/Image, mono16, mm)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


class DepthFormatConverter(Node):
    def __init__(self):
        super().__init__("depth_format_converter")

        self._sub = self.create_subscription(
            Image, "/g1/camera/depth", self._callback,
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST),
        )
        self._pub = self.create_publisher(
            Image, "/g1/camera/depth_uint16",
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                        history=QoSHistoryPolicy.KEEP_LAST),
        )
        self.get_logger().info("depth_format_converter started")

    def _callback(self, msg: Image):
        """Convert 32FC1 metres → mono16 millimetres."""
        if msg.encoding != "32FC1":
            self.get_logger().warn(
                f"Unexpected encoding '{msg.encoding}', expected 32FC1")
            return

        depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height, msg.width)

        # Clamp to [0, 65.535] m and convert to uint16 mm
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

        out = Image()
        out.header = msg.header  # preserve frame_id + stamp
        out.height = msg.height
        out.width = msg.width
        out.encoding = "mono16"
        out.is_bigendian = False
        out.step = 2 * msg.width
        out.data = depth_mm.tobytes()
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DepthFormatConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
