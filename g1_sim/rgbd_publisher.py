"""Publish a colorized RGB-D PointCloud2 (RealSense-style) for the D435.

``ROS2CameraHelper``'s ``type`` input has no RGB-fusion option - checked
against its own OGN schema (``OgnROS2CameraHelper.ogn``): the allowed tokens
are ``rgb``/``rgb_h264``/``depth``/``depth_pcl``/segmentation/bbox variants
only, no combined XYZRGB output. ``depth_pcl`` gives XYZ-only points (see
``g1_sim.rtx_camera``'s docstring for the frame-convention bug that was also
found there). This module fuses the ``rgb`` and ``distance_to_image_plane``
annotators by hand, projecting through the pinhole model directly into the
optical frame (X-right, Y-down, Z-forward) - the same frame
``attach_camera_publishers`` now publishes a static TF for - and packs color
into the point cloud's ``rgb`` field using the standard PCL/RViz convention
(a uint32 0x00RRGGBB reinterpreted as float32), same as RealSense's own
``/camera/depth/color/points``.
"""

from __future__ import annotations

import numpy as np


class RgbdPointCloudPublisher:
    """Reads RGB + depth annotators from one render product, publishes a
    colorized ``sensor_msgs/PointCloud2``.

    Args:
        camera_prim_path: the D435 camera prim, e.g. from
            :func:`g1_sim.rtx_camera.spawn_camera`.
        width, height: render resolution; must match the camera's intended
            aspect (focal_length/aperture below assume the D435 defaults in
            :func:`g1_sim.rtx_camera.spawn_camera`).
        stride: subsample every ``stride``-th pixel in each axis - a full
            640x480 frame is 300k points/publish, more than RViz needs and
            enough to stall it on an 8 GB GPU already carrying the sim.
        topic, frame_id, publish_rate: as published.
    """

    def __init__(
        self,
        camera_prim_path: str,
        width: int = 640,
        height: int = 480,
        focal_length: float = 1.93,
        horizontal_aperture: float = 2.682,
        max_range_m: float = 20.0,
        stride: int = 2,
        topic: str = "/g1/camera/depth/color/points",
        frame_id: str = "d435_color_optical_frame",
        publish_rate: float = 10.0,
    ):
        import omni.replicator.core as rep
        from rclpy.node import Node
        from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import PointCloud2, PointField

        self._PointCloud2 = PointCloud2
        self.frame_id = frame_id
        self.stride = stride
        self.max_range_m = max_range_m
        self.publish_period = 1.0 / publish_rate
        self._last_publish = -float("inf")

        # Same pinhole convention ROS2CameraInfoHelper uses internally, so
        # this cloud's geometry matches what /g1/camera/camera_info implies.
        self.fx = width * focal_length / horizontal_aperture
        # Renderer assumes square pixels (vertical aperture is ignored).
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0

        self._rp = rep.create.render_product(camera_prim_path, [width, height], name="rgbd_pub")
        self._rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annot.attach([self._rp])
        self._depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self._depth_annot.attach([self._rp])

        self.node = Node("g1_rgbd_publisher")
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.node.create_publisher(PointCloud2, topic, qos)

        self._fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.node.get_logger().info(f"publishing {topic} in frame {frame_id}")

    def publish(self, sim_time: float) -> int:
        """Publish one colorized cloud if due. Returns points sent."""
        if sim_time - self._last_publish < self.publish_period:
            return 0

        depth = self._depth_annot.get_data()
        rgb = self._rgb_annot.get_data()
        if depth is None or rgb is None or getattr(depth, "size", 0) == 0 or getattr(rgb, "size", 0) == 0:
            return 0
        self._last_publish = sim_time

        s = self.stride
        depth = np.asarray(depth)[::s, ::s]
        rgb = np.asarray(rgb)[::s, ::s, :3]
        h, w = depth.shape
        uu, vv = np.meshgrid(np.arange(w) * s, np.arange(h) * s)

        valid = np.isfinite(depth) & (depth > 0) & (depth < self.max_range_m)
        if not np.any(valid):
            return 0

        z = depth[valid]
        x = (uu[valid] - self.cx) * z / self.fx
        y = (vv[valid] - self.cy) * z / self.fy

        r = rgb[..., 0][valid].astype(np.uint32)
        g = rgb[..., 1][valid].astype(np.uint32)
        b = rgb[..., 2][valid].astype(np.uint32)
        rgb_float = ((r << 16) | (g << 8) | b).astype(np.uint32).view(np.float32)

        n = len(z)
        data = np.empty((n, 4), dtype=np.float32)
        data[:, 0] = x
        data[:, 1] = y
        data[:, 2] = z
        data[:, 3] = rgb_float

        msg = self._PointCloud2()
        msg.header.frame_id = self.frame_id
        msg.header.stamp.sec = int(sim_time)
        msg.header.stamp.nanosec = int((sim_time - int(sim_time)) * 1e9)
        msg.height = 1
        msg.width = n
        msg.fields = self._fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * n
        msg.is_dense = False
        msg.data = data.tobytes()
        self._pub.publish(msg)
        return n

    def destroy(self) -> None:
        self.node.destroy_node()
