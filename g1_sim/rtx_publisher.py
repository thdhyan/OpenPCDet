"""Publish raw per-prim RTX LiDAR returns to ROS2 via the ``LidarSensor`` runtime class.

Was reading ``IsaacExtractRTXSensorPointCloud`` straight off a manually
attached ``rep.create.render_product()`` until 2026-08-11: that produced a
valid annotator (real keys, no error) but an always-empty ``"data"`` array
for every one of our hand-authored ``OmniLidar`` prims. Per Isaac Sim
6.0.1's own docs (isaacsim_sensors_rtx_lidar.html, fetched live): "Raw prim
creation alone is insufficient. You must use the ``LidarSensor`` runtime
wrapper to enable annotators" - manually calling
``rep.AnnotatorRegistry.get_annotator(...).attach(...)`` skips whatever
setup ``LidarSensor`` does to actually wire the render product into the RTX
sensor engine. ``LidarSensor(path, annotators=["generic-model-output"])`` +
``get_data("generic-model-output")`` + ``parse_generic_model_output_data()``
is the currently-supported path.

GMO coordinate semantics (ISO8855, from GenericModelOutput.rst,
omni.sensors.nv.common-3.0.0, confirmed 2026-08-13):
  - Default elementsCoordsType=SPHERICAL: x=azimuth(deg), y=elevation(deg),
    z=range(m). Treating these as Cartesian x/y/z was the root cause of the
    partial-band / garbage point positions bug.
  - spawn_mid360 now sets elementsCoordsType="CARTESIAN" on every prim, so
    gmo.x/y/z are real metric sensor-frame Cartesian (ISO8855: +x=front,
    +y=left, +z=up). The publisher checks the actual elementsCoordsType
    at runtime and converts if needed, so it degrades gracefully if a prim
    was not authored with CARTESIAN.
  - gmo.scalar is the normalised intensity (Lidar modality). Previously
    assumed absent (checked .pyi stub, which didn't list it) - but the RST
    docs confirm it. Now used directly.

Each prim publishes its own raw cloud to its own topic - no cross-prim
merge. Verifying prim-by-prim coverage (2026-08-11) needed to see each
prim's returns in isolation, since the previous merged-single-topic design
made it impossible to tell whether a coverage gap came from one broken prim
or all of them.

In the uv ``isaac`` env (Python 3.12) the bundled ROS2 jazzy rclpy is used
(set up by Projects/IsaacLab/isaac6/.envrc).
"""

from __future__ import annotations

import numpy as np


class RtxLidarPublisher:
    """Publishes each RTX LiDAR prim's raw returns as its own ``sensor_msgs/PointCloud2``.

    Args:
        prim_paths: sensor prims from :func:`g1_sim.rtx_lidar.spawn_mid360`.
        topic: base topic; each prim publishes to ``{topic}/{a,b,c,...}``.
        frame_id: TF frame the points are expressed in.
        publish_rate: target rate in Hz; should match the sensor's scan rate.
        max_points: cap on points per prim per message.
    """

    def __init__(
        self,
        prim_paths: list[str],
        topic: str = "/livox/mid360/points",
        frame_id: str = "mid360_link",
        publish_rate: float = 10.0,
        max_points: int = 20000,
        max_range_m: float = 40.0,
    ):
        from isaacsim.sensors.experimental.rtx import LidarSensor
        from rclpy.node import Node
        from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import PointCloud2, PointField

        self._PointCloud2 = PointCloud2
        self.frame_id = frame_id
        self.max_points = max_points
        # Hard cap matching the profile's own farRangeM (see
        # gen_mid360_rtx_config.py) - a real return can never exceed it.
        # Points beyond this are provably not real hits: live-verified
        # 2026-08-11, points at 95-165m appeared against a farRangeM=40
        # sensor, exactly coincident with "GMO magic number is not correct"
        # buffer-corruption warnings (isaacsim.sensors.experimental.rtx.impl.
        # utils) - a known CUDA buffer race (IsaacSim discussion #685) reading
        # the RTX sensor's buffer while the GPU is still writing it. This
        # filter discards the resulting garbage rather than fixing the race
        # itself (which is upstream, in the RTX sensor plugin).
        self.max_range_m = max_range_m
        self.publish_period = 1.0 / publish_rate
        self._last_publish = -float("inf")

        self.prim_paths = list(prim_paths)
        self.sensors = [LidarSensor(p, annotators=["generic-model-output"]) for p in prim_paths]
        self._logged_keys = False
        # Slices of each prim's sweep awaiting the next publish.
        self._pending: list[list[tuple]] = [[] for _ in prim_paths]
        # Raw (pre-filter) point counts and az/el ranges for the window
        # currently being accumulated, and the most recently completed one -
        # diagnostic only, not used to build the published messages.
        self.window_prim_counts = [0] * len(prim_paths)
        self.window_az_range = [[float("inf"), float("-inf")] for _ in prim_paths]
        self.window_el_range = [[float("inf"), float("-inf")] for _ in prim_paths]
        self.last_window_prim_counts = [0] * len(prim_paths)
        self.last_window_az_range = [[0.0, 0.0] for _ in prim_paths]
        self.last_window_el_range = [[0.0, 0.0] for _ in prim_paths]

        self.node = Node("g1_rtx_lidar_publisher")
        # Best-effort matches how sensor streams are normally consumed: a
        # subscriber that falls behind should drop scans, not queue them.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.topics = [f"{topic}/{chr(ord('a') + i)}" for i in range(len(prim_paths))]
        self.publishers = [self.node.create_publisher(PointCloud2, t, qos) for t in self.topics]

        self._fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.node.get_logger().info(
            f"publishing raw per-prim clouds {self.topics} in frame {frame_id} "
            f"from {len(prim_paths)} prims (LidarSensor, no merge)"
        )

    def accumulate(self) -> int:
        """Collect one render's worth of returns from every prim. Call every simulation step.

        Each render yields only the emitters that fired in that frame, so a
        single read is a thin slice of the sweep. Accumulating between
        publishes is what turns those slices into a full scan.

        Returns the total number of points collected this call, across all prims.
        """
        total = 0
        for prim_idx, sensor in enumerate(self.sensors):
            points, intensities = self._gather_one(sensor, prim_idx)
            if points is None:
                continue
            self._pending[prim_idx].append((points, intensities))
            total += len(points)
        return total

    def publish(self, sim_time: float) -> int:
        """Publish each prim's accumulated scan if due. Returns total points sent."""
        if sim_time - self._last_publish < self.publish_period:
            return 0
        self._last_publish = sim_time

        self.last_window_prim_counts = self.window_prim_counts
        self.last_window_az_range = self.window_az_range
        self.last_window_el_range = self.window_el_range
        self.window_prim_counts = [0] * len(self.prim_paths)
        self.window_az_range = [[float("inf"), float("-inf")] for _ in self.prim_paths]
        self.window_el_range = [[float("inf"), float("-inf")] for _ in self.prim_paths]

        sent_total = 0
        for prim_idx in range(len(self.prim_paths)):
            pending = self._pending[prim_idx]
            if not pending:
                continue
            points = np.vstack([p for p, _ in pending])
            intensities = np.concatenate([i for _, i in pending])
            pending.clear()

            if len(points) > self.max_points:
                # Stride rather than slice: a contiguous slice would take one
                # part of the sweep, whereas striding preserves the pattern's
                # shape.
                idx = np.linspace(0, len(points) - 1, self.max_points).astype(np.int64)
                points, intensities = points[idx], intensities[idx]

            msg = self._to_message(points, intensities, sim_time)
            self.publishers[prim_idx].publish(msg)
            sent_total += len(points)
        return sent_total

    def _gather_one(self, sensor, prim_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Collect and filter one prim's returns.

        GMO coordinate layout depends on elementsCoordsType set on the prim:
          CARTESIAN (set by spawn_mid360): gmo.x/y/z = metric Cartesian in
            the sensor/world frame (ISO8855: +x=front, +y=left, +z=up).
          SPHERICAL (schema default, wrong path): gmo.x=azimuth(deg),
            gmo.y=elevation(deg), gmo.z=range(m) — treated as Cartesian x/y/z
            this produced the partial-band / wrong-elevation bug.
        This method reads elementsCoordsType from the GMO and converts
        SPHERICAL → CARTESIAN if the prim was not authored with CARTESIAN.
        """
        from isaacsim.sensors.experimental.rtx import parse_generic_model_output_data

        data, _info = sensor.get_data("generic-model-output")
        if data is None:
            return None, None
        gmo = parse_generic_model_output_data(data)
        if gmo.x is None or len(gmo.x) == 0:
            return None, None

        if not self._logged_keys:
            foref = getattr(gmo, "frameOfReference", None)
            for_name = getattr(foref, "name", str(foref))
            coords = getattr(gmo, "elementsCoordsType", None)
            coords_name = getattr(coords, "name", str(coords)) if coords is not None else "unknown"
            if for_name == "WORLD" and self.frame_id != "World":
                self.node.get_logger().warn(
                    f"GMO frameOfReference=WORLD; relabeling cloud frame_id "
                    f"'{self.frame_id}' -> 'World'"
                )
                self.frame_id = "World"
            self.node.get_logger().info(
                f"GenericModelOutput numElements={gmo.numElements} "
                f"frameOfReference={for_name} elementsCoordsType={coords_name}"
            )
            self._logged_keys = True

        coords = getattr(gmo, "elementsCoordsType", None)
        coords_name = getattr(coords, "name", str(coords)) if coords is not None else "CARTESIAN"

        if "SPHERICAL" in coords_name.upper():
            # gmo.x=azimuth(deg), gmo.y=elevation(deg), gmo.z=range(m).
            # ISO8855: az=0 is +x (front), positive CCW; el=0 is horizon,
            # positive up. Convert to sensor-frame Cartesian.
            az_rad = np.radians(np.asarray(gmo.x, dtype=np.float32))
            el_rad = np.radians(np.asarray(gmo.y, dtype=np.float32))
            r_m = np.asarray(gmo.z, dtype=np.float32)
            cos_el = np.cos(el_rad)
            cx = r_m * cos_el * np.cos(az_rad)  # +x = front
            cy = r_m * cos_el * np.sin(az_rad)  # +y = left
            cz = r_m * np.sin(el_rad)            # +z = up
            xyz = np.stack([cx, cy, cz], axis=1)
            r = r_m  # range for the filter below
        else:
            # CARTESIAN: gmo.x/y/z are already metric sensor-frame.
            xyz = np.stack(
                [np.asarray(gmo.x), np.asarray(gmo.y), np.asarray(gmo.z)], axis=1
            ).astype(np.float32)
            r = np.linalg.norm(xyz, axis=1)

        # gmo.scalar is the normalised intensity for Lidar (GMO RST docs,
        # omni.sensors.nv.common-3.0.0). Fall back to constant if absent/empty.
        scalar = getattr(gmo, "scalar", None)
        if scalar is not None and len(scalar) == len(r):
            intensities = np.asarray(scalar, dtype=np.float32)
        else:
            intensities = np.full(len(r), 100.0, dtype=np.float32)

        # Diagnostics — az/el computed from Cartesian xyz.
        self.window_prim_counts[prim_idx] += len(xyz)
        nonzero_r = r > 1e-6
        if np.any(nonzero_r):
            az = np.degrees(np.arctan2(xyz[nonzero_r, 1], xyz[nonzero_r, 0]))
            el = np.degrees(np.arcsin(np.clip(xyz[nonzero_r, 2] / r[nonzero_r], -1, 1)))
            az_lo, az_hi = self.window_az_range[prim_idx]
            self.window_az_range[prim_idx] = [min(az_lo, float(az.min())), max(az_hi, float(az.max()))]
            el_lo, el_hi = self.window_el_range[prim_idx]
            self.window_el_range[prim_idx] = [min(el_lo, float(el.min())), max(el_hi, float(el.max()))]

        # Drop non-finite, exact-origin (no-hit), and beyond-farRangeM points.
        finite = np.isfinite(xyz).all(axis=1)
        in_range = r <= self.max_range_m
        keep = finite & nonzero_r & in_range
        xyz, intensities = xyz[keep], intensities[keep]
        if len(xyz) == 0:
            return None, None

        return xyz, intensities

    def _to_message(self, points: np.ndarray, intensities: np.ndarray, sim_time: float):
        msg = self._PointCloud2()
        msg.header.frame_id = self.frame_id
        msg.header.stamp.sec = int(sim_time)
        msg.header.stamp.nanosec = int((sim_time - int(sim_time)) * 1e9)

        msg.height = 1  # unorganised cloud
        msg.width = len(points)
        msg.fields = self._fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = np.hstack([points, intensities[:, None]]).astype(np.float32).tobytes()
        return msg

    def diagnostics_str(self) -> str:
        """Per-prim raw counts + azimuth/elevation ranges (degrees) from the
        most recently completed publish window. Diagnostic only."""
        lines = []
        for i, prim_path in enumerate(self.prim_paths):
            n = self.last_window_prim_counts[i]
            az_lo, az_hi = self.last_window_az_range[i]
            el_lo, el_hi = self.last_window_el_range[i]
            lines.append(
                f"  [{i}] {prim_path}: raw_points={n} "
                f"azimuth=[{az_lo:.1f},{az_hi:.1f}] elevation=[{el_lo:.1f},{el_hi:.1f}]"
            )
        return "\n".join(lines)

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        """Service pending rclpy callbacks without blocking the sim loop."""
        import rclpy

        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def destroy(self) -> None:
        self.node.destroy_node()
