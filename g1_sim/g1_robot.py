"""Unitree G1 loader: one call spawns the robot + every sensor, all ON by default.

Designed to be the only import a consumer needs::

    from g1_sim.g1_robot import load_g1

    g1 = load_g1()                       # camera + lidar + 4 IMUs + ROS2 topics/TF
    g1 = load_g1(camera=False, ros2=False)   # bare robot for IsaacLab training
    while stepping:
        g1.step(sim.current_time)        # lidar + RGB-D cloud publishing
    g1.destroy()

Everything defaults to **on**; each sensor disables individually via kwargs.
ROS2 wiring (topics, ``/tf``, ``/clock``, joint states) is lazy - ``ros2=False``
never imports rclpy, so the module is safe to use headless in IsaacLab training
(the caller owns ``SimulationApp``; this module only authors prims on the
already-open stage).

Sensor -> ROS topic map (frame_id in parentheses):

    ============================  =========================  =================
    sensor                        topic                      frame
    ============================  =========================  =================
    D435 RGB                      /g1/camera/rgb             d435_color_optical_frame
    D435 depth (32FC1, metres)    /g1/camera/depth           d435_color_optical_frame
    D435 depth_pcl (XYZ)          /g1/camera/depth/points    d435_color_optical_frame
    D435 RGB-fused XYZRGB cloud   /g1/camera/depth/color/points
                                                             d435_color_optical_frame
    D435 segmentation             /g1/camera/semantic        d435_color_optical_frame
    D435 camera_info              /g1/camera/camera_info     d435_color_optical_frame
    Mid-360 LiDAR (per prim)      /livox/mid360/points/{a..} mid360_link
    pelvis IMU                    /g1/imu/pelvis             imu_in_pelvis
    torso IMU                     /g1/imu/torso              imu_in_torso
    torso IMU (legacy alias)      /g1/imu                    imu_in_torso
    Mid-360 built-in IMU          /livox/mid360/imu          mid360_link
    RealSense IMU                 /g1/camera/imu             d435_link
    robot TF / joints / clock     /tf, /g1/joint_states, /clock
    ============================  =========================  =================

``/g1/imu`` keeps its historical name because ``launch/perception_launch.py``
remaps ``visual_slam/imu`` onto it; the torso IMU additionally publishes on
the symmetric ``/g1/imu/torso``.

IMU datasheet parameters (researched 2026-09-23, see :data:`IMU_DATASHEET`)
are exported for downstream consumers (EKF configuration, training-time noise
models). Isaac's ``IsaacImuSensor`` schema only exposes rolling-average filter
widths - it has **no noise model** - so the numbers here are documentation +
config source, not injected into the sim by this module.

Placement: each IMU sensor prim is parented directly on its real USD frame
prim (``/World/G1/imu_in_pelvis`` etc.). Those prims are articulation bodies
(they appear in the published TF joint tree), so the sensors track robot
dynamics exactly at the URDF-specified locations. This supersedes the old
warehouse behaviour of spawning under a phantom ``torso_link/imu_in_torso``
chain that USD auto-created because the converted USD keeps all sensor frames
as flat siblings of ``pelvis``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Research results - one entry per physical IMU on the platform:
#
# * Unitree G1 pelvis/torso: MJCF in unitreerobotics/unitree_ros (issue #121)
#   models both as gyro noise=5e-4 cutoff=34.9 (rad/s units, lowpass Hz) and
#   accelerometer noise=1e-2 cutoff=157 (m/s^2, Hz). The real G1 LowState
#   exposes ONE waist IMU; both frames exist in the model.
# * Livox Mid-360 built-in IMU = TDK ICM-40609 (Livox spec page): gyro noise
#   density 4.5 mdps/sqrt(Hz), accel 100 ug/sqrt(Hz), gyro FSR +/-2000 dps,
#   accel FSR +/-32 g, ODR up to 32 kHz (Livox streams at ~100 Hz? unused).
# * RealSense D435i IMU = Bosch BMI055 (Intel D400 datasheet Table 4-23 +
#   BMI055 datasheet): gyro noise density 0.014 deg/s/sqrt(Hz) at +/-1000
#   deg/s configured by librealsense, accel 150 ug/sqrt(Hz) at +/-4 g.
#   (Plain D435 has no IMU; D435i does - we model the D435i variant.)
#
# Noise density -> per-sample sigma at sample rate fs: sigma = density*sqrt(fs).
IMU_DATASHEET = {
    "pelvis": {
        "chip": "Unitree waist IMU (MJCF model in unitree_ros)",
        "gyro_noise_density": 5e-4,          # MJCF white-noise param (rad/s)
        "gyro_cutoff_hz": 34.9,
        "accel_noise_density": 1e-2,         # MJCF white-noise param (m/s^2)
        "accel_cutoff_hz": 157.0,
        "frame": "imu_in_pelvis",
        "topic": "/g1/imu/pelvis",
    },
    "torso": {
        "chip": "Unitree waist IMU (MJCF model in unitree_ros)",
        "gyro_noise_density": 5e-4,
        "gyro_cutoff_hz": 34.9,
        "accel_noise_density": 1e-2,
        "accel_cutoff_hz": 157.0,
        "frame": "imu_in_torso",
        "topic": "/g1/imu/torso",
        "legacy_topic": "/g1/imu",
    },
    "lidar": {
        "chip": "TDK ICM-40609 (Mid-360 built-in)",
        "gyro_noise_density": 4.5e-5,        # rad/s/sqrt(Hz)  (4.5 mdps)
        "gyro_range_dps": 2000.0,
        "accel_noise_density": 9.81e-4,      # m/s^2/sqrt(Hz)  (100 ug)
        "accel_range_g": 32.0,
        "frame": "mid360_link",
        "topic": "/livox/mid360/imu",
    },
    "camera": {
        "chip": "Bosch BMI055 (RealSense D435i)",
        "gyro_noise_density": 2.44e-4,       # rad/s/sqrt(Hz) (0.014 deg/s)
        "gyro_range_dps": 1000.0,            # librealsense config
        "accel_noise_density": 1.471e-3,     # m/s^2/sqrt(Hz) (150 ug)
        "accel_range_g": 4.0,                # librealsense config
        "frame": "d435_link",
        "topic": "/g1/camera/imu",
    },
}

# URDF-cleared spawn pose: z=0.8 matches the warehouse's proven standing
# height (spawn_g1) and works with the WBC Balance policy's 0.74 m target.
DEFAULT_USD = Path(__file__).resolve().parent.parent / "assets/g1_29dof_sensors.usd"


@dataclass
class G1Robot:
    """Handles returned by :func:`load_g1`."""

    prim_path: str
    articulation: object | None = None
    camera_prim: str | None = None
    lidar_prims: list[str] = field(default_factory=list)
    imu_prims: dict[str, str] = field(default_factory=dict)
    ros_graphs: list[str] = field(default_factory=list)
    _lidar_pub: object | None = None
    _rgbd_pub: object | None = None
    _pattern_cycler: object | None = None

    def step(self, current_time: float) -> int:
        """Advance sensor publishing; call once per sim step.

        Returns the number of LiDAR points sent this step (0 when no lidar).
        Safe to call when every sensor is disabled.
        """
        sent = 0
        if self._pattern_cycler is not None:
            self._pattern_cycler.step(current_time)
        if self._lidar_pub is not None:
            self._lidar_pub.accumulate()
            sent = self._lidar_pub.publish(current_time) or 0
            self._lidar_pub.spin_once()
        if self._rgbd_pub is not None:
            self._rgbd_pub.publish(current_time)
        return sent

    def destroy(self) -> None:
        """Tear down rclpy-backed publishers (idempotent)."""
        for pub in (self._lidar_pub, self._rgbd_pub):
            if pub is None:
                continue
            try:
                pub.destroy()
            except Exception:
                pass  # rclpy already down or double-destroy
        self._lidar_pub = self._rgbd_pub = None


def load_g1(
    prim_path: str = "/World/G1",
    usd_path: str | Path | None = None,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.8),
    *,
    # --- sensors: every one defaults ON ---
    camera: bool = True,
    lidar: bool = True,
    imu_pelvis: bool = True,
    imu_torso: bool = True,
    imu_lidar: bool = True,
    imu_camera: bool = True,
    # --- publishing ---
    ros2: bool = True,
    robot_state: bool = True,        # /tf + /g1/joint_states + /clock
    depth_colorized: bool = True,    # RGB-fused XYZRGB cloud (needs camera)
    semantics: dict[str, str] | None = None,  # prim_path -> label
    # --- tuning ---
    camera_width: int = 640,
    camera_height: int = 480,
    lidar_config_dir: str | Path | None = None,
    # DO NOT CHANGE the lidar/camera mount defaults without re-running
    # scripts/verify_sensor_tf.py - verified 2026-09-23, frames recorded in
    # docs/tf_snapshot_20260923.yaml. -0.15 in the (180 deg rolled)
    # mid360_link = +15 cm world-up: lower and the head mesh swallows the
    # downward rays (-0.03 kept ~5% of them).
    lidar_translation: tuple[float, float, float] = (0.0, 0.0, -0.15),
    lidar_orientation: tuple[float, float, float, float] | None = None,
    lidar_num_prims: int = 0,         # 0 = all
    lidar_max_points: int = 200_000,
    lidar_topic: str = "/livox/mid360/points",
    create_articulation: bool = True,
) -> G1Robot:
    """Load the G1 USD onto the open stage and wire requested sensors.

    Requires a running ``SimulationApp`` and the extensions enabled in
    ``enable_extensions()`` (ros2 bridge, rtx sensors, physics nodes) - see
    ``scripts/g1_warehouse_sim.py`` for the canonical enable list.

    Args:
        prim_path: where to instance the robot.
        usd_path: robot USD; defaults to the package's baked sensors asset.
        translation: world spawn pose of the robot root.
        camera, lidar, imu_pelvis, imu_torso, imu_lidar, imu_camera:
            per-sensor switches, all True.
        ros2: master switch - False skips every ROS2 graph/publisher (and any
            rclpy import), for IsaacLab training or pure-viewport use.
        robot_state: publish /tf, /g1/joint_states, /clock (needs ros2).
        depth_colorized: publish the RGB-fused XYZRGB cloud (needs camera+ros2).
        semantics: optional ``{prim_path: class}`` labels for segmentation.
        lidar_*: Mid-360 spawn tuning; defaults reproduce the validated
            warehouse mount (+3 cm world-up clearance to clear the head mesh).
        create_articulation: wrap pelvis in isaacsim's Articulation.

    Returns:
        :class:`G1Robot` handle (articulation, sensor prim paths, publishers).
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    from g1_sim.rtx_camera import (
        OPTICAL_FRAME,
        apply_semantics,
        attach_camera_publishers,
        attach_imu_publisher,
        attach_robot_state_publishers,
        spawn_camera,
        spawn_imu_sensor,
    )
    from g1_sim.rtx_lidar import ScanPatternCycler, spawn_mid360

    usd_path = Path(usd_path) if usd_path else DEFAULT_USD
    if not usd_path.exists():
        raise SystemExit(f"[G1] {usd_path} not found - run scripts/convert_g1_urdf_to_usd.py")

    stage = omni.usd.get_context().get_stage()
    handle = G1Robot(prim_path=prim_path)

    # Session-layer edit target: after IRA opens the remote warehouse USD as
    # the root layer, referencing a local POSIX path on that layer composes
    # but silently fails to resolve (Nucleus resolver vs https layer). The
    # session layer is always local, sidestepping it; harmless otherwise.
    # (Verbatim rationale from the old spawn_g1().)
    stage.SetEditTarget(stage.GetSessionLayer())
    robot = stage.DefinePrim(prim_path, "Xform")
    robot.GetReferences().AddReference(str(usd_path))
    xform = UsdGeom.Xformable(robot)
    translate = next((op for op in xform.GetOrderedXformOps() if "translate" in op.GetOpName()), None)
    if translate is None:
        translate = xform.AddTranslateOp()
    translate.Set(Gf.Vec3d(*translation))
    # Let the reference compose before anyone queries child prims.
    import omni.kit.app

    for _ in range(10):
        omni.kit.app.get_app().update()

    pelvis = stage.GetPrimAtPath(f"{prim_path}/pelvis")
    if not pelvis.IsValid():
        raise SystemExit(f"[G1] {prim_path}/pelvis missing - reference did not resolve")
    print(f"[G1] loaded          : {usd_path.name} @ {prim_path}  pose={translation}")

    if create_articulation:
        from isaacsim.core.prims import Articulation

        handle.articulation = Articulation(f"{prim_path}/pelvis", name="g1")
        print(f"[G1] articulation    : {handle.articulation.num_dof} DOF")

    if semantics:
        labelled = apply_semantics(semantics)
        print(f"[G1] semantics       : {labelled} prims labelled")

    # Sensor prims below always spawn (per the sensor kwargs); ROS graphs and
    # rclpy publishers are gated on `ros2`, so ros2=False stays import-safe
    # for IsaacLab training.
    if ros2:
        # Init BEFORE any rclpy node below: the camera block runs before the
        # lidar block, and RgbdPointCloudPublisher/RtxLidarPublisher both
        # create Nodes (NotInitializedException otherwise).
        import rclpy

        if not rclpy.ok():
            rclpy.init()
    if ros2 and robot_state:
        graph = attach_robot_state_publishers(prim_path)
        handle.ros_graphs.append(graph)
        print(f"[G1] state graph     : {graph}  (/tf, /g1/joint_states, /clock)")

    # ---- IMUs: one sensor prim + one publish graph each ----
    imu_specs: list[tuple[str, str, bool]] = [
        ("pelvis", f"{prim_path}/imu_in_pelvis", imu_pelvis),
        ("torso", f"{prim_path}/imu_in_torso", imu_torso),
        ("lidar", f"{prim_path}/mid360_link", imu_lidar),
        ("camera", f"{prim_path}/d435_link", imu_camera),
    ]
    for name, parent, enabled in imu_specs:
        if not enabled:
            continue
        if not stage.GetPrimAtPath(parent).IsValid():
            print(f"[G1] imu {name:<8}: SKIP - frame {parent} missing from USD")
            continue
        spec = IMU_DATASHEET[name]
        imu_prim = spawn_imu_sensor(parent, name="imu_sensor")
        handle.imu_prims[name] = imu_prim
        if not ros2:
            print(f"[G1] imu {name:<8}: prim={imu_prim}  (ros2=False, no topic)")
            continue
        graph = attach_imu_publisher(
            imu_prim,
            graph_path=f"/ActionGraph/Imu{name.capitalize()}ROS2",
            topic=spec["topic"],
            frame_id=spec["frame"],
        )
        handle.ros_graphs.append(graph)
        # Torso additionally feeds the legacy /g1/remap used by perception.
        if spec.get("legacy_topic"):
            legacy = attach_imu_publisher(
                imu_prim,
                graph_path="/ActionGraph/ImuLegacyROS2",
                topic=spec["legacy_topic"],
                frame_id=spec["frame"],
            )
            handle.ros_graphs.append(legacy)
        print(f"[G1] imu {name:<8}: {spec['topic']:<24} frame={spec['frame']}  ({spec['chip']})")

    # ---- D435 camera ----
    if camera:
        handle.camera_prim = spawn_camera(f"{prim_path}/torso_link", width=camera_width, height=camera_height)
        print(f"[G1] camera prim     : {handle.camera_prim}  ({camera_width}x{camera_height})")
        if ros2:
            graph = attach_camera_publishers(handle.camera_prim, width=camera_width, height=camera_height)
            handle.ros_graphs.append(graph)
            print(f"[G1] camera graph    : {graph}  (/g1/camera/{{rgb,depth,depth/points,semantic,camera_info}})")
            if depth_colorized:
                from g1_sim.rgbd_publisher import RgbdPointCloudPublisher

                handle._rgbd_pub = RgbdPointCloudPublisher(handle.camera_prim, frame_id=OPTICAL_FRAME)
                print("[G1] rgbd publisher  : /g1/camera/depth/color/points (frame=%s)" % OPTICAL_FRAME)

    # ---- Mid-360 LiDAR ----
    if lidar:
        mount = f"{prim_path}/mid360_link"
        if not stage.GetPrimAtPath(mount).IsValid():
            raise SystemExit(f"[G1] mount prim {mount} missing from the USD")
        kwargs: dict = {}
        if lidar_config_dir is not None:
            kwargs["config_dir"] = Path(lidar_config_dir)
        prim_paths = spawn_mid360(
            mount,
            translation=lidar_translation,
            # Identity: mid360_link already carries the URDF's 180 deg roll.
            # A second roll here flips the sensor upright (ceiling cloud).
            orientation=lidar_orientation if lidar_orientation is not None else (1.0, 0.0, 0.0, 0.0),
            **kwargs,
        )
        if lidar_num_prims and lidar_num_prims < len(prim_paths):
            prim_paths = prim_paths[:lidar_num_prims]
        handle.lidar_prims = prim_paths
        print(f"[G1] lidar prims     : {len(prim_paths)}  (mount {mount}, offset {lidar_translation} in mid360_link)")
        scan_type = stage.GetPrimAtPath(prim_paths[0]).GetAttribute("omni:sensor:Core:scanType").Get()
        if scan_type == "SOLID_STATE" and len(prim_paths) == 1:
            handle._pattern_cycler = ScanPatternCycler(prim_paths[0])
            print(f"[G1] lidar pattern   : {len(handle._pattern_cycler.frames)} recorded Mid-360 frames, 1 per scan")

        if ros2:
            # RtxLidarPublisher, NOT the ROS2RtxLidarHelper OmniGraph: the
            # helper advertises the topic but never emits on this build (see
            # g1_sim/rtx_publisher.py docstring).
            import rclpy

            from g1_sim.rtx_publisher import RtxLidarPublisher

            if not rclpy.ok():
                rclpy.init()
            handle._lidar_pub = RtxLidarPublisher(
                prim_paths,
                topic=lidar_topic,
                publish_rate=10.0,
                max_points=lidar_max_points,
                sensor_offset=lidar_translation,
            )
            print(f"[G1] lidar publisher : rclpy -> {handle._lidar_pub.topics}")

    return handle
