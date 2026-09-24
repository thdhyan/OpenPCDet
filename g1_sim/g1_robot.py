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
# Unitree G1 29-DoF + Dex3 hands (rev 1.0 URDF), the asset IsaacLab's
# G129_CFG_WITH_DEX3_BASE_FIX uses; load_g1 un-welds its fixed base.
DEX3_USD_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/0.5.0/132c82d"
    "/Robots/UnitreeG1/g1_29dof_with_dex3_base_fix/g1_29dof_with_dex3_base_fix.usd"
)
DEFAULT_USD = Path(__file__).resolve().parent.parent / "assets/robot/g1_29_dex3/g1_29dof_with_dex3_base_fix.usd"
# Matching URDF (same rev 1.0 kinematics + Dex3 hands) for /robot_description.
DEFAULT_URDF = Path(__file__).resolve().parent.parent / "assets/robot/g1_29/g1_29dof_with_hand_rev_1_0.urdf"
# mid360_joint in unitree_ros g1_29dof_with_hand_rev_1_0.urdf (current): the
# Mid-360 is mounted inverted with a slight pitch. The Dex3 USD above carries
# an older upright version, so load_g1 re-authors the joint to match.
MID360_JOINT_XYZ = (0.0002835, 0.00003, 0.428434)
MID360_JOINT_RPY = (3.141592653589793, 0.05112069379091391, 0.0)


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
    _tip_pub: object | None = None
    _description_node: object | None = None
    tip_contacts: dict[str, str] = field(default_factory=dict)

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
        if self._tip_pub is not None:
            self._tip_pub.publish(current_time)
        return sent

    def destroy(self) -> None:
        """Tear down rclpy-backed publishers (idempotent)."""
        for pub in (self._lidar_pub, self._rgbd_pub, self._tip_pub):
            if pub is None:
                continue
            try:
                pub.destroy()
            except Exception:
                pass  # rclpy already down or double-destroy
        self._lidar_pub = self._rgbd_pub = self._tip_pub = None
        if self._description_node is not None:
            self._description_node.destroy_node()
            self._description_node = None


def _publish_robot_description(urdf_path: Path):
    """Latch the URDF on /robot_description (transient-local, like
    robot_state_publisher) with mesh paths rewritten to file:// URIs so RViz
    resolves them without a ROS package. /tf comes from the sim itself."""
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile
    from std_msgs.msg import String

    text = urdf_path.read_text().replace('filename="meshes/', f'filename="file://{urdf_path.parent}/meshes/')
    node = Node("g1_robot_description")
    qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    node.create_publisher(String, "/robot_description", qos).publish(String(data=text))
    print(f"[G1] description     : {urdf_path.name} -> /robot_description (latched)")
    return node


def _align_mid360_mount(stage, robot) -> None:
    """Author mid360_joint (and mid360_link's initial pose) from the current
    Unitree URDF so the lidar frame and its built-in IMU match the real robot."""
    import math

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    joint = next((p for p in Usd.PrimRange(robot) if p.GetName() == "mid360_joint"), None)
    if joint is None:
        return
    j = UsdPhysics.Joint(joint)
    roll, pitch, yaw = (math.degrees(a) for a in MID360_JOINT_RPY)
    # URDF rpy = fixed-axis X, then Y, then Z (row vectors: Rx * Ry * Rz).
    rot = (Gf.Rotation(Gf.Vec3d(1, 0, 0), roll) * Gf.Rotation(Gf.Vec3d(0, 1, 0), pitch)
           * Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw))
    q = rot.GetQuat()
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*MID360_JOINT_XYZ))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
    parent = stage.GetPrimAtPath(j.GetBody0Rel().GetTargets()[0])
    child = stage.GetPrimAtPath(j.GetBody1Rel().GetTargets()[0])
    # Keep the body's initial pose consistent with the joint (both are
    # authored relative to the robot root, like every link in this USD).
    cache = UsdGeom.XformCache()
    root_w = cache.GetLocalToWorldTransform(robot)
    parent_rel = cache.GetLocalToWorldTransform(parent) * root_w.GetInverse()
    child_rel = Gf.Matrix4d().SetTransform(rot, Gf.Vec3d(*MID360_JOINT_XYZ)) * parent_rel
    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(child).GetOrderedXformOps()}
    ops["xformOp:translate"].Set(child_rel.ExtractTranslation())
    cq = child_rel.ExtractRotationQuat()
    ops["xformOp:orient"].Set(Gf.Quatd(cq.GetReal(), cq.GetImaginary()))
    print(f"[G1] mid360 mount    : {MID360_JOINT_XYZ} rpy {MID360_JOINT_RPY} (unitree_ros rev 1.0)")


def load_g1(
    prim_path: str = "/World/G1",
    usd_path: str | Path | None = None,
    urdf_path: str | Path | None = None,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.8),
    yaw_deg: float = 0.0,
    *,
    # --- sensors: every one defaults ON ---
    camera: bool = True,
    lidar: bool = True,
    imu_pelvis: bool = True,
    imu_torso: bool = True,
    imu_lidar: bool = True,
    imu_camera: bool = True,
    hand_contacts: bool = True,     # Dex3 fingertip contact sensors
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
    # docs/tf_snapshot_20260923.yaml. Relative to the inverted mid360_link:
    # -0.15 = +15 cm world-up; lower and the head mesh swallows the downward
    # rays (-0.03 kept ~5% of them).
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
        usd_path: robot USD; defaults to the Dex3 G1 asset (DEFAULT_USD).
        urdf_path: URDF latched on /robot_description for RViz (needs ros2);
            defaults to the matching Dex3 URDF (DEFAULT_URDF).
        translation: world spawn position of the robot root.
        yaw_deg: spawn heading about world Z (counter-clockwise, degrees).
        camera, lidar, imu_pelvis, imu_torso, imu_lidar, imu_camera,
        hand_contacts: per-sensor switches, all True.
        ros2: master switch - False skips every ROS2 graph/publisher (and any
            rclpy import), for IsaacLab training or pure-viewport use.
        robot_state: publish /tf, /g1/joint_states, /clock (needs ros2).
        depth_colorized: publish the RGB-fused XYZRGB cloud (needs camera+ros2).
        semantics: optional ``{prim_path: class}`` labels for segmentation.
        lidar_*: Mid-360 spawn tuning; defaults reproduce the validated
            warehouse mount (+15 cm world-up clearance above mid360_link).
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
        raise SystemExit(f"[G1] {usd_path} not found - download it: curl -o {usd_path} {DEX3_USD_URL}")

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
    if yaw_deg:
        yaw = next((op for op in xform.GetOrderedXformOps() if op.GetOpName() == "xformOp:rotateZ"), None)
        (yaw or xform.AddRotateZOp()).Set(float(yaw_deg))
    # Let the reference compose before anyone queries child prims.
    import omni.kit.app

    for _ in range(10):
        omni.kit.app.get_app().update()

    pelvis = stage.GetPrimAtPath(f"{prim_path}/pelvis")
    if not pelvis.IsValid():
        raise SystemExit(f"[G1] {prim_path}/pelvis missing - reference did not resolve")
    print(f"[G1] loaded          : {usd_path.name} @ {prim_path}  pose={translation}")

    # The Dex3 asset (g1_29dof_with_dex3_base_fix) welds the pelvis to the
    # world with root_joint and roots the articulation there; WBC needs a
    # floating base, so drop the weld and root the articulation at the pelvis.
    from pxr import PhysxSchema, Usd, UsdPhysics

    welds = [
        p for p in Usd.PrimRange(robot)
        if p.IsA(UsdPhysics.FixedJoint) and not UsdPhysics.Joint(p).GetBody0Rel().GetTargets()
    ]
    for prim in welds:
        weld_attrs = {a.GetName(): a.Get() for a in prim.GetAttributes() if a.GetName().startswith("physxArticulation:")}
        prim.SetActive(False)
        UsdPhysics.ArticulationRootAPI.Apply(pelvis)
        PhysxSchema.PhysxArticulationAPI.Apply(pelvis)
        for name, value in weld_attrs.items():
            if value is not None:
                pelvis.GetAttribute(name).Set(value)
        print(f"[G1] floating base   : deactivated world weld {prim.GetPath()}")

    _align_mid360_mount(stage, robot)

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
        handle._description_node = _publish_robot_description(Path(urdf_path) if urdf_path else DEFAULT_URDF)

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

    # ---- Dex3 fingertip contacts ----
    if hand_contacts:
        from g1_sim.dex3_contacts import TipContactPublisher, spawn_tip_contacts

        handle.tip_contacts = spawn_tip_contacts(prim_path)
        if handle.tip_contacts and ros2:
            handle._tip_pub = TipContactPublisher(handle.tip_contacts)
        print(f"[G1] tip contacts    : {len(handle.tip_contacts)} Dex3 fingertips -> /g1/dex3/<side>/<finger>/contact")

    # ---- D435 camera ----
    if camera:
        handle.camera_prim = spawn_camera(f"{prim_path}/torso_link", width=camera_width, height=camera_height)
        print(f"[G1] camera prim     : {handle.camera_prim}  ({camera_width}x{camera_height})")
        if ros2:
            graph = attach_camera_publishers(
                handle.camera_prim, f"{prim_path}/d435_link", width=camera_width, height=camera_height
            )
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
        sensor_t = tuple(lidar_translation)
        # Identity: mid360_link carries the inverted mount (see MID360_JOINT_RPY).
        sensor_q = tuple(lidar_orientation or (1.0, 0.0, 0.0, 0.0))
        prim_paths = spawn_mid360(mount, translation=sensor_t, orientation=sensor_q, **kwargs)
        if lidar_num_prims and lidar_num_prims < len(prim_paths):
            prim_paths = prim_paths[:lidar_num_prims]
        handle.lidar_prims = prim_paths
        print(f"[G1] lidar prims     : {len(prim_paths)}  (mount {mount}, pose in link t={tuple(round(v, 4) for v in sensor_t)} q={tuple(round(v, 4) for v in sensor_q)})")
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
                sensor_offset=sensor_t,
                sensor_rotation=sensor_q,
            )
            print(f"[G1] lidar publisher : rclpy -> {handle._lidar_pub.topics}")

    return handle
