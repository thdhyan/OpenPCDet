#!/usr/bin/env python3
"""G1 in a populated warehouse: RTX LiDAR + camera + IMU + decoupled_wbc
locomotion, plus IRA-driven wandering humans and Nova Carters, on Isaac Sim 6.0.

    conda activate isaac
    python scripts/g1_warehouse_sim.py --headless --steps 200

Extends ``g1_rtx_sim.py``'s proven sensor/locomotion pipeline (kept
untouched and still the thing to run for a bare-warehouse-free scene) with:

- a real warehouse environment, via ``g1_sim.warehouse`` (fixes a Nucleus
  path-resolution bug found while building this: ``g1_rtx_sim.py``'s Nucleus
  room loader passed a catalog-relative path straight into
  ``add_reference_to_stage``, which silently failed and fell back to flat
  ground on every run - see that module's docstring for the fix)
- wandering humans and Nova Carters spawned via IRA
  (``g1_sim.ira_actors`` - see its docstring for a hard-coded-frame-budget
  bug found and worked around there)
- an IMU publisher (``g1_sim.rtx_camera.spawn_imu_sensor`` /
  ``attach_imu_publisher``) - there wasn't one before this script

Scene bootstrap order matters: IRA's ``setup_simulation()`` opens the
warehouse **as a new root stage** (a full replace, not a reference), so it
must run before anything else touches the stage. If IRA is disabled
(``--no-ira``) or fails, falls back to a plain warehouse reference (still
using the fixed loader) with no dynamic actors - the pipeline degrades
gracefully rather than aborting.
"""

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser(description="G1 in a populated warehouse.")
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--xr",
    action="store_true",
    help="Launch with isaacsim.exp.base.xr.vr kit (OpenXR/CloudXR mode). "
    "Requires 'python -m isaacteleop.cloudxr' running first as the system "
    "OpenXR compositor, and NVIDIA CloudXR client on Quest 3. "
    "Renders the scene into the VR headset; implies not headless.",
)
parser.add_argument("--steps", type=int, default=0, help="Stop after N steps; 0 runs forever.")
parser.add_argument("--no-ros2", action="store_true")
parser.add_argument("--no-camera", action="store_true", help="Skip the D435 camera (saves render time).")
parser.add_argument("--no-locomotion", action="store_true", help="Skip the decoupled_wbc cmd_vel bridge.")
parser.add_argument(
    "--wbc-mode",
    type=str,
    default="internal",
    choices=["internal", "external"],
    help="Actuation source when locomotion is enabled: 'internal' runs the "
    "in-sim decoupled_wbc ONNX policies; 'external' listens for per-joint "
    "Float64 position targets on /g1/joint/<name> (29 topics, same contract "
    "as the Gazebo wbc_node) and applies them - your own ROS node drives the "
    "robot. Gains are set in-sim in both modes.",
)
parser.add_argument("--no-ira", action="store_true", help="Skip IRA humans/carters; warehouse + G1 only.")
parser.add_argument("--num-humans", type=int, default=2, help="IRA wandering characters. 0 disables the group.")
parser.add_argument("--num-carters", type=int, default=1, help="IRA wandering Nova Carters. 0 disables the group.")
parser.add_argument("--ira-seed", type=int, default=42)
parser.add_argument(
    "--cache-scene",
    type=str,
    default="assets/warehouse_ira_baked.usd",
    help="Load IRA's post-bake scene from here if it exists, instead of "
    "re-downloading the warehouse and re-baking the navmesh every run "
    "(~60-80s). Saved automatically the first time IRA setup succeeds. "
    "Characters/carter sit static at their saved pose - not a substitute "
    "for a real run when actor wander motion matters.",
)
parser.add_argument("--rebake", action="store_true", help="Ignore --cache-scene and force a real IRA setup + navmesh bake, overwriting the cache.")
parser.add_argument(
    "--navmesh-max-frames",
    type=int,
    default=3000,
    help="Raised navmesh-bake poll budget - see g1_sim/ira_actors.py docstring for why the stock 100 fails.",
)
parser.add_argument(
    "--config-dir",
    type=str,
    default="assets/lidar_configs_rotary",
    help="Mid-360 LiDAR profile dir. Default is the ROTARY matched-envelope "
    "profile (Livox_Mid360_R, 128 emitters = 32 el ch x 4 groups, -7.2..+52.2 "
    "deg @10 Hz) - the emitter-state solidState path is dead on this Isaac "
    "Sim build (see docs/RTX_LIDAR_INIT.md). Pass assets/lidar_configs for "
    "the exact-pattern solidState configs (non-functional here).",
)
parser.add_argument("--num-prims", type=int, default=0, help="Use only the first N LiDAR sensor prims.")
parser.add_argument(
    "--keep-carter-cameras",
    action="store_true",
    help="Keep the Nova Carter Hawk/Owl camera rigs. Off by default: they are never "
    "published and their RTX passes OOM-crash an 8 GB GPU (see ira_actors.strip_carter_cameras).",
)
parser.add_argument(
    "--freeze-robot",
    action="store_true",
    help="Disable gravity on the G1 so it holds its spawn (standing) pose without a "
    "balance policy - use with --no-locomotion to test the sensors on an upright robot.",
)
args_cli = parser.parse_args()

# --xr implies rendering into the VR headset (not headless)
if args_cli.xr and args_cli.headless:
    print("[WH] WARNING: --xr and --headless are mutually exclusive; ignoring --headless")
    args_cli.headless = False

SIM_RATE_HZ = 60.0

# XR mode: use the OpenXR/CloudXR experience kit so IsaacSim renders into
# the VR compositor (CloudXR runtime from `python -m isaacteleop.cloudxr`).
_experience = "isaacsim.exp.base.xr.vr" if args_cli.xr else "isaacsim.exp.full"

simulation_app = SimulationApp(
    {
        "headless": args_cli.headless,
        # CPU-side LiDAR return buffer — avoids CUDA race (discussion #685).
        # ALSO passed via extra_args below: several /rtx and /app/sensors
        # settings are read at extension-init time, before SimulationApp's
        # config-dict carb settings land - those dict entries silently no-op
        # (supportMultiTickRate proven no-op live 2026-08-24).
        "/app/sensors/nv/lidar/outputBufferOnGPU": False,
        # Motion BVH MUST be on when multi-tick rendering is active (6.0
        # default), else every RTX LiDAR read fails with "GMO magic number is
        # not correct" and returns nothing ("Multi-tick ... motion BVH is not
        # active. This is not supported.").
        "extra_args": [
            # EXPERIMENT (rotary bring-up 2026-08-25, kept): multi-tick ON is
            # fine for static mounts but was implicated in early bring-up
            # failures; multi-tick OFF renders sensors every sim frame without
            # the motion-BVH path. tickRate is then ignored (60 Hz firing,
            # publisher rate-limits to 10 Hz).
            "--/rtx/hydra/supportMultiTickRate=false",
            "--/renderer/raytracingMotion/enabled=true",
            "--/renderer/raytracingMotion/enableHydraEngineMasking=true",
            "--/renderer/raytracingMotion/enabledForHydraEngines=0,1,2,3",
            "--/app/sensors/nv/lidar/outputBufferOnGPU=false",
        ],
    },
    experience=_experience,
)

# Verify the sensor-critical settings ACTUALLY took effect (they are read at
# extension init; dict-only authoring has silently no-op'd before).
import carb  # noqa: E402


def _log_sensor_settings() -> None:
    s = carb.settings.get_settings()
    print(
        "[WH] sensor settings : "
        f"outputBufferOnGPU={s.get('/app/sensors/nv/lidar/outputBufferOnGPU')} "
        f"multiTickRate={s.get('/rtx/hydra/supportMultiTickRate')} "
        f"motionBVH={s.get('/renderer/raytracingMotion/enabled')}"
    )


"""Rest everything follows."""

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics, Usd

from g1_sim.rtx_camera import (
    OPTICAL_FRAME,
    apply_semantics,
    attach_camera_publishers,
    attach_cmd_vel_subscriber,
    attach_imu_publisher,
    attach_robot_state_publishers,
    spawn_camera,
    spawn_imu_sensor,
)
from g1_sim.rtx_lidar import (
    MID360_POS,
    MID360_QUAT_WXYZ,
    blind_radius,
    spawn_mid360,
)
from g1_sim.warehouse import WAREHOUSE_USD, build_flat_ground, load_environment

ENABLE_ROS2 = not args_cli.no_ros2
ENABLE_LOCOMOTION = not args_cli.no_locomotion
ENABLE_IRA = not args_cli.no_ira
WBC_CONTROL_HZ = 50.0  # decoupled_wbc's trained control rate

G1_USD = REPO / "assets/g1_29dof_sensors.usd"
ROBOT_PRIM = "/World/G1"
# Fallback-only static targets (used when IRA is off/unavailable, so the
# scene still has something LiDAR-visible besides the robot itself).
PEDESTRIANS = [(3.0, 0.0), (4.5, -2.0), (6.0, 2.5)]

GUI_EXTENSIONS = [
    "omni.graph.window.action",
    "omni.graph.window.generic",
    "omni.kit.widget.stage",
]


def enable_extensions() -> None:
    manager = omni.kit.app.get_app().get_extension_manager()

    # Must precede any rclpy import - see g1_rtx_sim.py.
    manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    manager.set_extension_enabled_immediate("isaacsim.sensors.rtx", True)
    # isaacsim.sensors.physics.nodes registers the IsaacReadIMU OmniGraph node.
    # Deliberately NOT enabling the deprecated isaacsim.sensors.physics
    # extension - see spawn_imu_sensor()'s docstring for the Kit-command
    # ambiguity that causes.
    manager.set_extension_enabled_immediate("isaacsim.sensors.physics.nodes", True)

    if ENABLE_IRA:
        import g1_sim.ira_actors as ira_actors

        ira_actors.enable_extension()

    if not args_cli.headless:
        for ext in GUI_EXTENSIONS:
            manager.set_extension_enabled_immediate(ext, True)

    for _ in range(20):
        omni.kit.app.get_app().update()


def build_scene_ira() -> bool:
    """Try IRA: warehouse + humans + Nova Carters, all in one stage-owning
    call. Returns True on success."""
    import g1_sim.ira_actors as ira_actors

    config_path = REPO / "assets/ira_warehouse_config.yaml"
    ira_actors.write_config(
        config_path,
        warehouse_rel=WAREHOUSE_USD.lstrip("/"),
        num_humans=args_cli.num_humans,
        num_carters=args_cli.num_carters,
        seed=args_cli.ira_seed,
        duration_s=200.0,
    )
    print(f"[WH] IRA config      : {config_path}  ({args_cli.num_humans} humans, {args_cli.num_carters} carters)")

    return ira_actors.run_setup_blocking(
        simulation_app, config_path, max_navmesh_frames=args_cli.navmesh_max_frames
    )


def build_scene_fallback(stage) -> None:
    """Warehouse with no dynamic actors - used when --no-ira is passed or
    IRA's setup failed."""
    stage.DefinePrim("/World", "Xform")
    try:
        resolved = load_environment(stage, WAREHOUSE_USD, "/World/Env")
        print(f"[WH] environment     : {resolved}")
    except Exception as e:
        print(f"[WH] warehouse load failed ({e}), using flat ground")
        build_flat_ground(stage)

    for i, (x, y) in enumerate(PEDESTRIANS):
        box = UsdGeom.Cube.Define(stage, f"/World/targets/pedestrian_{i}")
        box.CreateSizeAttr(1.0)
        box.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.875))
        box.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 1.75))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())


def spawn_g1(stage) -> None:
    if not G1_USD.exists():
        raise SystemExit(f"[WH] {G1_USD} not found - run scripts/convert_g1_urdf_to_usd.py first")
    # After IRA's setup, the stage's root layer *is* the remote warehouse USD
    # (a full stage-open, not a reference - see g1_sim/ira_actors.py) and
    # becomes the default edit target. Authoring a reference to a local
    # filesystem path there composed the arc (AddReference returns True,
    # HasAuthoredReferences() is True) but its content silently failed to
    # resolve (pelvis and every other child prim missing) - almost certainly
    # the Nucleus/HTTP-aware asset resolver mishandling a plain absolute
    # POSIX path while anchored to an https:// layer. The session layer is
    # always a local anonymous layer regardless of what the root layer is,
    # so switching the edit target there sidesteps the resolver mismatch
    # entirely; harmless in the non-IRA fallback path too, where the root
    # layer is already local anonymous.
    stage.SetEditTarget(stage.GetSessionLayer())
    robot = stage.DefinePrim(ROBOT_PRIM, "Xform")
    robot.GetReferences().AddReference(str(G1_USD))
    xform = UsdGeom.Xformable(robot)
    translate = next((op for op in xform.GetOrderedXformOps() if "translate" in op.GetOpName()), None)
    if translate is None:
        translate = xform.AddTranslateOp()
    translate.Set(Gf.Vec3d(0.0, 0.0, 0.8))


def main() -> None:
    from isaacsim.core.api import SimulationContext

    enable_extensions()
    _log_sensor_settings()

    cache_path = REPO / args_cli.cache_scene
    ira_ok = False
    if ENABLE_IRA:
        if not args_cli.rebake and cache_path.exists():
            import g1_sim.ira_actors as ira_actors

            ira_actors.load_baked_scene(cache_path)
            ira_ok = True
        else:
            ira_ok = build_scene_ira()
            if not ira_ok:
                print("[WH] IRA setup failed - falling back to warehouse with no dynamic actors")
            else:
                import g1_sim.ira_actors as ira_actors

                ira_actors.save_baked_scene(omni.usd.get_context().get_stage(), cache_path)

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    if not ira_ok:
        build_scene_fallback(stage)
    elif not args_cli.keep_carter_cameras:
        import g1_sim.ira_actors as ira_actors

        stripped = ira_actors.strip_carter_cameras(stage)
        print(f"[WH] carter cameras  : {stripped} deactivated (keep with --keep-carter-cameras)")

    # G1 always goes on top, regardless of which path built the environment.
    spawn_g1(stage)
    # A few frames of margin for the reference's composition to settle before
    # anything queries child prims (Articulation's pelvis lookup below).
    for _ in range(10):
        simulation_app.update()
    pelvis_prim = stage.GetPrimAtPath(f"{ROBOT_PRIM}/pelvis")
    print(f"[WH] G1 prim valid   : {stage.GetPrimAtPath(ROBOT_PRIM).IsValid()}  pelvis valid: {pelvis_prim.IsValid()}")

    if args_cli.freeze_robot:
        # Disable gravity on every G1 rigid body *before* sim.reset() initializes
        # the physics view, so a policy-less robot holds its standing spawn pose
        # (otherwise it collapses and the LiDAR ends up buried in the floor).
        from pxr import PhysxSchema, Usd, UsdPhysics

        frozen = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
                frozen += 1
        print(f"[WH] freeze-robot    : gravity disabled on {frozen} bodies (upright, no policy)")

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / SIM_RATE_HZ,
        rendering_dt=1.0 / SIM_RATE_HZ,
    )

    from isaacsim.core.prims import Articulation

    robot_articulation = Articulation(f"{ROBOT_PRIM}/pelvis", name="g1")
    sim.reset()
    print(f"[WH] articulation    : {robot_articulation.num_dof} DOF")
    print(f"[WH] IRA actors      : {'yes' if ira_ok else 'no'}")

    wbc_bridge = None
    joint_cmd_sub = None
    if ENABLE_LOCOMOTION:
        from g1_sim.wbc_bridge import (
            ARM_JOINTS,
            ARM_KD,
            ARM_KP,
            ALL_JOINTS,
            KD,
            KP,
            LEG_WAIST_JOINTS,
        )

        dof_names = set(robot_articulation.dof_names or [])
        missing = [j for j in ALL_JOINTS if j not in dof_names]
        if missing:
            print(f"[WH] WBC bridge      : DISABLED - USD is missing joints {missing}")
        else:
            # Trained per-joint PD gains are required in BOTH modes - the
            # uniform URDF defaults collapse the robot (see Plan.md history).
            robot_articulation.set_gains(kps=KP[None, :], kds=KD[None, :], joint_names=LEG_WAIST_JOINTS)
            robot_articulation.set_gains(
                kps=np.full((1, len(ARM_JOINTS)), ARM_KP, dtype=np.float32),
                kds=np.full((1, len(ARM_JOINTS)), ARM_KD, dtype=np.float32),
                joint_names=ARM_JOINTS,
            )
            if args_cli.wbc_mode == "external":
                import rclpy

                from g1_sim.joint_cmd_sub import JointCommandSubscriber

                # rclpy may not be initialised yet - the publisher block's
                # own init comes later, so guard here.
                if not rclpy.ok():
                    rclpy.init()
                _wbc_ros_node = rclpy.create_node("g1_joint_cmd_listener")
                joint_cmd_sub = JointCommandSubscriber(_wbc_ros_node, ALL_JOINTS)
                print(
                    "[WH] WBC mode        : EXTERNAL - waiting for /g1/joint/<name> "
                    f"Float64 targets on {len(ALL_JOINTS)} joints (robot holds default "
                    "pose until every joint receives one)"
                )
            else:
                from g1_sim.wbc_bridge import WbcBridge

                wbc_bridge = WbcBridge(
                    REPO / "assets/policy/GR00T-WholeBodyControl-Balance.onnx",
                    REPO / "assets/policy/GR00T-WholeBodyControl-Walk.onnx",
                )
                print("[WH] WBC bridge      : loaded (decoupled_wbc Balance/Walk, gains overridden)")

    # Mount ON the mid360_link prim, which already carries the URDF's
    # torso->sensor pose (translation + 180deg roll). The sensor therefore uses
    # an identity local transform and its returns come out in the exact frame
    # PublishTransformTree emits as `mid360_link`, so the published frame_id and
    # the point origin are the same prim. Mounting on torso_link with a
    # hand-authored quat put the points in a *different* frame than the one they
    # were published in, applying the roll twice and dropping the whole cloud
    # below the ground (fixed 2026-08-11 - see rtx_lidar.py MID360_POS note).
    mount = f"{ROBOT_PRIM}/mid360_link"
    if not omni.usd.get_context().get_stage().GetPrimAtPath(mount).IsValid():
        raise SystemExit(f"[WH] mount prim {mount} missing from the USD")

    prim_paths = spawn_mid360(
        mount,
        config_dir=REPO / args_cli.config_dir,
        # +5cm WORLD-Z clearance (live-swept 2026-08-25): at the stock URDF
        # position the OmniLidar origin is buried inside the G1 head mesh -
        # every ray self-intersects and the GMO buffer never forms (0 points
        # + "GMO magic number" spam). Sweep results: dz=-1cm upper band only,
        # +3cm missing bottom 6 deg, +5cm FULL -7.3..+52.3 cone at 442k
        # pts/render, +7cm identical. NOTE: rays now originate 5cm above the
        # mid360_link origin while the published frame_id stays mid360_link -
        # negligible for detection; update the URDF mid360_joint if exact
        # frame-origin parity ever matters.
        translation=(0.0, 0.0, -0.05),  # -5cm: flipped lidar (identity orient) → local +z = world-UP, so negative moves UP
        # torso_link, so the link frame has +z pointing world-down (dome faces
        # scene). The OmniLidar child prims must carry the same 180° roll so
        # their emitter elevation angles are authored in a frame that matches
        # the physical mounting - without this, the prim's +z = link's +z =
        # world-down, but the engine interprets elevation +θ as rays above the
        # prim's +z, which points into the floor rather than the scene.
        # MID360_QUAT_WXYZ = (cos(π/2), sin(π/2), 0, 0) = 180° roll about X.
        orientation=MID360_QUAT_WXYZ,
    )
    # Debug: print actual world position of the first sensor prim
    if prim_paths:
        from pxr import UsdGeom as _UsdGeom
        _stage = omni.usd.get_context().get_stage()
        _sp = _stage.GetPrimAtPath(prim_paths[0])
        if _sp.IsValid():
            _xf = _UsdGeom.Xformable(_sp)
            _wm = _xf.ComputeLocalToWorldTransform(0.0)
            _wp = _wm.ExtractTranslation()
            print(f"[WH] sensor world pos: ({_wp[0]:.4f}, {_wp[1]:.4f}, {_wp[2]:.4f})")
    if args_cli.num_prims and args_cli.num_prims < len(prim_paths):
        prim_paths = prim_paths[: args_cli.num_prims]
    print(f"[WH] lidar prims     : {len(prim_paths)}")


    mount_height = 0.8 + MID360_POS[2]
    print(f"[WH] mount height    : {mount_height:.2f} m")
    print(f"[WH] blind radius    : {blind_radius(mount_height):.2f} m")

    publisher = None
    rgbd_publisher = None
    cmd_vel_graph_path = None
    if ENABLE_ROS2:
        # NOT attach_ros2_publishers() here: that OG-graph path's
        # ROS2RtxLidarHelper advertises /livox/mid360/points but never
        # actually emits on this Isaac Sim build (see rtx_publisher.py's
        # docstring - RtxLidarPublisher below is what really publishes it).
        # Calling it anyway was pure waste: 4 extra IsaacCreateRenderProduct
        # nodes/render-products for a graph whose output nothing reads,
        # competing for GPU memory with the 4 render products
        # RtxLidarPublisher itself creates, the camera's, and (new)
        # RgbdPointCloudPublisher's - live-verified 2026-08-11 as a
        # contributing cause of a mid-run crash
        # (AnnotatorRegistryError: "not attached to any render products",
        # consistent with Replicator evicting a render product under
        # memory pressure on an 8 GB GPU).
        state_graph = attach_robot_state_publishers(ROBOT_PRIM)
        print(f"[WH] state graph     : {state_graph}  (/tf, /g1/joint_states, /clock)")

        if ira_ok:
            import g1_sim.ira_actors as ira_actors

            # Re-enabled 2026-08-10: briefly disabled on the theory that this
            # graph's "[PoseTree] eInvalid" spam was uniquely responsible for
            # a stalled boot, but the same spam (just "parent .../World"
            # eInvalid, no human targets) kept firing with this disabled -
            # so it isn't the (sole) cause and disabling it bought nothing.
            # The eInvalid warning itself is still unexplained/unfixed - see
            # FUTURE_STEPS.md - but it doesn't block this from being useful.
            actor_prims = ira_actors.discover_actor_prims(stage)
            actors_graph = ira_actors.attach_actor_tf_publishers(actor_prims)
            print(f"[WH] actors tf graph : {actors_graph}  ({len(actor_prims)} actors: {actor_prims})")

            carter_prims = ira_actors.discover_prims_at(stage, "/World/Robots/carters")
            if carter_prims:
                # Publishes IRA's own existing carter IMU (read-only) -
                # see find_imu_prim()'s docstring for why we don't create one.
                carter_imu_graphs = ira_actors.attach_carter_imu_publishers(stage, carter_prims)
                print(f"[WH] carter imus     : {len(carter_imu_graphs)}/{len(carter_prims)} found -> /carter_N/imu")

        imu_prim = spawn_imu_sensor(f"{ROBOT_PRIM}/torso_link/imu_in_torso")
        imu_graph = attach_imu_publisher(imu_prim)
        print(f"[WH] imu graph       : {imu_graph}  (/g1/imu, prim={imu_prim})")

        if wbc_bridge is not None:
            cmd_vel_graph_path = attach_cmd_vel_subscriber()
            print(f"[WH] cmd_vel graph   : {cmd_vel_graph_path}  (/g1/cmd_vel)")

        if not args_cli.no_camera:
            camera_prim = spawn_camera(f"{ROBOT_PRIM}/torso_link")
            cam_graph = attach_camera_publishers(camera_prim)
            print(f"[WH] camera graph    : {cam_graph}")
            print("[WH] camera topics   : /g1/camera/{rgb,depth,semantic,camera_info}")

            labels = {ROBOT_PRIM: "robot"}
            if not ira_ok:
                labels.update({f"/World/targets/pedestrian_{i}": "pedestrian" for i in range(len(PEDESTRIANS))})
            print(f"[WH] semantics       : {apply_semantics(labels)} prims labelled")

        import rclpy

        from g1_sim.rtx_publisher import RtxLidarPublisher

        if not rclpy.ok():  # external-WBC mode already initialised it
            rclpy.init()
        publisher = RtxLidarPublisher(
            prim_paths, topic="/livox/mid360/points", publish_rate=10.0,
            # Verification pass 2026-08-11: publish the full accumulated
            # sweep instead of the real-device-matching 20k cap, so RViz
            # shows everything the non-repetitive pattern actually
            # collected per window. Real-device-rate subsampling can be
            # restored once the coverage itself is confirmed correct.
            max_points=200_000,
        )
        print(f"[WH] publisher       : rclpy (annotator, raw per-prim) -> {publisher.topics}")

        if not args_cli.no_camera:
            from g1_sim.rgbd_publisher import RgbdPointCloudPublisher

            rgbd_publisher = RgbdPointCloudPublisher(camera_prim, frame_id=OPTICAL_FRAME)
            print(f"[WH] rgbd publisher  : /g1/camera/depth/color/points (frame={OPTICAL_FRAME})")
    else:
        print("[WH] ROS2 disabled")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print(f"[WH] timeline        : playing={timeline.is_playing()}")
    print("[WH] running\n")

    step = 0
    scans = 0
    last_points = 0
    wbc_updates = 0
    wbc_control_period = 1.0 / WBC_CONTROL_HZ
    wbc_next_time = 0.0
    try:
        while simulation_app.is_running():
            sim.step(render=True)
            step += 1

            if joint_cmd_sub is not None and sim.current_time >= wbc_next_time:
                # EXTERNAL WBC: apply whatever the outside ROS node sent.
                wbc_next_time = sim.current_time + wbc_control_period
                joint_cmd_sub.spin_once()
                targets = joint_cmd_sub.get_targets()
                if targets is not None:
                    robot_articulation.set_joint_position_targets(
                        targets[None, :].astype(np.float32), joint_names=ALL_JOINTS
                    )
                    wbc_updates += 1
                    if wbc_updates % 50 == 0:
                        print(
                            f"[WH] ext-wbc updates={wbc_updates}  "
                            f"pelvis_z={robot_articulation.get_world_poses()[0][0][2]:.3f}"
                        )

            if wbc_bridge is not None and sim.current_time >= wbc_next_time:
                from g1_sim.wbc_bridge import quat_rotate_inverse

                wbc_next_time = sim.current_time + wbc_control_period

                cmd_vx, cmd_vy, cmd_wz = 0.0, 0.0, 0.0
                if cmd_vel_graph_path is not None:
                    from g1_sim.action_graph import read_cmd_vel

                    cmd_vx, cmd_vy, cmd_wz = read_cmd_vel(graph_path=cmd_vel_graph_path)

                qpos_all = np.asarray(robot_articulation.get_joint_positions(joint_names=ALL_JOINTS))[0]
                qvel_all = np.asarray(robot_articulation.get_joint_velocities(joint_names=ALL_JOINTS))[0]
                _, quat_wxyz = robot_articulation.get_world_poses()
                quat_wxyz = np.asarray(quat_wxyz)[0]
                ang_vel_world = np.asarray(robot_articulation.get_angular_velocities())[0]
                ang_vel_body = quat_rotate_inverse(quat_wxyz, ang_vel_world)

                target = wbc_bridge.step(qpos_all, qvel_all, quat_wxyz, ang_vel_body, cmd_vx, cmd_vy, cmd_wz)
                robot_articulation.set_joint_position_targets(
                    target[None, :].astype(np.float32), joint_names=LEG_WAIST_JOINTS
                )
                if wbc_updates == 0:
                    # Shoulders stay at the default URDF zero pose (arms
                    # hanging straight down) per 2026-08-11 request - only
                    # bend the elbows (index 3 left, 10 right in ARM_JOINTS)
                    # so the forearms/palms swing forward into the D435's
                    # downward-pitched FOV instead of raising the whole arm.
                    # +1.2 bent the forearms backward instead of forward
                    # (live visual check 2026-08-11) - negated.
                    arm_pose = np.zeros((1, len(ARM_JOINTS)), dtype=np.float32)
                    arm_pose[0, 3] = -1.2    # left_elbow_joint
                    arm_pose[0, 10] = -1.2   # right_elbow_joint
                    robot_articulation.set_joint_position_targets(
                        arm_pose, joint_names=ARM_JOINTS
                    )
                wbc_updates += 1
                if wbc_updates % 50 == 0:
                    print(
                        f"[WH] wbc cmd=({cmd_vx:.2f},{cmd_vy:.2f},{cmd_wz:.2f})  "
                        f"updates={wbc_updates}  pelvis_z={robot_articulation.get_world_poses()[0][0][2]:.3f}"
                    )

            if publisher is not None:
                publisher.accumulate()
                sent = publisher.publish(sim.current_time)
                if sent:
                    scans += 1
                    last_points = sent
                    if scans % 10 == 0:
                        print(f"[WH] lidar per-prim diagnostics (scan {scans}, published {sent} pts):")
                        print(publisher.diagnostics_str())
                publisher.spin_once()

            if rgbd_publisher is not None:
                rgbd_publisher.publish(sim.current_time)

            if step % 100 == 0:
                print(f"[WH] step {step:>6}  scans {scans}  points {last_points}")
            if args_cli.steps and step >= args_cli.steps:
                break
    except KeyboardInterrupt:
        print("\n[WH] interrupted")
    finally:
        if publisher is not None:
            import rclpy

            publisher.destroy()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
